// SPDX-FileCopyrightText: 2026 SPHARX Ltd.
// SPDX-License-Identifier: AGPL-3.0-or-later OR Apache-2.0
//! LLM 客户端基础设施 — 与 Python `openlab.core.llm` 对齐。
//!
//! - [`LlmClient`]       — 异步 chat trait
//! - [`OpenAiLlmClient`] — 真实客户端（OpenAI Chat Completions 兼容协议，reqwest + rustls）
//! - [`MockLlmClient`]   — 离线确定性 mock（无 API key 亦可跑通，含 50ms 模拟延迟）
//! - [`make_llm_client`] — 环境感知工厂（`$OPENAI_API_KEY` 存在 → 真实，否则 → mock）
//!
//! 设计原则 (Simplicity is beauty)：行为与 `openlab.core.llm` 严格对齐，
//! 保证 Python / Rust 两端在基准测试中 LLM 侧开销一致，差异即框架开销。

use crate::error::{AgentError, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// 一条聊天消息（OpenAI `messages` 数组元素）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// `"system"` | `"user"` | `"assistant"` | `"tool"`
    pub role: String,
    pub content: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: Some(content.into()),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: Some(content.into()),
        }
    }
}

/// LLM 调用结果（OpenAI 兼容响应子集 + 客户端补填耗时）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    /// 最终文本内容。
    pub content: String,
    /// 实际使用的模型名。
    pub model: String,
    /// 总 token 数（mock 或响应缺失时为 None）。
    pub total_tokens: Option<u64>,
    /// 请求端到端耗时（毫秒）。
    pub elapsed_ms: u64,
}

/// LLM 客户端 trait，与 Python `openlab.core.llm.LLMClient.chat` 对齐。
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// 调用一次 chat，返回最终文本内容。
    async fn chat(&self, messages: &[ChatMessage], model: &str) -> Result<LlmResponse>;
}

/// OpenAI 兼容的真实异步 LLM 客户端。
///
/// 配置优先级：显式参数 > 环境变量 > 默认值。
/// - `api_key`  : `$OPENAI_API_KEY`
/// - `base_url` : `$OPENAI_BASE_URL` 或 `https://api.openai.com/v1`
/// - `model`    : 调用时传入，未传用 `default_model`
pub struct OpenAiLlmClient {
    api_key: String,
    base_url: String,
    default_model: String,
    http: reqwest::Client,
}

impl OpenAiLlmClient {
    pub fn new(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        default_model: impl Into<String>,
    ) -> Self {
        let mut base = base_url.into();
        while base.ends_with('/') {
            base.pop();
        }
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .expect("reqwest client build");
        Self {
            api_key: api_key.into(),
            base_url: base,
            default_model: default_model.into(),
            http,
        }
    }
}

#[async_trait]
impl LlmClient for OpenAiLlmClient {
    async fn chat(&self, messages: &[ChatMessage], model: &str) -> Result<LlmResponse> {
        let started = Instant::now();
        let url = format!("{}/chat/completions", self.base_url);
        // model 为空时回退到构造时默认（与 Python LLMClient.chat(model=None) 对齐）
        let used_model = if model.is_empty() { &self.default_model } else { model };
        let body = serde_json::json!({
            "model": used_model,
            "messages": messages,
            "temperature": 0.7,
        });

        let resp = self
            .http
            .post(&url)
            .header("Content-Type", "application/json")
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| AgentError::Llm(format!("LLM request failed: {e}")))?;

        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| AgentError::Llm(format!("LLM response read failed: {e}")))?;
        if !status.is_success() {
            let snippet = text.chars().take(500).collect::<String>();
            return Err(AgentError::Llm(format!(
                "LLM request failed (HTTP {status}): {snippet}"
            )));
        }

        let v: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| AgentError::Llm(format!("LLM bad JSON response: {e}")))?;
        let content = v["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        let resp_model = v["model"].as_str().unwrap_or(used_model).to_string();
        let total_tokens = v["usage"]["total_tokens"].as_u64();

        Ok(LlmResponse {
            content,
            model: resp_model,
            total_tokens,
            elapsed_ms: started.elapsed().as_millis() as u64,
        })
    }
}

impl Default for OpenAiLlmClient {
    fn default() -> Self {
        Self::new(
            std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".into()),
            "gpt-4o-mini",
        )
    }
}

/// 离线确定性 mock LLM 客户端。
///
/// 行为对齐 Python `MockLLMClient`：
/// - 模拟 50ms 网络延迟（async 流程更真实）
/// - 按 system 首行关键词差异化角色输出（architect / coding）
/// - 否则返回基于 user 输入的模板化文本
pub struct MockLlmClient {
    default_model: String,
}

impl MockLlmClient {
    pub fn new(default_model: impl Into<String>) -> Self {
        Self {
            default_model: default_model.into(),
        }
    }
}

impl Default for MockLlmClient {
    fn default() -> Self {
        Self::new("mock-model")
    }
}

#[async_trait]
impl LlmClient for MockLlmClient {
    async fn chat(&self, messages: &[ChatMessage], model: &str) -> Result<LlmResponse> {
        let started = Instant::now();
        // 模拟网络延迟，让 async 流程更真实（与 Python MockLLMClient 对齐）
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let mut system_text = "";
        let mut last_user = "";
        for m in messages {
            match m.role.as_str() {
                "system" => system_text = m.content.as_deref().unwrap_or_default(),
                "user" => last_user = m.content.as_deref().unwrap_or_default(),
                _ => {}
            }
        }

        // 按 system 首行判定角色，让多 Agent 输出差异化（与 Python 对齐）
        let first_line = system_text.lines().next().unwrap_or_default().to_lowercase();
        let is_architect = first_line.contains("架构师") || first_line.contains("architect");
        let is_coding = first_line.contains("编码") || first_line.contains("coding");

        let preview = last_user.chars().take(60).collect::<String>().replace('\n', " ");
        let content = if is_architect {
            format!(
                "【Mock 架构设计】\n## 系统架构\n- 前端：React + TypeScript\n- 后端：FastAPI + SQLAlchemy\n- 数据库：PostgreSQL\n- 缓存：Redis\n## 模块划分\n1. 认证模块\n2. 任务模块\n3. 通知模块\n## 关键决策\n- 采用 RESTful API\n- 异步任务用 Celery\n## 依据需求\n{preview}\n（Mock 生成，仅用于验证 Agent 执行链路）"
            )
        } else if is_coding {
            format!(
                "【Mock 代码实现】\n```python\ndef solve():\n    \"\"\"针对需求：{preview}\"\"\"\n    # TODO: 由真实 LLM 生成实现\n    return None\n```\n（Mock 生成，仅用于验证 Agent 执行链路）"
            )
        } else {
            format!(
                "【Mock 响应】已收到你的输入并完成处理。\n输入摘要: {preview}\n这是一个离线 mock 响应，用于在无 LLM API key 时验证 Agent 流程。\n设置 OPENAI_API_KEY 后将自动切换到真实 LLM。"
            )
        };

        let used_model = if model.is_empty() {
            self.default_model.clone()
        } else {
            model.to_string()
        };
        let total_tokens = Some(50u64 + (content.len() as u64 / 3));
        Ok(LlmResponse {
            content,
            model: used_model,
            total_tokens,
            elapsed_ms: started.elapsed().as_millis() as u64,
        })
    }
}

/// 环境感知工厂，对齐 Python `make_llm_client`。
///
/// - `force_mock`             → 始终返回 [`MockLlmClient`]
/// - 环境变量 `OPENAI_API_KEY` 存在 → [`OpenAiLlmClient`]（真实）
/// - 否则                        → [`MockLlmClient`]（离线）
pub fn make_llm_client(default_model: &str, force_mock: bool) -> Box<dyn LlmClient> {
    if force_mock {
        return Box::new(MockLlmClient::new(default_model));
    }
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    if !api_key.trim().is_empty() {
        return Box::new(OpenAiLlmClient::default());
    }
    Box::new(MockLlmClient::new(default_model))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_chat_returns_content_and_model() {
        let client = MockLlmClient::new("mock-model");
        let msgs = vec![
            ChatMessage::system("你是 coding Agent"),
            ChatMessage::user("写一个斐波那契函数"),
        ];
        let resp = client.chat(&msgs, "mock-model").await.expect("chat");
        assert!(resp.content.contains("Mock"));
        assert_eq!(resp.model, "mock-model");
        assert!(resp.total_tokens.is_some());
        // 含 50ms 模拟延迟
        assert!(resp.elapsed_ms >= 45, "elapsed={}", resp.elapsed_ms);
    }

    #[tokio::test]
    async fn mock_chat_role_detection() {
        let client = MockLlmClient::new("mock-model");
        let arch = client
            .chat(&[ChatMessage::system("你是架构师 Architect"), ChatMessage::user("设计系统")], "m")
            .await
            .expect("chat");
        assert!(arch.content.contains("架构设计"), "got: {}", arch.content);
        let coding = client
            .chat(&[ChatMessage::system("你是编码 Coding Agent"), ChatMessage::user("生成代码")], "m")
            .await
            .expect("chat");
        assert!(coding.content.contains("代码实现"), "got: {}", coding.content);
    }

    #[test]
    fn chat_message_builders() {
        let s = ChatMessage::system("sys");
        assert_eq!(s.role, "system");
        assert_eq!(s.content.as_deref(), Some("sys"));
        let u = ChatMessage::user("usr");
        assert_eq!(u.role, "user");
    }
}
