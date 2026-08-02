// SPDX-FileCopyrightText: 2026 SPHARX Ltd.
// SPDX-License-Identifier: AGPL-3.0-or-later OR Apache-2.0
//! CodingAgent 实现：基于 `agent-core::Agent` trait 的 LLM 驱动 Coding Agent。
//!
//! 与 Python `airymax_agents/coding/agent.py` 对齐：
//! - `initialize()` 从 `contract.json` 加载契约，从 `prompts/system.md` 加载系统提示词
//! - `execute()` 构建 [system, user] 消息并调用 [`LlmClient`]，
//!   返回 LLM 输出与 metrics（model / elapsed_ms / tokens）
//! - `terminate()` 重置状态

use agent_contract::Contract;
use agent_core::{
    Agent, AgentContext, AgentError, AgentIdentity, AgentInput, AgentStatus, ChatMessage,
    LlmClient, Result, TaskResult, make_llm_client,
};
use std::path::PathBuf;
use std::sync::Mutex;

/// 编译期内置的默认系统提示词（`prompts/system.md` 缺失时的 fallback）。
const FALLBACK_SYSTEM_PROMPT: &str = include_str!("../prompts/system.md");

/// Airymax Coding Agent — LLM 驱动的代码生成 Agent。
pub struct CodingAgent {
    identity: AgentIdentity,
    contract_path: PathBuf,
    contract: Option<Contract>,
    system_prompt: String,
    llm: Box<dyn LlmClient>,
    status: Mutex<AgentStatus>,
}

impl CodingAgent {
    /// 使用默认 LLM 客户端（环境感知：`$OPENAI_API_KEY` → 真实，否则 mock）。
    pub fn new(contract_path: PathBuf) -> Self {
        Self::with_llm(contract_path, make_llm_client("gpt-4o-mini", false))
    }

    /// 显式指定 LLM 客户端（测试 / 基准对比用）。
    pub fn with_llm(contract_path: PathBuf, llm: Box<dyn LlmClient>) -> Self {
        let identity = AgentIdentity {
            // 占位值，initialize 后由契约覆盖
            agent_id: "coding_v1".to_string(),
            role: "coding".to_string(),
            version: "0.1.0".to_string(),
        };
        Self {
            identity,
            contract_path,
            contract: None,
            system_prompt: FALLBACK_SYSTEM_PROMPT.to_string(),
            llm,
            status: Mutex::new(AgentStatus::Created),
        }
    }

    /// 暴露已加载契约的不可变引用（测试用）。
    pub fn contract(&self) -> Option<&Contract> {
        self.contract.as_ref()
    }

    /// 从 `<contract_dir>/prompts/system.md` 加载系统提示词；
    /// 文件缺失时回退到编译期内置提示词。
    fn load_system_prompt(&self) -> String {
        if let Some(dir) = self.contract_path.parent() {
            let prompt_path = dir.join("prompts").join("system.md");
            if prompt_path.is_file() {
                if let Ok(s) = std::fs::read_to_string(&prompt_path) {
                    let t = s.trim();
                    if !t.is_empty() {
                        return t.to_string();
                    }
                }
            }
        }
        FALLBACK_SYSTEM_PROMPT.to_string()
    }
}

#[async_trait::async_trait]
impl Agent for CodingAgent {
    fn identity(&self) -> &AgentIdentity {
        &self.identity
    }

    async fn initialize(&mut self) -> Result<()> {
        let contract = Contract::from_file(&self.contract_path).map_err(|e| {
            AgentError::InvalidContract(format!("load {}: {e}", self.contract_path.display()))
        })?;
        // 用契约中的 agent_id/role/version 覆盖身份
        self.identity = AgentIdentity {
            agent_id: contract.agent_id.clone(),
            role: contract.role.clone(),
            version: contract.version.clone(),
        };
        self.system_prompt = self.load_system_prompt();
        self.contract = Some(contract);
        *self.status.lock().unwrap() = AgentStatus::Running;
        tracing::info!(
            "CodingAgent initialized: agent_id={}",
            self.identity.agent_id
        );
        Ok(())
    }

    async fn execute(&mut self, input: AgentInput, _ctx: &AgentContext) -> Result<TaskResult> {
        let status = *self.status.lock().unwrap();
        if status != AgentStatus::Running {
            return Err(AgentError::StateError(format!(
                "agent not running (status={:?})",
                status
            )));
        }
        tracing::info!(
            "CodingAgent.execute called: input_len={}",
            input.data.len()
        );

        // 模型：契约 models.system2（t2 主思考，与 Python LLMAgent._model_system2 对齐）
        let model = self
            .contract
            .as_ref()
            .and_then(|c| Some(c.models.system2.clone()))
            .filter(|m| !m.is_empty())
            .unwrap_or_else(|| "gpt-4o-mini".to_string());

        let messages = vec![
            ChatMessage::system(self.system_prompt.clone()),
            ChatMessage::user(input.data.clone()),
        ];

        let resp = self
            .llm
            .chat(&messages, &model)
            .await
            .map_err(|e| AgentError::Llm(format!("execute failed: {e}")))?;

        let mut metrics = serde_json::Map::new();
        metrics.insert("model".to_string(), serde_json::Value::String(resp.model.clone()));
        metrics.insert(
            "elapsed_ms".to_string(),
            serde_json::Value::Number(resp.elapsed_ms.into()),
        );
        metrics.insert(
            "tokens".to_string(),
            serde_json::Value::Number(resp.total_tokens.unwrap_or_default().into()),
        );

        let mut result = TaskResult::ok(resp.content);
        result.metrics = serde_json::Value::Object(metrics);
        Ok(result)
    }

    async fn terminate(&mut self) -> Result<()> {
        *self.status.lock().unwrap() = AgentStatus::Terminated;
        tracing::info!(
            "CodingAgent terminated: agent_id={}",
            self.identity.agent_id
        );
        Ok(())
    }

    fn status(&self) -> AgentStatus {
        *self.status.lock().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_core::MockLlmClient;

    fn dummy_contract_path() -> PathBuf {
        // 指向 crate 内置的 contract.json（位于 crate 根目录）
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("contract.json")
    }

    #[tokio::test]
    async fn lifecycle_initialized_to_terminated() {
        let mut agent = CodingAgent::new(dummy_contract_path());
        assert_eq!(agent.status(), AgentStatus::Created);
        agent.initialize().await.expect("initialize");
        assert_eq!(agent.status(), AgentStatus::Running);
        assert_eq!(agent.identity().agent_id, "coding_rs_v1");
        assert!(agent.contract().is_some());

        let ctx = AgentContext::new("coding-001", "coding");
        let r = agent
            .execute(AgentInput::new("写一个斐波那契函数"), &ctx)
            .await
            .expect("execute");
        assert!(r.success);
        let out = r.output.unwrap();
        assert!(out.contains("Mock"), "output: {out}");
        // metrics 含 model / elapsed_ms / tokens
        let m = r.metrics.as_object().expect("metrics object");
        assert!(m.contains_key("model"));
        assert!(m.contains_key("elapsed_ms"));
        assert!(m.contains_key("tokens"));

        agent.terminate().await.expect("terminate");
        assert_eq!(agent.status(), AgentStatus::Terminated);
    }

    #[tokio::test]
    async fn execute_rejected_when_not_running() {
        let mut agent = CodingAgent::new(dummy_contract_path());
        let ctx = AgentContext::new("coding-002", "coding");
        let err = agent.execute(AgentInput::new("x"), &ctx).await.unwrap_err();
        assert!(matches!(err, AgentError::StateError(_)));
    }

    #[tokio::test]
    async fn missing_contract_file_errors() {
        let mut agent = CodingAgent::new(PathBuf::from("/nonexistent/contract.json"));
        let err = agent.initialize().await.unwrap_err();
        assert!(matches!(err, AgentError::InvalidContract(_)));
    }

    #[tokio::test]
    async fn custom_llm_injection() {
        let mut agent = CodingAgent::with_llm(
            dummy_contract_path(),
            Box::new(MockLlmClient::new("mock-model")),
        );
        agent.initialize().await.expect("initialize");
        let ctx = AgentContext::new("coding-003", "coding");
        let r = agent
            .execute(AgentInput::new("hello"), &ctx)
            .await
            .expect("execute");
        assert!(r.success);
    }
}
