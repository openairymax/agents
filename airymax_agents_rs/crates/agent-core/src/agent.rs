// SPDX-FileCopyrightText: 2026 SPHARX Ltd.
// SPDX-License-Identifier: AGPL-3.0-or-later OR Apache-2.0
//! Agent trait 与相关类型，对齐 Python `AirymaxAgent` 与
//! `airymax-agent-standard.md` §3.2.

use crate::context::AgentContext;
use crate::error::Result;
use crate::status::AgentStatus;
use serde::{Deserialize, Serialize};

/// Agent 身份标识。
///
/// 与 Python 侧 `contract.json` 中的 `agent_id`/`role` 对齐
/// （`airymax-agent-standard.md` §9：格式 `X_v1`）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentIdentity {
    /// 全仓唯一的 agent_id（如 `architect_v1`）
    pub agent_id: String,
    /// 角色（如 `architect`）
    pub role: String,
    /// 契约版本（如 `1.0.0`）
    pub version: String,
}

/// Agent 输入载荷。
///
/// 当前为字符串透传，后续可扩展为带 schema 的强类型。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInput {
    pub data: String,
}

impl AgentInput {
    pub fn new(data: impl Into<String>) -> Self {
        Self { data: data.into() }
    }
}

/// Agent 执行结果。
///
/// 与 Python 侧 `openlab.core.agent.TaskResult` 字段对齐：
/// success/output/error/metrics。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TaskResult {
    pub success: bool,
    pub output: Option<String>,
    pub error: Option<String>,
    #[serde(default)]
    pub metrics: serde_json::Value,
}

impl TaskResult {
    pub fn ok(output: impl Into<String>) -> Self {
        Self {
            success: true,
            output: Some(output.into()),
            error: None,
            metrics: serde_json::Value::Null,
        }
    }

    pub fn fail(error: impl Into<String>) -> Self {
        Self {
            success: false,
            output: None,
            error: Some(error.into()),
            metrics: serde_json::Value::Null,
        }
    }
}

/// Agent 核心 trait，对齐 `airymax-agent-standard.md` §3.2 Rust 定义。
///
/// 所有 Rust 实现 Agent 须实现此 trait。
#[async_trait::async_trait]
pub trait Agent: Send + Sync {
    /// 返回 Agent 身份（agent_id / role / version）。
    fn identity(&self) -> &AgentIdentity;

    /// 初始化 Agent（加载契约、提示词、建立 LLM 客户端等）。
    async fn initialize(&mut self) -> Result<()>;

    /// 执行一次任务，返回 `TaskResult`。
    async fn execute(&mut self, input: AgentInput, ctx: &AgentContext) -> Result<TaskResult>;

    /// 终止 Agent，释放资源。
    async fn terminate(&mut self) -> Result<()>;

    /// 当前生命周期状态。
    fn status(&self) -> AgentStatus;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_result_ok_fail() {
        let ok = TaskResult::ok("done");
        assert!(ok.success);
        assert_eq!(ok.output.as_deref(), Some("done"));
        assert!(ok.error.is_none());

        let err = TaskResult::fail("boom");
        assert!(!err.success);
        assert_eq!(err.error.as_deref(), Some("boom"));
        assert!(err.output.is_none());
    }

    #[test]
    fn agent_input_new() {
        let i = AgentInput::new("hello");
        assert_eq!(i.data, "hello");
    }

    #[test]
    fn identity_serde() {
        let id = AgentIdentity {
            agent_id: "architect_v1".to_string(),
            role: "architect".to_string(),
            version: "1.0.0".to_string(),
        };
        let s = serde_json::to_string(&id).unwrap();
        let v: AgentIdentity = serde_json::from_str(&s).unwrap();
        assert_eq!(v, id);
    }
}
