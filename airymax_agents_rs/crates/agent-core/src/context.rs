// SPDX-FileCopyrightText: 2026 SPHARX Ltd.
// SPDX-License-Identifier: AGPL-3.0-or-later OR Apache-2.0
//! Agent 执行上下文，与 Python 侧 `AgentContext` 字段对齐
//! （见 airymax-agent-standard.md §9 跨语言映射）。

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Agent 执行上下文。
///
/// 携带一次 `Agent::execute` 调用所需的环境信息：
/// agent_id、task_id、会话元数据、共享变量等。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentContext {
    /// 调用方分配的 Agent ID（与 mem_d/agent_d 中的运行时 ID 对齐）
    pub agent_id: String,
    /// 任务 ID（若由 task_d 调度，否则可空）
    pub task_id: Option<String>,
    /// 会话 ID（用于跨调用状态共享）
    pub session_id: Option<String>,
    /// 当前 Agent 的 role（如 product_manager / architect）
    pub role: String,
    /// 自定义元数据（透传给 LLM / 工具）
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl AgentContext {
    pub fn new(agent_id: impl Into<String>, role: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            role: role.into(),
            task_id: None,
            session_id: None,
            metadata: HashMap::new(),
        }
    }

    pub fn with_task_id(mut self, task_id: impl Into<String>) -> Self {
        self.task_id = Some(task_id.into());
        self
    }

    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_builder() {
        let ctx = AgentContext::new("pm-001", "product_manager")
            .with_task_id("task-42")
            .with_session("sess-7");
        assert_eq!(ctx.agent_id, "pm-001");
        assert_eq!(ctx.role, "product_manager");
        assert_eq!(ctx.task_id.as_deref(), Some("task-42"));
        assert_eq!(ctx.session_id.as_deref(), Some("sess-7"));
    }

    #[test]
    fn serde_roundtrip() {
        let ctx = AgentContext::new("a-1", "architect").with_task_id("t-1");
        let s = serde_json::to_string(&ctx).unwrap();
        let v: AgentContext = serde_json::from_str(&s).unwrap();
        assert_eq!(v.agent_id, "a-1");
        assert_eq!(v.task_id.as_deref(), Some("t-1"));
    }
}
