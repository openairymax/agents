// SPDX-FileCopyrightText: 2026 SPHARX Ltd.
// SPDX-License-Identifier: AGPL-3.0-or-later OR Apache-2.0
//! Agent 状态枚举，与 C 侧 agent_d `status` 字段对齐
//! （见 airymax-agent-standard.md §3.1）。

use serde::{Deserialize, Serialize};

/// Agent 生命周期状态。
///
/// 数值与 `agent_d/src/service.c` 中 `agent_entry_internal_t.status` 一致：
///   - 0 = CREATED
///   - 1 = RUNNING（spawn 后）
///   - 2 = COMPLETED
///   - 3 = TERMINATED
///   - 4 = FAILED
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(i32)]
pub enum AgentStatus {
    Created = 0,
    Running = 1,
    Completed = 2,
    Terminated = 3,
    Failed = 4,
}

impl Default for AgentStatus {
    fn default() -> Self {
        Self::Created
    }
}

impl AgentStatus {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Terminated | Self::Failed)
    }

    pub fn as_i32(self) -> i32 {
        self as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_repr_matches_c() {
        assert_eq!(AgentStatus::Created.as_i32(), 0);
        assert_eq!(AgentStatus::Running.as_i32(), 1);
        assert_eq!(AgentStatus::Completed.as_i32(), 2);
        assert_eq!(AgentStatus::Terminated.as_i32(), 3);
        assert_eq!(AgentStatus::Failed.as_i32(), 4);
    }

    #[test]
    fn terminal_states() {
        assert!(AgentStatus::Completed.is_terminal());
        assert!(AgentStatus::Terminated.is_terminal());
        assert!(AgentStatus::Failed.is_terminal());
        assert!(!AgentStatus::Running.is_terminal());
        assert!(!AgentStatus::Created.is_terminal());
    }

    #[test]
    fn serde_roundtrip() {
        // serde 默认以字符串形式序列化枚举变体名称
        let s = serde_json::to_string(&AgentStatus::Running).unwrap();
        assert_eq!(s, "\"Running\"");
        let v: AgentStatus = serde_json::from_str("\"Terminated\"").unwrap();
        assert_eq!(v, AgentStatus::Terminated);
    }
}
