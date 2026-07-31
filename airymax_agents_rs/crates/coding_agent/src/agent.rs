// SPDX-FileCopyrightText: 2026 SPHARX Ltd.
// SPDX-License-Identifier: AGPL-3.0-or-later OR Apache-2.0
//! CodingAgent 实现：基于 `agent-core::Agent` trait 的样板。

use agent_contract::Contract;
use agent_core::{
    Agent, AgentContext, AgentError, AgentIdentity, AgentInput, AgentStatus, Result, TaskResult,
};
use async_trait::AsyncTrait;
use std::path::PathBuf;
use std::sync::Mutex;

/// Airymax Coding Agent 样板。
///
/// 当前为骨架实现：
/// - `initialize()` 从同目录 `contract.json` 加载契约
/// - `execute()` 仅记录日志并返回固定输出（待接入 LLM）
/// - `terminate()` 重置状态
pub struct CodingAgent {
    identity: AgentIdentity,
    contract_path: PathBuf,
    contract: Option<Contract>,
    status: Mutex<AgentStatus>,
}

impl CodingAgent {
    pub fn new(contract_path: PathBuf) -> Self {
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
            status: Mutex::new(AgentStatus::Created),
        }
    }

    /// 暴露已加载契约的不可变引用（测试用）。
    pub fn contract(&self) -> Option<&Contract> {
        self.contract.as_ref()
    }
}

#[async_trait]
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
        // TODO: 接入 LLM 客户端 + 工具回路（参见 airymax-agent-standard.md §4）
        Ok(TaskResult::ok(format!(
            "[coding_agent stub] received input of {} bytes",
            input.data.len()
        )))
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
        assert_eq!(agent.identity().agent_id, "coding_v1");
        assert!(agent.contract().is_some());

        let ctx = AgentContext::new("coding-001", "coding");
        let r = agent
            .execute(AgentInput::new("hello"), &ctx)
            .await
            .expect("execute");
        assert!(r.success);
        assert!(r.output.unwrap().contains("hello"));

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
}
