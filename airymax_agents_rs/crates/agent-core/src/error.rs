// SPDX-FileCopyrightText: 2026 SPHARX Ltd.
// SPDX-License-Identifier: AGPL-3.0-or-later OR Apache-2.0
//! Agent 错误类型，统一 `Result<T>` 别名。

use thiserror::Error;

/// Agent 执行过程中可能抛出的错误。
#[derive(Debug, Error)]
pub enum AgentError {
    #[error("invalid contract: {0}")]
    InvalidContract(String),

    #[error("agent not found: {0}")]
    NotFound(String),

    #[error("agent state error: {0}")]
    StateError(String),

    #[error("syscall FFI failure: {0}")]
    Syscall(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("agent terminated externally")]
    Terminated,
}

pub type Result<T> = std::result::Result<T, AgentError>;
