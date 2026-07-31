// SPDX-FileCopyrightText: 2026 SPHARX Ltd.
// SPDX-License-Identifier: AGPL-3.0-or-later OR Apache-2.0
//! Airymax Agent core: trait + context + status + error types.
//!
//! 与 Python `airymax_agents/base.py::AirymaxAgent` 对齐
//! （见 docs-closed/agentrt/01-designs/airymax-agent-standard.md §3, §9）。

pub mod agent;
pub mod context;
pub mod error;
pub mod status;

pub use agent::{Agent, AgentIdentity, AgentInput, TaskResult};
pub use context::AgentContext;
pub use error::{AgentError, Result};
pub use status::AgentStatus;
