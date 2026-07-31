// SPDX-FileCopyrightText: 2026 SPHARX Ltd.
// SPDX-License-Identifier: AGPL-3.0-or-later OR Apache-2.0
//! `agent-contract` — Airymax Agent 契约 (contract.json) 解析与校验。
//!
//! 与 Python 侧 `airymax_agents/<role>/contract.json` 共享同一 schema
//! （见 `airymax-agent-standard.md` §1）。

pub mod schema;

pub use schema::{
    Capability, Contract, ContractError, CostProfile, ModelConfig, Result, TrustMetrics,
};
