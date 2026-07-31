// SPDX-FileCopyrightText: 2026 SPHARX Ltd.
// SPDX-License-Identifier: AGPL-3.0-or-later OR Apache-2.0
//! `coding_agent` — Airymax Rust Agent 样板（骨架实现）。
//!
//! 作为 `airymax_agents_rs/` workspace 中第一个具体 Rust Agent 样板，
//! 演示如何基于 `agent-core` trait + `agent-contract` 契约 + `agent-ffi`
//! 系统调用搭建一个最小可用的 Agent。
//!
//! 实际业务逻辑（LLM 调用、工具回路）待后续填充。

pub mod agent;

pub use agent::CodingAgent;
