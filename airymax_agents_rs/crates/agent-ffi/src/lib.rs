// SPDX-FileCopyrightText: 2026 SPHARX Ltd.
// SPDX-License-Identifier: AGPL-3.0-or-later OR Apache-2.0
//! `agent-ffi` — Rust FFI 绑定 agentrt C 系统调用。
//!
//! 与 Python 侧 `sdk-python/agentrt/syscall.py::SyscallProxy` 对齐
//! （`airymax-agent-standard.md` §8.3 syscall ABI）。
//!
//! # 安全
//! 所有 `extern "C"` 函数签名严格匹配
//! `agentrt/atoms/syscall/include/syscalls.h`。调用前需链接 `libagentrt.so`。

pub mod syscalls;

pub use syscalls::{
    agent_invoke, agent_list, agent_spawn, agent_terminate, memory_delete, memory_get,
    memory_search, memory_write, SyscallError, SyscallHandle,
};
