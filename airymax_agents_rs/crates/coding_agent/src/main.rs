// SPDX-FileCopyrightText: 2026 SPHARX Ltd.
// SPDX-License-Identifier: AGPL-3.0-or-later OR Apache-2.0
//! `coding_agent` 二进制入口 — agent_d 子进程。
//!
//! 由 agent_d 的 `service.c::agent_spawn_child` 通过 execvp 启动：
//!
//!     coding_agent --spec <spec_json>
//!
//! 通信协议（行分隔 JSON，与 Python `airymax_agents.runner` 严格对齐）：
//!
//!   - 请求（从 stdin 读取）：
//!
//!       {"agent_id": "...", "input": "..."}
//!
//!   - 响应（写入 stdout）：
//!
//!       {"success": true, "output": "..."}
//!       {"success": false, "error": "..."}
//!
//! 生命周期：
//!
//!   1. 解析 `--spec` 参数（Agent 规格 JSON）
//!   2. 实例化 `CodingAgent` 并加载契约
//!   3. 主循环：逐行读 stdin → 执行 → 写 stdout
//!   4. EOF（stdin 关闭）时调用 `terminate()` 后正常退出
//!
//! 与 Python runner 不同：Rust 端暂不接入 agentrt SyscallProxy（IPC），
//! 仅作骨架级端到端验证；后续可经 `agent-ffi` crate 接入 syscall。

use std::env;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use agent_core::{Agent, AgentContext, AgentInput};
use coding_agent::CodingAgent;
use tokio::runtime::Runtime;

fn main() -> ExitCode {
    let spec_str = match parse_spec_arg() {
        Some(s) => s,
        None => {
            eprintln!("usage: coding_agent --spec <spec_json>");
            return ExitCode::from(2);
        }
    };

    let spec: serde_json::Value = match serde_json::from_str(&spec_str) {
        Ok(v) => v,
        Err(e) => {
            let resp = serde_json::json!({"success": false, "error": format!("bad spec: {e}")});
            println!("{resp}");
            return ExitCode::from(1);
        }
    };

    // 定位 contract.json：spec.contract_path 优先，否则用 crate 内置
    let contract_path = spec
        .get("contract_path")
        .and_then(|v| v.as_str())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("contract.json"));

    let runtime = match Runtime::new() {
        Ok(r) => r,
        Err(e) => {
            let resp = serde_json::json!({"success": false, "error": format!("tokio init: {e}")});
            println!("{resp}");
            return ExitCode::from(1);
        }
    };

    let mut agent = CodingAgent::new(contract_path);
    if let Err(e) = runtime.block_on(async { agent.initialize().await }) {
        let resp = serde_json::json!({"success": false, "error": format!("agent init failed: {e:?}")});
        println!("{resp}");
        return ExitCode::from(1);
    }

    eprintln!(
        "[coding_agent] ready: agent_id={}, role={}",
        agent.identity().agent_id,
        agent.identity().role
    );

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break, // stdin 错误，视为 EOF
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let req: serde_json::Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                let resp = serde_json::json!({"success": false, "error": format!("bad request: {e}")});
                let _ = writeln!(out, "{resp}");
                let _ = out.flush();
                continue;
            }
        };

        let req_agent_id = req.get("agent_id").and_then(|v| v.as_str()).unwrap_or("unknown");
        let user_input = req.get("input").and_then(|v| v.as_str()).unwrap_or("");

        let ctx = AgentContext::new(req_agent_id, &agent.identity().role);
        let result = runtime.block_on(async { agent.execute(AgentInput::new(user_input), &ctx).await });

        let resp = match result {
            Ok(r) if r.success => serde_json::json!({
                "success": true,
                "output": r.output.unwrap_or_default()
            }),
            Ok(r) => serde_json::json!({
                "success": false,
                "error": r.error.unwrap_or_else(|| "agent execution failed".to_string())
            }),
            Err(e) => serde_json::json!({
                "success": false,
                "error": format!("{e:?}")
            }),
        };

        if writeln!(out, "{resp}").is_err() {
            break; // stdout 关闭，无需继续
        }
        let _ = out.flush();
    }

    // EOF — 优雅终止
    let _ = runtime.block_on(async { agent.terminate().await });
    ExitCode::from(0)
}

/// 从命令行解析 `--spec <value>`。
fn parse_spec_arg() -> Option<String> {
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--spec" {
            return args.next();
        }
        // 容忍 `--spec=value` 形式
        if let Some(rest) = arg.strip_prefix("--spec=") {
            return Some(rest.to_string());
        }
    }
    None
}
