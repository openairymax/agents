# airymax_agents_rs

Airymax Rust Agent 执行体（Cargo workspace）。

仿 codex-rs 多 crate 划分；与 Python `airymax_agents/` 共享同一
`contract.json` schema 与 agent_id 规则（见
`docs-closed/agentrt/01-designs/airymax-agent-standard.md` §9 跨语言映射）。

## Crates

| crate | 职责 |
|-------|------|
| `agent-core` | `Agent` trait + `AgentContext` + `AgentStatus` + 错误类型 |
| `agent-ffi` | extern "C" 绑定 `airy_sys_*`（链接 `libagentrt.so`） |
| `agent-contract` | serde 解析 contract.json + agent_id 格式校验 |
| `coding_agent` | CodingAgent 样板（骨架实现） |

## 构建

```sh
cargo build --workspace
cargo test  --workspace
```

链接 `libagentrt.so` 仅 `agent-ffi` crate 需要；其他 crate 可独立编译。

## 状态

骨架（v0.1.0）。`coding_agent` 仅演示生命周期 trait 实现，未接入真实
LLM 客户端与工具回路。后续按需扩展。
