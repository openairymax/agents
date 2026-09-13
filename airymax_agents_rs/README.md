# airymax_agents_rs

Airymax Rust Agent 执行体（Cargo workspace）。

采用 Cargo workspace 多 crate 划分，按职责边界解耦（核心抽象 /
FFI 绑定 / 契约解析 / 执行体）；与 Python `airymax_agents/`
共享同一契约 schema（`shared/contracts/agent.schema.json`）与
`<role>_v1` agent_id 命名规则，保证跨语言契约一致。

## Crates

| crate | 职责 |
|-------|------|
| `agent-core` | `Agent` trait + `AgentContext` + `AgentStatus` + `LlmClient`/`MockLlmClient` + 错误类型 |
| `agent-ffi` | extern "C" 绑定 `airy_sys_*`（链接 `libagentrt.so`） |
| `agent-contract` | serde 解析 contract.json + agent_id 格式校验 |
| `coding_agent` | CodingAgent（LLM 驱动，OpenAI 兼容协议 + mock 降级） |

## 构建

```sh
cargo build --workspace
cargo test  --workspace
```

链接 `libagentrt.so` 仅 `agent-ffi` crate 需要；其他 crate 可独立编译。

## 状态

v0.1.3（LLM 已接入）。`coding_agent` 已具备真实 LLM 调用能力：

- **OpenAI 兼容协议**：`LlmClient` 经 `$OPENAI_API_KEY`（可选 `$OPENAI_BASE_URL`）调用真实 LLM；
- **Mock 降级**：无 API Key 时 `MockLlmClient` 提供确定性响应，保证 CI/本地可运行；
- **契约**：`crates/coding_agent/contract.json` — `agent_id: coding_rs_v1`，`implementation_status: llm_ready`；
- **性能对比**：与 Python `CodingAgent` 的延迟基准见 `agents/tests/`（Python vs Rust 对比）。

AgentRT 运行时可经 `agent_d` spawn/invoke 驱动本 Rust agent 真实执行。
