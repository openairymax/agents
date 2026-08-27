# Airymax Agents

Airymax 内置 Agent 执行体集合 — 11 个开箱即用、遵循
[`01-agent-contract.md`](https://gitcode.com/openairymax/docs/blob/main/AirymaxRT/20-modules/10-contracts/01-agent-contract.md)
的智能体，覆盖软件交付全链路。Python 实现位于 `airymax_agents/`，
Rust 实现位于 `airymax_agents_rs/`（当前为 `coding` 角色）。

## 设计原则

- **契约驱动**：每个 Agent 遵循 [`01-agent-contract.md`](https://gitcode.com/openairymax/docs/blob/main/AirymaxRT/20-modules/10-contracts/01-agent-contract.md) 规范（schema_version / agent_id / capabilities / models / required_permissions / cost_profile / trust_metrics），`contract.json` 是 Agent 与运行时之间的唯一可信接口
- **零样板**：`AirymaxAgent` 基类通过 `__module__` 自动定位子类源文件目录，自动装配契约与提示词，子类只需声明 `ROLE`
- **可独立运行**：无 `OPENAI_API_KEY` 时自动启用 `MockLLMClient`（确定性响应），便于 CI 与本地开发；配置 `OPENAI_API_KEY`（及可选 `OPENAI_BASE_URL`）后切换真实 LLM
- **分层清晰**：本包依赖同仓 `orchestration/`（`LLMAgent` + `LLMClient`），不耦合 `manager` / `skills` / 运行时基础设施

## 包含的 Agent

| Role | 类 | 职责 |
|------|----|------|
| `product_manager` | `ProductManagerAgent` | 需求分析、PRD 撰写、用户故事拆解 |
| `architect` | `ArchitectAgent` | 系统架构设计、技术选型、架构评审 |
| `backend` | `BackendAgent` | API 设计、数据库实现、服务端逻辑 |
| `frontend` | `FrontendAgent` | UI 实现、组件设计、状态管理 |
| `devops` | `DevOpsAgent` | CI/CD 流水线、部署自动化、基础设施即代码 |
| `security` | `SecurityAgent` | 安全审计、漏洞扫描、威胁建模 |
| `tester` | `TesterAgent` | 测试用例生成、测试执行、覆盖率分析 |
| `coding` | `CodingAgent` | 代码生成、解释、重构（Python + Rust 双实现，均接入 LLM） |
| `data_engineer` | `DataEngineerAgent` | 数据处理、ETL 管道、数据建模 |
| `reviewer` | `ReviewerAgent` | 代码评审、最佳实践建议、重构建议 |
| `analyst` | `AnalystAgent` | 数据分析、趋势检测、可视化报告 |

> 角色清单的运行时权威为 `airymax_agents/__init__.py` 的 `AGENT_REGISTRY`；
> 执行体注册表见 `registry/agents.yaml`（coordinator / custom_template 仍为规划中）。

## 目录结构

```
agents/
├── README.md
├── airymax_agents/                # Python 实现（11 角色）
│   ├── __init__.py            # 统一导出 + get_agent() 工厂 + AGENT_REGISTRY（角色 SSoT）
│   ├── base.py                # AirymaxAgent 基类 (自动加载契约+提示词)
│   ├── product_manager/
│   │   ├── __init__.py
│   │   ├── agent.py            # ProductManagerAgent(AirymaxAgent)
│   │   ├── contract.json
│   │   └── prompts/system.md
│   ├── architect/...
│   ├── backend/...
│   ├── frontend/...
│   ├── devops/...
│   ├── security/...
│   ├── tester/...
│   ├── coding/                   # CodingAgent (Python)
│   ├── data_engineer/...
│   ├── reviewer/...
│   └── analyst/...
├── orchestration/                # 多智能体编排内核（v0.1.1，AirymaxAgent 基座）
│   ├── core/                     # Agent / Task / Tool / Storage / LLM 抽象
│   ├── agents/                   # LLMAgent 执行体基类
│   ├── strategies/               # 调度 / 规划策略
│   ├── protocols/  utils/        # 协议与通用工具
│   ├── config.yaml  run.sh  requirements.txt
│   └── README.md
├── airymax_agents_rs/             # Rust 实现（cargo workspace）
│   └── crates/
│       ├── agent-core/            # 核心 trait / LLM 客户端 / 错误类型
│       ├── agent-contract/        # contract.json 解析与校验
│       ├── agent-ffi/             # syscall FFI 绑定
│       └── coding_agent/          # Rust CodingAgent (LLM 驱动)
├── registry/
│   └── agents.yaml              # 执行体注册表（运行时单一可信源）
├── shared/
│   └── contracts/
│       └── agent.schema.json    # 契约 JSON Schema（权威）
├── tests/                        # pytest 套件（含基准测试）
└── examples/
    └── run_pm.py              # 端到端示例
```

## 快速开始

### 前置条件

依赖同仓 `orchestration/` 包（`LLMAgent` + `LLMClient`），需在 `PYTHONPATH` 上。

### 端到端示例

```bash
cd airymaxhub/ecosystem/agents
PYTHONPATH=. python3 examples/run_pm.py
```

无 `OPENAI_API_KEY` 时自动启用 Mock 模式，输出确定性响应验证流程；
设置 `OPENAI_API_KEY` (及可选 `OPENAI_BASE_URL`) 后切换真实 LLM。

### 在代码中使用

```python
import asyncio
from airymax_agents import get_agent
from orchestration.core.agent import AgentContext

async def main():
    agent = get_agent("product_manager")
    await agent.initialize()
    result = await agent.execute("写一份待办应用 PRD", AgentContext("pm-001"))
    print(result.output)
    await agent.shutdown()

asyncio.run(main())
```

### 多 Agent 协作

```python
from airymax_agents import get_agent
from orchestration.core.agent import AgentContext, Message

pm = get_agent("product_manager")
arch = get_agent("architect")
await pm.initialize()
await arch.initialize()

# PM 完成 PRD
prd = await pm.execute("电商应用 PRD", AgentContext("pm-001"))

# 将 PRD 委托给 Architect
msg = Message(message_type="delegate", content=prd.output, sender="pm-001", receiver="arch-001")
arch_reply = await arch.handle_message(msg)
```

## 契约规范

每个 Agent 的 `contract.json` 遵循 Airymax 智能体契约规范 v1.0.0：

- `schema_version`: 契约规范版本
- `agent_id`: 全局唯一标识 (`<role>_v1`，如 `product_manager_v1`)
- `role`: 角色分类
- `capabilities[]`: 能力列表 (含 input/output JSON Schema)
- `models`: Thinkdual 双思考模型配置 (`system1` t1-f 快思考 + `system2` t2 主思考)
- `required_permissions[]`: 所需权限 (最小权限原则)
- `cost_profile`: 成本预估
- `trust_metrics`: 信任指标

## License

AGPL-3.0-or-later OR Apache-2.0

© 2025-2026 SPHARX Ltd. — Airymax Team
