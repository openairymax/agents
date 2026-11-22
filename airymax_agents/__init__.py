"""Airymax Agents — 内置 11 个 Agent 执行体

每个 Agent 子目录包含:
- ``agent.py``       — 继承 :class:`AirymaxAgent` (→ :class:`LLMAgent`) 的具体实现
- ``contract.json``  — 遵循 ``01-agent-contract.md`` 的契约
- ``prompts/``       — 系统提示词

使用方式::

    from airymax_agents import get_agent

    agent = get_agent("product_manager")   # 自动装配契约 + 提示词
    await agent.initialize()
    result = await agent.execute("写一份待办应用 PRD", AgentContext("pm-001"))

模型连接按 ``$AIRY_HOME/config/model.yaml`` 解析：本地条目免 key 直连，
无本地条目且无任何厂商 key 时才启用 :class:`MockLLMClient`（离线兜底）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import AirymaxAgent
from .product_manager.agent import ProductManagerAgent
from .architect.agent import ArchitectAgent
from .backend.agent import BackendAgent
from .frontend.agent import FrontendAgent
from .devops.agent import DevOpsAgent
from .security.agent import SecurityAgent
from .tester.agent import TesterAgent
from .coding.agent import CodingAgent
from .data_engineer.agent import DataEngineerAgent
from .reviewer.agent import ReviewerAgent
from .analyst.agent import AnalystAgent

#: role → Agent 类
AGENT_REGISTRY: Dict[str, type] = {
    "product_manager": ProductManagerAgent,
    "architect": ArchitectAgent,
    "backend": BackendAgent,
    "frontend": FrontendAgent,
    "devops": DevOpsAgent,
    "security": SecurityAgent,
    "tester": TesterAgent,
    "coding": CodingAgent,
    "data_engineer": DataEngineerAgent,
    "reviewer": ReviewerAgent,
    "analyst": AnalystAgent,
}

#: 规划器抽象角色 → 执行体角色 归一化映射
#:
#: 认知规划器（reactive 规则表 / GRAD s2 LLM 计划 / 降级路径）产出的是
#: 抽象角色（creator/verifier/analyst...），而 agent 运行时只注册具体
#: 工程角色。此处作为「策略 → 机制」的绑定层，将抽象角色归一化到
#: 最近的执行体；映射表之外的字符串由驱动层显式兜底并留痕。
ROLE_ALIASES: Dict[str, str] = {
    "creator": "coding",
    "generator": "coding",
    "writer": "coding",
    "translator": "coding",
    "explainer": "coding",
    "processor": "coding",
    "formatter": "coding",
    "reactive-agent": "coding",
    "verifier": "tester",
    "validator": "tester",
    "executor": "devops",
    "retriever": "architect",
    "summarizer": "product_manager",
}

#: 兜底执行体：仅用于**驱动**路径（计划缺角色信息时保证节点可执行）
ROLE_FALLBACK = "coding"


def resolve_role(role: str) -> Optional[str]:
    """严格解析角色：注册名原样返回，别名映射到具体执行体，未登记返回 None。

    与 agentrt ``agent_vocab_resolve`` 同语义（fail-closed）：未知角色不得
    被静默当成某个执行体，否则会以错误的能力集执行任务且无迹可查。
    需要「计划总能被驱动」的调用方自行使用 :data:`ROLE_FALLBACK` 并留痕。
    """
    if role in AGENT_REGISTRY:
        return role
    return ROLE_ALIASES.get(role)


def get_agent(
    role: str,
    llm: Optional[Any] = None,
    contract_overrides: Optional[Dict[str, Any]] = None,
    syscall_proxy: Optional[Any] = None,
) -> AirymaxAgent:
    """按 role 名实例化 Agent。

    参数:
        role:              角色 (product_manager/architect/backend/frontend/devops/
                            security/tester/coding/data_engineer/reviewer/analyst)
                           或 :data:`ROLE_ALIASES` 中的抽象角色别名
        llm:               LLM 客户端；None 时用 :func:`make_llm_client`
        contract_overrides: 契约字段覆盖 (例如临时换 model)
        syscall_proxy:     agentrt 系统调用代理 (SyscallProxy)；
                           None 时为纯 Python LLM 模式（向后兼容）

    返回:
        已就绪的 Agent 实例 (尚未调用 :meth:`initialize`)

    异常:
        ValueError: role 既非注册名也非已登记别名（不猜测调用方意图）
    """
    resolved = resolve_role(role)
    if resolved is None:
        raise ValueError(
            f"unknown agent role {role!r}; valid roles: {', '.join(AGENT_REGISTRY)}"
        )
    return AGENT_REGISTRY[resolved](
        llm=llm,
        contract_overrides=contract_overrides,
        syscall_proxy=syscall_proxy,
    )


def list_agents() -> List[str]:
    """返回所有可用 role 名。"""
    return list(AGENT_REGISTRY)


__all__ = [
    "AirymaxAgent",
    "AGENT_REGISTRY",
    "ROLE_ALIASES",
    "ROLE_FALLBACK",
    "resolve_role",
    "get_agent",
    "list_agents",
    "ProductManagerAgent",
    "ArchitectAgent",
    "BackendAgent",
    "FrontendAgent",
    "DevOpsAgent",
    "SecurityAgent",
    "TesterAgent",
    "CodingAgent",
    "DataEngineerAgent",
    "ReviewerAgent",
    "AnalystAgent",
]
