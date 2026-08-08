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

无 ``OPENAI_API_KEY`` 时自动启用 :class:`MockLLMClient`，端到端可跑通。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

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
#: 最近的执行体；LLM 计划可能产出任意角色串，因此保留兜底。
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

#: 兜底执行体：未知角色归一化至此，保证计划总能被驱动、不因角色失配中断
ROLE_FALLBACK = "coding"


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
        llm:               LLM 客户端；None 时用 :func:`make_llm_client`
        contract_overrides: 契约字段覆盖 (例如临时换 model)
        syscall_proxy:     agentrt 系统调用代理 (SyscallProxy)；
                           None 时为纯 Python LLM 模式（向后兼容）

    返回:
        已就绪的 Agent 实例 (尚未调用 :meth:`initialize`)
    """
    cls = AGENT_REGISTRY.get(role)
    if cls is None:
        # 策略→机制 绑定层：抽象角色归一化到具体执行体
        resolved = ROLE_ALIASES.get(role, ROLE_FALLBACK)
        logger.warning("unknown agent role %r, normalized to %r", role, resolved)
        cls = AGENT_REGISTRY[resolved]
    return cls(
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
