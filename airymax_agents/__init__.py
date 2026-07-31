"""Airymax Agents — 内置 7 个 Agent 执行体

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

from typing import Any, Dict, List, Optional

from .base import AirymaxAgent
from .product_manager.agent import ProductManagerAgent
from .architect.agent import ArchitectAgent
from .backend.agent import BackendAgent
from .frontend.agent import FrontendAgent
from .devops.agent import DevOpsAgent
from .security.agent import SecurityAgent
from .tester.agent import TesterAgent

#: role → Agent 类
AGENT_REGISTRY: Dict[str, type] = {
    "product_manager": ProductManagerAgent,
    "architect": ArchitectAgent,
    "backend": BackendAgent,
    "frontend": FrontendAgent,
    "devops": DevOpsAgent,
    "security": SecurityAgent,
    "tester": TesterAgent,
}


def get_agent(
    role: str,
    llm: Optional[Any] = None,
    contract_overrides: Optional[Dict[str, Any]] = None,
    syscall_proxy: Optional[Any] = None,
) -> AirymaxAgent:
    """按 role 名实例化 Agent。

    参数:
        role:              角色 (product_manager/architect/backend/frontend/devops/security/tester)
        llm:               LLM 客户端；None 时用 :func:`make_llm_client`
        contract_overrides: 契约字段覆盖 (例如临时换 model)
        syscall_proxy:     agentrt 系统调用代理 (SyscallProxy)；
                           None 时为纯 Python LLM 模式（向后兼容）

    返回:
        已就绪的 Agent 实例 (尚未调用 :meth:`initialize`)
    """
    cls = AGENT_REGISTRY.get(role)
    if cls is None:
        raise KeyError(
            f"unknown agent role: {role!r}; "
            f"available: {list(AGENT_REGISTRY)}"
        )
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
]
