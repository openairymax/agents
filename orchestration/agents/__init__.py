"""Airymax Agents — Agent 执行体基类集合

Agent 执行体基类统一定义于 ``orchestration.agents``：

- ``llm``  — :class:`LLMAgent`，调 LLM + 工具执行的具体执行体基类

具体的 Airymax 内置 Agent 执行体位于 ``ecosystem/agents/airymax_agents/``，
每个 Agent 子目录含 ``agent.py`` + ``contract.json`` + ``prompts/system.md``，
继承 :class:`AirymaxAgent` (→ :class:`LLMAgent`)。

历史 ``architect/`` 实现已删除（签名与基类不兼容）；如需架构师 Agent，
请使用 :class:`airymax_agents.architect.ArchitectAgent`。
"""

from .llm import LLMAgent

__all__ = [
    "LLMAgent",
]
