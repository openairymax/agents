"""Coding Agent — 编码开发

LLM 驱动的代码生成 Agent（与 Rust `coding_agent` 对称，用于基准对比）。
"""

from __future__ import annotations

from ..base import AirymaxAgent


class CodingAgent(AirymaxAgent):
    """编码开发 Agent — 基于需求生成可运行代码。"""

    ROLE = "coding"


__all__ = ["CodingAgent"]
