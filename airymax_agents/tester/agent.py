"""Tester Agent — 测试

测试用例生成、测试执行规划、覆盖率分析。
"""

from __future__ import annotations

from ..base import AirymaxAgent


class TesterAgent(AirymaxAgent):
    """测试 Agent — 用例生成、测试执行、覆盖率分析。"""

    ROLE = "tester"


__all__ = ["TesterAgent"]
