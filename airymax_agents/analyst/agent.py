# SPDX-FileCopyrightText: 2025-2026 SPHARX Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later OR Apache-2.0

"""Analyst Agent — 数据分析

数据分析、趋势检测、报告生成与可视化。继承 :class:`AirymaxAgent`，契约与提示词随包发布。
"""

from __future__ import annotations

from ..base import AirymaxAgent


class AnalystAgent(AirymaxAgent):
    """数据分析 Agent — 数据分析、趋势检测、报告生成与数据可视化。"""

    ROLE = "analyst"


__all__ = ["AnalystAgent"]
