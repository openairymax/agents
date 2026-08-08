# SPDX-FileCopyrightText: 2025-2026 SPHARX Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later OR Apache-2.0

"""Data Engineer Agent — 数据工程师

数据处理、ETL 开发与数据建模。继承 :class:`AirymaxAgent`，契约与提示词随包发布。
"""

from __future__ import annotations

from ..base import AirymaxAgent


class DataEngineerAgent(AirymaxAgent):
    """数据工程师 Agent — 数据管道构建、ETL 开发、数据建模与分析。"""

    ROLE = "data_engineer"


__all__ = ["DataEngineerAgent"]
