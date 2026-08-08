# SPDX-FileCopyrightText: 2025-2026 SPHARX Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later OR Apache-2.0

"""Reviewer Agent — 代码审查

代码质量审查、最佳实践检查与重构建议。继承 :class:`AirymaxAgent`，契约与提示词随包发布。
"""

from __future__ import annotations

from ..base import AirymaxAgent


class ReviewerAgent(AirymaxAgent):
    """代码审查 Agent — 代码质量审查、最佳实践检查、重构建议与文档评审。"""

    ROLE = "reviewer"


__all__ = ["ReviewerAgent"]
