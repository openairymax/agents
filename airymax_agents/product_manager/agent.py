"""Product Manager Agent — 产品经理

理解用户需求，撰写产品需求文档 (PRD)，拆解用户故事。
继承 :class:`AirymaxAgent`，契约与提示词随包发布，开箱即用。
"""

from __future__ import annotations

from ..base import AirymaxAgent


class ProductManagerAgent(AirymaxAgent):
    """产品经理 Agent — 需求分析、PRD 撰写、用户故事拆解。"""

    ROLE = "product_manager"


__all__ = ["ProductManagerAgent"]
