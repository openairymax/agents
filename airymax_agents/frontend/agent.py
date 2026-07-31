"""Frontend Agent — 前端开发

UI 实现、组件设计、状态管理、交互逻辑。
"""

from __future__ import annotations

from ..base import AirymaxAgent


class FrontendAgent(AirymaxAgent):
    """前端 Agent — UI 实现、组件设计、状态管理。"""

    ROLE = "frontend"


__all__ = ["FrontendAgent"]
