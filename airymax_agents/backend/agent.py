"""Backend Agent — 后端开发

API 设计、数据库实现、服务端业务逻辑。
"""

from __future__ import annotations

from ..base import AirymaxAgent


class BackendAgent(AirymaxAgent):
    """后端 Agent — API 设计、数据库开发、服务实现。"""

    ROLE = "backend"


__all__ = ["BackendAgent"]
