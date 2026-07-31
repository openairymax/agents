"""Architect Agent — 架构师

系统架构设计、技术选型、架构评审。
"""

from __future__ import annotations

from ..base import AirymaxAgent


class ArchitectAgent(AirymaxAgent):
    """架构师 Agent — 系统设计、技术选型、架构评审。"""

    ROLE = "architect"


__all__ = ["ArchitectAgent"]
