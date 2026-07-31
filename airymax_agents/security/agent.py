"""Security Agent — 安全审计

安全审计、漏洞扫描、威胁建模、合规检查。
"""

from __future__ import annotations

from ..base import AirymaxAgent


class SecurityAgent(AirymaxAgent):
    """安全 Agent — 安全审计、漏洞扫描、威胁建模。"""

    ROLE = "security"


__all__ = ["SecurityAgent"]
