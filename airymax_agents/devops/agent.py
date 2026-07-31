"""DevOps Agent — DevOps 自动化

CI/CD 流水线、部署自动化、基础设施即代码、监控配置。
"""

from __future__ import annotations

from ..base import AirymaxAgent


class DevOpsAgent(AirymaxAgent):
    """DevOps Agent — CI/CD、部署、基础设施即代码。"""

    ROLE = "devops"


__all__ = ["DevOpsAgent"]
