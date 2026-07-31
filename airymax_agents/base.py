"""AirymaxAgent — LLMAgent 子类基类

所有具体 Agent (ProductManagerAgent / ArchitectAgent / ...) 继承本类，
无需重复样板代码：自动从子类所在目录加载 ``contract.json`` 与
``prompts/system.md``，再调用 :class:`LLMAgent` 的初始化。

约定目录结构::

    airymax_agents/<role>/
        ├── agent.py          # 子类实现 (class XxxAgent(AirymaxAgent))
        ├── contract.json     # 契约 (遵循 01-agent-contract.md)
        └── prompts/
            └── system.md     # 系统提示词

子类最小实现::

    class XxxAgent(AirymaxAgent):
        ROLE = "xxx"

设计原则 (Simplicity is beauty)：
- 零样板：子类只需声明 ROLE，余下由基类自动装配
- 零配置：契约 + 提示词随包发布，开箱即用
- 与 openlab 解耦：仅依赖 ``openlab.agents.LLMAgent`` 与 ``openlab.core.llm``
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from openlab.agents import LLMAgent
from openlab.core.llm import make_llm_client

logger = logging.getLogger(__name__)


class AirymaxAgent(LLMAgent):
    """所有 Airymax 内置 Agent 的基类。

    子类只需:
      1. 声明 ``ROLE`` 类属性 (与 contract.json 中 ``role`` 一致)
      2. 放置 ``contract.json`` 与 ``prompts/system.md`` 于同类目录

    构造时自动解析契约与提示词，再委托 :class:`LLMAgent` 完成初始化。
    """

    #: 子类必填：角色名 (与 contract.json 中 ``role`` 字段一致)
    ROLE: str = "agent"

    def __init__(
        self,
        llm: Optional[Any] = None,
        contract_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        agent_dir = self._resolve_agent_dir()
        contract = self._load_contract(agent_dir)
        if contract_overrides:
            contract = {**contract, **contract_overrides}
        system_prompt = self._load_system_prompt(agent_dir)

        super().__init__(
            contract=contract,
            llm=llm if llm is not None else make_llm_client(),
            system_prompt=system_prompt,
        )
        logger.debug(
            "AirymaxAgent[%s] loaded from %s (role=%s)",
            self.agent_id,
            agent_dir,
            self.ROLE,
        )

    # ── 内部工具 ──────────────────────────────────────────

    @classmethod
    def _resolve_agent_dir(cls) -> Path:
        """通过子类 ``__module__`` 定位其源文件所在目录。

        子类定义在 ``airymax_agents/<role>/agent.py``，故源文件所在目录
        即为契约 / 提示词的根目录。
        """
        module = sys.modules.get(cls.__module__)
        if module is None or not getattr(module, "__file__", None):
            raise RuntimeError(
                f"cannot locate source file for {cls.__module__}; "
                "AirymaxAgent subclass must be defined in a normal .py module"
            )
        return Path(module.__file__).resolve().parent

    @staticmethod
    def _load_contract(agent_dir: Path) -> Dict[str, Any]:
        contract_path = agent_dir / "contract.json"
        if not contract_path.is_file():
            raise FileNotFoundError(
                f"contract.json not found in {agent_dir}; "
                "every AirymaxAgent subclass requires a contract.json beside agent.py"
            )
        return json.loads(contract_path.read_text(encoding="utf-8"))

    @staticmethod
    def _load_system_prompt(agent_dir: Path) -> str:
        prompt_path = agent_dir / "prompts" / "system.md"
        if not prompt_path.is_file():
            return ""
        return prompt_path.read_text(encoding="utf-8").strip()


__all__ = ["AirymaxAgent"]
