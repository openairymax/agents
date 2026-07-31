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
- agentrt 接入可选：注入 ``SyscallProxy`` 后自动持久化上下文/结果，
  ``None`` 时退化为纯 Python LLM 模式（向后兼容）
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

    agentrt 系统调用代理（``syscall_proxy``）是可选的：

      - ``None``（默认）：纯 Python LLM 模式，行为与重构前完全一致，
        MockLLMClient 可跑、无 agentrt 依赖。
      - ``SyscallProxy(...)``：execute() 在 LLM 推理前后将输入上下文与
        最终结果写入 mem_d 守护进程持久化，便于跨会话/跨 Agent 检索。
        持久化失败不阻断 LLM 推理（仅记录 warning），保证业务可用性。
    """

    #: 子类必填：角色名 (与 contract.json 中 ``role`` 字段一致)
    ROLE: str = "agent"

    def __init__(
        self,
        llm: Optional[Any] = None,
        contract_overrides: Optional[Dict[str, Any]] = None,
        syscall_proxy: Optional[Any] = None,
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
        # agentrt 系统调用代理（None 时为纯 Python 向后兼容模式）
        self._sys = syscall_proxy
        logger.debug(
            "AirymaxAgent[%s] loaded from %s (role=%s, syscall=%s)",
            self.agent_id,
            agent_dir,
            self.ROLE,
            "on" if self._sys is not None else "off",
        )

    # ── 核心执行（增强：可选 agentrt 记忆持久化） ───────────────

    async def execute(self, input_data: Any, context: Any) -> Any:
        """执行任务：在 LLM 推理前后可选持久化到 agentrt mem_d。

        持久化采用 best-effort 策略：失败仅记录日志，不阻断 LLM 推理。
        这样即便 mem_d 不可达（如 daemon 未启动、socket 错误），
        Agent 仍能完成核心推理任务。

        参数:
            input_data: 任务输入（任意可 stringify 对象）
            context:   AgentContext（含 agent_id / task_id 等）

        返回:
            TaskResult（与 :meth:`LLMAgent.execute` 一致）
        """
        # 1. 可选：将输入上下文写入 agentrt 记忆
        if self._sys is not None:
            await self._persist_input(input_data, context)

        # 2. LLM 推理（沿用父类完整逻辑：LLM + 工具回路）
        result = await super().execute(input_data, context)

        # 3. 可选：将最终结果持久化到 agentrt 记忆
        if self._sys is not None and result is not None:
            await self._persist_result(result, context)

        return result

    async def _persist_input(self, input_data: Any, context: Any) -> None:
        """将输入上下文持久化到 agentrt mem_d（best-effort）。"""
        try:
            payload = {
                "phase": "input",
                "agent_id": getattr(context, "agent_id", self.agent_id),
                "task_id": getattr(context, "task_id", None),
                "role": self.ROLE,
                "input": self._stringify_input(input_data),
            }
            metadata = json.dumps(
                {"agent_id": payload["agent_id"], "task_id": payload["task_id"],
                 "role": self.ROLE, "phase": "input"},
                ensure_ascii=False,
            )
            record_id = self._sys.memory_write(
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                metadata,
            )
            logger.debug(
                "AirymaxAgent[%s] input persisted (record_id=%s)",
                self.agent_id, record_id,
            )
        except Exception as e:
            logger.warning(
                "AirymaxAgent[%s] input persistence failed (non-fatal): %s",
                self.agent_id, e,
            )

    async def _persist_result(self, result: Any, context: Any) -> None:
        """将最终结果持久化到 agentrt mem_d（best-effort）。"""
        try:
            payload = {
                "phase": "output",
                "agent_id": getattr(context, "agent_id", self.agent_id),
                "task_id": getattr(context, "task_id", None),
                "role": self.ROLE,
                "success": getattr(result, "success", None),
                "output": getattr(result, "output", None),
                "error": getattr(result, "error", None),
                "metrics": getattr(result, "metrics", None),
            }
            metadata = json.dumps(
                {"agent_id": payload["agent_id"], "task_id": payload["task_id"],
                 "role": self.ROLE, "phase": "output"},
                ensure_ascii=False,
            )
            record_id = self._sys.memory_write(
                json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"),
                metadata,
            )
            logger.debug(
                "AirymaxAgent[%s] output persisted (record_id=%s)",
                self.agent_id, record_id,
            )
        except Exception as e:
            logger.warning(
                "AirymaxAgent[%s] output persistence failed (non-fatal): %s",
                self.agent_id, e,
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
