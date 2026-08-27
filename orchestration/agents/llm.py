"""Airymax Agents — LLM Agent 执行体

:class:`LLMAgent` 继承 ``orchestration.core.agent.Agent`` 基类，补全三要素：
  1. LLM 调用         — 通过 ``orchestration.core.llm.LLMClient`` (OpenAI 兼容)
  2. 工具执行         — OpenAI function-calling 协议，自动回路
  3. Agent 间协作     — 复用基类 ``Message`` + ``handle_message``

执行循环 (``execute``)::

    ┌─ build messages [system, user] ─────────────────────┐
    │                                                     │
    ▼                                                     │
   LLM.chat(tools=...)                                    │
    │                                                     │
    ├── tool_calls? ──yes──► run each tool ──append──►───┘
    │                                  tool result
    ▼
   返回最终 content → TaskResult

工具调用最大 16 轮，避免无限循环（loop-hint 连续重复 3 次注入警告、
4 次强制终止；60% 轮数时注入一次性收敛提示）。契约遵循
``01-agent-contract.md``：``models.system2`` 用于深度思考 (t2)，
``system1`` 用于快思考 (t1-f)。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Set

from ..core.agent import (
    Agent,
    AgentCapability,
    AgentContext,
    AgentStatus,
    Message,
    TaskResult,
)
from ..core.llm import LLMClient, MockLLMClient, make_llm_client
from ..core.project_context import inject_project_context

logger = logging.getLogger(__name__)

# 契约中声明的默认思考模型
#   system2 (t2 深度思考) — 复杂任务主模型
#   system1 (t1-f 快思考) — 简单任务快模型（省 token）
# 保持 provider 中立：留空 "" 由运行时 default_model（SSoT model.yaml）决定，
# 避免硬编码厂商模型名（如 gpt-4o-mini）导致 provider 切换时 400。
DEFAULT_MODEL_SYSTEM2 = ""
DEFAULT_MODEL_SYSTEM1 = ""
# 简单任务判定阈值（≤ 此字符数的输入走 system1 快思考）
SIMPLE_TASK_MAX_CHARS = 600
# 工具回路最大轮数，防止 LLM 反复调用工具
# （16 轮：多步任务如"写→读→改→复核→删→glob 确认"每轮单工具也留足余量；
#  loop-hint 在连续重复 3 次注入警告、4 次强制终止，仍能防死循环，
#  硬上限只是兜底，见 execute() 内 loop 保护。q8i 实测 8 轮不足以完成
#  六步文件增删改查闭环，LLM 每轮通常只调一个工具）
MAX_TOOL_ROUNDS = 16
# 单次工具结果回灌 LLM 上下文的上限（参照 Atom Code 64KiB 上限；防止大文件读取污染窗口）
MAX_TOOL_RESULT_CHARS = 64 * 1024


def _truncate_tool_result(result: str) -> str:
    """按 MAX_TOOL_RESULT_CHARS 截断工具结果并追加显式标记。"""
    if len(result) <= MAX_TOOL_RESULT_CHARS:
        return result
    return (
        result[:MAX_TOOL_RESULT_CHARS]
        + f"\n...[tool result truncated at {MAX_TOOL_RESULT_CHARS} chars]"
    )


class LLMAgent(Agent):
    """调用 LLM + 执行工具的具体 Agent。

    参数:
        contract:       遵循 ``01-agent-contract.md`` 的契约 dict
                        必含 ``agent_id``；``role`` / ``models`` / ``description`` 可选
        llm:            LLM 客户端 (LLMClient / MockLLMClient)；None 时用工厂
        system_prompt:  自定义系统提示词；空则按 ``role`` 生成默认
        capabilities:   能力集合 (透传基类)

    示例::

        contract = {
            "agent_id": "pm-001",
            "role": "product_manager",
            "models": {"system1": "deepseek-v4-flash", "system2": "deepseek-v4-flash"},
        }
        agent = LLMAgent(contract, llm=make_llm_client())
        await agent.initialize()
        result = await agent.execute("写一份待办应用的 PRD", AgentContext("pm-001"))
    """

    def __init__(
        self,
        contract: Dict[str, Any],
        llm: Optional[Any] = None,
        system_prompt: str = "",
        capabilities: Optional[Set[AgentCapability]] = None,
    ) -> None:
        super().__init__(
            agent_id=contract.get("agent_id", "anonymous-agent"),
            capabilities=capabilities,
        )
        self.contract = contract
        self.llm = llm if llm is not None else make_llm_client()
        self.system_prompt = system_prompt or self._default_system_prompt()

        # 工具 schema (OpenAI function-calling 格式)
        # name -> {"description":..., "parameters": {...json schema...}}
        self._tool_schemas: Dict[str, Dict[str, Any]] = {}

    # ── 工具注册 (扩展基类) ────────────────────────────────

    def register_tool(
        self,
        name: str,
        tool: Callable[..., Any],
        schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        """注册工具。

        参数:
            name:   工具名 (LLM 看到的函数名)
            tool:   可调用对象，支持 sync / async；签名为 ``(params: dict) -> Any``
            schema: 工具描述，形如::

                    {
                      "description": "获取当前时间",
                      "parameters": {
                        "type": "object",
                        "properties": {...},
                        "required": [...]
                      }
                    }

                若省略则仅注册可调用对象，不暴露给 LLM (不可被 function-calling 调用)。
        """
        super().register_tool(name, tool)
        if schema is not None:
            self._tool_schemas[name] = schema

    # ── Agent 生命周期 ────────────────────────────────────

    async def initialize(self) -> None:
        self._status = AgentStatus.READY
        logger.debug("LLMAgent[%s] initialized, llm=%s", self.agent_id, type(self.llm).__name__)

    async def shutdown(self) -> None:
        self._status = AgentStatus.SHUTDOWN
        logger.debug("LLMAgent[%s] shutdown", self.agent_id)

    # ── 核心执行 ──────────────────────────────────────────

    async def execute(self, input_data: Any, context: AgentContext) -> TaskResult:
        """执行任务：调 LLM + 工具回路，返回 TaskResult。"""
        self._status = AgentStatus.RUNNING
        self._context = context
        self._update_activity()

        started = time.time()
        rounds = 0
        # 中途收敛提示按单次 execute 计数（runner 主循环复用 agent 处理
        # 多个请求，标志须在每次执行时重置，避免跨任务误跳提示）。
        self._mid_progress_hinted = False
        model = self._select_model(input_data)
        tool_defs = self._build_tool_definitions()

        # 项目上下文（AGENTS.md 等价物）作为 system 消息注入最前；
        # 未找到约定文件时保持原消息序列不变
        messages: List[Dict[str, Any]] = inject_project_context(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self._stringify_input(input_data)},
            ]
        )

        try:
            while rounds <= MAX_TOOL_ROUNDS:
                resp = await self.llm.chat(
                    messages=messages,
                    model=model,
                    tools=tool_defs if tool_defs else None,
                )
                choice = resp.get("choices", [{}])[0]
                msg = choice.get("message", {})
                tool_calls = msg.get("tool_calls")

                if not tool_calls:
                    content = msg.get("content", "")
                    return TaskResult(
                        success=True,
                        output=content,
                        metrics={
                            "model": resp.get("model", model),
                            "rounds": rounds,
                            "tokens": resp.get("usage", {}).get("total_tokens"),
                            "elapsed_ms": int((time.time() - started) * 1000),
                        },
                    )

                # 有 tool_calls：执行工具，结果回流到 messages
                messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
                # 结果感知 loop 保护（参照 Atom Code ToolLoopPolicy）：
                # 同一工具 + 同一参数 + 同一结果连续出现 3 次注入警告，4 次强制终止，
                # 防止模型对失败调用空转直至硬上限。
                # OpenAI 兼容 API 要求 tool 结果消息紧跟带 tool_calls 的 assistant
                # 消息（中间不得插入 system 等其它 role），因此 loop-hint 在全部
                # tool 消息追加完成后统一注入。
                loop_hint = None
                # 中途收敛提示（q8i 生产实测）：模型可能在"不同参数/不同工具"间
                # 空转（指纹不同，不触发上面的连续重复检测）直至硬上限。到
                # MAX_TOOL_ROUNDS 的 60% 仍未收尾时注入一次性收敛提醒，引导
                # 模型停止无进展重试、基于已有工具结果给出结论。
                if rounds >= int(MAX_TOOL_ROUNDS * 0.6) and not getattr(
                    self, "_mid_progress_hinted", False
                ):
                    loop_hint = (
                        f"[progress-hint] 已执行 {rounds} 轮工具调用。若任务关键步骤"
                        "已完成，请停止继续调用工具，基于已有结果给出最终结论；"
                        "若某次调用持续失败，请更换策略而非重试同一思路。"
                    )
                    self._mid_progress_hinted = True
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    raw_args = fn.get("arguments", "{}")
                    tool_result = await self._invoke_tool(name, raw_args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id", name),
                            "content": _truncate_tool_result(tool_result),
                        }
                    )
                    self._track_tool_call(name, raw_args, tool_result)
                    repeated = self._consecutive_duplicate_tool_calls()
                    if repeated >= 3 and loop_hint is None:
                        loop_hint = (
                            "[loop-hint] 工具调用 "
                            f"{name}({raw_args[:120]}) 连续 {repeated} 次返回"
                            " 相同结果，未取得进展。请更换策略、调整参数或"
                            " 停止重复该调用。"
                        )
                    if repeated >= 4:
                        return TaskResult(
                            success=False,
                            output=None,
                            error=(
                                f"tool loop detected: {name} returned identical result "
                                f"{repeated} times"
                            ),
                            error_code="TOOL_LOOP_DETECTED",
                        )
                if loop_hint is not None:
                    messages.append({"role": "system", "content": loop_hint})
                rounds += 1
                logger.debug("LLMAgent[%s] tool round %d done", self.agent_id, rounds)

            return TaskResult(
                success=False,
                output=None,
                error=f"tool loop exceeded {MAX_TOOL_ROUNDS} rounds",
                error_code="TOOL_LOOP_OVERFLOW",
            )
        except Exception as e:
            logger.exception("LLMAgent[%s] execute failed", self.agent_id)
            return TaskResult(success=False, error=str(e), error_code="LLM_EXEC_ERROR")
        finally:
            if self._status == AgentStatus.RUNNING:
                self._status = AgentStatus.READY

    # ── Agent 间协作 ──────────────────────────────────────

    async def handle_message(self, message: Message) -> Optional[Message]:
        """处理来自其他 Agent 的消息。

        扩展基类：支持 ``task`` 与 ``delegate`` 两种类型，
        均转发到 :meth:`execute` 并以结果回信。
        """
        if message.type in ("task", "delegate"):
            ctx = self._context or AgentContext(self.agent_id)
            result = await self.execute(message.content, ctx)
            return Message(
                message_type="result",
                content=result,
                sender=self.agent_id,
                receiver=message.sender,
            )
        return None

    # ── 内部辅助 ──────────────────────────────────────────

    def _model_system1(self) -> str:
        """契约 ``models.system1`` (t1-f 快思考)；缺省用客户端 default。

        部署级覆盖：环境变量 ``AIRY_AGENT_MODEL`` 优先级最高（强制单模型）。
        """
        env_model = os.environ.get("AIRY_AGENT_MODEL", "").strip()
        if env_model:
            return env_model
        models = self.contract.get("models") or {}
        return models.get("system1") or getattr(self.llm, "default_model", DEFAULT_MODEL_SYSTEM1)

    def _select_model(self, input_data: Any) -> str:
        """按任务复杂度选择思考模型（双思考分级）：

        - 简单任务（短输入 ≤ SIMPLE_TASK_MAX_CHARS）→ system1（t1-f 快思考，省 token）
        - 复杂任务 → system2（t2 深度思考）

        环境变量 ``AIRY_AGENT_MODEL`` 覆盖一切（部署级强制单模型）。
        契约保持 provider 中立（不在契约中写死厂商模型名）。
        """
        if os.environ.get("AIRY_AGENT_MODEL", "").strip():
            return self._model_system2()
        text = self._stringify_input(input_data)
        if len(text) <= SIMPLE_TASK_MAX_CHARS:
            return self._model_system1()
        return self._model_system2()

    def _model_system2(self) -> str:
        """契约 ``models.system2`` (t2 主思考)；缺省用客户端 default。

        部署级覆盖：环境变量 ``AIRY_AGENT_MODEL`` 优先级最高，契约保持
        provider 中立（不在契约中写死厂商模型名）。
        """
        env_model = os.environ.get("AIRY_AGENT_MODEL", "").strip()
        if env_model:
            return env_model
        models = self.contract.get("models") or {}
        return models.get("system2") or getattr(self.llm, "default_model", DEFAULT_MODEL_SYSTEM2)

    def _default_system_prompt(self) -> str:
        role = self.contract.get("role", "assistant")
        desc = self.contract.get("description", "")
        name = self.contract.get("agent_name", self.agent_id)
        lines = [
            f"你是 {name}，担任 {role} 角色。",
        ]
        if desc:
            lines.append(f"职责: {desc}")
        lines.append("请清晰、准确、有条理地完成任务。")
        return "\n".join(lines)

    def _build_tool_definitions(self) -> List[Dict[str, Any]]:
        """转换 _tool_schemas 为 OpenAI function-calling tools 列表。"""
        defs = []
        for name, schema in self._tool_schemas.items():
            defs.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": schema.get("description", ""),
                        "parameters": schema.get("parameters", {"type": "object", "properties": {}}),
                    },
                }
            )
        return defs

    async def _invoke_tool(self, name: str, raw_args: str) -> str:
        """执行单个工具调用，返回字符串化的结果。"""
        tool = self.get_tool(name)
        if tool is None:
            return json.dumps({"error": f"tool '{name}' not registered"}, ensure_ascii=False)

        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            args = {"_raw": raw_args}

        try:
            if inspect.iscoroutinefunction(tool):
                result = await tool(args)
            else:
                result = await asyncio.to_thread(tool, args)
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            logger.warning("LLMAgent[%s] tool '%s' raised: %s", self.agent_id, name, e)
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    # ── 工具结果反哺控制与 loop 保护（参照 Atom Code 64KiB 上限 + ToolLoopPolicy） ──

    def _track_tool_call(self, name: str, raw_args: str, result: str) -> None:
        """记录最近一次 (工具, 参数, 结果) 指纹，供连续重复检测。"""
        if not hasattr(self, "_tool_call_fingerprints"):
            self._tool_call_fingerprints = []
        fingerprint = (name, raw_args, result[:256])
        self._tool_call_fingerprints.append(fingerprint)

    def _consecutive_duplicate_tool_calls(self) -> int:
        """返回最近连续重复（指纹相同）的工具调用次数。"""
        fps = getattr(self, "_tool_call_fingerprints", [])
        if not fps:
            return 0
        n = 1
        for i in range(len(fps) - 1, 0, -1):
            if fps[i] == fps[i - 1]:
                n += 1
            else:
                break
        return n

    @staticmethod
    def _stringify_input(input_data: Any) -> str:
        if isinstance(input_data, str):
            return input_data
        if isinstance(input_data, dict):
            # 若是 {"task": "..."} 形式，直接取 task
            if "task" in input_data and isinstance(input_data["task"], str):
                return input_data["task"]
            return json.dumps(input_data, ensure_ascii=False, default=str)
        return str(input_data)


__all__ = [
    "LLMAgent",
    "MAX_TOOL_ROUNDS",
    "DEFAULT_MODEL_SYSTEM1",
    "DEFAULT_MODEL_SYSTEM2",
    "SIMPLE_TASK_MAX_CHARS",
]
