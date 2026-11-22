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

工具调用最大 16 轮，避免无限循环：同指纹连续重复 3 次注入警告、
4 次强制终止；连败（参数/错误内容各不相同也计入）达阈值
（默认 3，``AIRY_TOOL_FAIL_LIMIT`` 可调）立即熔断并回报
``TOOL_LOOP_CONSECUTIVE_FAILURES``（含最后 K 个失败原因）；60% 轮数时
注入一次性收敛提示。契约遵循 ``01-agent-contract.md``：
``models.system2`` 用于深度思考 (t2)，``system1`` 用于快思考 (t1-f)。
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
from ..core.llm_cfg import model_tool_rounds
from ..core.project_context import inject_project_context
from ..core.tool import tool_timeout

logger = logging.getLogger(__name__)

# 简单任务判定阈值（≤ 此字符数的输入走 system1 快思考）
SIMPLE_TASK_MAX_CHARS = 600
# 工具回路轮次预算，防止 LLM 反复调用工具。
# 机制与策略分离：机制给硬上限（TOOL_ROUNDS_MAX，与 agent_d 的
# rounds_resolve 同界 32）；策略经 AIRY_AGENT_TOOL_ROUNDS 收紧到具体值；
# 缺省 16（多步任务如"写→读→改→复核→删→glob 确认"每轮单工具也留足余量；
# loop-hint 在连续重复 3 次注入警告、4 次强制终止，仍能防死循环，硬上限
# 只是兜底。q8i 实测 8 轮不足以完成六步文件增删改查闭环）。
TOOL_ROUNDS_DEFAULT = 16
TOOL_ROUNDS_MAX = 32
# 单次工具结果回灌 LLM 上下文的上限（参照 Atom Code 64KiB 上限；防止大文件读取污染窗口）
MAX_TOOL_RESULT_CHARS = 64 * 1024
# 连败熔断默认阈值：连续 N 次工具调用失败即终止。
# q8i 生产实测：模型会在"不同参数/不同工具"间空转——每次失败内容都
# 不同，同指纹重复检测不触发，只能烧完全部轮数预算。连败达阈值几乎
# 必然是系统性故障（工具缺失/策略错误），继续重试没有意义。
# 环境变量 AIRY_TOOL_FAIL_LIMIT 可覆盖（非法值回退默认）。
DEFAULT_TOOL_FAIL_LIMIT = 3


def _rounds_limit(model: str = "") -> int:
    """解析一次执行的工具轮次预算。

    优先级：``AIRY_AGENT_TOOL_ROUNDS``（运维显式收紧/放宽）>
    model.yaml 当前条目的 ``tool_rounds``（配置 SSoT）>
    :data:`TOOL_ROUNDS_DEFAULT`；结果夹取到 :data:`TOOL_ROUNDS_MAX`，
    与 agent_d 的 ``rounds_resolve`` 同名同界。
    单次 execute 解析一次，运行期改配置不影响进行中的任务。
    """
    try:
        n = int(os.environ.get("AIRY_AGENT_TOOL_ROUNDS", ""))
    except ValueError:
        n = 0
    if n <= 0:
        n = model_tool_rounds(model)
    if n <= 0:
        n = TOOL_ROUNDS_DEFAULT
    return min(n, TOOL_ROUNDS_MAX)


def _tool_fail_limit() -> int:
    """连败熔断阈值（env ``AIRY_TOOL_FAIL_LIMIT``，缺省 3，下限 1）。"""
    try:
        return max(1, int(os.environ.get("AIRY_TOOL_FAIL_LIMIT", "")))
    except ValueError:
        return DEFAULT_TOOL_FAIL_LIMIT


def _result_is_error(result: str) -> bool:
    """判定工具结果是否为错误载荷（JSON 对象含 error 键）。

    纯文本结果无法判定语义，宁可漏判也不误杀正常输出；JSON 解析
    失败（含截断）同样不判失败。
    """
    try:
        parsed = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(parsed, dict) and "error" in parsed


def _error_brief(result: str, limit: int = 160) -> str:
    """提取工具错误载荷中的可判读原因（供连败熔断回报聚合）。

    工具错误统一为 ``{"error": ...}`` JSON；解析失败时回退原文首段，
    空白折叠为单空格，便于拼进单行错误信息。
    """
    try:
        parsed = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    text = parsed.get("error") if isinstance(parsed, dict) else None
    if not isinstance(text, str) or not text:
        text = result
    return " ".join(text.split())[:limit]


def _truncate_tool_result(result: str) -> str:
    """按 MAX_TOOL_RESULT_CHARS 截断工具结果并追加显式标记。"""
    if len(result) <= MAX_TOOL_RESULT_CHARS:
        return result
    return (
        result[:MAX_TOOL_RESULT_CHARS]
        + f"\n...[tool result truncated at {MAX_TOOL_RESULT_CHARS} chars]"
    )


def _history_messages(context: AgentContext) -> List[Dict[str, Any]]:
    """取上下文携带的会话历史（OpenAI messages 形式）。

    只重放 user/assistant 轮次：system 提示词由执行体本轮自行注入，
    避免调用方历史里的 system 与本轮策略冲突。条目须为带字符串
    role/content 的 dict，其余丢弃（协议边界校验，不猜测修形）。
    """
    out: List[Dict[str, Any]] = []
    for item in getattr(context, "history", None) or []:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in ("user", "assistant") and isinstance(content, str):
            out.append({"role": role, "content": content})
    return out


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
            "models": {"system1": "deepseek-flash", "system2": "deepseek-flash"},
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
        self._update_activity()

        started = time.time()
        rounds = 0
        model = self._select_model(input_data)
        max_rounds = _rounds_limit(model)
        fail_limit = _tool_fail_limit()
        consecutive_failures = 0
        # 最近 K 个失败原因（K = fail_limit），熔断时一并回报，
        # 使上层无需回捞工具日志即可判读系统性故障的类型。
        recent_failures: List[str] = []
        # 工具结果指纹按单次 execute 归零：runner 主循环复用 agent
        # 处理多个请求，上一任务尾部的重复调用不得污染本任务的
        # loop 检测（与 _mid_progress_hinted 同理）。
        self._tool_call_fingerprints = []
        # 中途收敛提示按单次 execute 计数（runner 主循环复用 agent 处理
        # 多个请求，标志须在每次执行时重置，避免跨任务误跳提示）。
        self._mid_progress_hinted = False
        tool_defs = self._build_tool_definitions()

        # 项目上下文（AGENTS.md 等价物）作为 system 消息注入最前；
        # 未找到约定文件时保持原消息序列不变。会话历史紧随 system、
        # 位于本轮输入之前，使追问能读到上一轮的问答（S-P1-19）。
        messages: List[Dict[str, Any]] = inject_project_context(
            [
                {"role": "system", "content": self.system_prompt},
                *_history_messages(context),
                {"role": "user", "content": self._stringify_input(input_data)},
            ]
        )

        try:
            while rounds <= max_rounds:
                resp = await self.llm.chat(
                    messages=messages,
                    model=model,
                    tools=tool_defs if tool_defs else None,
                )
                choice = resp.get("choices", [{}])[0]
                msg = choice.get("message", {})
                tool_calls = msg.get("tool_calls")

                # 厂商输出上限截断检测（OpenAI finish_reason=length，
                # Anthropic 兼容值 max_tokens）：长生成（大文件 fs_write
                # 的 arguments JSON）会被厂商在输出上限处截成半截。历史
                # 实现不检测截断、解析失败后静默降级 {"_raw": ...}，LLM
                # 收到误导性的"缺参数"错误后以不同内容反复重试（指纹检
                # 测不触发），直至烧完 16 轮预算——蓝图节点空转超时的
                # 直接根源。截断属确定性失败（同请求必然再次截断），
                # 一经实锤立即 fast-fail，不消耗工具轮次。
                finish = str(choice.get("finish_reason") or "")
                output_truncated = finish in ("length", "max_tokens")

                if not tool_calls:
                    content = msg.get("content", "") or ""
                    if output_truncated:
                        content += (
                            "\n...[output truncated at provider token limit "
                            f"(finish_reason={finish})]"
                        )
                    return TaskResult(
                        success=True,
                        output=content,
                        metrics={
                            "model": resp.get("model", model),
                            "rounds": rounds,
                            "tokens": resp.get("usage", {}).get("total_tokens"),
                            "elapsed_ms": int((time.time() - started) * 1000),
                            "truncated": output_truncated,
                        },
                    )

                # 有 tool_calls：执行工具，结果回流到 messages
                messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
                # 结果感知 loop 保护（参照 Atom Code ToolLoopPolicy）：
                # 同一工具 + 同一参数 + 同一结果连续出现 3 次注入警告，4 次强制终止，
                # 防止模型对失败调用空转直至硬上限。
                # OpenAI 兼容 API 要求 tool 结果消息紧跟带 tool_calls 的 assistant
                # 消息，且本地 chat 模板对多 system 消息容错差；loop-hint 因此
                # 折叠进最后一条 tool 结果，不新增 system 消息（S-P1-11）。
                loop_hint = None
                # 中途收敛提示（q8i 生产实测）：模型可能在"不同参数/不同工具"间
                # 空转（指纹不同，不触发上面的连续重复检测）直至硬上限。到
                # max_rounds 的 60% 仍未收尾时注入一次性收敛提醒，引导
                # 模型停止无进展重试、基于已有工具结果给出结论。
                if rounds >= int(max_rounds * 0.6) and not getattr(
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
                    if output_truncated:
                        try:
                            json.loads(raw_args)
                        except json.JSONDecodeError as e:
                            return TaskResult(
                                success=False,
                                output=None,
                                error=(
                                    f"tool '{name}' arguments truncated by "
                                    f"provider output limit (finish_reason="
                                    f"{finish}): {e}. Split the content into "
                                    "smaller writes or raise "
                                    "$AIRY_LLM_MAX_TOKENS."
                                ),
                                error_code="TOOL_ARGS_TRUNCATED",
                            )
                    tool_result = await self._invoke_tool(name, raw_args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id", name),
                            "content": _truncate_tool_result(tool_result),
                        }
                    )
                    self._track_tool_call(name, raw_args, tool_result)
                    # 通用连败熔断：与同指纹重复检测互补——后者只捕获
                    # "同参同果"空转，这里捕获"次次失败但内容各异"的
                    # 系统性故障（q8i 实测主烧轮路径），达限立即终止。
                    if _result_is_error(tool_result):
                        consecutive_failures += 1
                        recent_failures.append(f"{name}: {_error_brief(tool_result)}")
                        del recent_failures[:-fail_limit]
                        if consecutive_failures >= fail_limit:
                            return TaskResult(
                                success=False,
                                output=None,
                                error=(
                                    f"tool calls failed {consecutive_failures} "
                                    "consecutive times; aborting instead of "
                                    "burning the round budget on a systemic "
                                    "failure. recent causes: "
                                    + " | ".join(recent_failures)
                                ),
                                error_code="TOOL_LOOP_CONSECUTIVE_FAILURES",
                            )
                    else:
                        consecutive_failures = 0
                        recent_failures.clear()
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
                if loop_hint is not None and messages[-1].get("role") == "tool":
                    messages[-1]["content"] = (
                        f"{messages[-1].get('content') or ''}\n\n{loop_hint}"
                    )
                rounds += 1
                logger.debug("LLMAgent[%s] tool round %d done", self.agent_id, rounds)

            return TaskResult(
                success=False,
                output=None,
                error=f"tool loop exceeded {max_rounds} rounds",
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
            # 上下文按消息独立构造：runner 复用同一 Agent 实例处理多个请求，
            # 沿用上次 execute 的上下文会把上一请求的会话标识与元数据带进
            # 本轮（跨请求串话）。会话标识随消息携带，缺省即无会话。
            meta = dict(message.metadata or {})
            ctx = AgentContext(
                agent_id=self.agent_id,
                session_id=meta.get("session_id"),
                metadata=meta,
            )
            result = await self.execute(message.content, ctx)
            return Message(
                message_type="result",
                content=result,
                sender=self.agent_id,
                receiver=message.sender,
            )
        return None

    # ── 内部辅助 ──────────────────────────────────────────

    def _contract_model(self, key: str) -> str:
        """契约 ``models.<key>`` 模型名解析（部署级覆盖 > 契约 > 客户端缺省）。

        环境变量 ``AIRY_AGENT_MODEL`` 覆盖一切（部署级强制单模型）；契约
        保持 provider 中立（不写死厂商模型名），缺省回落客户端 default
        （SSoT model.yaml 经 core 解析）。
        """
        env_model = os.environ.get("AIRY_AGENT_MODEL", "").strip()
        if env_model:
            return env_model
        models = self.contract.get("models") or {}
        return models.get(key) or getattr(self.llm, "default_model", "")

    def _model_system1(self) -> str:
        """契约 ``models.system1`` (t1-f 快思考)。"""
        return self._contract_model("system1")

    def _select_model(self, input_data: Any) -> str:
        """按任务复杂度选择思考模型（双思考分级）：

        - 简单任务（短输入 ≤ SIMPLE_TASK_MAX_CHARS）→ system1（t1-f 快思考，省 token）
        - 复杂任务 → system2（t2 深度思考）

        部署级 env ``AIRY_AGENT_MODEL`` 覆盖一切（见 :meth:`_contract_model`）。
        """
        text = self._stringify_input(input_data)
        if len(text) <= SIMPLE_TASK_MAX_CHARS:
            return self._model_system1()
        return self._model_system2()

    def _model_system2(self) -> str:
        """契约 ``models.system2`` (t2 主思考)。"""
        return self._contract_model("system2")

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
        except json.JSONDecodeError as e:
            # 畸形 arguments 显式回传（历史实现静默降级 {"_raw": ...}，
            # 工具侧 validator 只报"缺参数"，掩盖了 JSON 本身损坏的事实，
            # LLM 无法据此换策略）。错误头进入工具结果指纹，同一畸形参数
            # 重试可触发连续重复检测兜底 fast-fail。
            return json.dumps(
                {
                    "error": f"invalid tool arguments JSON: {e}",
                    "arguments_head": raw_args[:200],
                },
                ensure_ascii=False,
            )

        timeout = tool_timeout()
        try:
            if inspect.iscoroutinefunction(tool):
                result = await asyncio.wait_for(tool(args), timeout=timeout)
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(tool, args), timeout=timeout
                )
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False, default=str)
        except asyncio.TimeoutError:
            logger.warning(
                "LLMAgent[%s] tool '%s' timed out after %.0fs",
                self.agent_id, name, timeout,
            )
            return json.dumps(
                {
                    "error": f"tool '{name}' timed out after {timeout:.0f}s",
                    "error_code": "TOOL_TIMEOUT",
                },
                ensure_ascii=False,
            )
        except Exception as e:
            logger.warning("LLMAgent[%s] tool '%s' raised: %s", self.agent_id, name, e)
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    # ── 工具结果反哺控制与 loop 保护（参照 Atom Code 64KiB 上限 + ToolLoopPolicy） ──

    def _track_tool_call(self, name: str, raw_args: str, result: str) -> None:
        """记录最近一次 (工具, 参数, 结果) 指纹，供连续重复检测。

        指纹列表由 :meth:`execute` 在每次执行开始时归零。
        """
        self._tool_call_fingerprints.append((name, raw_args, result[:256]))

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
    "TOOL_ROUNDS_DEFAULT",
    "TOOL_ROUNDS_MAX",
    "SIMPLE_TASK_MAX_CHARS",
    "DEFAULT_TOOL_FAIL_LIMIT",
]
