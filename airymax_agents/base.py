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
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from openlab.agents import LLMAgent
from openlab.core.llm import make_llm_client

logger = logging.getLogger(__name__)


#: tool_d 内置工具 schema（与 gateway GW_TOOLS_JSON 一一对应）。
#:
#: 注意：``required`` 必须与 tool_d 注册的参数集合一致——tool_d validator 对
#: 所有注册参数一律校验「必须存在」，若 schema 声明可选而 tool_d 要求必填
#: （如 fs_list 的 path），LLM 按 schema 不传参会导致工具校验失败。
BUILTIN_TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "fs_read": {
        "description": "Read a file's content from the local filesystem",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    "fs_write": {
        "description": "Write content to a local file (creates or overwrites)",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    "fs_list": {
        "description": "List entries of a local directory (JSON array)",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    "shell_run": {
        "description": "Execute a shell command and capture its output",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    "web_fetch": {
        "description": "Fetch a web page over HTTP(S) and return its body text",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    "fs_glob": {
        "description": (
            "List files matching a glob pattern under a base directory "
            "(supports * ? and ** for recursive match; returns newline-separated paths)"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "base": {"type": "string"},
            },
            "required": ["pattern"],
        },
    },
    "fs_grep": {
        "description": (
            "Search file contents with a POSIX extended regular expression "
            "(returns relpath:lineno:text lines; skips .git/node_modules/build etc.)"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "glob": {"type": "string"},
                "max_results": {"type": "integer"},
            },
            "required": ["pattern"],
        },
    },
    "fs_edit": {
        "description": (
            "Replace an exact string in a file (search-and-replace edit). "
            "Use for surgical edits instead of rewriting whole files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old": {"type": "string"},
                "new": {"type": "string"},
                "count": {"type": "integer"},
            },
            "required": ["path", "old", "new"],
        },
    },
    "web_search": {
        "description": "Search the web (DuckDuckGo) and return ranked title/url/snippet results",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
}


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
        同时把 9 个 tool_d 内置工具（fs_read/fs_write/fs_list/shell_run/
        web_fetch/fs_glob/fs_grep/fs_edit/web_search）注册为
        function-calling 工具，让 LLM 真正具备读写文件、搜索代码、
        执行命令、抓取/搜索网页的能力（dispatch 直连 tool_d，见
        :meth:`_register_builtin_tools`）。
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
            # 决策 D（2026-08-09）：执行体（worker）主推理默认 t1-p——优先读
            # $AIRY_MODEL_T1P，未设置回落 make_llm_client 既有解析（AIRY_AGENT_MODEL
            # > OPENAI_MODEL > gpt-4o-mini）。
            llm=llm
            if llm is not None
            else make_llm_client(default_model=os.environ.get("AIRY_MODEL_T1P")),
            system_prompt=system_prompt,
        )
        # agentrt 系统调用代理（None 时为纯 Python 向后兼容模式）
        self._sys = syscall_proxy
        # 注入代理后注册 tool_d 内置工具（注册失败降级为纯 LLM，不阻断）
        self._register_builtin_tools()
        logger.debug(
            "AirymaxAgent[%s] loaded from %s (role=%s, syscall=%s)",
            self.agent_id,
            agent_dir,
            self.ROLE,
            "on" if self._sys is not None else "off",
        )

    # ── 内置工具（tool_d 接线） ───────────────────────────

    def _register_builtin_tools(self) -> None:
        """把 5 个 tool_d 内置工具注册为 LLM function-calling 工具。

        仅当注入 ``syscall_proxy`` 时注册（纯 Python 向后兼容模式跳过）；
        dispatch 经 ``syscall_proxy.tool_execute`` 直连 tool_d。单个工具
        注册失败仅记录 warning 并跳过该工具，不阻断 Agent 初始化
        （降级为纯 LLM 模式）。
        """
        if self._sys is None:
            return
        for tool_id, schema in BUILTIN_TOOL_SCHEMAS.items():
            try:
                self.register_tool(
                    tool_id,
                    self._make_tool_dispatcher(tool_id),
                    schema=schema,
                )
                logger.debug(
                    "AirymaxAgent[%s] builtin tool registered: %s",
                    self.agent_id, tool_id,
                )
            except Exception as e:
                logger.warning(
                    "AirymaxAgent[%s] register builtin tool '%s' failed "
                    "(non-fatal, degraded to pure LLM): %s",
                    self.agent_id, tool_id, e,
                )

    def _make_tool_dispatcher(
        self, tool_id: str
    ) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """构造单个工具的 dispatch 闭包。

        签名遵循 ``LLMAgent.register_tool`` 约定 ``(params: dict) -> Any``
        （LLMAgent._invoke_tool 会把它包装进 asyncio.to_thread 执行）。
        结果由 tool_d 直接返回：``{success, output, error, exit_code}``。
        """

        def _dispatch(params: Dict[str, Any]) -> Dict[str, Any]:
            # P0 交互式审批：透传真实 agent_id，tool_d 按该主体做 ACL 判定，
            # 未授权工具进入 pending（AIRY_TOOL_APPROVAL_MODE=interactive）。
            return self._sys.tool_execute(
                tool_id, params if isinstance(params, dict) else {}, self.agent_id
            )

        return _dispatch

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


__all__ = ["AirymaxAgent", "BUILTIN_TOOL_SCHEMAS"]
