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
- 与 orchestration 内核解耦：仅依赖 ``orchestration.agents.LLMAgent`` 与 ``orchestration.core.llm``
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

from orchestration.agents import LLMAgent
from orchestration.core.llm import make_llm_client

logger = logging.getLogger(__name__)


#: tool_d 内置工具 schema（与 gateway GW_TOOLS_JSON 一一对应）。
#:
#: 注意：``required`` 必须与 tool_d 注册的参数集合一致——tool_d validator 对
#: 所有注册参数一律校验「必须存在」，若 schema 声明可选而 tool_d 要求必填
#: （如 fs_list 的 path），LLM 按 schema 不传参会导致工具校验失败。
BUILTIN_TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "fs_read": {
        "description": "Read a file's content",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    "fs_write": {
        "description": "Write content to a file (create/overwrite)",
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
        "description": "List a directory's entries",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    "shell_run": {
        "description": "Run a shell command, capture output",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "cwd": {"type": "string", "description": "Working directory (default: task dir)"},
            },
            "required": ["command"],
        },
    },
    "web_fetch": {
        "description": "Fetch a URL, return page body text",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    "fs_glob": {
        "description": "List files by glob pattern",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern (* ? **), relative to base"},
                "base": {"type": "string"},
            },
            "required": ["pattern"],
        },
    },
    "fs_grep": {
        "description": "Regex search file contents",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "POSIX ERE; skips .git/node_modules/build"},
                "path": {"type": "string"},
                "glob": {"type": "string"},
                "max_results": {"type": "integer"},
            },
            "required": ["pattern"],
        },
    },
    "fs_edit": {
        "description": "Replace exact text in a file (search-and-replace)",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old": {"type": "string", "description": "Exact text to match"},
                "new": {"type": "string", "description": "Replacement text"},
                "count": {"type": "integer"},
            },
            "required": ["path", "old", "new"],
        },
    },
    "web_search": {
        "description": "Search the web, return ranked results",
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
        # 任务工作目录（runner 每次 invoke 前设置，可随请求变化）：工具
        # dispatch 把相对路径解析为该目录内的绝对路径（tool_d 保持无状态，
        # 文件落点跟随任务而非 daemon cwd）。
        self.workspace_dir: Optional[str] = None
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

        2026-08-26 修复（防静默丢失）：SyscallProxy 不可用（_sys is None）
        时不再让工具凭空消失——那样 LLM 完全不知道自己有工具能力缺失，
        function-calling 直接无工具可用，subagent"无法使用工具"且无任何
        可诊断信号。改为注册显式报错的分发器：LLM 调用时返回明确的失败
        原因（SDK 缺失/损坏 → 提示修复路径），可诊断、可自愈。
        """
        if self._sys is None:
            for tool_id, schema in BUILTIN_TOOL_SCHEMAS.items():
                try:
                    self.register_tool(
                        tool_id,
                        self._make_unavailable_dispatcher(tool_id),
                        schema=schema,
                    )
                    logger.warning(
                        "AirymaxAgent[%s] SyscallProxy 不可用：工具 '%s' 已注册为"
                        "显式报错（调用将返回失败原因）",
                        self.agent_id, tool_id,
                    )
                except Exception as e:
                    logger.warning(
                        "AirymaxAgent[%s] register builtin tool '%s' failed: %s",
                        self.agent_id, tool_id, e,
                    )
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

    def _make_unavailable_dispatcher(
        self, tool_id: str
    ) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """SyscallProxy 缺失时工具的显式报错分发器（非桩实现）。

        返回与 tool_d 一致的失败结构 ``{success:false, error, exit_code}``，
        让 LLM 能把"工具不可用"转述给用户而不是无提示地消失。
        """
        def _dispatch(params: Dict[str, Any]) -> Dict[str, Any]:
            del params
            return {
                "success": False,
                "error": (
                    f"无法执行工具 {tool_id}：agentrt SyscallProxy 未初始化"
                    "（Python SDK 缺失、损坏或 socket 不可达）。"
                    "请检查 agentrt SDK 安装：pip install -e sdk-python，"
                    "并确认 mem_d/agent_d/tool_d daemon 正常运行。"
                ),
                "exit_code": 1,
            }

        return _dispatch

    #: 工具参数中含路径的字段（相对路径 → workspace 内绝对路径）。
    #: fs_glob 的 base 与 fs_grep 的 path 同为目录/文件定位字段。
    _PATH_TOOL_PARAMS = {
        "fs_read": ("path",),
        "fs_write": ("path",),
        "fs_list": ("path",),
        "fs_glob": ("base",),
        "fs_grep": ("path",),
        "fs_edit": ("path",),
    }
    #: 以命令形式执行的工具：注入 cwd 使相对路径在任务 workspace 内解析
    #: （tool_d 的 shell_run 支持可选 cwd 参数，子进程 chdir 后执行）。
    _CWD_TOOL_IDS = ("shell_run",)

    def _resolve_tool_path(self, path: str) -> str:
        """把工具路径参数解析为 workspace 内的绝对路径。

        相对路径（如 ``src/main.py``、``.``）基于任务 workspace 解析，
        绝对路径原样保留（agent 显式访问 workspace 之外的路径不被改写）。
        workspace 未设置或路径为空时原样返回，保证与 daemon cwd 兼容。
        """
        if not path or not self.workspace_dir:
            return path
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(self.workspace_dir, path))

    def _make_tool_dispatcher(
        self, tool_id: str
    ) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """构造单个工具的 dispatch 闭包。

        签名遵循 ``LLMAgent.register_tool`` 约定 ``(params: dict) -> Any``
        （LLMAgent._invoke_tool 会把它包装进 asyncio.to_thread 执行）。
        结果由 tool_d 直接返回：``{success, output, error, exit_code}``。
        """
        path_keys = self._PATH_TOOL_PARAMS.get(tool_id, ())

        def _dispatch(params: Dict[str, Any]) -> Dict[str, Any]:
            # 路径类工具：把相对路径解析为任务 workspace 内的绝对路径，
            # 使文件落点跟随任务（tool_d 的 fs_* 以自身 cwd 为基准执行，
            # 相对路径会落到 daemon cwd，见 builtin_fs.c）。
            if path_keys and isinstance(params, dict):
                params = dict(params)
                for key in path_keys:
                    val = params.get(key)
                    if isinstance(val, str) and val:
                        params[key] = self._resolve_tool_path(val)
            # 命令类工具：注入 cwd，使命令内的相对路径在任务 workspace 内
            # 解析（tool_d 的 shell_run 支持可选 cwd 参数，子进程 chdir）。
            if tool_id in self._CWD_TOOL_IDS and self.workspace_dir:
                params = dict(params) if isinstance(params, dict) else {}
                params.setdefault("cwd", self.workspace_dir)
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
