"""Airymax Agent runner — agent_d 子进程入口。

由 agent_d 的 service.c::agent_spawn_child 通过 execvp 启动::

    python3 -m airymax_agents.runner --spec <spec_json>

通信协议（行分隔 JSON，每行一个请求/响应）:

  - 请求（从 stdin 读取）::

      {"agent_id": "...", "input": "...", "session_id": "...", "history": [...]}

    ``session_id`` 与 ``history`` 均可选：前者标识会话，后者为
    OpenAI messages 形式的历史轮次（user/assistant），在本次输入前
    重放以支持多轮追问。每次请求独立构造上下文，不跨请求复用。

  - 响应（写入 stdout）::

      {"success": true, "output": "..."}
      {"success": false, "error": "..."}

生命周期:

  1. 解析 ``--spec`` 参数（Agent 规格 JSON）
  2. 构建 :class:`SyscallProxy`（IPC 后端，连接 mem_d/agent_d）
  3. 通过 :func:`get_agent` 实例化 Agent
  4. 调用 ``agent.initialize()``
  5. 主循环：逐行读取 stdin 请求 → ``agent.execute()`` → 写 stdout 响应
  6. EOF（stdin 关闭）时正常退出

设计原则 (Simplicity is beauty):

  - 与 agent_d 的 C 侧协议严格对齐（行分隔 JSON）
  - SyscallProxy 初始化失败不阻断运行（退化为纯 Python LLM 模式），
    但降级事实必须显式上报：stderr 记 error，ready 帧带 ``degraded``/``reason``，
    agent_d 据此在守护日志里留痕（0.1.21 S-P0-11）。
  - 单次 execute 异常不退出进程（写错误响应后继续下一轮）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from typing import Any, Optional, Tuple

logger = logging.getLogger("runner")

# Spawn-time cwd (the agent_d working directory): the baseline a request
# without workspace_dir chdirs back to. The runner process is reused across
# requests; without this reset the second task would silently run inside the
# first task's workspace (cwd drift).
_RUNNER_BASE_CWD = os.getcwd()

# 内容级失败信号（防 L2 缓存中毒）：LLM 最终回复（无工具调用）被 orchestration
# 无条件判为 success=True，即使内容明确表达「无法完成/工具被拒/推诿用户」。
# 命中任一模式即把 execute 结果降级为失败，避免 agent_d 将失败回复原样
# 上报、CLI 误 absorb 为 SUCCESS 写入 L2 语义缓存（失败建议被当作成功重放）。
# 只匹配「陈述失败/推诿」的强信号，条件句（如「如果无法运行则…」）不命中。
# 2026-08-19：收窄宽泛模式——"我无法""无法执行"会被只读验证者（validator）
# 的能力说明命中（"我无法直接执行 fs_write，但验证通过"是角色受限，不是任务
# 失败）。失败信号改为目标导向（无法完成/无法创建/...），"无法执行"要求
# 后接任务目标动词。
_FAILURE_RE = re.compile(
    r"(我无法(完成|创建|写入|生成|安装|启动|停止|提供|执行任务)|我暂时无法(完成|创建|写入|"
    r"生成|安装|启动|停止|提供|执行任务)|无法完成任务|无法完成|无法创建|无法写入|无法生成|"
    r"无法安装|无法启动|无法停止|无法提供|未能完成|未能创建|未能执行|"
    r"被禁用|被拒绝|禁用了|不允许使用|permission denied|"
    r"请自行|请手动|请在终端|你可以直接在|你可以在终端|需要你手动|需要您手动)",
    re.IGNORECASE,
)

# 明确成功结论：输出同时给出通过/达标结论时，上面的失败信号只指工具能力
# 受限（如"我无法直接执行写操作，但验证通过"），不代表任务失败，不降级。
_SUCCESS_OVERRIDE_RE = re.compile(
    r"(验证通过|校验通过|检查通过|任务目标已达成|目标已达成|已完成|成功达成|满足要求)",
    re.IGNORECASE,
)


def _content_declares_failure(text: str) -> bool:
    """回复文本是否明确表达任务未能完成（供调用方降级 success）。"""
    if not _FAILURE_RE.search(text):
        return False
    # 明确的成功结论覆盖：只读验证者能力说明 + 通过结论 → 判定成功。
    if _SUCCESS_OVERRIDE_RE.search(text):
        return False
    return True


# 只读验证角色（归一化后的具体执行体角色）：验证者与认知审查者只允许
# 读取与联网，禁止任何写操作。除系统提示词约束外，此处做能力隔离——
# 直接从 agent 的工具注册表移除写工具，LLM 看不到也调不到（模型行为
# 不可控，仅靠提示词无法可靠阻止越界，见 2026-08-19 tester_v1 越界
# fs_write 根因）。判定先经角色归一化（validator/verifier → tester），
# 抽象别名直传不再漏网。权威词汇经 A-IPC agent.vocab 加载（C 侧
# agent_vocab SSoT），下列本地表仅为 agent_d 不可达时的启动回退。
_READONLY_LOCAL = frozenset({"tester", "reviewer"})

# 写操作工具：从只读角色的工具注册表与 schema 中移除。工具清单由 tool_d
# 目录生成（S-P1-15），故凡是会改文件/改仓库/改系统状态的一律列入本表，
# 新增写工具时必须同步追加，否则只读角色会被目录自动放行。
WRITE_TOOLS = frozenset({
    "fs_write", "fs_edit", "fs_delete", "shell_run",
    "git_exec", "git_apply", "tool_create", "priv_exec",
})

# 运行期角色词汇表（agent.vocab 返回值），None 表示未加载/加载失败。
_ROLE_VOCAB: Optional[dict] = None


def _load_role_vocab(syscall_proxy: Optional[Any]) -> None:
    """经 A-IPC agent.vocab 加载角色词汇表（失败回退本地，非致命）。"""
    global _ROLE_VOCAB
    if syscall_proxy is None:
        return
    try:
        vocab = syscall_proxy.agent_vocab()
        if isinstance(vocab, dict) and vocab.get("roles"):
            _ROLE_VOCAB = vocab
            logger.info("role vocabulary loaded via agent.vocab (%d roles)",
                        len(vocab["roles"]))
    except Exception as e:
        logger.warning("agent.vocab unavailable, keep local vocab: %s", e)


def _canonical_role(role: str) -> str:
    """角色归一化（驱动语义）：别名 → 具体执行体，未登记 → 兜底并留痕。

    与 ``get_agent`` 的严格解析分工明确：这里保证计划节点总能被驱动；
    真正构造执行体时不再二次兜底，避免把未知角色悄悄当成 coding 跑。
    """
    if _ROLE_VOCAB:
        for entry in _ROLE_VOCAB.get("aliases", []):
            if isinstance(entry, dict) and entry.get("alias") == role:
                return entry["role"]
        if role in _ROLE_VOCAB.get("roles", []):
            return role
        fallback = _ROLE_VOCAB.get("fallback", "coding")
        logger.warning("unknown agent role %r, driven as %r", role, fallback)
        return fallback
    from airymax_agents import ROLE_ALIASES, ROLE_FALLBACK, AGENT_REGISTRY

    if role in AGENT_REGISTRY:
        return role
    if role in ROLE_ALIASES:
        return ROLE_ALIASES[role]
    logger.warning("unknown agent role %r, driven as %r", role, ROLE_FALLBACK)
    return ROLE_FALLBACK


def _readonly_roles() -> frozenset:
    if _ROLE_VOCAB and _ROLE_VOCAB.get("readonly"):
        return frozenset(_ROLE_VOCAB["readonly"])
    return _READONLY_LOCAL


def _apply_role_tool_limits(agent: Any, role: str) -> None:
    """按角色收紧 agent 可用工具（判定基于归一化后的具体角色）。

    只读角色移除写工具（fs_write/fs_edit/fs_delete/shell_run）：LLM
    function-calling 的 schema 与其可调用注册表同时移除，实现能力隔离。
    其他角色保持完整工具集。
    """
    canon = _canonical_role(role)
    if canon not in _readonly_roles():
        return
    tools = getattr(agent, "_tools", None)
    schemas = getattr(agent, "_tool_schemas", None)
    for tid in WRITE_TOOLS:
        if isinstance(tools, dict):
            tools.pop(tid, None)
        if isinstance(schemas, dict):
            schemas.pop(tid, None)
    logger.info("read-only role %s (canon=%s): write tools isolated (%d)",
                role, canon, len(WRITE_TOOLS))


def _parse_spec(spec_str: str) -> dict:
    """解析 --spec 参数为 dict，失败抛 ValueError。"""
    try:
        spec = json.loads(spec_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid spec JSON: {e}") from e
    if not isinstance(spec, dict):
        raise ValueError(f"spec must be a JSON object, got {type(spec).__name__}")
    return spec


def _build_syscall_proxy() -> Tuple[Optional[Any], str]:
    """构建 agentrt SyscallProxy（IPC 后端）。

    返回 ``(proxy, reason)``：失败时 proxy 为 None、reason 为失败原因，
    且以 error 级别留痕——降级不能是静默的（0.1.21 S-P0-11）。
    """
    try:
        from agentrt.syscall import SyscallProxy

        return SyscallProxy(backend="ipc"), ""
    except Exception as e:
        logger.error("SyscallProxy init failed (degraded, no tools): %s", e)
        # ready 帧受 agent_d 的定长读缓冲约束，超长会被截断成非法 JSON；
        # 这里截到 120 字符，完整堆栈仍在 stderr 日志里。
        return None, str(e)[:120]


def _make_response(output: str = "", success: bool = True, error: str = "") -> str:
    """构建单行 JSON 响应。"""
    resp: dict = {"success": success}
    if success:
        resp["output"] = output
    else:
        resp["error"] = error
    return json.dumps(resp, ensure_ascii=False)


def _make_ready(role: str, degraded: Optional[list] = None,
                reason: str = "") -> str:
    """构建 ready 信号（spawn 存活校验协议，P0-2）。

    agent_d 的 service.c::agent_service_spawn 在 fork 后读取子进程
    stdout 的第一行，校验 ``{"ready":true}`` 确认子进程存活，失败则
    判定 spawn 失败并回收，不再静默回退 stub。该行在 spawn 阶段被
    消费，不会与 invoke 的 ``{"success":...}`` 响应混淆。

    ``degraded`` 非空时附加降级清单与原因：agent_d 只把该帧当日志与
    存活信号消费，不认识新字段时仍判 ready（向后兼容），认识的版本
    会把降级写进守护日志（0.1.21 S-P0-11）。
    """
    frame: dict = {"ready": True, "role": role}
    if degraded:
        frame["degraded"] = list(degraded)
        if reason:
            frame["reason"] = reason
    return json.dumps(frame, ensure_ascii=False)


async def _execute_once(agent: Any, agent_id: str, user_input: str,
                        session_id: Optional[str] = None,
                        history: Optional[list] = None) -> str:
    """执行一次 Agent 调用，返回 JSON 响应字符串。

    上下文按请求独立构造：会话标识与历史随请求携带（缺省即无会话、
    无历史），不复用上一请求的上下文（S-P1-19 跨请求串话根因）。
    """
    from orchestration.core.agent import AgentContext

    ctx = AgentContext(
        agent_id=agent_id,
        session_id=session_id,
        history=list(history or []),
    )
    result = await agent.execute(user_input, ctx)
    output = getattr(result, "output", None)
    if output is None:
        output = ""
    success = bool(getattr(result, "success", True))
    if not success:
        err = getattr(result, "error", None) or "agent execution failed"
        return _make_response(success=False, error=str(err))
    # 内容级失败降级：LLM 回复明确表达未能完成任务（工具被拒/无法执行/
    # 推诿用户）时按失败上报，防止 L2 缓存把失败建议当成功吸收。
    if _content_declares_failure(str(output)):
        return _make_response(
            success=False,
            error="agent reported it could not complete the task: " + str(output)[:200],
        )
    return _make_response(output=str(output), success=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Airymax Agent runner (agent_d subprocess)",
    )
    parser.add_argument(
        "--spec",
        required=True,
        help='Agent 规格 JSON 字符串，如 {"role":"product_manager"}',
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=os.environ.get("AIRY_RUNNER_LOG_LEVEL", "WARNING"),
        format="[%(asctime)s.%(msecs)03d %(levelname)s runner] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stderr,
    )

    # 1. 解析 spec
    try:
        spec = _parse_spec(args.spec)
    except ValueError as e:
        print(_make_response(success=False, error=f"bad spec: {e}"), flush=True)
        return 1

    role = spec.get("role")
    if not role:
        print(_make_response(success=False, error="spec missing 'role' field"), flush=True)
        return 1

    # 2. 构建 SyscallProxy（可选，失败退化为纯 Python 模式并显式上报）
    syscall_proxy, proxy_error = _build_syscall_proxy()
    # 角色词汇 SSoT：优先经 A-IPC 取权威词汇，失败保持本地回退表
    _load_role_vocab(syscall_proxy)

    # 3. 实例化 Agent
    try:
        from airymax_agents import get_agent

        agent = get_agent(role, syscall_proxy=syscall_proxy)
        # 角色工具白名单（只读角色能力隔离）：须在 initialize 前应用，
        # 确保 function-calling 始终看不到被移除的工具。
        _apply_role_tool_limits(agent, role)
        asyncio.run(agent.initialize())
    except Exception as e:
        logger.exception("agent init failed (role=%s)", role)
        print(_make_response(success=False, error=f"agent init failed: {e}"), flush=True)
        return 1

    logger.info("runner ready (role=%s, agent_id=%s)", role, agent.agent_id)

    # 项目上下文（AGENTS.md 等价物）加载状态：启动即注入 Agent 系统上下文
    from orchestration.core.project_context import find_project_context

    _project_ctx = find_project_context()
    if _project_ctx:
        logger.info("project context loaded (%d chars)", len(_project_ctx))
    else:
        logger.info("no project context file found (AGENTS.md/CLAUDE.md)")

    # P0-2：spawn 存活校验协议 — 初始化完成后向 agent_d 发送 ready 信号。
    # 降级清单取自 agent 自身（实例属性），保证「上报的能力状态」与
    # 「实际注册表」同源，不靠调用方二次推断（0.1.21 S-P0-11）。
    degraded = list(getattr(agent, "degraded", []) or [])
    print(_make_ready(role, degraded, proxy_error), flush=True)

    # 4. 主循环：逐行读取请求 → 执行 → 写响应
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            agent_id = req.get("agent_id", "unknown")
            user_input = req.get("input", "")
            session_id = req.get("session_id") or None
            history = req.get("history")
            if not isinstance(history, list):
                history = None
        except (json.JSONDecodeError, AttributeError) as e:
            print(_make_response(success=False, error=f"bad request: {e}"), flush=True)
            continue

        # Decision E workspace isolation: when agent_d forwards an isolated
        # workspace_dir (wh_agent_invoke -> agent.invoke -> child request),
        # chdir into it and only then expose it as agent.workspace_dir, so
        # both the tool cwd base (tool_d fs_* resolve relative to the runner
        # cwd) and the per-request cwd param point at a live directory.
        # chdir failure fails THIS request fast: continuing would resolve
        # every relative path against a stale cwd and burn the whole
        # tool-round budget on [Errno 2] errors (incident 0.1.16).
        # Requests without workspace_dir chdir back to the spawn-time cwd
        # (_RUNNER_BASE_CWD) to undo the previous task's chdir — the runner
        # process is reused across requests (cwd drift).
        ws_dir = req.get("workspace_dir") or ""
        if ws_dir:
            try:
                os.chdir(ws_dir)
            except OSError as e:
                logger.error("workspace_dir %s unavailable: %s", ws_dir, e)
                print(
                    _make_response(success=False, error=f"workspace_dir unavailable: {e}"),
                    flush=True,
                )
                continue
            agent.workspace_dir = ws_dir
        else:
            os.chdir(_RUNNER_BASE_CWD)
            agent.workspace_dir = None
        logger.debug("runner workspace: %s", agent.workspace_dir or _RUNNER_BASE_CWD)

        try:
            resp = asyncio.run(_execute_once(agent, agent_id, user_input, session_id, history))
        except Exception as e:
            logger.exception("execute failed (agent_id=%s)", agent_id)
            resp = _make_response(success=False, error=f"execute failed: {e}")

        print(resp, flush=True)

    # EOF：正常退出
    return 0


if __name__ == "__main__":
    sys.exit(main())
