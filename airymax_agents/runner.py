"""Airymax Agent runner — agent_d 子进程入口。

由 agent_d 的 service.c::agent_spawn_child 通过 execvp 启动::

    python3 -m airymax_agents.runner --spec <spec_json>

通信协议（行分隔 JSON，每行一个请求/响应）:

  - 请求（从 stdin 读取）::

      {"agent_id": "...", "input": "..."}

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
  - SyscallProxy 初始化失败不阻断运行（退化为纯 Python LLM 模式）
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
from typing import Any, Optional

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


# 只读验证角色：任务管线中的 validator/verifier 与认知审查 reviewer 只允许
# 读取与联网，禁止任何写操作。除系统提示词约束外，此处做能力隔离——直接
# 从 agent 的工具注册表移除写工具，LLM 看不到也调不到（模型行为不可控，
# 仅靠提示词无法可靠阻止越界，见 2026-08-19 tester_v1 越界 fs_write 根因）。
READONLY_ROLES = frozenset({"validator", "verifier", "reviewer"})

# 写操作工具：从只读角色的工具注册表与 schema 中移除。
WRITE_TOOLS = frozenset({"fs_write", "fs_edit", "fs_delete", "shell_run"})


def _apply_role_tool_limits(agent: Any, role: str) -> None:
    """按角色收紧 agent 可用工具。

    只读角色移除写工具（fs_write/fs_edit/fs_delete/shell_run）：LLM
    function-calling 的 schema 与其可调用注册表同时移除，实现能力隔离。
    其他角色保持完整工具集。
    """
    if role not in READONLY_ROLES:
        return
    tools = getattr(agent, "_tools", None)
    schemas = getattr(agent, "_tool_schemas", None)
    for tid in WRITE_TOOLS:
        if isinstance(tools, dict):
            tools.pop(tid, None)
        if isinstance(schemas, dict):
            schemas.pop(tid, None)
    logger.info("read-only role %s: write tools isolated (%d)", role, len(WRITE_TOOLS))


def _parse_spec(spec_str: str) -> dict:
    """解析 --spec 参数为 dict，失败抛 ValueError。"""
    try:
        spec = json.loads(spec_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid spec JSON: {e}") from e
    if not isinstance(spec, dict):
        raise ValueError(f"spec must be a JSON object, got {type(spec).__name__}")
    return spec


def _build_syscall_proxy() -> Optional[Any]:
    """构建 agentrt SyscallProxy（IPC 后端）。失败返回 None（非致命）。"""
    try:
        from agentrt.syscall import SyscallProxy

        return SyscallProxy(backend="ipc")
    except Exception as e:
        logger.warning("SyscallProxy init failed (non-fatal): %s", e)
        return None


def _make_response(output: str = "", success: bool = True, error: str = "") -> str:
    """构建单行 JSON 响应。"""
    resp: dict = {"success": success}
    if success:
        resp["output"] = output
    else:
        resp["error"] = error
    return json.dumps(resp, ensure_ascii=False)


def _make_ready(role: str) -> str:
    """构建 ready 信号（spawn 存活校验协议，P0-2）。

    agent_d 的 service.c::agent_service_spawn 在 fork 后读取子进程
    stdout 的第一行，校验 ``{"ready":true}`` 确认子进程存活，失败则
    判定 spawn 失败并回收，不再静默回退 stub。该行在 spawn 阶段被
    消费，不会与 invoke 的 ``{"success":...}`` 响应混淆。
    """
    return json.dumps({"ready": True, "role": role}, ensure_ascii=False)


async def _execute_once(agent: Any, agent_id: str, user_input: str) -> str:
    """执行一次 Agent 调用，返回 JSON 响应字符串。"""
    from orchestration.core.agent import AgentContext

    ctx = AgentContext(agent_id=agent_id)
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

    # 2. 构建 SyscallProxy（可选，失败退化为纯 Python 模式）
    syscall_proxy = _build_syscall_proxy()

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

    # P0-2：spawn 存活校验协议 — 初始化完成后向 agent_d 发送 ready 信号
    print(_make_ready(role), flush=True)

    # 4. 主循环：逐行读取请求 → 执行 → 写响应
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            agent_id = req.get("agent_id", "unknown")
            user_input = req.get("input", "")
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
            resp = asyncio.run(_execute_once(agent, agent_id, user_input))
        except Exception as e:
            logger.exception("execute failed (agent_id=%s)", agent_id)
            resp = _make_response(success=False, error=f"execute failed: {e}")

        print(resp, flush=True)

    # EOF：正常退出
    return 0


if __name__ == "__main__":
    sys.exit(main())
