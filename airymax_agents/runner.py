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

# 内容级失败信号（防 L2 缓存中毒）：LLM 最终回复（无工具调用）被 openlab
# 无条件判为 success=True，即使内容明确表达「无法完成/工具被拒/推诿用户」。
# 命中任一模式即把 execute 结果降级为失败，避免 agent_d 将失败回复原样
# 上报、CLI 误 absorb 为 SUCCESS 写入 L2 语义缓存（失败建议被当作成功重放）。
# 只匹配「陈述失败/推诿」的强信号，条件句（如「如果无法运行则…」）不命中。
_FAILURE_RE = re.compile(
    r"(我无法|我暂时无法|无法完成|无法创建|无法执行|无法写入|无法生成|"
    r"无法安装|无法启动|无法停止|无法提供|未能完成|未能创建|未能执行|"
    r"被禁用|被拒绝|禁用了|不允许使用|permission denied|"
    r"请自行|请手动|请在终端|你可以直接在|你可以在终端|需要你手动|需要您手动)",
    re.IGNORECASE,
)


def _content_declares_failure(text: str) -> bool:
    """回复文本是否明确表达任务未能完成（供调用方降级 success）。"""
    return bool(_FAILURE_RE.search(text))


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
    from openlab.core.agent import AgentContext

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
        format="[runner %(levelname)s] %(message)s",
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
        asyncio.run(agent.initialize())
    except Exception as e:
        logger.exception("agent init failed (role=%s)", role)
        print(_make_response(success=False, error=f"agent init failed: {e}"), flush=True)
        return 1

    logger.info("runner ready (role=%s, agent_id=%s)", role, agent.agent_id)

    # 项目上下文（AGENTS.md 等价物）加载状态：启动即注入 Agent 系统上下文
    from openlab.core.project_context import find_project_context

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
            # Decision E workspace isolation: when agent_d forwards an isolated
            # workspace_dir (wh_agent_invoke -> agent.invoke -> child request),
            # chdir into it so the agent acts inside the task workspace instead
            # of the daemon's cwd (avoids tool-round exhaustion exploring the
            # host tree). Directory missing is non-fatal (best-effort).
            ws_dir = req.get("workspace_dir") or ""
            if ws_dir:
                try:
                    os.chdir(ws_dir)
                    logger.debug("runner chdir to workspace: %s", ws_dir)
                except OSError as e:
                    logger.warning("runner chdir to workspace %s failed: %s", ws_dir, e)
        except (json.JSONDecodeError, AttributeError) as e:
            print(_make_response(success=False, error=f"bad request: {e}"), flush=True)
            continue

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
