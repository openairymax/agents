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
import sys
from typing import Any, Optional

logger = logging.getLogger("runner")


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
