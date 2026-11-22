#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025-2026 SPHARX Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later OR Apache-2.0

# -*- coding: utf-8 -*-
"""子代理编排测试（§25.6 六层判据与 §25.7 B5 出口）。

覆盖：
  - 协议层：子进程请求携带 session_id/workspace_dir/parent_task_id；
  - 工具层：spawn/wait/cancel 的并发、深度与总量上限显式拒绝；
  - 隔离层：子代理由父级授权决定只读，独立 workspace；
  - 回收层：终态回传耗时与 token；父取消级联终止子进程；
  - 端到端：父任务并行 2 个子任务、其一失败、父汇总后继续工作；
  - 崩溃容错：子进程异常退出不拖垮父进程。
"""

import json
import os
import sys
import time

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_AGENTS_DIR = os.path.dirname(_HERE)
_ORCH_DIR = os.path.join(_AGENTS_DIR, "orchestration")
for _p in (_AGENTS_DIR, _ORCH_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from airymax_agents import resolve_role  # noqa: E402
from orchestration.core.subagent import SubagentManager  # noqa: E402

# 真实子进程用 runner；这些测试不依赖网络，MockLLM 在无 key 时确定性输出。
_ECHO_CHILD = [
    sys.executable, "-u", "-c",
    "import json,sys\n"
    "print(json.dumps({'ready': True}), flush=True)\n"
    "req = json.loads(sys.stdin.readline())\n"
    "print(json.dumps({'success': True, 'output': json.dumps(req),"
    " 'metrics': {'tokens': 7, 'rounds': 1, 'model': 'echo'}}), flush=True)\n",
]
_CRASH_CHILD = [
    sys.executable, "-u", "-c",
    "import json,sys\n"
    "print(json.dumps({'ready': True}), flush=True)\n"
    "sys.stdin.readline()\n"
    "sys.exit(7)\n",
]
_SLOW_CHILD = [
    sys.executable, "-u", "-c",
    "import json,sys,time\n"
    "print(json.dumps({'ready': True}), flush=True)\n"
    "sys.stdin.readline()\n"
    "time.sleep(60)\n",
]


@pytest.fixture(autouse=True)
def _offline(monkeypatch, tmp_path):
    """隔离宿主密钥与配置，保证子进程走 MockLLM 且不触网。

    不设厂商时工厂会回落到 model.yaml 的本地条目（127.0.0.1:11435），
    干净的 CI 上没有该进程，子代理会连不上。钉住一个云端厂商并清掉
    它的 key，才是文档化的离线路径：无 key → MockLLMClient。
    """
    monkeypatch.setenv("AIRY_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("AIRY_LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("AIRY_SUBAGENT_DEPTH", raising=False)


def _manager(tmp_path, **kwargs):
    defaults = dict(
        parent_agent_id="parent-agent",
        parent_task_id="task-1",
        parent_session_id="sess-1",
        role_resolver=resolve_role,
        workspace_dir=str(tmp_path / "workspace"),
        readonly=True,
    )
    defaults.update(kwargs)
    return SubagentManager(**defaults)


def _terminal(manager, child_id, tries=3):
    """等待子代理终态；wait 每次最多阻塞一个切片。"""
    result = None
    for _ in range(tries):
        result = manager.wait({"child_id": child_id})
        if result.get("status") != "running":
            return result
    return result


def test_child_completes_with_metrics(tmp_path):
    """真实 runner 子进程跑通，终态回传输出与用量指标。"""
    manager = _manager(tmp_path)
    try:
        spawned = manager.spawn({"role": "coding", "input": "实现一个求和函数"})
        assert spawned["status"] == "running", spawned.get("error") or spawned
        assert spawned["readonly"] is True

        result = _terminal(manager, spawned["child_id"])

        assert result["status"] == "completed", result.get("error") or result
        assert result["output"]
        assert isinstance(result["tokens"], int)
        assert result["elapsed_ms"] >= 0
    finally:
        manager.shutdown()


def test_request_carries_parent_scope(tmp_path):
    """协议层：子请求带会话、父任务与独立 workspace。"""
    manager = _manager(tmp_path, child_argv=_ECHO_CHILD)
    try:
        spawned = manager.spawn({"role": "coding", "input": "回显请求"})
        result = _terminal(manager, spawned["child_id"])

        assert result["status"] == "completed", result
        request = json.loads(result["output"])
        assert request["parent_task_id"] == "task-1"
        assert request["session_id"].startswith("sess-1:")
        assert request["workspace_dir"] == spawned["workspace"]
        assert spawned["workspace"].startswith(os.path.join(str(tmp_path), "workspace"))
        assert result["tokens"] == 7
    finally:
        manager.shutdown()


def test_unknown_role_rejected_without_spawn(tmp_path):
    """未知角色确定性失败：不创建子进程，错误可判读。"""
    manager = _manager(tmp_path)
    try:
        rejected = manager.spawn({"role": "not-a-role", "input": "任务"})

        assert rejected["status"] == "rejected"
        assert rejected["child_id"] is None
        assert "not-a-role" in rejected["error"]
    finally:
        manager.shutdown()


def test_depth_limit_rejected(tmp_path):
    """递归防护：深度达到上限后不再派生孙代理。"""
    manager = _manager(tmp_path, depth=1)
    try:
        rejected = manager.spawn({"role": "coding", "input": "任务"})

        assert rejected["status"] == "rejected"
        assert "depth" in rejected["error"]
    finally:
        manager.shutdown()


def test_writable_requires_parent_authority(tmp_path):
    """隔离层：父只读时即使请求 writable 也强制只读。"""
    manager = _manager(tmp_path, readonly=True)
    try:
        spawned = manager.spawn({"role": "coding", "input": "任务", "writable": True})

        assert spawned["status"] == "running", spawned.get("error") or spawned
        assert spawned["readonly"] is True
    finally:
        manager.shutdown()


def test_parallel_children_one_fails_parent_summarizes(tmp_path):
    """B5 判据：父任务并行 2 子任务，其一失败，父汇总后继续。"""
    manager = _manager(tmp_path)
    try:
        first = manager.spawn({"role": "coding", "input": "子任务 A"})
        second = manager.spawn({"role": "not-a-role", "input": "子任务 B"})

        assert first["status"] == "running", first.get("error") or first
        assert second["status"] == "rejected"

        first_result = _terminal(manager, first["child_id"])
        second_result = manager.wait({"child_id": second["child_id"]}) if second["child_id"] else second

        summary = {
            "children": [
                {"status": first_result["status"], "output": first_result["output"]},
                {"status": second_result["status"], "error": second_result["error"]},
            ],
            "failed": sum(1 for item in (first_result, second_result)
                          if item["status"] != "completed"),
            "tokens": first_result["tokens"],
        }

        assert summary["failed"] == 1, summary
        assert summary["children"][0]["status"] == "completed", first_result
        assert summary["children"][0]["output"]
        assert summary["children"][1]["error"]

        # 父进程与编排器仍可继续工作（失败不拖垮父任务）
        third = manager.spawn({"role": "coding", "input": "子任务 C"})
        assert third["status"] == "running", third.get("error") or third
        third_result = _terminal(manager, third["child_id"])
        assert third_result["status"] == "completed", third_result.get("error") or third_result
    finally:
        manager.shutdown()


def test_child_crash_is_contained(tmp_path):
    """崩溃容错：子进程异常退出只让本次派生失败，父进程继续。"""
    crashing = _manager(tmp_path, child_argv=_CRASH_CHILD)
    try:
        spawned = crashing.spawn({"role": "coding", "input": "任务"})
        result = _terminal(crashing, spawned["child_id"])

        assert result["status"] == "failed"
        assert "exit_code=7" in result["error"]
    finally:
        crashing.shutdown()

    # 崩溃不影响后续任务：新编排器仍能正常派生并拿到结果
    recovered = _manager(tmp_path)
    try:
        again = recovered.spawn({"role": "coding", "input": "恢复后的任务"})
        outcome = _terminal(recovered, again["child_id"])
        assert outcome["status"] == "completed", outcome.get("error") or outcome
    finally:
        recovered.shutdown()


def test_cancel_cascade_stops_child(tmp_path):
    """取消级联：cancel 终止存活子进程，不留下孤儿。"""
    manager = _manager(tmp_path, child_argv=_SLOW_CHILD)
    try:
        spawned = manager.spawn({"role": "coding", "input": "长任务"})
        assert spawned["status"] == "running"

        result = manager.cancel({"child_id": spawned["child_id"]})

        assert result["status"] == "cancelled"
        assert result["elapsed_ms"] < 30_000
        assert manager._children[spawned["child_id"]]["proc"].poll() is not None
        assert manager.wait({"child_id": spawned["child_id"]})["status"] == "cancelled"
    finally:
        manager.shutdown()


def test_shutdown_cascades_to_live_children(tmp_path):
    """父任务结束时 shutdown 终止所有存活子进程（finally 路径）。"""
    manager = _manager(tmp_path, child_argv=_SLOW_CHILD)
    spawned = manager.spawn({"role": "coding", "input": "长任务"})
    proc = manager._children[spawned["child_id"]]["proc"]

    manager.shutdown()

    assert proc.poll() is not None
    assert manager.spawn({"role": "coding", "input": "任务"})["status"] == "rejected"


def test_wait_slice_returns_running(tmp_path, monkeypatch):
    """等待切片到期返回 running，父任务可先做别的事。"""
    monkeypatch.setenv("AIRY_SUBAGENT_WAIT_SLICE_S", "0.3")
    manager = _manager(tmp_path, child_argv=_SLOW_CHILD)
    try:
        spawned = manager.spawn({"role": "coding", "input": "长任务"})
        started = time.monotonic()

        result = manager.wait({"child_id": spawned["child_id"]})

        assert result["status"] == "running"
        assert result["remaining_ms"] >= 0
        assert time.monotonic() - started < 10
    finally:
        manager.shutdown()


def test_depth_limits_tool_registration(tmp_path, monkeypatch):
    """派生工具只在任务作用域内存在，深度上限处不注册。"""
    from types import SimpleNamespace

    from airymax_agents import get_agent

    monkeypatch.setenv("AIRY_SUBAGENT_DEPTH", "1")
    child = get_agent("coding")
    child.workspace_dir = str(tmp_path)
    child._begin_subagent_scope(SimpleNamespace(task_id="task-c", session_id=None))
    assert "agent_spawn" not in child._tools
    child._end_subagent_scope()

    monkeypatch.delenv("AIRY_SUBAGENT_DEPTH")
    parent = get_agent("coding")
    parent.workspace_dir = str(tmp_path)
    assert "agent_spawn" not in parent._tools
    parent._begin_subagent_scope(SimpleNamespace(task_id="task-p", session_id=None))
    assert {"agent_spawn", "agent_wait", "agent_cancel"} <= set(parent._tools)
    parent._end_subagent_scope()
    assert "agent_spawn" not in parent._tools


def test_task_scope_opens_and_closes_manager(tmp_path, monkeypatch):
    """每次任务建立独立作用域，退出时连同派生工具一起回收。"""
    from types import SimpleNamespace

    from airymax_agents import get_agent

    monkeypatch.delenv("AIRY_SUBAGENT_DEPTH", raising=False)
    agent = get_agent("coding")
    agent.workspace_dir = str(tmp_path)
    assert "agent_spawn" not in agent._tools

    agent._begin_subagent_scope(SimpleNamespace(task_id="task-9", session_id="sess-9"))
    assert agent._subagent_manager is not None
    assert agent._subagent_manager.depth == 0
    assert "agent_spawn" in agent._tools

    agent._end_subagent_scope()
    assert agent._subagent_manager is None
    assert "agent_spawn" not in agent._tools
