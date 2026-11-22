#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025-2026 SPHARX Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later OR Apache-2.0

"""runner 请求上下文隔离与追问历史（0.1.21 S-P1-19）。

runner 进程跨请求复用同一 Agent 实例，历史实现把上下文缓存在实例上，
第二轮回读的是第一轮的会话标识与元数据（跨请求串话）。此处从协议
边界验证：每次请求独立构造上下文、会话标识与历史随请求携带、
未携带历史时上一轮不会被带入。
"""

from __future__ import annotations

import json

import pytest

from airymax_agents.runner import _execute_once


class _EchoAgent:
    """记录每次 execute 收到的上下文，并按输入回显。"""

    def __init__(self):
        self.contexts = []

    async def execute(self, input_data, context):
        from orchestration.core.agent import TaskResult

        self.contexts.append(context)
        return TaskResult(success=True, output=f"echo:{input_data}")


@pytest.mark.asyncio
async def test_second_turn_carries_history():
    """第二轮请求携带会话标识与历史，上下文与第一轮彼此独立。"""
    agent = _EchoAgent()
    history = [
        {"role": "user", "content": "第一轮问题"},
        {"role": "assistant", "content": "第一轮答案"},
    ]

    first = await _execute_once(agent, "a-1", "第一轮问题", "s-1", None)
    second = await _execute_once(agent, "a-1", "第二轮追问", "s-1", history)

    assert json.loads(first)["success"] is True
    assert json.loads(second)["output"] == "echo:第二轮追问"

    ctx1, ctx2 = agent.contexts
    assert ctx1 is not ctx2, "每次请求必须独立构造上下文"
    assert ctx1.session_id == "s-1" and ctx2.session_id == "s-1"
    assert ctx1.history == []
    assert ctx2.history == history


@pytest.mark.asyncio
async def test_no_history_carries_nothing():
    """未携带历史的请求不得继承上一轮历史。"""
    agent = _EchoAgent()
    history = [{"role": "user", "content": "上一轮问题"}]

    await _execute_once(agent, "a-1", "第一轮", "s-1", history)
    await _execute_once(agent, "a-1", "第二轮", "s-1", None)

    assert agent.contexts[0].history == history
    assert agent.contexts[1].history == []


@pytest.mark.asyncio
async def test_parent_task_id_is_per_request():
    """父任务标识随请求落地，顶层任务不带上一轮的父任务。"""
    agent = _EchoAgent()

    await _execute_once(agent, "a-1", "子任务", "s-1", None, "task-parent")
    await _execute_once(agent, "a-1", "顶层任务", "s-1", None)

    assert agent.contexts[0].parent_task_id == "task-parent"
    assert agent.contexts[1].parent_task_id is None


@pytest.mark.asyncio
async def test_usage_metrics_are_returned():
    """成功响应回传 TaskResult.metrics，供父任务汇总子任务用量。"""
    from orchestration.core.agent import TaskResult

    class _UsageAgent:
        async def execute(self, input_data, context):
            return TaskResult(
                success=True,
                output="done",
                metrics={"model": "mock-model", "rounds": 1, "tokens": 12},
            )

    resp = json.loads(await _execute_once(_UsageAgent(), "a-1", "任务", "s-1"))

    assert resp["success"] is True
    assert resp["metrics"]["tokens"] == 12
    assert resp["metrics"]["rounds"] == 1
