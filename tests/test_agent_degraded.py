#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025-2026 SPHARX Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later OR Apache-2.0

# -*- coding: utf-8 -*-
"""S-P0-11：SyscallProxy 缺失时的显式降级契约。

断言四件事：

- ``degraded`` 由构造条件派生，与工具暴露集合同源，不靠调用方二次推断；
- 降级时不把 schema 交给 LLM（模型看到的清单 = 实际可执行集合），
  避免模型按「有工具」的假设把工具轮次耗在必然失败的调用上；
- 工具入口仍保留显式报错分发器（可诊断，不静默消失）；
- runner 的 ready 帧携带 degraded/reason，agent_d 据此写守护日志。
"""

import json

from airymax_agents import ProductManagerAgent
from airymax_agents.base import BUILTIN_TOOL_SCHEMAS
from airymax_agents.runner import _make_ready


def test_missing_proxy_withholds_schemas(mock_llm):
    """无 proxy：工具入口保留，但一条 schema 都不交给模型。"""
    agent = ProductManagerAgent(llm=mock_llm)

    assert agent.degraded == ["syscall_proxy"]
    assert agent._tool_schemas == {}
    assert set(agent._tools) == set(BUILTIN_TOOL_SCHEMAS)


def test_wired_proxy_exposes_every_tool(mock_llm, mock_syscall_proxy):
    """有 proxy：能力清单完整，且无降级信号。"""
    agent = ProductManagerAgent(llm=mock_llm, syscall_proxy=mock_syscall_proxy)

    assert agent.degraded == []
    assert set(agent._tool_schemas) == set(BUILTIN_TOOL_SCHEMAS)


def test_prompt_states_the_degradation(mock_llm):
    """降级说明必须进入系统提示词，且只有降级时进入。"""
    degraded = ProductManagerAgent(llm=mock_llm)
    assert "工具后端不可用" in degraded.system_prompt

    wired = ProductManagerAgent(llm=mock_llm, syscall_proxy=object())
    assert "工具后端不可用" not in wired.system_prompt


def test_unavailable_tool_names_the_cause(mock_llm):
    """直接调用（诊断脚本/单测）仍能拿到确定原因，而不是 KeyError。"""
    agent = ProductManagerAgent(llm=mock_llm)
    result = agent._tools["fs_read"]({"path": "/tmp/irrelevant"})

    assert result["success"] is False
    assert "SyscallProxy" in result["error"]
    assert result["exit_code"] == 1


def test_ready_frame_carries_degradation():
    """ready 帧：默认形态不变，降级时追加 degraded/reason。"""
    frame = json.loads(
        _make_ready("pm", ["syscall_proxy"], "ImportError: no sdk")
    )
    assert frame == {
        "ready": True,
        "role": "pm",
        "degraded": ["syscall_proxy"],
        "reason": "ImportError: no sdk",
    }
    assert json.loads(_make_ready("pm")) == {"ready": True, "role": "pm"}
