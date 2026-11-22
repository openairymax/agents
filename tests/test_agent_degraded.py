#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025-2026 SPHARX Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later OR Apache-2.0

# -*- coding: utf-8 -*-
"""S-P0-11 / S-P1-15：SyscallProxy 缺失时的显式降级契约。

断言四件事：

- ``degraded`` 由构造条件派生，与工具暴露集合同源，不靠调用方二次推断；
- 降级时不注册任何工具（模型看到的清单 = 实际可执行集合），
  避免模型按「有工具」的假设把工具轮次耗在必然失败的调用上；
- 有 proxy 时工具清单完全由 tool_d 目录生成，策略侧无平行静态表；
- runner 的 ready 帧携带 degraded/reason，agent_d 据此写守护日志。
"""

import json

from airymax_agents import ProductManagerAgent
from airymax_agents.runner import _make_ready


def test_missing_proxy_withholds_schemas(mock_llm):
    """无 proxy：一条 schema 都不交给模型，也不留不可执行的入口。"""
    agent = ProductManagerAgent(llm=mock_llm)

    assert agent.degraded == ["syscall_proxy"]
    assert agent._tool_schemas == {}
    assert agent._tools == {}


def test_wired_proxy_exposes_every_tool(mock_llm, mock_syscall_proxy):
    """有 proxy：模型清单与目录条目一致，且无降级信号。"""
    agent = ProductManagerAgent(llm=mock_llm, syscall_proxy=mock_syscall_proxy)

    assert agent.degraded == []
    assert set(agent._tool_schemas) == set(agent._tools)
    assert len(agent._tool_schemas) == len(mock_syscall_proxy.tool_list.return_value)


def test_prompt_states_the_degradation(mock_llm, mock_syscall_proxy):
    """降级说明必须进入系统提示词，且只有降级时进入。"""
    degraded = ProductManagerAgent(llm=mock_llm)
    assert "工具后端不可用" in degraded.system_prompt

    wired = ProductManagerAgent(llm=mock_llm, syscall_proxy=mock_syscall_proxy)
    assert "工具后端不可用" not in wired.system_prompt


def test_missing_proxy_reports_no_tool(mock_llm):
    """直接查询（诊断脚本/单测）得到 None 而不是一个必然失败的桩。"""
    agent = ProductManagerAgent(llm=mock_llm)

    assert agent.get_tool("fs_read") is None
    assert agent.degraded == ["syscall_proxy"]


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
