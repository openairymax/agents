#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025-2026 SPHARX Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later OR Apache-2.0

# -*- coding: utf-8 -*-
"""AirymaxAgent 测试套件共享 fixtures 与 sys.path 装配。

将 ``airymax_agents/`` 与 ``orchestration/`` 包根目录加入 ``sys.path``，
使测试可经 ``from airymax_agents import get_agent`` 与
``from orchestration.core.agent import TaskResult`` 直接导入，无需额外环境变量。
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# ── sys.path 装配 ─────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
# tests/ → agents/ (含 airymax_agents/ 包)
_AGENTS_DIR = os.path.dirname(_HERE)
# agents/ → orchestration/ 包（多智能体编排内核，原 openlab 叶子仓并入）
_ORCH_DIR = os.path.join(_AGENTS_DIR, "orchestration")

for _p in (_AGENTS_DIR, _ORCH_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── fixtures ─────────────────────────────────────────────


@pytest.fixture
def mock_syscall_proxy():
    """Mock 的 agentrt SyscallProxy，memory_* 方法返回合理预设值。"""
    proxy = MagicMock()
    proxy.memory_write.return_value = "record-mock-1"
    proxy.memory_get.return_value = b"mock-data"
    proxy.memory_search.return_value = [("record-mock-1", 0.95)]
    proxy.memory_delete.return_value = True
    proxy.agent_spawn.return_value = "agent-mock-1"
    proxy.agent_terminate.return_value = True
    proxy.agent_invoke.return_value = "mock-invoke-output"
    proxy.agent_list.return_value = ["agent-mock-1"]
    return proxy


@pytest.fixture
def mock_llm():
    """Mock 的 LLM 客户端，避免真实 OpenAI 调用。

    AirymaxAgent.__init__ 仅将其存为 ``self.llm``；测试中通过 mock
    父类 ``LLMAgent.execute`` 跳过实际 LLM 调用，故此 mock 仅需满足
    构造器对 ``llm`` 参数的存取需求。
    """
    llm = MagicMock()
    llm.default_model = "mock-model"
    return llm


@pytest.fixture
def agent_context():
    """简单 namespace 上下文，含 agent_id / task_id 供持久化 metadata 使用。"""
    return SimpleNamespace(agent_id="test-agent-001", task_id="test-task-001")
