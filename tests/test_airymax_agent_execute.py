#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025-2026 SPHARX Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later OR Apache-2.0

# -*- coding: utf-8 -*-
"""``AirymaxAgent.execute`` best-effort 持久化行为测试套件。

覆盖 ``airymax_agents/base.py`` 中：
- ``execute()`` 在 ``syscall_proxy`` 非 None 时调用 ``_persist_input`` /
  ``_persist_result``；为 None 时跳过
- ``_persist_input`` / ``_persist_result`` 调用 ``syscall_proxy.memory_write``，
  传入 JSON payload + metadata
- 持久化失败仅 warning 不抛（best-effort），LLM 推理不受阻断
- metadata 含 agent_id / task_id / role / phase 字段
- 契约与系统提示词从子类目录自动加载
- ``get_agent`` 工厂与 ``AGENT_REGISTRY`` 行为

对齐 ``airymax-agent-standard.md`` §5.2
持久化协议与 §10 验证清单 "AirymaxAgent.execute 持久化 best-effort 行为有测试覆盖"。
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from airymax_agents import (
    AGENT_REGISTRY,
    ArchitectAgent,
    CodingAgent,
    ProductManagerAgent,
    get_agent,
)
from airymax_agents.base import AirymaxAgent
from orchestration.core.agent import TaskResult


# ── 辅助上下文管理器 ──────────────────────────────────────


class _MockParentExecute:
    """上下文管理器：mock 父类 ``LLMAgent.execute`` 返回固定 TaskResult。

    避免 LLM 真实调用，使测试聚焦于持久化逻辑。
    """

    def __init__(self, result=None):
        self.result = result or TaskResult(success=True, output="mock-llm-output")
        self._patch = None

    def __enter__(self):
        self._patch = patch(
            "airymax_agents.base.LLMAgent.execute",
            new=AsyncMock(return_value=self.result),
        )
        self.mock = self._patch.__enter__()
        return self.mock

    def __exit__(self, *exc):
        if self._patch is not None:
            self._patch.__exit__(*exc)


# ── execute() 持久化调度 ─────────────────────────────────


@pytest.mark.asyncio
async def test_execute_calls_persist_input_and_result(
    mock_llm, mock_syscall_proxy, agent_context
):
    """syscall_proxy 非 None 时，execute() 应调用 _persist_input 与 _persist_result。"""
    agent = ProductManagerAgent(llm=mock_llm, syscall_proxy=mock_syscall_proxy)
    with _MockParentExecute(), \
         patch.object(agent, "_persist_input", new=AsyncMock()) as spy_in, \
         patch.object(agent, "_persist_result", new=AsyncMock()) as spy_out:
        await agent.execute("input data", agent_context)
    spy_in.assert_awaited_once()
    spy_out.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_skips_persistence_when_syscall_proxy_is_none(
    mock_llm, agent_context
):
    """syscall_proxy=None 时，execute() 不调用 _persist_input / _persist_result。"""
    agent = ProductManagerAgent(llm=mock_llm, syscall_proxy=None)
    with _MockParentExecute(), \
         patch.object(agent, "_persist_input", new=AsyncMock()) as spy_in, \
         patch.object(agent, "_persist_result", new=AsyncMock()) as spy_out:
        await agent.execute("input data", agent_context)
    spy_in.assert_not_awaited()
    spy_out.assert_not_awaited()


# ── _persist_input / _persist_result 调用 memory_write ──


@pytest.mark.asyncio
async def test_persist_input_calls_memory_write(
    mock_llm, mock_syscall_proxy, agent_context
):
    """_persist_input 应调用 syscall_proxy.memory_write，传入 bytes payload + metadata。"""
    agent = ProductManagerAgent(llm=mock_llm, syscall_proxy=mock_syscall_proxy)
    await agent._persist_input("test input", agent_context)
    mock_syscall_proxy.memory_write.assert_called_once()
    args = mock_syscall_proxy.memory_write.call_args.args
    assert isinstance(args[0], bytes)  # payload
    assert isinstance(args[1], str)    # metadata JSON string
    payload = json.loads(args[0])
    assert payload["phase"] == "input"
    assert payload["input"] == "test input"


@pytest.mark.asyncio
async def test_persist_result_calls_memory_write(
    mock_llm, mock_syscall_proxy, agent_context
):
    """_persist_result 应调用 syscall_proxy.memory_write，传入 bytes payload + metadata。"""
    agent = ProductManagerAgent(llm=mock_llm, syscall_proxy=mock_syscall_proxy)
    result = TaskResult(success=True, output="done")
    await agent._persist_result(result, agent_context)
    mock_syscall_proxy.memory_write.assert_called_once()
    args = mock_syscall_proxy.memory_write.call_args.args
    assert isinstance(args[0], bytes)
    assert isinstance(args[1], str)
    payload = json.loads(args[0])
    assert payload["phase"] == "output"
    assert payload["success"] is True
    assert payload["output"] == "done"


# ── best-effort：持久化失败不抛 ─────────────────────────


@pytest.mark.asyncio
async def test_persist_input_failure_does_not_raise(mock_llm, agent_context):
    """memory_write 抛异常时，_persist_input 仅 warning，不向上抛。"""
    proxy = MagicMock()
    proxy.memory_write.side_effect = Exception("daemon unreachable")
    agent = ProductManagerAgent(llm=mock_llm, syscall_proxy=proxy)
    # 不应抛出任何异常
    await agent._persist_input("input", agent_context)
    assert proxy.memory_write.call_count == 1


@pytest.mark.asyncio
async def test_persist_result_failure_does_not_raise(mock_llm, agent_context):
    """memory_write 抛异常时，_persist_result 仅 warning，不向上抛。"""
    proxy = MagicMock()
    proxy.memory_write.side_effect = Exception("daemon unreachable")
    agent = ProductManagerAgent(llm=mock_llm, syscall_proxy=proxy)
    result = TaskResult(success=True, output="ok")
    await agent._persist_result(result, agent_context)
    assert proxy.memory_write.call_count == 1


@pytest.mark.asyncio
async def test_execute_continues_on_persistence_failure(mock_llm, agent_context):
    """持久化失败时，execute() 仍完成 LLM 推理并返回结果。"""
    proxy = MagicMock()
    proxy.memory_write.side_effect = Exception("daemon down")
    agent = ProductManagerAgent(llm=mock_llm, syscall_proxy=proxy)
    expected = TaskResult(success=True, output="llm-result-text")
    with _MockParentExecute(result=expected):
        result = await agent.execute("input", agent_context)
    assert result.success is True
    assert result.output == "llm-result-text"
    # 持久化至少被尝试过（input 阶段）
    assert proxy.memory_write.call_count >= 1


# ── metadata 字段验证 ────────────────────────────────────


@pytest.mark.asyncio
async def test_persist_input_metadata_contains_agent_id_task_id_role_phase(
    mock_llm, mock_syscall_proxy
):
    """_persist_input 的 metadata 含 agent_id/task_id/role/phase=input。"""
    ctx = SimpleNamespace(agent_id="pm-001", task_id="task-42")
    agent = ProductManagerAgent(llm=mock_llm, syscall_proxy=mock_syscall_proxy)
    await agent._persist_input("input data", ctx)
    args = mock_syscall_proxy.memory_write.call_args.args
    metadata = json.loads(args[1])
    assert metadata["agent_id"] == "pm-001"
    assert metadata["task_id"] == "task-42"
    assert metadata["role"] == "product_manager"
    assert metadata["phase"] == "input"


@pytest.mark.asyncio
async def test_persist_result_metadata_contains_phase_output(
    mock_llm, mock_syscall_proxy
):
    """_persist_result 的 metadata 含 phase=output。"""
    ctx = SimpleNamespace(agent_id="pm-001", task_id="task-42")
    agent = ProductManagerAgent(llm=mock_llm, syscall_proxy=mock_syscall_proxy)
    result = TaskResult(success=True, output="done")
    await agent._persist_result(result, ctx)
    args = mock_syscall_proxy.memory_write.call_args.args
    metadata = json.loads(args[1])
    assert metadata["phase"] == "output"
    assert metadata["agent_id"] == "pm-001"
    assert metadata["role"] == "product_manager"


# ── 契约与提示词自动装配 ─────────────────────────────────


def test_airymax_agent_loads_contract_from_subclass_dir(mock_llm):
    """ProductManagerAgent 从子类目录加载 contract.json (agent_id=product_manager_v1)。"""
    agent = ProductManagerAgent(llm=mock_llm)
    assert agent.agent_id == "product_manager_v1"
    assert agent.contract["role"] == "product_manager"


def test_airymax_agent_loads_system_prompt(mock_llm):
    """ProductManagerAgent 从 prompts/system.md 加载非空系统提示词。"""
    agent = ProductManagerAgent(llm=mock_llm)
    assert agent.system_prompt
    assert len(agent.system_prompt) > 0


# ── get_agent 工厂与注册表 ───────────────────────────────


def test_get_agent_returns_correct_class(mock_llm):
    """get_agent('architect') 返回 ArchitectAgent 实例。"""
    agent = get_agent("architect", llm=mock_llm)
    assert isinstance(agent, ArchitectAgent)
    assert isinstance(agent, AirymaxAgent)
    assert agent.agent_id == "architect_v1"


def test_get_agent_unknown_role_normalizes_to_coding(mock_llm):
    """get_agent 传入未知 role 时归一化到 coding 并返回 AirymaxAgent 实例。"""
    agent = get_agent("unknown_role", llm=mock_llm)
    assert isinstance(agent, CodingAgent)
    assert isinstance(agent, AirymaxAgent)


def test_agent_registry_has_eleven_roles():
    """AGENT_REGISTRY 含 11 个 role 条目。"""
    assert len(AGENT_REGISTRY) == 11
    expected_roles = {
        "product_manager", "architect", "backend",
        "frontend", "devops", "security", "tester", "coding",
        "data_engineer", "reviewer", "analyst",
    }
    assert set(AGENT_REGISTRY.keys()) == expected_roles


# ── 内置工具注册（tool_d 接线） ─────────────────────────


BUILTIN_TOOL_IDS = (
    "fs_read",
    "fs_write",
    "fs_list",
    "shell_run",
    "web_fetch",
    "fs_glob",
    "fs_grep",
    "fs_edit",
    "fs_delete",
    "web_search",
)


def test_builtin_tools_registered_when_syscall_proxy_provided(
    mock_llm, mock_syscall_proxy
):
    """注入 syscall_proxy 时，10 个 tool_d 内置工具应注册为 function-calling 工具。"""
    agent = ProductManagerAgent(llm=mock_llm, syscall_proxy=mock_syscall_proxy)
    for tool_id in BUILTIN_TOOL_IDS:
        assert agent.get_tool(tool_id) is not None, f"{tool_id} not registered"
        assert tool_id in agent._tool_schemas, f"{tool_id} schema missing"
        assert "parameters" in agent._tool_schemas[tool_id]


def test_builtin_tools_registered_with_unavailable_dispatcher_without_syscall_proxy(
    mock_llm,
):
    """syscall_proxy=None 时注册显式报错分发器（非静默消失），调用返回失败原因。"""
    agent = ProductManagerAgent(llm=mock_llm, syscall_proxy=None)
    dispatcher = agent.get_tool("fs_read")
    assert dispatcher is not None, "工具不应静默消失，应注册显式报错分发器"
    result = dispatcher({"path": "/tmp/a.txt"})
    assert result["success"] is False
    assert result["exit_code"] == 1
    assert "未初始化" in result["error"]


@pytest.mark.asyncio
async def test_builtin_tool_dispatcher_calls_tool_execute(
    mock_llm, mock_syscall_proxy
):
    """dispatch 应调用 syscall_proxy.tool_execute 并返回其结果。"""
    mock_syscall_proxy.tool_execute.return_value = {
        "success": True, "output": "file content", "error": "", "exit_code": 0,
    }
    agent = ProductManagerAgent(llm=mock_llm, syscall_proxy=mock_syscall_proxy)
    dispatcher = agent.get_tool("fs_read")
    result = dispatcher({"path": "/tmp/a.txt"})
    mock_syscall_proxy.tool_execute.assert_called_once_with(
        "fs_read", {"path": "/tmp/a.txt"}, agent.agent_id
    )
    assert result["output"] == "file content"


def test_builtin_tool_schemas_required_aligned_with_tool_d():
    """BUILTIN_TOOL_SCHEMAS 的 required 必须齐备（tool_d validator 全参数必填）。"""
    from airymax_agents.base import BUILTIN_TOOL_SCHEMAS

    assert set(BUILTIN_TOOL_SCHEMAS) == set(BUILTIN_TOOL_IDS)
    for tool_id, schema in BUILTIN_TOOL_SCHEMAS.items():
        params = schema["parameters"]
        assert params["type"] == "object"
        required = params.get("required") or []
        assert required, f"{tool_id} missing required"
        for field in required:
            assert field in params["properties"], (
                f"{tool_id}.{field} required but not declared"
            )
