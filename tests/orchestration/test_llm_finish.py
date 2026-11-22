# Copyright (c) 2026 SPHARX. All Rights Reserved.
"""finish_reason 归一与终结映射（S-P1-09）。

厂商原生标记（end_turn / max_tokens / SAFETY / refusal / tool_use 等）
按 llm_service_types.h 的规范词汇归一；无工具调用的终结轮按截断、拦截、
完成三态映射，截断不得当作最终答案。
"""

from __future__ import annotations

import pytest

from orchestration.agents.llm import LLMAgent, _finish_norm
from orchestration.core.agent import AgentContext


class ScriptedLLM:
    """返回单条脚本响应并记录调用次数的 LLM 测试替身。"""

    def __init__(self, response):
        self._response = response
        self.calls = 0

    async def chat(self, messages, model=None, tools=None, **kwargs):
        self.calls += 1
        return self._response


def _resp(content=None, tool_calls=None, finish="stop"):
    """构造最小 OpenAI chat 响应。"""
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "choices": [{"message": msg, "finish_reason": finish}],
        "usage": {"total_tokens": 7},
        "model": "scripted-model",
    }


async def _run(response, tool=None):
    """执行一次任务，返回 (TaskResult, ScriptedLLM)。"""
    agent = LLMAgent({"agent_id": "finish-agent"}, llm=ScriptedLLM(response))
    if tool is not None:
        agent.register_tool(
            "fs_write",
            tool,
            schema={"description": "write", "parameters": {"type": "object"}},
        )
    result = await agent.execute("task", AgentContext("t"))
    return result, agent.llm


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, "stop"),
        ("", "stop"),
        ("stop", "stop"),
        ("STOP", "stop"),
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("length", "length"),
        ("max_tokens", "length"),
        ("MAX_TOKENS", "length"),
        ("content_filter", "content_filter"),
        ("SAFETY", "content_filter"),
        ("refusal", "content_filter"),
        ("tool_calls", "tool_calls"),
        ("tool_use", "tool_calls"),
        ("vendor_odd", "vendor_odd"),
    ],
)
def test_finish_norm(raw, expected):
    """原生标记逐一归一到规范词汇，未识别标记原样保留。"""
    assert _finish_norm(raw) == expected


@pytest.mark.asyncio
async def test_max_tokens_alias_is_incomplete():
    """max_tokens 视同 length：前缀答案按不完整返回。"""
    result, llm = await _run(_resp(content="half", finish="max_tokens"))
    assert result.success is False
    assert result.error_code == "LLM_OUTPUT_TRUNCATED"
    assert result.metrics["finish_reason"] == "length"
    assert result.metrics["truncated"] is True
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_content_filter_terminal():
    """安全拦截且无正文 → 显式失败，不产生空答案成功。"""
    result, _ = await _run(_resp(content=None, finish="content_filter"))
    assert result.success is False
    assert result.error_code == "LLM_CONTENT_FILTERED"
    assert result.output is None
    assert "safety filter" in (result.error or "")


@pytest.mark.asyncio
async def test_safety_discards_tool_calls():
    """拦截轮的 tool_calls 不得执行，直接按失败返回。"""
    invoked = []
    call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "fs_write", "arguments": "{}"},
    }
    result, llm = await _run(
        _resp(tool_calls=[call], finish="SAFETY"),
        tool=lambda params: invoked.append(params) or "ok",
    )
    assert result.success is False
    assert result.error_code == "LLM_CONTENT_FILTERED"
    assert "discarded" in (result.error or "")
    assert invoked == []
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_unknown_marker_passes_through():
    """未识别标记按完成放行，实际标记记录进 metrics。"""
    result, _ = await _run(_resp(content="done", finish="vendor_odd"))
    assert result.success is True
    assert result.output == "done"
    assert result.metrics["finish_reason"] == "vendor_odd"
    assert result.metrics["truncated"] is False


@pytest.mark.asyncio
async def test_missing_finish_is_stop():
    """缺失终结标记按 stop 处理（与 C 侧同判）。"""
    result, _ = await _run(_resp(content="done", finish=""))
    assert result.success is True
    assert result.metrics["finish_reason"] == "stop"
