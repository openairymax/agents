# Copyright (c) 2026 SPHARX. All Rights Reserved.
"""LLMAgent 工具回路单测：厂商输出上限截断检测与畸形参数处理。

覆盖 0.1.16 蓝图挂死事故的根因路径（orchestration/agents/llm.py）：
  - finish_reason=length/max_tokens 且 arguments JSON 损坏
    → 立即 fast-fail TOOL_ARGS_TRUNCATED（确定性失败不烧工具轮次）
  - 截断但 arguments 合法 → 照常执行工具，不误杀
  - 纯 content 截断 → 显式截断标记 + metrics.truncated
  - 非截断畸形 arguments → 显式 "invalid tool arguments JSON" 回传
    （替代历史 {"_raw": ...} 静默降级），工具不被调用

0.1.17 R1-b 补充：通用连败熔断（TOOL_CONSECUTIVE_FAILURES）——
次次失败但参数/错误内容各不相同的空转不触发同指纹检测，
连败达阈值（默认 3，AIRY_TOOL_FAIL_LIMIT 可调）立即熔断。
"""

from __future__ import annotations

import json

import pytest

from orchestration.agents.llm import LLMAgent
from orchestration.core.agent import AgentContext


class ScriptedLLM:
    """按脚本顺序返回响应的最小 LLM 测试替身，记录每轮消息。"""

    def __init__(self, responses, default_model="scripted-model"):
        self._responses = list(responses)
        self.default_model = default_model
        self.calls = 0
        self.seen_messages = []

    async def chat(
        self,
        messages,
        model=None,
        tools=None,
        tool_choice="auto",
        temperature=0.7,
        max_tokens=None,
    ):
        self.calls += 1
        self.seen_messages.append(messages)
        assert self._responses, "ScriptedLLM: unexpected extra chat() call"
        return self._responses.pop(0)


def _resp(content=None, tool_calls=None, finish="stop"):
    """构造最小 OpenAI chat 响应。"""
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "choices": [{"message": msg, "finish_reason": finish}],
        "usage": {"total_tokens": 10},
        "model": "scripted-model",
    }


def _tool_call(name, raw_args, call_id="call_1"):
    """构造单个 OpenAI tool_call。"""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": raw_args},
    }


def _agent(llm, tool):
    contract = {"agent_id": "test-agent", "role": "processor"}
    agent = LLMAgent(contract, llm=llm)
    agent.register_tool(
        "fs_write",
        tool,
        schema={"description": "write file", "parameters": {"type": "object"}},
    )
    return agent


TOOL_MESSAGES = "invalid tool arguments JSON"


class TestLLMAgentTruncation:
    """厂商输出上限截断的 fast-fail 与降级路径。"""

    @pytest.mark.asyncio
    async def test_truncated_args_fast_fail(self):
        """截断 + arguments JSON 损坏 → TOOL_ARGS_TRUNCATED，不烧轮次。"""
        bad_json = '{"path": "big.txt", "content": "AAAA'
        llm = ScriptedLLM(
            [_resp(tool_calls=[_tool_call("fs_write", bad_json)], finish="length")]
        )
        invoked = []

        def tool(params):
            invoked.append(params)
            return "ok"

        result = await _agent(llm, tool).execute("write big file", AgentContext("t"))
        assert result.success is False
        assert result.error_code == "TOOL_ARGS_TRUNCATED"
        assert "truncated" in (result.error or "")
        assert "$AIRY_LLM_MAX_TOKENS" in (result.error or "")
        assert invoked == []
        assert llm.calls == 1

    @pytest.mark.asyncio
    async def test_truncated_but_valid_args_executes(self):
        """截断但 arguments 恰好完整 → 照常执行，不误杀。"""
        llm = ScriptedLLM(
            [
                _resp(
                    tool_calls=[
                        _tool_call("fs_write", '{"path": "a.txt", "content": "hi"}')
                    ],
                    finish="length",
                ),
                _resp(content="done"),
            ]
        )
        invoked = []

        def tool(params):
            invoked.append(params)
            return "written"

        result = await _agent(llm, tool).execute("write a.txt", AgentContext("t"))
        assert result.success is True
        assert result.output == "done"
        assert result.metrics["truncated"] is False
        assert invoked == [{"path": "a.txt", "content": "hi"}]
        assert llm.calls == 2
        # 第二轮回灌的 tool 结果消息可见
        tool_msgs = [m for m in llm.seen_messages[1] if m.get("role") == "tool"]
        assert any("written" in m["content"] for m in tool_msgs)

    @pytest.mark.asyncio
    async def test_content_truncation_marked(self):
        """纯 content 截断 → 显式标记 + metrics.truncated，不 fast-fail。"""
        llm = ScriptedLLM([_resp(content="partial analysis...", finish="length")])
        result = await _agent(llm, lambda p: "x").execute("analyze", AgentContext("t"))
        assert result.success is True
        assert "[output truncated" in result.output
        assert "finish_reason=length" in result.output
        assert result.metrics["truncated"] is True
        assert llm.calls == 1

    @pytest.mark.asyncio
    async def test_malformed_args_non_truncated(self):
        """非截断畸形 arguments → 显式错误回传，工具不被调用。"""
        llm = ScriptedLLM(
            [
                _resp(
                    tool_calls=[_tool_call("fs_write", '{"path": ')],
                    finish="tool_calls",
                ),
                _resp(content="reported and stopped"),
            ]
        )
        invoked = []

        def tool(params):
            invoked.append(params)
            return "ok"

        result = await _agent(llm, tool).execute("write", AgentContext("t"))
        assert result.success is True
        assert result.output == "reported and stopped"
        assert result.metrics["truncated"] is False
        assert invoked == []
        # 第二轮消息中的 tool 结果为显式 JSON 错误（非历史 {"_raw": ...} 降级）
        tool_msgs = [m for m in llm.seen_messages[1] if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        payload = json.loads(tool_msgs[0]["content"])
        assert payload["error"].startswith(TOOL_MESSAGES)
        assert payload["arguments_head"] == '{"path": '


class TestLLMAgentConsecutiveFailures:
    """通用连败熔断：不同参数/不同错误内容的空转 fast-fail。"""

    @pytest.mark.asyncio
    async def test_distinct_failures_fast_fail(self):
        """连败 3 次（参数各异，指纹不重复）→ TOOL_CONSECUTIVE_FAILURES。"""

        def tool(params):
            return {"error": f"no such file: {params['path']}"}

        llm = ScriptedLLM(
            [
                _resp(tool_calls=[_tool_call("fs_write", '{"path": "a.txt"}', "c1")]),
                _resp(tool_calls=[_tool_call("fs_write", '{"path": "b.txt"}', "c2")]),
                _resp(tool_calls=[_tool_call("fs_write", '{"path": "c.txt"}', "c3")]),
            ]
        )
        result = await _agent(llm, tool).execute("write", AgentContext("t"))
        assert result.success is False
        assert result.error_code == "TOOL_CONSECUTIVE_FAILURES"
        assert "consecutive" in (result.error or "")
        assert llm.calls == 3

    @pytest.mark.asyncio
    async def test_success_resets_counter(self):
        """成功清零连败计数：2 败 + 1 成 + 2 败不熔断，正常收尾。"""
        state = {"n": 0}

        def tool(params):
            state["n"] += 1
            if state["n"] != 3:
                return {"error": f"transient failure {state['n']}"}
            return "ok"

        llm = ScriptedLLM(
            [
                _resp(tool_calls=[_tool_call("fs_write", '{"p": 1}', "c1")]),
                _resp(tool_calls=[_tool_call("fs_write", '{"p": 2}', "c2")]),
                _resp(tool_calls=[_tool_call("fs_write", '{"p": 3}', "c3")]),
                _resp(tool_calls=[_tool_call("fs_write", '{"p": 4}', "c4")]),
                _resp(tool_calls=[_tool_call("fs_write", '{"p": 5}', "c5")]),
                _resp(content="done"),
            ]
        )
        result = await _agent(llm, tool).execute("write", AgentContext("t"))
        assert result.success is True
        assert result.output == "done"
        assert llm.calls == 6

    @pytest.mark.asyncio
    async def test_plain_text_result_not_counted(self):
        """纯文本结果不算失败：文本 + 文本 + 文本正常收尾，不熔断。"""
        llm = ScriptedLLM(
            [
                _resp(tool_calls=[_tool_call("fs_write", '{"p": 1}', "c1")]),
                _resp(tool_calls=[_tool_call("fs_write", '{"p": 2}', "c2")]),
                _resp(tool_calls=[_tool_call("fs_write", '{"p": 3}', "c3")]),
                _resp(content="done"),
            ]
        )
        result = await _agent(llm, lambda p: "plain text ok").execute(
            "write", AgentContext("t")
        )
        assert result.success is True
        assert result.output == "done"

    @pytest.mark.asyncio
    async def test_env_limit_override_and_fallback(self, monkeypatch):
        """AIRY_TOOL_FAIL_LIMIT=1 单败即熔断；非法值回退默认 3。"""
        from orchestration.agents.llm import _tool_fail_limit

        monkeypatch.setenv("AIRY_TOOL_FAIL_LIMIT", "1")

        def tool(params):
            return {"error": "boom"}

        llm = ScriptedLLM(
            [_resp(tool_calls=[_tool_call("fs_write", '{"p": 1}', "c1")])]
        )
        result = await _agent(llm, tool).execute("write", AgentContext("t"))
        assert result.success is False
        assert result.error_code == "TOOL_CONSECUTIVE_FAILURES"
        assert llm.calls == 1

        monkeypatch.setenv("AIRY_TOOL_FAIL_LIMIT", "not-a-number")
        assert _tool_fail_limit() == 3

        monkeypatch.setenv("AIRY_TOOL_FAIL_LIMIT", "0")
        assert _tool_fail_limit() == 1
