# Copyright (c) 2026 SPHARX. All Rights Reserved.
"""契约能力声明校验与实测成功率（S-P1-14）。

覆盖三条要求：
  - 契约 capabilities 在构造期校验 schema 结构，坏契约拒绝装载；
  - 指定能力时按 input_schema 校验结构化输入，未知能力显式拒绝；
  - 成功率由执行体实测，调度读取实测值（0.0 不得被先验抹平）。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from orchestration.agents.llm import LLMAgent
from orchestration.core.agent import AgentContext
from orchestration.core.cap_schema import check_input, load_caps
from orchestration.strategies.dispatching.dispatching import _coerce_agent

CAP = {
    "name": "code_write",
    "description": "Write a file",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "lines": {"type": "integer"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["path"],
    },
    "output_schema": {
        "type": "object",
        "properties": {"written": {"type": "boolean"}},
        "required": ["written"],
    },
}


class _LLM:
    """单轮应答替身。"""

    default_model = "fast-model"

    async def chat(self, messages, model=None, tools=None, tool_choice="auto",
                   temperature=0.7, max_tokens=None):
        return {
            "choices": [
                {"message": {"role": "assistant", "content": "ok"},
                 "finish_reason": "stop"}
            ],
            "usage": {"total_tokens": 3},
            "model": model or self.default_model,
        }


def _agent(caps=None):
    contract = {
        "agent_id": "cap-agent",
        "models": {"system1": "fast-model", "system2": "deep-model"},
        "capabilities": [CAP] if caps is None else caps,
    }
    return LLMAgent(contract, llm=_LLM())


def _ctx(capability=None):
    ctx = AgentContext("cap-agent")
    if capability is not None:
        ctx.metadata["capability"] = capability
    return ctx


def test_load_caps_valid():
    caps = load_caps([CAP])
    assert set(caps) == {"code_write"}


def test_load_caps_none_and_empty():
    assert load_caps(None) == {}
    assert load_caps([]) == {}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.update(input_schema={"type": "object", "properties": {}, "required": ["x"]}),
        lambda c: c.update(output_schema={"type": "object", "properties": {"a": {"type": "blob"}}}),
        lambda c: c.update(name=""),
        lambda c: c.update(description=""),
        lambda c: c.pop("input_schema"),
        lambda c: c.update(input_schema={"type": "string", "items": {"type": "string"}}),
    ],
)
def test_load_caps_rejects_malformed(mutate):
    broken = {k: v for k, v in CAP.items()}
    broken["input_schema"] = dict(CAP["input_schema"])
    broken["output_schema"] = dict(CAP["output_schema"])
    mutate(broken)
    with pytest.raises(ValueError):
        load_caps([broken])


def test_load_caps_rejects_duplicate():
    with pytest.raises(ValueError):
        load_caps([CAP, dict(CAP)])


def test_agent_refuses_malformed_contract():
    bad = dict(CAP)
    bad["input_schema"] = {"type": "object", "properties": {}, "required": ["nope"]}
    with pytest.raises(ValueError):
        _agent([bad])


def test_check_input_accepts_valid():
    check_input(CAP, {"path": "a.c", "lines": 3, "tags": ["x"]})


@pytest.mark.parametrize(
    "value",
    [
        {"lines": 1},
        {"path": "a.c", "lines": True},
        {"path": "a.c", "tags": "x"},
        {"path": 5},
        ["a.c"],
    ],
)
def test_check_input_rejects(value):
    with pytest.raises(ValueError):
        check_input(CAP, value)


def test_unknown_capability_is_explicit():
    out = asyncio.run(_agent().execute("hello", _ctx("nope")))
    assert out.success is False
    assert out.error_code == "CAPABILITY_UNKNOWN"


def test_invalid_input_is_explicit():
    out = asyncio.run(_agent().execute({"lines": 1}, _ctx("code_write")))
    assert out.success is False
    assert out.error_code == "CAPABILITY_INPUT_INVALID"


def test_named_capability_passes_to_execution():
    out = asyncio.run(_agent().execute({"path": "a.c"}, _ctx("code_write")))
    assert out.success is True


def test_free_text_skips_capability_gate():
    assert asyncio.run(_agent().execute("hello", _ctx())).success is True


def test_success_rate_is_measured():
    agent = _agent()
    asyncio.run(agent.execute("hello", _ctx("nope")))
    assert agent.success_rate == 0.0
    asyncio.run(agent.execute("hello", _ctx()))
    assert agent.success_rate == 0.5


def test_new_agent_keeps_neutral_prior():
    assert _agent().success_rate == 1.0


def test_dispatch_reads_measured_rate():
    metrics = _coerce_agent(SimpleNamespace(agent_id="a", success_rate=0.0))
    assert metrics is not None and metrics.success_rate == 0.0
