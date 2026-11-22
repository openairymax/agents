# Copyright (c) 2026 SPHARX. All Rights Reserved.
"""任务分级回归样例集（S-P1-08）。

锁定双思考选型不再按输入长度一刀切：
  - 短而多步/改码的请求必须升级 t2；
  - 长而机械的请求不因长度升级，长度最多计 1 分；
  - 显式 metadata（complexity/tier/task_type）优先于内容信号；
  - 单个弱信号不足以升级。

同时验证 :class:`LLMAgent` 的分级结果确实落到契约 models.system1/2。
"""

from __future__ import annotations

import pytest

from orchestration.agents.llm import LLMAgent
from orchestration.core.agent import AgentContext
from orchestration.core.tier import (
    DEEP_SCORE,
    TIER_DEEP,
    TIER_FAST,
    pick_tier,
    tier_score,
)


class _StubLLM:
    """只提供 default_model 的最小客户端；分级不触发网络调用。"""

    default_model = "stub-model"


def _agent() -> LLMAgent:
    contract = {
        "agent_id": "tier-agent",
        "role": "engineer",
        "models": {"system1": "fast-model", "system2": "deep-model"},
    }
    return LLMAgent(contract, llm=_StubLLM())


@pytest.mark.parametrize(
    "text,meta,want",
    [
        ("你好", None, TIER_FAST),
        ("解释一下什么是 HTTP 缓存", None, TIER_FAST),
        ("分析一下这句话的语法", None, TIER_FAST),
        ("帮我看看", None, TIER_FAST),
        ("修复 daemon 启动空指针崩溃", None, TIER_DEEP),
        ("重构 llm.py 的重试逻辑", None, TIER_DEEP),
        ("审查 llm.py 的锁实现并给出建议", None, TIER_DEEP),
        (
            "请按以下步骤操作：\n1. 备份配置\n2. 升级服务\n3. 验证结果",
            None,
            TIER_DEEP,
        ),
        ("请把下面这段文字原样重复十遍：" + "abc " * 300, None, TIER_FAST),
        ("你好", {"complexity": "fast"}, TIER_FAST),
        ("重构全部模块", {"complexity": "t1-f"}, TIER_FAST),
        ("解释", {"complexity": "deep"}, TIER_DEEP),
        ("解释", {"tier": "t2"}, TIER_DEEP),
        ("解释", {"task_type": "refactor"}, TIER_DEEP),
        ("解释一下", {"task_type": "chat"}, TIER_FAST),
        ("帮我看看", {"requires_tools": True}, TIER_FAST),
        ("帮我看看", {"requires_tools": "true", "complexity": "deep"}, TIER_DEEP),
    ],
)
def test_pick_tier_samples(text, meta, want):
    assert pick_tier(text, meta) == want


def test_length_is_weak_feature():
    """超长但无动作/多步信号的输入停在一分，不达升级阈值。"""
    text = "重复输出：" + "资料" * 800
    assert len(text) > 600
    assert tier_score(text) == 1
    assert pick_tier(text) == TIER_FAST


def test_change_verb_alone_upgrades():
    """改动类动作单独即达到升级阈值，短请求不被降级。"""
    assert tier_score("修复登录失败") >= DEEP_SCORE
    assert pick_tier("修复登录失败") == TIER_DEEP


def test_selector_maps_to_contract_models():
    agent = _agent()
    ctx = AgentContext(agent_id="tier-agent")
    assert agent._select_model("你好", ctx) == "fast-model"
    assert agent._select_model("修复 llm.py 的锁泄漏", ctx) == "deep-model"


def test_selector_metadata_and_env(monkeypatch):
    agent = _agent()
    ctx = AgentContext(agent_id="tier-agent", metadata={"complexity": "fast"})
    assert agent._select_model("修复 llm.py 的锁泄漏", ctx) == "fast-model"

    monkeypatch.setenv("AIRY_AGENT_MODEL", "forced-model")
    assert agent._select_model("修复 llm.py 的锁泄漏", ctx) == "forced-model"
