# Copyright (c) 2026 SPHARX. All Rights Reserved.
"""S-P0-04：model.yaml 的 ``tool_rounds`` 必须成为运行期生效预算。

历史缺口是该键只被解析、无人消费，执行体恒用内置 16 轮。现由
``model_tool_rounds`` 读配置、``_rounds_limit`` 兑现，优先级为
``AIRY_AGENT_TOOL_ROUNDS`` > model.yaml 条目 > 缺省，且任何来源都受
``TOOL_ROUNDS_MAX`` 机制硬界约束。

用例以 ``AIRY_HOME/config/model.yaml``（安装面第一候选）构造配置，
不依赖仓库源码树里的真实 model.yaml，因此断言的是映射关系本身。
"""

from __future__ import annotations

import pytest

from orchestration.agents.llm import TOOL_ROUNDS_DEFAULT, TOOL_ROUNDS_MAX, _rounds_limit
from orchestration.core.llm_cfg import model_tool_rounds

YAML = """\
models:
  - name: localtest
    model_id: test-model-r5
    mode: local
    base_url: http://127.0.0.1:11434/v1
    context_window: 8k
    tool_rounds: 5
  - name: overmax
    model_id: test-model-r1000
    api_key_env: TEST_KEY
    tool_rounds: 1000
  - name: broken
    model_id: test-model-bad
    api_key_env: TEST_KEY
    tool_rounds: not-a-number
"""


@pytest.fixture()
def model_home(tmp_path, monkeypatch):
    """把 AIRY_HOME 指向只含本用例 model.yaml 的临时安装面。"""
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "model.yaml").write_text(YAML, encoding="utf-8")
    monkeypatch.setenv("AIRY_HOME", str(tmp_path))
    monkeypatch.delenv("AIRY_AGENT_TOOL_ROUNDS", raising=False)
    # 源码树候选与本节无关；确保 AIRY_MODEL_CONFIG 不引入第二来源。
    monkeypatch.delenv("AIRY_MODEL_CONFIG", raising=False)
    monkeypatch.delenv("AIRYMAXHUB_ROOT", raising=False)
    return tmp_path


class TestModelYamlToolRounds:
    """配置项与被执行预算一一映射，不接受「解析了但不用」。"""

    def test_entry_value_is_consumed(self, model_home):
        assert model_tool_rounds("test-model-r5") == 5
        assert _rounds_limit("test-model-r5") == 5

    def test_entry_above_ceiling_is_clamped(self, model_home):
        """配置再大也不得越过机制硬界。"""
        assert model_tool_rounds("test-model-r1000") == 1000
        assert _rounds_limit("test-model-r1000") == TOOL_ROUNDS_MAX

    def test_illegal_value_falls_back_to_default(self, model_home):
        assert model_tool_rounds("test-model-bad") == 0
        assert _rounds_limit("test-model-bad") == TOOL_ROUNDS_DEFAULT

    def test_undeclared_model_falls_back_to_default(self, model_home):
        assert model_tool_rounds("no-such-model") == 0
        assert _rounds_limit("no-such-model") == TOOL_ROUNDS_DEFAULT

    def test_model_id_match_is_case_insensitive(self, model_home):
        assert _rounds_limit("TEST-MODEL-R5") == 5

    def test_env_overrides_entry(self, model_home, monkeypatch):
        """运维显式收紧优先于配置 SSoT。"""
        monkeypatch.setenv("AIRY_AGENT_TOOL_ROUNDS", "4")
        assert _rounds_limit("test-model-r5") == 4

    def test_env_also_clamped(self, model_home, monkeypatch):
        monkeypatch.setenv("AIRY_AGENT_TOOL_ROUNDS", "999")
        assert _rounds_limit("test-model-r5") == TOOL_ROUNDS_MAX

    def test_execute_uses_entry_budget(self, model_home):
        """端到端：耗尽条目声明的轮次即报溢出，错误体回报生效值。"""
        import asyncio

        from orchestration.agents.llm import LLMAgent
        from orchestration.core.agent import AgentContext

        class LoopLLM:
            """每轮参数不同，避开同指纹终止，只让轮次预算生效。"""

            default_model = "test-model-r5"

            def __init__(self):
                self.n = 0

            async def chat(self, messages, model=None, tools=None, **kw):
                self.n += 1
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": f"c{self.n}",
                                        "type": "function",
                                        "function": {
                                            "name": "noop",
                                            "arguments": '{"n": %d}' % self.n,
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                }

        agent = LLMAgent({"agent_id": "rounds"}, llm=LoopLLM())
        agent.register_tool(
            "noop",
            lambda params: "ok",
            schema={"description": "noop", "parameters": {"type": "object"}},
        )
        result = asyncio.run(agent.execute("loop", AgentContext("t")))
        assert result.success is False
        assert result.error_code == "TOOL_LOOP_OVERFLOW"
        assert "5 rounds" in (result.error or "")
