# Copyright (c) 2026 SPHARX. All Rights Reserved.
"""LLM 客户端统一 model/base_url/key/max_tokens 解析单测。

覆盖 ``orchestration/core/llm.py`` 的 model.yaml SSoT 单一解析入口：
  - ``resolve_default_model`` : model.yaml 顶层 default_model > 条目 model_id > 环境变量
  - ``resolve_base_url``      : model.yaml 条目端点（跟随 model）> 环境变量兜底
  - ``_resolve_api_key``      : OPENAI_API_KEY → DEEPSEEK_API_KEY → ANTHROPIC_API_KEY
  - ``_resolve_max_tokens``   : 显式参数 → AIRY_LLM_MAX_TOKENS → 不发送
  - ``make_llm_client``       : 任一兼容 key 存在即真实客户端，否则 mock

与 ecosystem/manager/model/model.yaml 的 models 表字段保持一致。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import types

import pytest

import orchestration.core.llm as llm
from orchestration.core.llm import (
    LLMClient,
    MockLLMClient,
    _resolve_api_key,
    _resolve_max_tokens,
    make_llm_client,
    resolve_base_url,
    resolve_default_model,
)

# LLM 解析相关环境变量全集（用例前清除，保证 SSoT 语义确定性）
_LLM_ENV_VARS = (
    "AIRY_HOME", "AIRY_MODEL_CONFIG", "AIRYMAXHUB_ROOT",
    "AIRY_LLM_PROVIDER", "AIRY_LLM_BASE_URL", "OPENAI_BASE_URL",
    "AIRY_AGENT_MODEL", "OPENAI_MODEL",
    "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY",
    "MODEL_1_API_KEY", "MODEL_2_API_KEY",
)


def _clear_llm_env(monkeypatch):
    """清除 LLM 解析相关环境变量。"""
    for var in _LLM_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _disable_model_yaml(monkeypatch):
    """屏蔽全部 model.yaml 候选路径（模拟安装面缺失）。"""
    _clear_llm_env(monkeypatch)
    monkeypatch.setattr(llm, "_model_yaml_candidates",
                        lambda: ["/nonexistent/model.yaml"])


class TestResolveApiKey:
    """统一 API key 解析（变量名映射 SSoT）。"""

    def test_empty_when_no_key(self, monkeypatch):
        for var in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        assert _resolve_api_key() == ""

    def test_fallback_to_deepseek(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert _resolve_api_key() == "ds-key"

    def test_openai_priority(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "oa-key")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "an-key")
        assert _resolve_api_key() == "oa-key"

    def test_fallback_to_anthropic(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "an-key")
        assert _resolve_api_key() == "an-key"

    def test_explicit_wins(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "oa-key")
        assert _resolve_api_key("explicit-key") == "explicit-key"


class TestResolveDefaultModel:
    """统一 model 解析（SSoT 单一入口：model.yaml 第一权威，R3）。"""

    def test_yaml_default_model_wins(self, monkeypatch):
        """model.yaml 顶层 default_model 是第一权威源。"""
        _clear_llm_env(monkeypatch)
        assert resolve_default_model() == "deepseek-flash"

    def test_env_does_not_shadow_yaml(self, monkeypatch):
        """R3 回归：残留环境变量不得遮蔽 model.yaml。"""
        _clear_llm_env(monkeypatch)
        monkeypatch.setenv("AIRY_AGENT_MODEL", "stale-legacy-model")
        monkeypatch.setenv("OPENAI_MODEL", "stale-legacy-model")
        assert resolve_default_model() == "deepseek-flash"

    def test_env_fallback_without_yaml(self, monkeypatch):
        """model.yaml 不可用时环境变量兜底。"""
        _disable_model_yaml(monkeypatch)
        monkeypatch.setenv("AIRY_AGENT_MODEL", "fallback-model")
        assert resolve_default_model() == "fallback-model"

    def test_builtin_fallback(self, monkeypatch):
        """model.yaml 与环境变量均不可用时的内置兜底。"""
        _disable_model_yaml(monkeypatch)
        assert resolve_default_model() == "deepseek-flash"

    def test_explicit_wins(self, monkeypatch):
        assert resolve_default_model("explicit-model") == "explicit-model"


class TestResolveBaseUrl:
    """统一 base_url 解析（model.yaml 第一权威，端点跟随模型条目）。"""

    def test_yaml_entry_endpoint(self, monkeypatch):
        """默认解析命中 model.yaml DeepSeek 条目端点。"""
        _clear_llm_env(monkeypatch)
        assert resolve_base_url() == "https://api.deepseek.com"

    def test_env_does_not_shadow_yaml(self, monkeypatch):
        """R3 回归：残留环境变量不得遮蔽 model.yaml 端点。"""
        _clear_llm_env(monkeypatch)
        monkeypatch.setenv("AIRY_LLM_BASE_URL", "https://stale.example/v1")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://stale.example/v1")
        assert resolve_base_url() == "https://api.deepseek.com"

    def test_follows_selected_model(self, monkeypatch):
        """base_url 跟随 model 所在条目（model/端点同源一致）。"""
        _clear_llm_env(monkeypatch)
        assert resolve_base_url(model="GLM-4.7-Flash") == \
            "https://open.bigmodel.cn/api/paas/v4"

    def test_env_fallback_without_yaml(self, monkeypatch):
        """model.yaml 不可用时环境变量兜底（尾斜杠剥离）。"""
        _disable_model_yaml(monkeypatch)
        monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1/")
        assert resolve_base_url() == "https://gateway.example/v1"

    def test_explicit_wins(self, monkeypatch):
        monkeypatch.setenv("AIRY_LLM_BASE_URL", "https://env.example/v1")
        assert resolve_base_url("https://explicit.example/v1") == \
            "https://explicit.example/v1"


class TestMakeLLMClient:
    """环境感知工厂：任一兼容 key → 真实客户端，否则 mock。"""

    def test_force_mock_always_mock(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "oa-key")
        client = make_llm_client(force_mock=True)
        assert isinstance(client, MockLLMClient)

    def test_real_client_with_openai_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "oa-key")
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        client = make_llm_client()
        assert isinstance(client, LLMClient)

    def test_real_client_with_deepseek_key_only(self, monkeypatch):
        """默认提供商 DeepSeek：仅有 DEEPSEEK_API_KEY 也应切换真实客户端。"""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        client = make_llm_client()
        assert isinstance(client, LLMClient)
        assert client.api_key == "ds-key"

    def test_mock_when_no_key(self, monkeypatch):
        for var in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        client = make_llm_client()
        assert isinstance(client, MockLLMClient)

    def test_env_does_not_shadow_model_yaml(self, monkeypatch):
        """R3 回归：残留 AIRY_AGENT_MODEL 不遮蔽 model.yaml 默认模型，
        且 base_url 跟随选中模型的条目端点（model/端点同源一致）。"""
        _clear_llm_env(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "oa-key")
        monkeypatch.setenv("AIRY_AGENT_MODEL", "stale-legacy-model")
        client = make_llm_client()
        assert isinstance(client, LLMClient)
        assert client.default_model == "deepseek-flash"
        assert client.base_url == "https://api.deepseek.com"


class TestLLMClientInit:
    """LLMClient 构造时即解析统一变量（构造/工厂路径行为一致）。"""

    def test_init_resolves_deepseek_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
        client = LLMClient()
        assert client.api_key == "ds-key"

    def test_init_follows_model_yaml(self, monkeypatch):
        """构造路径与工厂一致：model/base_url 均来自 model.yaml SSoT。"""
        _clear_llm_env(monkeypatch)
        client = LLMClient(api_key="k")
        assert client.default_model == "deepseek-flash"
        assert client.base_url == "https://api.deepseek.com"

    def test_init_strips_base_url_slash(self):
        client = LLMClient(api_key="k", base_url="https://example.com/v1/")
        assert client.base_url == "https://example.com/v1"


class TestResolveMaxTokens:
    """统一输出上限解析：显式参数 > AIRY_LLM_MAX_TOKENS > 不发送。"""

    def test_default_none(self, monkeypatch):
        monkeypatch.delenv("AIRY_LLM_MAX_TOKENS", raising=False)
        assert _resolve_max_tokens() is None

    def test_env_positive(self, monkeypatch):
        monkeypatch.delenv("AIRY_LLM_MAX_TOKENS", raising=False)
        monkeypatch.setenv("AIRY_LLM_MAX_TOKENS", "16384")
        assert _resolve_max_tokens() == 16384

    def test_env_invalid_returns_none(self, monkeypatch):
        monkeypatch.setenv("AIRY_LLM_MAX_TOKENS", "abc")
        assert _resolve_max_tokens() is None
        monkeypatch.setenv("AIRY_LLM_MAX_TOKENS", "0")
        assert _resolve_max_tokens() is None
        monkeypatch.setenv("AIRY_LLM_MAX_TOKENS", "  ")
        assert _resolve_max_tokens() is None

    def test_explicit_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("AIRY_LLM_MAX_TOKENS", "1024")
        assert _resolve_max_tokens(4096) == 4096


# chat() HTTP 路径捕获用的最小 OpenAI 响应体
_CHAT_OK = json.dumps(
    {
        "choices": [
            {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        ],
        "model": "scripted",
        "usage": {"total_tokens": 1},
    }
)


def _make_fake_aiohttp(captured: dict):
    """构造注入 sys.modules 的 fake aiohttp，捕获 chat() 发出的请求。

    chat() 内懒导入 ``import aiohttp``，故经 monkeypatch.setitem 注入
    sys.modules 即可命中，无需真实网络。
    """
    fake = types.ModuleType("aiohttp")

    class FakeClientTimeout:
        def __init__(self, total=None):
            self.total = total

    class FakeResponse:
        status = 200

        async def text(self):
            return _CHAT_OK

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class FakeSession:
        def __init__(self, timeout=None):
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def post(self, url, json=None, headers=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return FakeResponse()

    fake.ClientTimeout = FakeClientTimeout
    fake.ClientSession = FakeSession
    return fake


class TestChatPayload:
    """chat() 请求 payload：max_tokens 按解析结果发送/省略。"""

    def _run_chat(self, monkeypatch, captured, **kwargs):
        monkeypatch.setitem(sys.modules, "aiohttp", _make_fake_aiohttp(captured))
        client = LLMClient(api_key="k", base_url="https://llm.example/v1")
        return asyncio.run(
            client.chat(messages=[{"role": "user", "content": "hi"}], **kwargs)
        )

    def test_env_max_tokens_sent(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setenv("AIRY_LLM_MAX_TOKENS", "16384")
        self._run_chat(monkeypatch, captured)
        assert captured["url"] == "https://llm.example/v1/chat/completions"
        assert captured["json"]["max_tokens"] == 16384

    def test_no_env_omits_max_tokens(self, monkeypatch):
        """不发送 max_tokens 时沿用厂商默认（键不得出现）。"""
        captured: dict = {}
        monkeypatch.delenv("AIRY_LLM_MAX_TOKENS", raising=False)
        self._run_chat(monkeypatch, captured)
        assert "max_tokens" not in captured["json"]

    def test_explicit_overrides_env(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setenv("AIRY_LLM_MAX_TOKENS", "1024")
        self._run_chat(monkeypatch, captured, max_tokens=4096)
        assert captured["json"]["max_tokens"] == 4096
