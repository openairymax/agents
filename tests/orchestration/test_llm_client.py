# Copyright (c) 2026 SPHARX. All Rights Reserved.
"""LLM 客户端统一 key/base_url 解析单测。

覆盖 ``orchestration/core/llm.py`` 的变量名映射 SSoT：
  - ``_resolve_api_key``    : OPENAI_API_KEY → DEEPSEEK_API_KEY → ANTHROPIC_API_KEY
  - ``_resolve_base_url``   : AIRY_LLM_BASE_URL → OPENAI_BASE_URL → 官方默认
  - ``make_llm_client``     : 任一兼容 key 存在即真实客户端，否则 mock

与 ecosystem/manager/model/model.yaml 的 providers[].api_key_env
及 devtools/scripts/ops/templates/secrets.env.example 保持一致。
"""

from __future__ import annotations

import os

import pytest

from orchestration.core.llm import (
    LLMClient,
    MockLLMClient,
    _resolve_api_key,
    _resolve_base_url,
    make_llm_client,
)


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


class TestResolveBaseUrl:
    """统一 base_url 解析。"""

    def test_default_official(self, monkeypatch):
        """无 DeepSeek key、无 base_url 覆盖时默认 OpenAI 官方。"""
        monkeypatch.delenv("AIRY_LLM_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        assert _resolve_base_url() == "https://api.openai.com/v1"

    def test_default_follows_deepseek_key(self, monkeypatch):
        """仅有 DEEPSEEK_API_KEY（默认提供商）时默认 DeepSeek 端点。"""
        monkeypatch.delenv("AIRY_LLM_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert _resolve_base_url() == "https://api.deepseek.com/v1"

    def test_airy_preferred(self, monkeypatch):
        monkeypatch.setenv("AIRY_LLM_BASE_URL", "https://api.deepseek.com/v1")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        assert _resolve_base_url() == "https://api.deepseek.com/v1"

    def test_openai_base_url_fallback(self, monkeypatch):
        monkeypatch.delenv("AIRY_LLM_BASE_URL", raising=False)
        monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
        assert _resolve_base_url() == "https://gateway.example/v1"

    def test_explicit_wins(self, monkeypatch):
        monkeypatch.setenv("AIRY_LLM_BASE_URL", "https://env.example/v1")
        assert _resolve_base_url("https://explicit.example/v1") == "https://explicit.example/v1"

    def test_trailing_slash_stripped(self, monkeypatch):
        monkeypatch.delenv("AIRY_LLM_BASE_URL", raising=False)
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1/")
        assert _resolve_base_url() == "https://api.openai.com/v1"


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

    def test_default_model_airy_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "oa-key")
        monkeypatch.setenv("AIRY_AGENT_MODEL", "deepseek-v4-flash")
        client = make_llm_client()
        assert client.default_model == "deepseek-v4-flash"


class TestLLMClientInit:
    """LLMClient 构造时即解析统一变量（构造/工厂路径行为一致）。"""

    def test_init_resolves_deepseek_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
        client = LLMClient()
        assert client.api_key == "ds-key"

    def test_init_resolves_airy_base_url(self, monkeypatch):
        monkeypatch.setenv("AIRY_LLM_BASE_URL", "https://api.deepseek.com/v1")
        client = LLMClient(api_key="k")
        assert client.base_url == "https://api.deepseek.com/v1"

    def test_init_strips_base_url_slash(self):
        client = LLMClient(api_key="k", base_url="https://example.com/v1/")
        assert client.base_url == "https://example.com/v1"
