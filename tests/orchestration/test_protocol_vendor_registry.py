# SPDX-FileCopyrightText: 2026 SPHARX Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later OR Apache-2.0
"""L3 厂商策略注入面消费器测试（orchestration.protocols.vendor_registry）。

覆盖:
- 全部厂商 manifest 通过 ``_schema.json``（L3 契约）校验
- 索引 / 别名解析 / 分类与传输过滤
- 内置 handler 从 manifest 取默认超时（回归: 修复缺失 ``import os``）
"""

from __future__ import annotations

import pytest

from orchestration.protocols import JSONRPCHandler, MCPHandler
from orchestration.protocols.vendor_registry import (
    VendorRegistry,
    get_vendor_registry,
    validate_document,
)

EXPECTED_TOTAL = 11
EXPECTED_KIND_COUNTS = {"core": 1, "standard": 3, "integration": 5, "framework": 2}


@pytest.fixture()
def registry() -> VendorRegistry:
    return VendorRegistry()


def test_all_manifests_conform_to_schema(registry: VendorRegistry) -> None:
    report = registry.validate()
    assert report == {}, f"schema violations: {report}"
    assert len(registry) == EXPECTED_TOTAL


def test_kind_distribution(registry: VendorRegistry) -> None:
    for kind, expected in EXPECTED_KIND_COUNTS.items():
        assert len(registry.list(kind=kind)) == expected, kind


def test_resolve_by_id_protocol_name_and_alias(registry: VendorRegistry) -> None:
    assert registry.resolve("jsonrpc").id == "jsonrpc"
    assert registry.resolve("JSON-RPC").id == "jsonrpc"
    assert registry.resolve("mcp").id == "mcp"
    assert registry.resolve("a2a").id == "a2a"
    assert registry.resolve("openai").id == "openai"
    assert registry.resolve("no-such-vendor") is None


def test_transport_index(registry: VendorRegistry) -> None:
    http_ids = {m.id for m in registry.by_transport("http")}
    assert http_ids == {"jsonrpc", "a2a", "agntcy", "china_eco", "claude", "openai"}
    assert {m.id for m in registry.by_transport("stdio")} == {"mcp"}
    assert {m.id for m in registry.by_transport("unix_socket")} == {"openclaw"}
    assert {m.id for m in registry.by_transport("custom_binary")} == {"openjiuwen"}


def test_manifest_fields_exposed(registry: VendorRegistry) -> None:
    mcp = registry.resolve("mcp")
    assert mcp.kind == "standard"
    assert mcp.protocol_version == "1.0.0"
    assert "tool_calling" in mcp.capabilities
    assert mcp.default_timeout_ms == 30000
    assert mcp.transport_kind == "stdio"


def test_handlers_default_timeout_from_manifest() -> None:
    """回归: 修复缺失 import os 后, handler 不再抛 NameError 并读 manifest 超时。"""
    jsonrpc = JSONRPCHandler()
    assert jsonrpc._timeout_ms == 30000
    assert jsonrpc._endpoint.startswith("http://")

    mcp = MCPHandler()
    assert mcp._timeout_ms == 30000


def test_explicit_timeout_overrides_manifest(registry: VendorRegistry) -> None:
    handler = JSONRPCHandler(timeout_ms=1234)
    assert handler._timeout_ms == 1234


def test_validate_document_rejects_missing_required_field() -> None:
    errors = validate_document({"manifest_version": 1, "id": "x"})
    assert any("kind" in e for e in errors)


def test_get_vendor_registry_is_shared_singleton() -> None:
    assert get_vendor_registry() is get_vendor_registry()
    assert len(get_vendor_registry()) == EXPECTED_TOTAL
