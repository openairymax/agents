# SPDX-FileCopyrightText: 2026 SPHARX Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later OR Apache-2.0
"""L3 厂商策略注入面消费器测试（orchestration.protocols.vendor_registry）。

覆盖:
- 全部厂商 manifest 通过 ``_schema.json``（L3 契约）校验
- 索引 / 别名解析 / 分类与传输过滤
- ``interface`` 方法面与线信封（L4 数据 → L3 消费）
- manifest 驱动的 ``ManifestHandler``：超时/端点取自 manifest，方法面逐厂商对齐
- 无 HTTP 调用面者诚实报边界错误（不伪造调用）
"""

from __future__ import annotations

import asyncio

import pytest

from orchestration.protocols import (
    ManifestHandler,
    ProtocolRequestContext,
    build_manifest_handlers,
)
from orchestration.protocols.vendor_registry import (
    VendorRegistry,
    get_vendor_registry,
    validate_document,
)

EXPECTED_TOTAL = 11
EXPECTED_KIND_COUNTS = {"core": 1, "standard": 3, "integration": 5, "framework": 2}
EXPECTED_METHOD_COUNTS = {
    "jsonrpc": 14, "mcp": 9, "a2a": 4, "agntcy": 5, "claude": 4,
    "openai": 3, "china_eco": 6, "openclaw": 12, "openjiuwen": 4,
}
EXPECTED_REGISTRY_IDS = set(EXPECTED_METHOD_COUNTS)


@pytest.fixture()
def registry() -> VendorRegistry:
    return VendorRegistry()


def _ctx(method: str, params: dict | None = None) -> ProtocolRequestContext:
    return ProtocolRequestContext(
        session_id="sess", trace_id="trace", source_protocol="orchestration",
        target_protocol="vendor", method=method, params=params or {},
    )


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


def test_interface_method_face(registry: VendorRegistry) -> None:
    for vendor_id, expected in EXPECTED_METHOD_COUNTS.items():
        manifest = registry.resolve(vendor_id)
        assert manifest.is_registry, vendor_id
        assert len(manifest.methods) == expected, vendor_id
        assert len(set(manifest.methods)) == len(manifest.methods), vendor_id


def test_interface_invoke_envelope(registry: VendorRegistry) -> None:
    assert registry.resolve("jsonrpc").invoke == {"style": "jsonrpc2", "path": "/rpc"}
    assert registry.resolve("mcp").invoke == {
        "style": "envelope", "path": "/api/v1/invoke", "version": "1.0",
    }
    assert registry.resolve("a2a").invoke == {
        "style": "envelope", "path": "/api/v1/invoke", "version": "0.3",
    }
    assert registry.resolve("openai").invoke == {
        "style": "rest", "path": "/v1/chat/completions",
    }
    for vendor_id in ("agntcy", "china_eco", "openclaw", "openjiuwen"):
        assert registry.resolve(vendor_id).invoke is None, vendor_id


def test_non_registry_manifests_have_no_face(registry: VendorRegistry) -> None:
    for vendor_id in ("autogen", "langchain"):
        manifest = registry.resolve(vendor_id)
        assert not manifest.is_registry
        assert manifest.methods == ()
        assert manifest.invoke is None


def test_build_manifest_handlers_covers_registry_vendors() -> None:
    handlers = build_manifest_handlers()
    assert {h.protocol_name for h in handlers} == EXPECTED_REGISTRY_IDS
    for handler in handlers:
        assert handler.supported_methods() == list(handler.manifest.methods)


def test_manifest_handler_field_defaults(registry: VendorRegistry) -> None:
    """回归: 修复缺失 import os 后, handler 不再抛 NameError 并读 manifest 超时。"""
    handler = ManifestHandler(registry.resolve("jsonrpc"))
    assert handler._timeout_ms == 30000
    assert handler._endpoint.startswith("http://")
    assert handler.protocol_name == "jsonrpc"


def test_manifest_handler_explicit_timeout_overrides(registry: VendorRegistry) -> None:
    handler = ManifestHandler(registry.resolve("jsonrpc"), timeout_ms=1234)
    assert handler._timeout_ms == 1234


def test_manifest_handler_payload_shape(registry: VendorRegistry) -> None:
    ctx = _ctx("agent.list", {"q": 1})

    jsonrpc = ManifestHandler(registry.resolve("jsonrpc"))._build_payload(ctx)
    assert jsonrpc["jsonrpc"] == "2.0"
    assert jsonrpc["method"] == "agent.list"
    assert jsonrpc["params"] == {"q": 1}

    envelope = ManifestHandler(registry.resolve("mcp"))._build_payload(ctx)
    assert envelope == {
        "protocol": "mcp", "version": "1.0", "method": "agent.list", "params": {"q": 1},
    }

    rest = ManifestHandler(registry.resolve("openai"))._build_payload(ctx)
    assert rest == {"q": 1}


def test_manifest_handler_rejects_missing_invoke_face(registry: VendorRegistry) -> None:
    handler = ManifestHandler(registry.resolve("openclaw"))
    response = asyncio.run(handler.handle_request(_ctx("openclaw_register_agent")))
    assert response.success is False
    assert response.error_code == 5010
    assert "no HTTP invocation face" in response.error_message
    assert response.protocol == "openclaw"


def test_validate_document_rejects_missing_required_field() -> None:
    errors = validate_document({"manifest_version": 1, "id": "x"})
    assert any("kind" in e for e in errors)


def test_get_vendor_registry_is_shared_singleton() -> None:
    assert get_vendor_registry() is get_vendor_registry()
    assert len(get_vendor_registry()) == EXPECTED_TOTAL
