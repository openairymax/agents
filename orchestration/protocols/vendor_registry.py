# SPDX-FileCopyrightText: 2026 SPHARX Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later OR Apache-2.0
"""
orchestration.protocols.vendor_registry — L3 厂商策略注入面消费器

按 0.1.19 架构文档 §5.1 四层分离：L4 厂商策略以数据文件（``vendors/*.json``）
承载，本模块是 L3 消费面 —— 读取并索引厂商 manifest，供运行时租户装配协议处理器，
使一类决策从「N 份代码」塌缩为「1 个件 + N 份数据」。

机制 / 策略边界（sdk/README 运行期租户律）：本模块只消费数据，不实现协议栈；
所有平台能力经 SDK over HTTP / JSON-RPC 调用，禁止在生态层平行实现厂商协议。

@since 0.1.19
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_VENDORS_DIR = Path(__file__).resolve().parent / "vendors"
_SCHEMA_FILE = _VENDORS_DIR / "_schema.json"

_REQUIRED_FIELDS = ("manifest_version", "id", "kind", "protocol")


def validate_document(doc: Dict[str, Any]) -> List[str]:
    """Validate one vendor manifest against the L3 schema.

    Returns a list of human-readable violations (empty list means valid). When
    ``jsonschema`` is unavailable, falls back to a structural required-field
    check so the consumer still rejects obviously malformed data.
    """
    errors: List[str] = []
    try:
        import jsonschema
    except ImportError:
        for key in _REQUIRED_FIELDS:
            if key not in doc:
                errors.append(f"missing required field: {key}")
        return errors

    if not _SCHEMA_FILE.is_file():
        return errors

    schema = json.loads(_SCHEMA_FILE.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    for err in sorted(validator.iter_errors(doc), key=lambda e: list(e.path)):
        loc = "/".join(str(p) for p in err.path) or "<root>"
        errors.append(f"{loc}: {err.message}")
    return errors


@dataclass(frozen=True)
class VendorManifest:
    """Immutable view over one vendor strategy manifest (L4 data)."""

    id: str
    kind: str
    protocol_name: str
    registry_name: str
    protocol_version: str
    aliases: Tuple[str, ...]
    category: Optional[str]
    type: str
    capabilities: Tuple[str, ...]
    transport: Dict[str, Any]
    limits: Dict[str, Any]
    timeouts: Dict[str, Any]
    retry: Dict[str, Any]
    rate_limit: Dict[str, Any]
    heartbeat_interval_sec: Optional[int]
    raw: Dict[str, Any]

    @property
    def transport_kind(self) -> str:
        """Declared transport kind, empty when the manifest omits transport."""
        return str(self.transport.get("kind", ""))

    @property
    def default_timeout_ms(self) -> int:
        return int(self.timeouts.get("default_ms", 30000))

    @property
    def endpoint_default(self) -> Optional[str]:
        return self.transport.get("endpoint_default") or self.transport.get("path_prefix")

    def matches(self, name: str) -> bool:
        """True when ``name`` matches this vendor's id / protocol / registry name / alias."""
        lowered = name.lower()
        if lowered in (self.id.lower(), self.protocol_name.lower(), self.registry_name.lower()):
            return True
        return lowered in (alias.lower() for alias in self.aliases)


def _to_manifest(doc: Dict[str, Any]) -> VendorManifest:
    proto = doc.get("protocol", {})
    registry = doc.get("registry", {})
    return VendorManifest(
        id=doc["id"],
        kind=doc["kind"],
        protocol_name=proto.get("name", doc["id"]),
        registry_name=registry.get("name", proto.get("name", doc["id"])),
        protocol_version=str(proto.get("version", "")),
        aliases=tuple(proto.get("aliases", []) or ()),
        category=registry.get("category"),
        type=registry.get("type", proto.get("name", doc["id"])),
        capabilities=tuple(registry.get("capabilities", []) or ()),
        transport=dict(doc.get("transport", {})),
        limits=dict(doc.get("limits", {})),
        timeouts=dict(doc.get("timeouts", {})),
        retry=dict(doc.get("retry", {})),
        rate_limit=dict(doc.get("rate_limit", {})),
        heartbeat_interval_sec=doc.get("heartbeat_interval_sec"),
        raw=doc,
    )


class VendorRegistry:
    """Loads and indexes vendor strategy manifests from a directory."""

    def __init__(self, vendors_dir: Optional[os.PathLike] = None):
        self._dir = Path(vendors_dir) if vendors_dir else _VENDORS_DIR
        self._vendors: Dict[str, VendorManifest] = {}
        self._aliases: Dict[str, str] = {}
        self.load()

    @property
    def vendors_dir(self) -> Path:
        return self._dir

    def load(self) -> int:
        """(Re)load every manifest in the vendors directory; returns the count."""
        self._vendors.clear()
        self._aliases.clear()
        if not self._dir.is_dir():
            logger.warning("vendor manifest dir not found: %s", self._dir)
            return 0

        for path in sorted(self._dir.glob("*.json")):
            if path.name.startswith("_"):
                continue
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
                manifest = _to_manifest(doc)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                logger.error("invalid vendor manifest %s: %s", path.name, exc)
                continue

            self._vendors[manifest.id] = manifest
            keys = (manifest.id, manifest.protocol_name, manifest.registry_name, *manifest.aliases)
            for key in keys:
                self._aliases[key.lower()] = manifest.id

        logger.debug("loaded %d vendor manifests from %s", len(self._vendors), self._dir)
        return len(self._vendors)

    def get(self, vendor_id: str) -> Optional[VendorManifest]:
        return self._vendors.get(vendor_id)

    def resolve(self, name: str) -> Optional[VendorManifest]:
        """Resolve by id / protocol name / alias, case-insensitively."""
        vendor_id = self._aliases.get(name.lower())
        return self._vendors.get(vendor_id) if vendor_id else None

    def list(self, kind: Optional[str] = None) -> List[VendorManifest]:
        items = list(self._vendors.values())
        if kind is not None:
            items = [m for m in items if m.kind == kind]
        return items

    def by_transport(self, kind: str) -> List[VendorManifest]:
        return [m for m in self._vendors.values() if m.transport_kind == kind]

    def validate(self) -> Dict[str, List[str]]:
        """Validate every manifest against the L3 schema; id -> violations."""
        report: Dict[str, List[str]] = {}
        for path in sorted(self._dir.glob("*.json")):
            if path.name.startswith("_"):
                continue
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                report[path.name] = [f"unreadable: {exc}"]
                continue
            errors = validate_document(doc)
            if errors:
                report[doc.get("id", path.name)] = errors
        return report

    def __len__(self) -> int:
        return len(self._vendors)

    def __contains__(self, vendor_id: str) -> bool:
        return vendor_id in self._vendors


_default_registry: Optional[VendorRegistry] = None


def get_vendor_registry() -> VendorRegistry:
    """Return the process-wide vendor registry, creating it on first use."""
    global _default_registry
    if _default_registry is None:
        _default_registry = VendorRegistry()
    return _default_registry
