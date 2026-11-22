# Copyright (c) 2026 SPHARX. All Rights Reserved.
"""Agent 契约能力声明的运行期校验（S-P1-14）。

契约 ``capabilities`` 不再只是描述性文本：装载时逐条校验 schema 结构
（类型、properties、required 可满足、数组 items），结构不合法即拒绝
构造，避免坏契约静默进入运行期。执行体按 ``metadata.capability``
校验结构化输入；成功率由执行体实测，契约不得声明。

自由文本任务是本项目的缺省执行形态，此时没有可校验的结构化输入，
能力声明用于发现与契约约束，不虚构输入校验。
"""

from __future__ import annotations

from typing import Any, Dict

_MAX_DEPTH = 4
_TYPES = {
    "string": str,
    "object": dict,
    "array": list,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
}


def _fail(path: str, msg: str) -> None:
    raise ValueError(f"{path}: {msg}")


def _schema(schema: Any, path: str, depth: int = 0) -> None:
    """校验单个 JSON Schema 子集节点。"""
    if not isinstance(schema, dict):
        _fail(path, "schema must be an object")
    if depth > _MAX_DEPTH:
        _fail(path, "schema nesting too deep")
    stype = schema.get("type", "object")
    if stype not in _TYPES:
        _fail(path, f"unsupported type {stype!r}")
    props = schema.get("properties")
    if props is not None:
        if not isinstance(props, dict):
            _fail(path, "properties must be an object")
        for key, sub in props.items():
            if not isinstance(key, str) or not key:
                _fail(path, "property names must be non-empty strings")
            _schema(sub, f"{path}.{key}", depth + 1)
    req = schema.get("required")
    if req is not None:
        if not isinstance(req, list) or not all(isinstance(x, str) for x in req):
            _fail(path, "required must be a list of strings")
        if props is None:
            _fail(path, "required without properties")
        missing = [x for x in req if x not in props]
        if missing:
            _fail(path, f"required keys not declared: {missing}")
    if schema.get("items") is not None:
        if stype != "array":
            _fail(path, "items declared on non-array schema")
        _schema(schema["items"], f"{path}[]", depth + 1)


def load_caps(caps: Any) -> Dict[str, Dict[str, Any]]:
    """校验并索引契约能力声明；缺省返回空表（自由文本角色无声明）。"""
    if caps is None:
        return {}
    if not isinstance(caps, list):
        raise ValueError("capabilities must be a list")
    out: Dict[str, Dict[str, Any]] = {}
    for idx, cap in enumerate(caps):
        path = f"capabilities[{idx}]"
        if not isinstance(cap, dict):
            _fail(path, "capability must be an object")
        name = cap.get("name")
        if not isinstance(name, str) or not name:
            _fail(path, "missing name")
        if name in out:
            _fail(path, f"duplicate capability {name!r}")
        desc = cap.get("description")
        if not isinstance(desc, str) or not desc:
            _fail(path, "missing description")
        for key in ("input_schema", "output_schema"):
            if key not in cap:
                _fail(path, f"missing {key}")
            _schema(cap[key], f"{path}.{key}")
        out[name] = cap
    return out


def check_input(cap: Dict[str, Any], value: Any) -> None:
    """按能力声明的 input_schema 校验结构化输入。"""
    _value(cap.get("input_schema") or {}, value, "input")


def _value(schema: Dict[str, Any], value: Any, path: str) -> None:
    stype = schema.get("type", "object")
    if stype == "object":
        if not isinstance(value, dict):
            _fail(path, "expected object")
        for key in schema.get("required") or []:
            if key not in value:
                _fail(path, f"missing required key {key!r}")
        for key, sub in (schema.get("properties") or {}).items():
            if key in value:
                _value(sub, value[key], f"{path}.{key}")
        return
    if stype == "array":
        if not isinstance(value, list):
            _fail(path, "expected array")
        items = schema.get("items")
        if items:
            for idx, item in enumerate(value):
                _value(items, item, f"{path}[{idx}]")
        return
    if stype == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            _fail(path, "expected integer")
        return
    if stype == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _fail(path, "expected number")
        return
    if not isinstance(value, _TYPES[stype]):
        _fail(path, f"expected {stype}")
