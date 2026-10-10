#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025-2026 SPHARX Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later OR Apache-2.0

# -*- coding: utf-8 -*-
"""执行体角色词汇三方一致性守卫（S-3 生态 SSoT 收敛门禁）。

守卫「运行时角色词汇权威 = 机制核 ``agentrt/commons/utils/cognition/
agent_vocab.c``；生态声明面 = ``registry/agents.yaml``；Python 策略运行时
经 ``agent_d.vocab``（A-IPC）消费同一词汇」的 SSoT，防止任一方漂移而无声：

  - V1: 角色集合 C == Python(AGENT_REGISTRY) == yaml(enabled role)
  - V2: 别名映射 C == Python(ROLE_ALIASES)
  - V3: 只读集合 C == runner._READONLY_LOCAL（agent_d 不可达时的启动回退）
  - V4: 兜底角色 C(AGENT_VOCAB_FALLBACK) == Python(ROLE_FALLBACK)
  - V5: 别名目标均登记为具体角色；planned 角色不得进入运行时词汇

解析全程为纯文本正则，不导入被测包，故可在无 pytest 环境下以
``python3 tests/test_vocab_consistency.py`` 独立驱动（``__main__`` 分支）。
独仓组装（agents 单独检出、无 agentrt 机制核）时 C 侧断言按理由跳过，
其余 Python↔yaml 断言仍生效；若 agentrt 已存在却缺词汇 SSoT 则判失败，
以暴露重命名/迁移而非静默。
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml

# ── 仓库内文件定位（相对本测试文件推导，不依赖绝对路径）─────────────
_TESTS_DIR = Path(__file__).resolve().parent
_AGENTS_DIR = _TESTS_DIR.parent
_ECOSYSTEM_DIR = _AGENTS_DIR.parent

PY_INIT = _AGENTS_DIR / "airymax_agents" / "__init__.py"
PY_RUNNER = _AGENTS_DIR / "airymax_agents" / "runner.py"
YAML_REGISTRY = _AGENTS_DIR / "registry" / "agents.yaml"

_VOCAB_REL = Path("commons") / "utils" / "cognition" / "agent_vocab.c"
_VOCAB_H_REL = Path("commons") / "utils" / "cognition" / "agent_vocab.h"


def _locate_agentrt() -> Optional[Path]:
    """在若干组装布局下定位 agentrt 机制核根目录。"""
    for base in (_ECOSYSTEM_DIR.parent, _ECOSYSTEM_DIR.parent.parent):
        cand = base / "agentrt"
        if cand.is_dir():
            return cand
    return None


_AGENTRT_DIR = _locate_agentrt()
_VOCAB_C = _AGENTRT_DIR / _VOCAB_REL if _AGENTRT_DIR else None
_VOCAB_H = _AGENTRT_DIR / _VOCAB_H_REL if _AGENTRT_DIR else None

_C_STR_RE = re.compile(r'"([^"]*)"')
_ALIAS_PAIR_RE = re.compile(r'\{\s*"([^"]*)"\s*,\s*"([^"]*)"\s*\}')


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _require_c() -> str:
    """返回 C 机制核词汇源；独仓组装缺机制核时按理由跳过。"""
    if _AGENTRT_DIR is None:
        raise unittest.SkipTest(
            "agentrt 机制核不在当前组装（agents 独仓检出），跳过 C↔策略一致性校验"
        )
    assert _VOCAB_C.is_file(), f"agentrt 已存在但缺少词汇 SSoT: {_VOCAB_C}"
    return _read(_VOCAB_C)


def _c_block(text: str, decl: str) -> str:
    """提取 C 数组 ``<decl>[] = { ... };`` 的花括号体（支持 [][2] 形态）。"""
    pat = re.escape(decl) + r"\s*\[\s*\]\s*(\[\s*2\s*\])?\s*=\s*\{(.*?)\};"
    m = re.search(pat, text, re.S)
    assert m, f"agent_vocab.c 缺少数组 {decl}"
    return m.group(2)


def _c_roles(text: str) -> List[str]:
    return _C_STR_RE.findall(_c_block(text, "VOCAB_ROLES"))


def _c_aliases(text: str) -> Dict[str, str]:
    return dict(_ALIAS_PAIR_RE.findall(_c_block(text, "VOCAB_ALIASES")))


def _c_readonly(text: str) -> Set[str]:
    return set(_C_STR_RE.findall(_c_block(text, "VOCAB_READONLY")))


def _c_fallback(text: str) -> str:
    m = re.search(r'#define\s+AGENT_VOCAB_FALLBACK\s+"([^"]*)"', text)
    assert m, "agent_vocab.h 缺少 AGENT_VOCAB_FALLBACK 定义"
    return m.group(1)


def _py_roles(text: str) -> List[str]:
    m = re.search(r"AGENT_REGISTRY[^=]*=\s*\{(.*?)\}", text, re.S)
    assert m, "__init__.py 缺少 AGENT_REGISTRY"
    return re.findall(r'"([^"]+)"\s*:', m.group(1))


def _py_aliases(text: str) -> Dict[str, str]:
    m = re.search(r"ROLE_ALIASES[^=]*=\s*\{(.*?)\}", text, re.S)
    assert m, "__init__.py 缺少 ROLE_ALIASES"
    return dict(re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', m.group(1)))


def _py_fallback(text: str) -> str:
    m = re.search(r'ROLE_FALLBACK\s*=\s*"([^"]+)"', text)
    assert m, "__init__.py 缺少 ROLE_FALLBACK"
    return m.group(1)


def _runner_readonly(text: str) -> Set[str]:
    m = re.search(r"_READONLY_LOCAL\s*=\s*frozenset\(\{(.*?)\}\)", text, re.S)
    assert m, "runner.py 缺少 _READONLY_LOCAL"
    return set(re.findall(r'"([^"]+)"', m.group(1)))


def _yaml_data() -> dict:
    with YAML_REGISTRY.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict) and "agents" in data, "agents.yaml 缺少 agents 列表"
    return data


def _enabled_roles(agents: List[dict]) -> Set[str]:
    return {a.get("role") for a in agents if a.get("enabled")}


def _planned_roles(agents: List[dict]) -> Set[str]:
    return {a.get("role") for a in agents if not a.get("enabled")}


def _diff(label: str, left: Set[str], right: Set[str]) -> str:
    return (
        f"{label} 漂移：only-in-left={sorted(left - right)} "
        f"only-in-right={sorted(right - left)}"
    )


class TestRoleVocabConsistency(unittest.TestCase):
    """V1/V5-partial: Python 运行时注册表 ↔ yaml 声明面（不依赖机制核）。"""

    def test_python_yaml_roles_aligned(self):
        agents = _yaml_data()["agents"]
        py_roles = set(_py_roles(_read(PY_INIT)))
        yaml_roles = _enabled_roles(agents)
        assert py_roles == yaml_roles, _diff("enabled role 集", py_roles, yaml_roles)
        assert len(py_roles) == 11, f"Python 角色数应为 11，实测 {len(py_roles)}"

    def test_python_alias_targets_registered(self):
        py_roles = set(_py_roles(_read(PY_INIT)))
        targets = set(_py_aliases(_read(PY_INIT)).values())
        stray = targets - py_roles
        assert not stray, f"Python 别名目标未登记为具体角色: {sorted(stray)}"

    def test_planned_roles_not_runtime(self):
        agents = _yaml_data()["agents"]
        py_roles = set(_py_roles(_read(PY_INIT)))
        planned = _planned_roles(agents)
        leaked = planned & py_roles
        assert not leaked, f"planned 角色不得进入运行时词汇: {sorted(leaked)}"


class TestMechanismVocabAlignment(unittest.TestCase):
    """V1-V4/V5: 机制核 C SSoT ↔ Python ↔ yaml 全量对齐（需 agentrt 机制核）。"""

    def test_roles_c_python_yaml_aligned(self):
        c_roles = set(_c_roles(_require_c()))
        py_roles = set(_py_roles(_read(PY_INIT)))
        yaml_roles = _enabled_roles(_yaml_data()["agents"])
        assert c_roles == py_roles, _diff("C↔Python 角色集", c_roles, py_roles)
        assert c_roles == yaml_roles, _diff("C↔yaml 角色集", c_roles, yaml_roles)
        assert len(c_roles) == 11, f"机制核角色数应为 11，实测 {len(c_roles)}"

    def test_aliases_c_python_aligned(self):
        c_alias = _c_aliases(_require_c())
        py_alias = _py_aliases(_read(PY_INIT))
        assert c_alias == py_alias, (
            "别名映射漂移：only-in-C="
            f"{sorted(set(c_alias) - set(py_alias))} "
            f"only-in-Python={sorted(set(py_alias) - set(c_alias))} "
            f"值不一致={sorted(k for k in set(c_alias) & set(py_alias) if c_alias[k] != py_alias[k])}"
        )

    def test_c_alias_targets_registered(self):
        c_roles = set(_c_roles(_require_c()))
        stray = set(_c_aliases(_require_c()).values()) - c_roles
        assert not stray, f"C 别名目标未登记为具体角色: {sorted(stray)}"

    def test_readonly_c_runner_aligned(self):
        c_ro = _c_readonly(_require_c())
        runner_ro = _runner_readonly(_read(PY_RUNNER))
        assert c_ro == runner_ro, _diff("只读集合 C↔runner", c_ro, runner_ro)

    def test_fallback_c_python_aligned(self):
        assert _VOCAB_H is not None and _VOCAB_H.is_file(), (
            f"agentrt 已存在但缺少词汇头文件: {_VOCAB_H}"
        )
        c_fb = _c_fallback(_read(_VOCAB_H))
        py_fb = _py_fallback(_read(PY_INIT))
        assert c_fb == py_fb, f"兜底角色漂移：C={c_fb!r} Python={py_fb!r}"


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if outcome.wasSuccessful() else 1)
