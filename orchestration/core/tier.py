# Copyright (c) 2026 SPHARX. All Rights Reserved.
"""任务分级：以任务类型与复杂度信号在 t1-f / t2 之间选型。

机制与策略分离：本模块只产出分级（机制），实际模型由契约
``models.system1``/``models.system2`` 与部署级 ``AIRY_AGENT_MODEL``
决定（策略）。判据顺序：

1. 上下文显式 ``complexity``/``tier``（fast/deep）优先，供调用方直指；
2. 显式 ``task_type`` 属于深度类型直接 t2；
3. 其余按内容信号计分，达到 :data:`DEEP_SCORE` 判 t2。

长度只作弱特征（S-P1-08）：长而机械的输入不升级，短而多步或改码的
输入必须升级，避免按字符数一刀切把复杂请求降级为快思考。
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

TIER_FAST = "t1f"
TIER_DEEP = "t2"

# 长度弱特征阈值：越线最多计 1 分，不单独触发升级。
LONG_INPUT_CHARS = 600
# 深度判定阈值：两个独立内容信号即升级。
DEEP_SCORE = 2

_FAST_WORDS = frozenset({"fast", "t1f", "t1-f", "low", "simple"})
_DEEP_WORDS = frozenset({"deep", "t2", "high", "complex"})

_DEEP_TYPES = frozenset(
    {
        "refactor",
        "debug",
        "design",
        "review",
        "analysis",
        "audit",
        "migration",
        "implementation",
        "feature",
        "test",
    }
)

# 改动类动作：直接改变代码/配置/运行面，短句也属深度任务（计 2 分）。
_CHANGE_VERBS = (
    "重构",
    "实现",
    "修复",
    "调试",
    "改造",
    "新增",
    "删除",
    "移除",
    "迁移",
    "优化",
    "补齐",
    "改成",
    "改为",
    "替换",
    "排查",
    "定位",
    "部署",
    "安装",
)
# 认知类动作：本身需要推理，但短问答仍可快思考（计 1 分）。
_ANALYZE_VERBS = ("审查", "分析", "评估", "验证", "回归", "测试", "对比", "设计")
_DEEP_VERB_EN = re.compile(
    r"\b(refactor|implement|debug|design|review|migrate|optimi[sz]e|audit|diagnose)\b",
    re.I,
)
# 源码/配置文件名出现即视为改码类任务（弱信号，与动作词配合）。
_CODE_REF = re.compile(
    r"\b[\w./-]+\.(py|rs|c|h|cc|cpp|go|ts|tsx|js|yaml|yml|json|toml|md)\b"
)
_STEP_MARK = re.compile(r"(?:^|\n)\s*(?:\d+[.、)]|[-*]\s)", re.M)
_STEP_WORDS = ("然后", "接着", "之后", "最后", "同时", "并且")


def _forced_tier(meta: Optional[Mapping[str, Any]]) -> Optional[str]:
    """读取显式分级元数据；未声明或值不可识别时返回 None。"""
    if not meta:
        return None
    raw = str(meta.get("complexity") or meta.get("tier") or "").strip().lower()
    if raw:
        if raw in _DEEP_WORDS:
            return TIER_DEEP
        if raw in _FAST_WORDS:
            return TIER_FAST
        return None
    task_type = str(meta.get("task_type") or "").strip().lower()
    return TIER_DEEP if task_type in _DEEP_TYPES else None


def tier_score(text: str, meta: Optional[Mapping[str, Any]] = None) -> int:
    """内容信号计分：动作词、改码引用、代码围栏、多步、工具、长度各计一次。"""
    score = 0
    meta = meta or {}
    if str(meta.get("requires_tools") or "").strip().lower() in ("1", "true", "yes"):
        score += 1
    if any(v in text for v in _CHANGE_VERBS) or _DEEP_VERB_EN.search(text):
        score += 2
    elif any(v in text for v in _ANALYZE_VERBS):
        score += 1
    if _CODE_REF.search(text):
        score += 1
    if "```" in text:
        score += 1
    steps = len(_STEP_MARK.findall(text))
    if steps >= 2 or sum(1 for w in _STEP_WORDS if w in text) >= 2:
        score += 1
    if len(text) > LONG_INPUT_CHARS:
        score += 1
    return score


def pick_tier(text: str, meta: Optional[Mapping[str, Any]] = None) -> str:
    """分级入口：显式元数据优先，否则内容信号分达阈值为深度思考。"""
    forced = _forced_tier(meta)
    if forced:
        return forced
    return TIER_DEEP if tier_score(text, meta) >= DEEP_SCORE else TIER_FAST


__all__ = [
    "TIER_FAST",
    "TIER_DEEP",
    "LONG_INPUT_CHARS",
    "DEEP_SCORE",
    "tier_score",
    "pick_tier",
]
