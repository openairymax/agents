# SPDX-License-Identifier: AGPL-3.0-or-later OR Apache-2.0
"""orchestration Core — 项目上下文文件加载（AGENTS.md 等价物）。

参照 Claude Code 的 CLAUDE.md / Codex 的 AGENTS.md 设计：Agent 启动或执行时
自动向上查找项目约定文件，注入系统上下文，让 LLM 感知项目级约束
（编码规范、目录约定、禁改路径等）。

- :data:`PROJECT_CONTEXT_FILENAMES`  等价物候选文件名（按优先级排列）
- :func:`find_project_context`       从 start_dir 向上逐级查找并拼接内容
- :func:`inject_project_context`     将项目上下文作为 system 消息插入最前
- 进程内 LRU 缓存（起始目录 → 内容），避免每次调用重复扫盘
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional

# 项目约定文件等价物候选（Codex: AGENTS.md；Claude Code: CLAUDE.md；
# Airymax 私有: .airymax/AGENTS.md）。按优先级从上到下匹配，
# 同一目录只取第一个命中文件。
PROJECT_CONTEXT_FILENAMES: List[str] = [
    "AGENTS.md",
    "agents.md",
    "CLAUDE.md",
    ".airymax/AGENTS.md",
]

# 注入 system 消息的前缀标签
_CONTEXT_LABEL = "项目约定（AGENTS.md）:"

# 进程内 LRU 缓存：起始目录绝对路径 → 拼接后的项目上下文内容
_CACHE_MAXSIZE = 64
_cache: "OrderedDict[str, str]" = OrderedDict()


def _cache_get(key: str) -> Optional[str]:
    """LRU 缓存读取，命中时刷新最近使用序。"""
    value = _cache.get(key)
    if value is not None:
        _cache.move_to_end(key)
    return value


def _cache_put(key: str, value: str) -> None:
    """LRU 缓存写入，超容量淘汰最久未用项。"""
    _cache[key] = value
    _cache.move_to_end(key)
    while len(_cache) > _CACHE_MAXSIZE:
        _cache.popitem(last=False)


def clear_project_context_cache() -> None:
    """清空进程内项目上下文缓存（测试隔离或热更新时调用）。"""
    _cache.clear()


def _read_candidates(dirpath: str) -> Optional[str]:
    """读取 dirpath 下第一个命中的候选文件，返回带路径标注的内容。"""
    for name in PROJECT_CONTEXT_FILENAMES:
        path = os.path.join(dirpath, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f"{path}\n{f.read().rstrip()}\n"
        except OSError:
            continue
    return None


def find_project_context(start_dir: Optional[str] = None) -> str:
    """从 start_dir（默认 cwd）向上查找项目约定文件。

    逐级向上扫描：每级目录按 :data:`PROJECT_CONTEXT_FILENAMES` 检查，
    命中即拼接（内容前带文件路径标注）；遇下列条件之一停止：

    1. 目录包含 ``.git/``（视为项目根，命中则在检查该目录后停止）
    2. 到达文件系统根

    找不到返回空串。结果按起始目录做进程内 LRU 缓存。
    """
    start = os.path.abspath(start_dir or os.getcwd())
    cached = _cache_get(start)
    if cached is not None:
        return cached

    parts: List[str] = []
    current = start
    while True:
        found = _read_candidates(current)
        if found is not None:
            parts.append(found)
        if os.path.isdir(os.path.join(current, ".git")):
            break
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    content = "".join(parts)
    _cache_put(start, content)
    return content


def inject_project_context(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """把项目上下文作为一条 system 消息插入 messages 最前。

    未找到项目上下文时原样返回原列表；找到时返回新列表：
    ``[{"role": "system", "content": "项目约定（AGENTS.md）:\\n<内容>"}, *messages]``
    """
    context = find_project_context()
    if not context:
        return messages
    return [
        {"role": "system", "content": f"{_CONTEXT_LABEL}\n{context}"},
        *messages,
    ]


__all__ = [
    "PROJECT_CONTEXT_FILENAMES",
    "find_project_context",
    "inject_project_context",
    "clear_project_context_cache",
]
