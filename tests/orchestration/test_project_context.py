# SPDX-License-Identifier: AGPL-3.0-or-later OR Apache-2.0
"""项目上下文文件加载（AGENTS.md 等价物）单测。

覆盖 ``orchestration/core/project_context.py``：
  - find_project_context    向上逐级查找、.git 项目根边界、拼接路径标注
  - inject_project_context  system 消息注入顺序、未命中原样返回
"""

from __future__ import annotations

import pytest

from orchestration.core import project_context


@pytest.fixture(autouse=True)
def _clear_cache():
    """每个用例前清空进程内缓存，保证真实扫盘断言。"""
    project_context.clear_project_context_cache()
    yield


class TestFindProjectContext:
    """find_project_context：向上查找与 .git 边界。"""

    def test_find_agents_md_in_start_dir(self, tmp_path):
        proj = tmp_path / "proj"
        (proj / ".git").mkdir(parents=True)
        (proj / "AGENTS.md").write_text("使用中文注释\n", encoding="utf-8")

        content = project_context.find_project_context(str(proj))

        assert "AGENTS.md" in content
        assert "使用中文注释" in content

    def test_find_upwards_walks_parent_dirs(self, tmp_path):
        proj = tmp_path / "proj"
        sub = proj / "sub"
        sub.mkdir(parents=True)
        (proj / ".git").mkdir()
        (proj / "CLAUDE.md").write_text("顶层约定\n", encoding="utf-8")

        content = project_context.find_project_context(str(sub))

        assert "CLAUDE.md" in content
        assert "顶层约定" in content

    def test_git_dir_is_project_root_boundary(self, tmp_path):
        proj = tmp_path / "proj"
        sub = proj / "sub"
        sub.mkdir(parents=True)
        (proj / ".git").mkdir()
        # AGENTS.md 位于 .git 边界（proj）之外，不应被扫到
        (tmp_path / "AGENTS.md").write_text("边界外约定\n", encoding="utf-8")

        assert project_context.find_project_context(str(sub)) == ""

    def test_not_found_returns_empty(self, tmp_path):
        proj = tmp_path / "empty"
        (proj / ".git").mkdir(parents=True)

        assert project_context.find_project_context(str(proj)) == ""


class TestInjectProjectContext:
    """inject_project_context：注入顺序与未命中原样返回。"""

    def test_inject_prepends_system_message(self, monkeypatch):
        monkeypatch.setattr(
            project_context, "find_project_context", lambda: "约定内容"
        )
        messages = [
            {"role": "system", "content": "系统提示"},
            {"role": "user", "content": "任务"},
        ]

        result = project_context.inject_project_context(messages)

        assert result[0]["role"] == "system"
        assert result[0]["content"] == "项目约定（AGENTS.md）:\n约定内容"
        assert result[1:] == messages

    def test_no_context_returns_messages_unchanged(self, monkeypatch):
        monkeypatch.setattr(
            project_context, "find_project_context", lambda: ""
        )
        messages = [{"role": "user", "content": "任务"}]

        result = project_context.inject_project_context(messages)

        assert result is messages
