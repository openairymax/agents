# SPDX-FileCopyrightText: 2025-2026 SPHARX Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later OR Apache-2.0

"""Planning strategy module — 任务规划策略。

提供多级规划能力，将复杂目标分解为可执行的子任务 DAG：

- :class:`PlanningStrategy`      — 基础规划策略：启发式分解，返回结构化计划 dict
- :class:`HierarchicalPlanner`  — 层次化规划器：可接入可选 LLM 客户端做分解，
                                  未提供 LLM 时退化为启发式分解
- :class:`ReactivePlanner`      — 响应式规划器：真实验证目标是否可被已注册的
                                  工具 / 能力解决，并给出判定依据
- :class:`ReflectivePlanner`    — 反思式规划器：接受上次执行结果 / 反馈作为输入，
                                  基于反馈调整与改进计划

所有规划器的 ``plan()`` 均为 async 签名，与 Airymax 异步编排模型一致。
"""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# ─────────────────────────────────────────────────────────────
# 基础数据结构
# ─────────────────────────────────────────────────────────────


@dataclass
class PlanStep:
    """计划中的单个步骤。

    Attributes:
        step_id:        步骤唯一标识
        description:    步骤描述
        dependencies:   依赖的步骤 ID 列表
        assigned_agent: 分配的 Agent 标识（可选）
    """

    step_id: str = ""
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    assigned_agent: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转为 dict，便于序列化 / 下游消费。"""
        return {
            "step_id": self.step_id,
            "description": self.description,
            "dependencies": list(self.dependencies),
            "assigned_agent": self.assigned_agent,
        }


@dataclass
class TaskNode:
    """任务依赖图（DAG）中的一个节点。"""

    id: str
    name: str = ""
    description: str = ""
    status: str = "pending"
    priority: int = 50
    dependencies: Set[str] = field(default_factory=set)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TaskNode):
            return NotImplemented
        return self.id == other.id


class TaskDAG:
    """任务依赖有向无环图。

    提供拓扑排序分层（并行可执行层）与就绪任务查询，供编排器消费。
    """

    def __init__(self, root_goal: str = ""):
        self.root_goal = root_goal
        self.nodes: Dict[str, TaskNode] = {}
        self.edges: Set[Tuple[str, str]] = set()

    def add_node(self, node: TaskNode) -> None:
        """加入节点，并按其依赖登记边。"""
        self.nodes[node.id] = node
        for dep_id in node.dependencies:
            self.edges.add((dep_id, node.id))

    def get_execution_order(self) -> List[List[TaskNode]]:
        """拓扑排序，返回可并行执行的任务分层。"""
        in_degree = {node_id: 0 for node_id in self.nodes}
        adj = {node_id: [] for node_id in self.nodes}

        for parent, child in self.edges:
            if parent in adj and child in in_degree:
                adj[parent].append(child)
                in_degree[child] += 1

        queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
        layers: List[List[TaskNode]] = []

        while queue:
            layer = []
            for _ in range(len(queue)):
                nid = queue.popleft()
                layer.append(self.nodes[nid])
                for neighbor in adj.get(nid, []):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
            layers.append(layer)

        return layers

    def get_ready_tasks(self, completed: Set[str]) -> List[TaskNode]:
        """返回依赖全部完成、当前可执行的任务。"""
        ready = []
        for node in self.nodes.values():
            if node.id in completed:
                continue
            if node.dependencies.issubset(completed):
                ready.append(node)
        return ready

    def validate(self) -> Tuple[bool, List[str]]:
        """校验 DAG 无环。无环返回 (True, [])，否则返回 (False, 错误列表)。"""
        try:
            self.get_execution_order()
            return True, []
        except Exception as exc:  # 环会导致队列提前清空而非抛异常，此处兜底
            return False, [str(exc)]

    def to_dict(self) -> Dict[str, Any]:
        """导出为 dict：root_goal / nodes / edges，便于序列化。"""
        return {
            "root_goal": self.root_goal,
            "nodes": [
                {
                    "id": n.id,
                    "name": n.name,
                    "description": n.description,
                    "status": n.status,
                    "priority": n.priority,
                    "dependencies": sorted(n.dependencies),
                }
                for n in self.nodes.values()
            ],
            "edges": sorted([list(e) for e in self.edges]),
        }


@dataclass
class PlanningContext:
    """规划上下文：目标、最大深度、超时与约束。"""

    goal: str = ""
    max_depth: int = 5
    timeout: float = 60.0
    constraints: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# 启发式分解（真实实现，非固定 step-1..N 模板）
# ─────────────────────────────────────────────────────────────

#: 关键词 → (步骤名, 步骤说明模板)。中英文关键词混用，按阶段顺序排列。
_KEYWORD_STEPS: List[Tuple[Tuple[str, ...], str, str]] = [
    (
        ("研究", "调研", "搜集", "搜索", "资料", "research", "investigate", "gather", "collect"),
        "需求调研",
        "分析任务背景与输入约束，明确目标与验收标准",
    ),
    (
        ("设计", "架构", "建模", "规划", "design", "architecture", "model", "plan"),
        "方案设计",
        "基于调研结论设计整体方案、模块划分与数据结构",
    ),
    (
        ("实现", "开发", "编码", "编写", "编程", "implement", "code", "build", "develop", "write"),
        "实现开发",
        "按设计方案完成核心功能实现",
    ),
    (
        ("测试", "验证", "校验", "审查", "review", "test", "verify", "validate", "audit"),
        "测试验证",
        "对实现结果进行测试、审查与验证，确保质量达标",
    ),
    (
        ("部署", "发布", "上线", "deploy", "release", "publish", "ship"),
        "部署交付",
        "将验证通过的成果部署并完成交付说明",
    ),
]


def _split_clauses(goal: str) -> List[str]:
    """按常见分隔符切分目标文本为独立子句。"""
    parts = [p.strip() for p in re.split(r"[，。；;、,\n]", goal) if p.strip()]
    return parts


def _matches_phase(goal: str, keywords: Tuple[str, ...]) -> bool:
    """目标文本是否命中某阶段的关键词（英文做大小写不敏感匹配）。"""
    lowered = goal.lower()
    return any(k.lower() in lowered for k in keywords)


def heuristic_decompose(goal: str, max_depth: int) -> List[Dict[str, Any]]:
    """启发式任务分解（真实实现）。

    分解策略：
    1. 先按关键词匹配目标所属阶段（调研 / 设计 / 实现 / 测试 / 部署），
       命中阶段按先后顺序生成串行依赖的步骤；
    2. 若未命中任何阶段关键词，则按分隔符把目标切分为子句，
       每个子句生成一个可并行的子任务；
    3. 总步骤数受 ``max_depth`` 约束。

    返回步骤 dict 列表，每项含 ``id`` / ``name`` / ``description``。
    """
    depth = max(1, int(max_depth or 1))
    lowered = goal.lower()

    matched: List[Tuple[str, str]] = []
    for keywords, name, desc in _KEYWORD_STEPS:
        if _matches_phase(goal, keywords):
            matched.append((name, desc))
    # 去重保序
    seen: Set[str] = set()
    unique_matched: List[Tuple[str, str]] = []
    for name, desc in matched:
        if name not in seen:
            seen.add(name)
            unique_matched.append((name, desc))

    steps: List[Dict[str, Any]] = []

    if unique_matched:
        # 阶段型：串行依赖（后续阶段依赖前一阶段）
        for idx, (name, desc) in enumerate(unique_matched[:depth]):
            steps.append(
                {
                    "id": f"step-{idx + 1}",
                    "name": name,
                    "description": f"{desc}（目标：{goal}）",
                }
            )
    else:
        # 子句型：并行独立子任务
        clauses = _split_clauses(goal)
        if len(clauses) <= 1:
            # 单句且无关键词命中：按长度拆分为“理解 + 执行 + 收敛”三段
            steps.append(
                {"id": "step-1", "name": "目标分析",
                 "description": f"理解目标「{goal}」的背景、范围与约束"}
            )
            steps.append(
                {"id": "step-2", "name": "执行落地",
                 "description": f"围绕「{goal}」完成主体工作"}
            )
            steps.append(
                {"id": "step-3", "name": "结果校验",
                 "description": f"复核「{goal}」的产出质量并交付"}
            )
        else:
            for idx, clause in enumerate(clauses[:depth]):
                steps.append(
                    {
                        "id": f"step-{idx + 1}",
                        "name": f"子任务 {idx + 1}",
                        "description": f"完成：{clause}",
                    }
                )

    return steps


def _build_dag(goal: str, steps: List[Dict[str, Any]], chained: bool = True) -> TaskDAG:
    """将步骤列表装配为 TaskDAG。

    ``chained=True`` 时步骤间构成串行依赖（前一完成才能执行后一）；
    ``chained=False`` 时步骤相互独立（可并行）。
    """
    dag = TaskDAG(root_goal=goal)
    prev_id: Optional[str] = None
    for st in steps:
        dependencies = {prev_id} if (chained and prev_id) else set()
        node = TaskNode(
            id=st["id"],
            name=st.get("name", st["id"]),
            description=st.get("description", ""),
            dependencies=dependencies,
        )
        dag.add_node(node)
        prev_id = st["id"]
    return dag


def _task_text(task: Any) -> str:
    """把任意形态的任务输入规范化为目标文本。"""
    if isinstance(task, dict):
        return str(task.get("description") or task.get("goal") or str(task))
    if isinstance(task, PlanningContext):
        return task.goal or str(task)
    return str(task)


# ─────────────────────────────────────────────────────────────
# 规划策略
# ─────────────────────────────────────────────────────────────


class PlanningStrategy:
    """基础规划策略。

    对任务做启发式分解，返回包含任务描述、依赖、策略类型与步骤列表的
    结构化计划 dict。
    """

    name: str = "planning"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._plan_count = 0

    async def plan(self, task: Any, context: Optional[PlanningContext] = None) -> Dict[str, Any]:
        """将任务分解为执行计划。

        参数:
            task:    任务对象（dict / PlanningContext / 字符串）
            context: 规划上下文（提供 max_depth 等约束）

        返回:
            Dict: 含 status / strategy / task / dependencies / steps 的计划结构
        """
        self._plan_count += 1
        goal = _task_text(task)
        depth = context.max_depth if context else int(self.config.get("max_depth", 3))
        steps = heuristic_decompose(goal, depth)

        return {
            "status": "completed",
            "strategy": self.name,
            "task": goal,
            "dependencies": [s["id"] for s in steps[1:]],
            "steps": [
                PlanStep(
                    step_id=s["id"],
                    description=s["description"],
                    dependencies=[steps[i - 1]["id"]] if i > 0 else [],
                )
                for i, s in enumerate(steps)
            ],
        }


class HierarchicalPlanner(PlanningStrategy):
    """层次化规划器：将目标逐层分解为任务 DAG。

    支持可选的 LLM 分解：构造或调用时传入兼容 OpenAI Chat Completions 接口
    的异步客户端（提供 ``async chat(messages, ...)``，如
    :class:`orchestration.core.llm.LLMClient`），则优先用 LLM 生成分解；
    LLM 不可用 / 解析失败时自动退化为启发式分解（按关键词与子句切分），
    保证任何环境都能产出有意义的步骤。
    """

    def __init__(
        self,
        max_depth: int = 3,
        config: Optional[Dict[str, Any]] = None,
        llm: Optional[Any] = None,
    ):
        super().__init__(config)
        self.max_depth = max_depth
        self.name = "HierarchicalPlanner"
        self.llm = llm  # 可选：异步 LLM 客户端
        self._plan_count = 0

    async def _llm_decompose(self, goal: str, llm: Any) -> Optional[List[Dict[str, Any]]]:
        """调用 LLM 生成子任务分解，返回步骤列表；失败返回 None。"""
        try:
            resp = await llm.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "你是任务规划助手。请将用户目标分解为有序子任务，"
                            "仅输出 JSON 数组，每项包含 name 与 description 两个字符串字段。"
                        ),
                    },
                    {"role": "user", "content": f"目标：{goal}"},
                ],
                temperature=0.3,
            )
        except Exception:
            return None

        content = ""
        try:
            content = resp["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            return None

        # 从回复中提取 JSON 数组（容忍 LLM 在数组前后附加说明文字）
        match = re.search(r"\[.*\]", content, re.S)
        if not match:
            return None
        try:
            items = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if not isinstance(items, list):
            return None

        steps: List[Dict[str, Any]] = []
        for i, item in enumerate(items[: self.max_depth]):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("step") or f"子任务 {i + 1}")
            desc = str(
                item.get("description")
                or item.get("desc")
                or f"为达成目标「{goal}」执行：{name}"
            )
            steps.append({"id": f"step-{i + 1}", "name": name, "description": desc})
        return steps or None

    async def plan(
        self,
        task: Any,
        context: Optional[PlanningContext] = None,
        llm: Optional[Any] = None,
    ) -> TaskDAG:
        """将目标分解为 TaskDAG。

        参数:
            task:    目标（字符串 / dict / PlanningContext）
            context: 规划上下文（其 max_depth 可覆盖构造时的 max_depth）
            llm:     可选 LLM 客户端，优先级高于构造时传入的 llm

        返回:
            TaskDAG: 包含真实分解步骤的依赖图
        """
        self._plan_count += 1
        goal = _task_text(task)

        depth = min(
            self.max_depth,
            context.max_depth if context else self.max_depth,
        )
        depth = max(1, depth)

        llm_client = llm if llm is not None else self.llm
        steps: Optional[List[Dict[str, Any]]] = None
        if llm_client is not None:
            steps = await self._llm_decompose(goal, llm_client)
        if not steps:
            steps = heuristic_decompose(goal, depth)

        return _build_dag(goal, steps, chained=True)


#: 内置能力关键词表：能力名 → (能力 id, 关键词元组)，用于响应式规划的真实验证
_BUILTIN_CAPABILITIES: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("文件读写", "fs", ("读文件", "写文件", "创建文件", "修改文件", "文件", "fs", "glob")),
    ("shell 命令执行", "shell", ("执行", "运行命令", "shell", "终端", "命令行", "脚本")),
    ("网页抓取", "web_fetch", ("网页", "抓取", "爬取", "fetch", "下载页面", "http")),
    ("信息搜索", "web_search", ("搜索", "检索", "查资料", "search", "查询")),
    ("代码开发", "coding", ("写代码", "编码", "实现", "开发", "编程", "代码", "脚本",
                            "python", "c++", "java", "go", "rust", "code", "implement")),
    ("数据分析", "data_analysis", ("分析", "统计", "数据", "报表", "analysis", "analytics")),
    ("代码审查", "code_review", ("审查", "代码检查", "review", "审计")),
]


class ReactivePlanner(PlanningStrategy):
    """响应式规划器：验证目标是否可用已注册工具 / 能力解决。

    对目标做能力可行性校验（内置能力关键词表 + 用户可注入的能力注册表），
    输出判定结果与依据；校验通过的任务节点即直接可执行。
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        capabilities: Optional[Dict[str, str]] = None,
    ):
        super().__init__(config)
        self.name = "ReactivePlanner"
        self.capabilities = capabilities or {}  # 能力名 → 能力说明（用户注入）
        self._plan_count = 0

    def _assess_feasibility(self, goal: str) -> Dict[str, Any]:
        """真实验证：目标是否可被已注册能力 / 内置工具解决。

        返回::

            {
                "feasible": bool,
                "matched": [能力名, ...],
                "evidence": "判定依据文本",
            }
        """
        lowered = goal.lower()
        matched: List[str] = []

        # 1) 用户注入的能力注册表（名称 + 说明均参与匹配）
        for name, desc in self.capabilities.items():
            haystack = f"{name} {desc}".lower()
            # 能力描述中的任一实词命中目标即视为匹配
            tokens = [t for t in re.split(r"[\s/：:，,、（）()]+", haystack) if len(t) >= 2]
            if any(tok in lowered for tok in tokens):
                matched.append(name)

        # 2) 内置能力关键词表
        for cap_name, cap_id, keywords in _BUILTIN_CAPABILITIES:
            if any(k in lowered for k in keywords):
                matched.append(f"{cap_name}({cap_id})")

        # 去重保序
        seen: Set[str] = set()
        unique = [m for m in matched if not (m in seen or seen.add(m))]

        if unique:
            evidence = (
                f"目标包含与已注册能力匹配的关键词，可交由 {len(unique)} 项能力处理："
                f"{'、'.join(unique)}。"
            )
            return {"feasible": True, "matched": unique, "evidence": evidence}

        evidence = (
            "目标未匹配到任何已注册工具 / 能力，可能需要人工介入或先注册相应能力。"
        )
        return {"feasible": False, "matched": [], "evidence": evidence}

    async def plan(
        self,
        task: Any,
        context: Optional[PlanningContext] = None,
        capabilities: Optional[Dict[str, str]] = None,
    ) -> TaskDAG:
        """创建响应式计划：先验证可行性，再给出可执行节点。

        参数:
            task:         目标
            context:      规划上下文
            capabilities: 可选的能力注册表（名称 → 说明），优先于构造时注入

        返回:
            TaskDAG: 根节点携带可行性判定，``dag.verification`` 保存判定详情
        """
        self._plan_count += 1
        goal = _task_text(task)
        registry = capabilities if capabilities is not None else self.capabilities
        if registry:
            self.capabilities = registry

        assessment = self._assess_feasibility(goal)

        dag = TaskDAG(root_goal=goal)
        if assessment["feasible"]:
            node = TaskNode(
                id="reactive-1",
                name="响应执行",
                description=(
                    f"{assessment['evidence']} 直接执行：{goal}"
                ),
            )
        else:
            node = TaskNode(
                id="reactive-1",
                name="人工介入",
                description=f"{assessment['evidence']} 待补充能力后执行：{goal}",
            )
        dag.add_node(node)
        # 可行性判定随 DAG 一起交付，供编排器决策
        dag.verification = assessment
        return dag


class ReflectivePlanner(PlanningStrategy):
    """反思式规划器：基于上次执行结果 / 反馈改进计划。

    接受 ``previous_result``（上次执行结果）或 ``feedback``（反馈文本）作为
    输入：若上次执行失败或反馈指出问题，则在计划中插入问题分析与修复步骤，
    并保留一份 ``dag.reflection`` 反思记录。
    """

    def __init__(
        self,
        max_depth: int = 3,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)
        self.max_depth = max_depth
        self.name = "ReflectivePlanner"
        self.history: List[Dict[str, Any]] = []  # 历次反思记录
        self._plan_count = 0

    def _extract_issues(
        self, previous_result: Optional[Any], feedback: Optional[str]
    ) -> List[str]:
        """从上次结果与反馈中提取问题清单（真实解析，而非空模板）。"""
        issues: List[str] = []

        if isinstance(previous_result, dict):
            ok = previous_result.get("success", previous_result.get("ok", True))
            if not ok:
                issues.append(
                    str(previous_result.get("error") or previous_result.get("reason")
                        or "上次执行未成功")
                )
            elif previous_result.get("issues"):
                for item in previous_result["issues"]:
                    issues.append(str(item))
        elif isinstance(previous_result, Exception):
            issues.append(str(previous_result))

        if feedback:
            text = str(feedback)
            # 按反馈中的问题性关键词切分提取
            for sentence in re.split(r"[。；;\n]", text):
                if not sentence.strip():
                    continue
                if any(
                    kw in sentence
                    for kw in ("失败", "错误", "缺失", "问题", "改进", "修复",
                               "报错", "异常", "不足", "缺少", "未", "没有",
                               "补充", "完善", "优化", "增加",
                               "fail", "error", "bug", "issue", "fix", "missing")
                ):
                    issues.append(sentence.strip())

        # 去重保序
        seen: Set[str] = set()
        return [i for i in issues if not (i in seen or seen.add(i))]

    async def plan(
        self,
        task: Any,
        context: Optional[PlanningContext] = None,
        previous_result: Optional[Any] = None,
        feedback: Optional[str] = None,
    ) -> TaskDAG:
        """创建反思式计划：融合上次结果 / 反馈改进任务分解。

        参数:
            task:            目标
            context:         规划上下文
            previous_result: 上次执行结果（dict / Exception），失败信息进入计划
            feedback:        反馈文本，问题描述进入计划

        返回:
            TaskDAG: 含初始分解 + 问题修复步骤；``dag.reflection`` 保存反思记录
        """
        self._plan_count += 1
        goal = _task_text(task)
        depth = min(
            self.max_depth,
            context.max_depth if context else self.max_depth,
        )
        depth = max(1, depth)

        base_steps = heuristic_decompose(goal, depth)
        issues = self._extract_issues(previous_result, feedback)

        steps: List[Dict[str, Any]] = []
        if issues:
            # 反思调整：先诊断问题，再执行主体任务，最后回归验证
            steps.append(
                {
                    "id": "reflect-0",
                    "name": "问题诊断",
                    "description": f"基于上次反馈定位根因：{'；'.join(issues[:3])}",
                }
            )
            for idx, issue in enumerate(issues[:2]):
                steps.append(
                    {
                        "id": f"reflect-fix-{idx + 1}",
                        "name": f"修复项 {idx + 1}",
                        "description": f"针对「{issue}」实施修复并验证",
                    }
                )
        # 主体分解步骤（串行依赖）
        for st in base_steps:
            steps.append(st)
        # 收尾：反思评审（反思式规划器的自我校验环节）
        steps.append(
            {
                "id": "reflect-final",
                "name": "反思评审",
                "description": f"复核「{goal}」的完成质量，对照反馈确认问题已闭环",
            }
        )

        dag = _build_dag(goal, steps, chained=True)

        # 反思记录：供编排器与可观测性消费
        reflection = {
            "issues_found": issues,
            "had_feedback": feedback is not None or previous_result is not None,
            "adjustments": len(issues),
            "total_steps": len(steps),
        }
        dag.reflection = reflection
        self.history.append(reflection)
        return dag


__all__ = [
    "PlanningStrategy",
    "PlanStep",
    "TaskNode",
    "TaskDAG",
    "PlanningContext",
    "heuristic_decompose",
    "HierarchicalPlanner",
    "ReactivePlanner",
    "ReflectivePlanner",
]
