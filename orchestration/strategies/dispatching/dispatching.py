# SPDX-FileCopyrightText: 2025-2026 SPHARX Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later OR Apache-2.0

"""Dispatching strategy module — 任务调度策略。

负责任务到 Agent 的分配决策。核心接口 :class:`DispatchingStrategy.dispatch`
返回真实派发结果（``assigned_agents`` 基于 :meth:`select` 的实际选择结果填充）。

内置四类调度策略：

- :class:`WeightedRoundRobinStrategy` — 加权轮询：按权重比例选择 Agent
- :class:`PriorityBasedStrategy`      — 优先级优先：选择优先级最高的 Agent
- :class:`LeastLoadedStrategy`        — 最小负载：选择当前负载最低的 Agent
- :class:`AdaptiveMLStrategy`         — 自适应评分：综合成功率 / 负载 / 响应时间打分

所有策略的 ``dispatch()`` 均感知 Agent 注册表（接受 ``agents`` 列表参数，
元素可以是 :class:`AgentMetrics`、dict 或任意带属性的 Agent 对象），
支持按任务所需能力过滤候选，再调用各自 :meth:`select` 组装结果。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────
# 基础数据结构
# ─────────────────────────────────────────────────────────────


@dataclass
class AgentMetrics:
    """参与派发决策的 Agent 度量。

    Attributes:
        agent_id:          Agent 唯一标识
        weight:            权重（加权轮询用）
        current_load:      当前负载（0.0 ~ 1.0，最小负载策略用）
        avg_response_time: 平均响应时间（秒）
        success_rate:      历史成功率（0.0 ~ 1.0）
        priority:          优先级（越大越优先）
        capabilities:      能力清单（能力匹配过滤用）
    """

    agent_id: str
    weight: float = 1.0
    current_load: float = 0.0
    avg_response_time: float = 0.0
    success_rate: float = 1.0
    priority: int = 0
    capabilities: List[str] = field(default_factory=list)


@dataclass
class TaskContext:
    """待派发任务的上下文。"""

    task_id: str
    task_type: str = "default"
    priority: int = 0
    required_capabilities: List[str] = field(default_factory=list)
    estimated_duration: float = 0.0
    deadline: Optional[float] = None


def _coerce_agent(item: Any) -> Optional[AgentMetrics]:
    """把任意形态的 Agent 条目规范化为 :class:`AgentMetrics`。

    支持：
    - 已是 ``AgentMetrics``：直接返回
    - ``dict``：按 ``agent_id`` / ``id`` / ``name`` 取标识，其余字段按名取值
    - 任意带属性的对象（如 AirymaxAgent）：经 ``getattr`` 取字段
    无法识别时返回 None。
    """
    if isinstance(item, AgentMetrics):
        return item

    if isinstance(item, dict):
        agent_id = (
            item.get("agent_id")
            or item.get("id")
            or item.get("name")
            or ""
        )
        if not agent_id:
            return None
        return AgentMetrics(
            agent_id=str(agent_id),
            weight=float(item.get("weight", 1.0)),
            current_load=float(item.get("current_load", item.get("load", 0.0))),
            avg_response_time=float(item.get("avg_response_time", 0.0)),
            success_rate=float(item.get("success_rate", 1.0)),
            priority=int(item.get("priority", 0)),
            capabilities=list(item.get("capabilities", []) or []),
        )

    agent_id = getattr(item, "agent_id", None) or getattr(item, "id", None)
    if agent_id is None:
        return None
    return AgentMetrics(
        agent_id=str(agent_id),
        weight=float(getattr(item, "weight", 1.0) or 1.0),
        current_load=float(getattr(item, "current_load", getattr(item, "load", 0.0)) or 0.0),
        avg_response_time=float(getattr(item, "avg_response_time", 0.0) or 0.0),
        success_rate=float(getattr(item, "success_rate", 1.0) or 1.0),
        priority=int(getattr(item, "priority", 0) or 0),
        capabilities=list(getattr(item, "capabilities", []) or []),
    )


def _coerce_task(task: Any) -> TaskContext:
    """把任意形态的任务规范化为 :class:`TaskContext`。"""
    if isinstance(task, TaskContext):
        return task
    if isinstance(task, dict):
        return TaskContext(
            task_id=str(task.get("id") or task.get("task_id") or str(task)),
            task_type=str(task.get("type") or task.get("task_type") or "default"),
            priority=int(task.get("priority", 0) or 0),
            required_capabilities=list(task.get("required_capabilities", []) or []),
            estimated_duration=float(task.get("estimated_duration", 0.0) or 0.0),
            deadline=task.get("deadline"),
        )
    task_id = getattr(task, "task_id", None) or getattr(task, "id", None) or str(task)
    return TaskContext(
        task_id=str(task_id),
        task_type=str(getattr(task, "task_type", "default") or "default"),
        priority=int(getattr(task, "priority", 0) or 0),
        required_capabilities=list(getattr(task, "required_capabilities", []) or []),
        estimated_duration=float(getattr(task, "estimated_duration", 0.0) or 0.0),
        deadline=getattr(task, "deadline", None),
    )


# ─────────────────────────────────────────────────────────────
# 基础派发策略
# ─────────────────────────────────────────────────────────────


class DispatchingStrategy:
    """基础派发策略。

    统一实现派发主流程：构建候选 Agent 度量 → 按任务能力过滤 → 调用
    子类的 :meth:`select` 选出目标 → 组装真实派发结果。子类仅需实现
    :meth:`select` 与自身的 :meth:`dispatch`（调用本类实现）。
    """

    name: str = "dispatching"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._total_selections = 0
        self._selection_distribution: Dict[str, int] = {}

    def select(
        self,
        candidates: List[AgentMetrics],
        context: Optional[TaskContext] = None,
    ) -> Optional[AgentMetrics]:
        """默认选择策略：优先返回首个候选（子类各自实现具体策略）。"""
        if not candidates:
            return None
        selected = candidates[0]
        self._total_selections += 1
        self._selection_distribution[selected.agent_id] = (
            self._selection_distribution.get(selected.agent_id, 0) + 1
        )
        return selected

    def get_stats(self) -> Dict[str, Any]:
        """返回策略统计：总选择次数与选择分布。"""
        return {
            "strategy": self.name,
            "total_selections": self._total_selections,
            "selection_distribution": dict(self._selection_distribution),
        }

    # ── 派发辅助（子类共享） ─────────────────────────────

    def _build_candidates(self, agents: Optional[List[Any]]) -> List[AgentMetrics]:
        """把 Agent 注册表条目统一规范化为 AgentMetrics 列表。"""
        if not agents:
            return []
        candidates: List[AgentMetrics] = []
        for item in agents:
            metrics = _coerce_agent(item)
            if metrics is not None:
                candidates.append(metrics)
        return candidates

    def _filter_by_capability(
        self,
        candidates: List[AgentMetrics],
        required: List[str],
    ) -> List[AgentMetrics]:
        """按任务所需能力过滤候选（大小写不敏感匹配）。"""
        if not required:
            return candidates
        lowered = [c.lower() for c in required]
        return [
            c for c in candidates
            if any(k.lower() in lowered for k in c.capabilities)
        ]

    def _assemble_result(
        self,
        task: Any,
        context: TaskContext,
        selected: Optional[AgentMetrics],
        candidates: List[AgentMetrics],
        reason: str = "",
    ) -> Dict[str, Any]:
        """组装真实派发结果：assigned_agents 来自 select() 的实际选择。"""
        result: Dict[str, Any] = {
            "status": "completed",
            "strategy": self.name,
            "task_id": context.task_id,
            "task_type": context.task_type,
            "assigned_agents": [selected.agent_id] if selected else [],
            "candidates": [c.agent_id for c in candidates],
            "dispatch_count": self._total_selections,
        }
        if selected is not None:
            result["selected"] = {
                "agent_id": selected.agent_id,
                "weight": selected.weight,
                "current_load": selected.current_load,
                "avg_response_time": selected.avg_response_time,
                "success_rate": selected.success_rate,
                "priority": selected.priority,
                "capabilities": list(selected.capabilities),
            }
            if reason:
                result["reason"] = reason
        return result

    async def dispatch(self, task: Any, agents: Optional[List[Any]] = None) -> Dict[str, Any]:
        """将任务派发给最合适的 Agent（统一派发主流程）。

        参数:
            task:   任务对象（TaskContext / dict / 字符串）
            agents: 可用 Agent 注册表（AgentMetrics / dict / Agent 对象列表）

        返回:
            Dict: 含 status / strategy / task_id / assigned_agents / selected 等字段
        """
        candidates = self._build_candidates(agents)
        context = _coerce_task(task)
        # 能力感知：按任务所需能力过滤候选
        filtered = self._filter_by_capability(candidates, context.required_capabilities)
        pool = filtered if filtered else candidates
        selected = self.select(pool, context)
        reason = (
            f"按能力过滤后剩余 {len(pool)} 个候选" if filtered else "未做能力过滤"
        )
        return self._assemble_result(task, context, selected, pool, reason)


# ─────────────────────────────────────────────────────────────
# 具体派发策略
# ─────────────────────────────────────────────────────────────


class WeightedRoundRobinStrategy(DispatchingStrategy):
    """加权轮询派发策略。

    按各 Agent 的权重比例做随机加权选择（权重可由构造参数 ``weights`` 覆盖
    每个 Agent 的默认权重）；单个候选时直接选中。
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)
        self.name = "weighted_round_robin"
        self.weights = weights or {}
        self._current_index = 0

    def select(
        self,
        candidates: List[AgentMetrics],
        context: Optional[TaskContext] = None,
    ) -> Optional[AgentMetrics]:
        """从候选中按权重选择一名 Agent。"""
        if not candidates:
            return None

        if len(candidates) == 1:
            selected = candidates[0]
        else:
            total_weight = sum(self.weights.get(c.agent_id, c.weight) for c in candidates)
            if total_weight <= 0:
                selected = candidates[self._current_index % len(candidates)]
                self._current_index += 1
            else:
                r = random.uniform(0, total_weight)
                cumulative = 0
                selected = candidates[0]
                for c in candidates:
                    cumulative += self.weights.get(c.agent_id, c.weight)
                    if r <= cumulative:
                        selected = c
                        break

        self._total_selections += 1
        self._selection_distribution[selected.agent_id] = (
            self._selection_distribution.get(selected.agent_id, 0) + 1
        )
        return selected

    async def dispatch(self, task: Any, agents: Optional[List[Any]] = None) -> Dict[str, Any]:
        """加权轮询派发：调用自身 select() 按权重选择后组装结果。"""
        return await super().dispatch(task, agents)


class PriorityBasedStrategy(DispatchingStrategy):
    """优先级派发策略：始终选择优先级最高的 Agent。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = "priority_based"

    def select(
        self,
        candidates: List[AgentMetrics],
        context: Optional[TaskContext] = None,
    ) -> Optional[AgentMetrics]:
        """选择优先级最高的 Agent。"""
        if not candidates:
            return None

        selected = max(candidates, key=lambda c: c.priority)

        self._total_selections += 1
        self._selection_distribution[selected.agent_id] = (
            self._selection_distribution.get(selected.agent_id, 0) + 1
        )
        return selected

    async def dispatch(self, task: Any, agents: Optional[List[Any]] = None) -> Dict[str, Any]:
        """优先级派发：调用自身 select() 选择最高优先级 Agent 后组装结果。"""
        return await super().dispatch(task, agents)


class LeastLoadedStrategy(DispatchingStrategy):
    """最小负载派发策略：始终选择当前负载最低的 Agent。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = "least_loaded"

    def select(
        self,
        candidates: List[AgentMetrics],
        context: Optional[TaskContext] = None,
    ) -> Optional[AgentMetrics]:
        """选择当前负载最低的 Agent。"""
        if not candidates:
            return None

        selected = min(candidates, key=lambda c: c.current_load)

        self._total_selections += 1
        self._selection_distribution[selected.agent_id] = (
            self._selection_distribution.get(selected.agent_id, 0) + 1
        )
        return selected

    async def dispatch(self, task: Any, agents: Optional[List[Any]] = None) -> Dict[str, Any]:
        """最小负载派发：调用自身 select() 选择负载最低 Agent 后组装结果。"""
        return await super().dispatch(task, agents)


class AdaptiveMLStrategy(DispatchingStrategy):
    """自适应评分派发策略。

    综合三个维度打分（无真实 ML 模型时的启发式自适应）：

    - 成功率权重 40%
    - 空闲度（1 - current_load）权重 30%
    - 响应时间归一化（1 / (1 + avg_response_time)）权重 30%
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = "adaptive_ml"

    def select(
        self,
        candidates: List[AgentMetrics],
        context: Optional[TaskContext] = None,
    ) -> Optional[AgentMetrics]:
        """按自适应评分选择综合表现最佳的 Agent。"""
        if not candidates:
            return None

        if len(candidates) == 1:
            selected = candidates[0]
        else:
            def score(agent: AgentMetrics) -> float:
                return (
                    agent.success_rate * 0.4
                    + (1.0 - agent.current_load) * 0.3
                    + (1.0 / (1.0 + agent.avg_response_time)) * 0.3
                )

            selected = max(candidates, key=score)

        self._total_selections += 1
        self._selection_distribution[selected.agent_id] = (
            self._selection_distribution.get(selected.agent_id, 0) + 1
        )
        return selected

    async def dispatch(self, task: Any, agents: Optional[List[Any]] = None) -> Dict[str, Any]:
        """自适应派发：调用自身 select() 按综合评分选择后组装结果。"""
        return await super().dispatch(task, agents)


__all__ = [
    "DispatchingStrategy",
    "AgentMetrics",
    "TaskContext",
    "WeightedRoundRobinStrategy",
    "PriorityBasedStrategy",
    "LeastLoadedStrategy",
    "AdaptiveMLStrategy",
]
