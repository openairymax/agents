# Dispatching — 任务调度策略

**模块路径**: `ecosystem/agents/orchestration/strategies/dispatching/`
**版本**: v0.1.3

> **Status**: 本模块作为 AgentRT 平台生态的组成部分，API 持续演进中。

## 概述

Dispatching 策略模块提供智能体的任务调度能力，负责将任务高效地分发给最合适的执行者。模块提供 `DispatchingStrategy` 基类与 4 个内置具体策略，均基于 `AgentMetrics` 指标进行选择决策。

## 目录结构

```
dispatching/
├── __init__.py                 # 模块导出
├── dispatching.py              # DispatchingStrategy 与 4 个具体策略实现
└── README.md                   # 本文件
```

## 核心组件

### 类一览 (`dispatching.py`)

| 类 | 说明 |
|----|------|
| `DispatchingStrategy` | 调度策略基类，提供 `select()` / `dispatch()` / `get_stats()` 标准接口 |
| `WeightedRoundRobinStrategy` | 加权轮询策略，按权重比例依次分配 |
| `PriorityBasedStrategy` | 优先级优先策略，优先分配给高优先级 Agent |
| `LeastLoadedStrategy` | 最小负载策略，分配给当前负载最低的 Agent |
| `AdaptiveMLStrategy` | 自适应评分策略，综合响应时间 / 成功率等指标动态打分 |
| `AgentMetrics` | Agent 运行指标数据类：agent_id / weight / current_load / avg_response_time / success_rate / priority / capabilities |
| `TaskContext` | 任务上下文数据类，供选择决策使用 |

## 接口说明

```python
class DispatchingStrategy:
    name: str = "dispatching"

    def __init__(self, config: Optional[Dict[str, Any]] = None)

    def select(self, candidates: List[AgentMetrics],
               context: Optional[TaskContext] = None) -> Optional[AgentMetrics]:
        """从候选 Agent 指标中选择最合适的一个"""

    def get_stats(self) -> Dict[str, Any]:
        """返回策略统计信息"""

    async def dispatch(self, task: Any, agents: Optional[List[Any]] = None) -> Dict[str, Any]:
        """分发任务到合适的 Agent

        Args:
            task: 任务对象
            agents: 可用 Agent 列表

        Returns:
            Dict: 包含 status / strategy / task_id / task_type /
                  assigned_agents / candidates / dispatch_count /
                  selected / reason 字段
        """
```

## 内置策略

| 策略类 | 说明 | 适用场景 |
|--------|------|----------|
| `WeightedRoundRobinStrategy` | 加权轮询，按权重比例依次分配 | 异构 Agent 集群 |
| `PriorityBasedStrategy` | 优先级优先，优先分配给高优先级 Agent | 分级执行体 |
| `LeastLoadedStrategy` | 最小负载，分配给当前负载最低的 Agent | 高负载场景 |
| `AdaptiveMLStrategy` | 自适应评分，综合多项指标动态决策 | 生产环境长期运行 |

## 依赖关系

- **核心依赖**: AgentRT orchestration Core
- **Python**: >= 3.10, typing

## 使用示例

```python
from orchestration.strategies.dispatching import LeastLoadedStrategy

dispatcher = LeastLoadedStrategy()

task = {
    "id": "task-001",
    "type": "code_review",
    "priority": "high",
    "payload": {"repo": "airymax", "branch": "main"}
}

result = await dispatcher.dispatch(task, agents=available_agents)
print(f"Task dispatched to: {result['assigned_agents']}")
```

---

© 2025-2026 SPHARX Ltd. All Rights Reserved.
