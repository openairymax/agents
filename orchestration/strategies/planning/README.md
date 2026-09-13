# Planning — 任务规划策略

**模块路径**: `ecosystem/agents/orchestration/strategies/planning/`
**版本**: v0.1.3

> **Status**: 本模块作为 AgentRT 平台生态的组成部分，API 持续演进中。

## 概述

Planning 策略模块提供智能体的任务规划能力，负责将复杂目标分解为可执行的步骤序列。模块提供启发式 `PlanningStrategy`（默认实现）与 3 个进阶规划器，并内置 `TaskDAG` 任务依赖图基础设施（拓扑排序 / 就绪任务 / 环检测）。

## 目录结构

```
planning/
├── __init__.py                 # 模块导出
├── planning.py                 # 规划策略与 TaskDAG 核心实现
└── README.md                   # 本文件
```

## 核心组件

### 类一览 (`planning.py`)

| 类 | 说明 |
|----|------|
| `PlanningStrategy` | 启发式任务分解规划器（默认实现），提供 `plan()` 方法 |
| `HierarchicalPlanner` | 分层规划器，可接入 OpenAI 兼容 LLM 客户端，调用失败时退化为启发式分解 |
| `ReactivePlanner` | 反应式规划器，验证目标可解性 |
| `ReflectivePlanner` | 反思式规划器，基于执行反馈改进计划 |
| `TaskDAG` | 任务依赖图：`get_execution_order()` 拓扑分层 / `get_ready_tasks()` 就绪任务 / `validate()` 环检测 |
| `TaskNode` | DAG 节点数据类 |
| `PlanningContext` | 规划上下文数据类 |
| `PlanStep` | 计划步骤数据类：step_id / description / dependencies / assigned_agent |

## 规划流程

```
目标输入 → 目标分解 → 依赖分析 → 资源评估 → 计划生成 → 计划执行
    ↓          ↓          ↓          ↓          ↓          ↓
 用户需求   子任务列表   DAG 图    资源分配   时间线    执行引擎
```

## 接口说明

```python
@dataclass
class PlanStep:
    step_id: str = ""
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    assigned_agent: Optional[str] = None

class PlanningStrategy:
    def __init__(self, config: Optional[Dict[str, Any]] = None)

    async def plan(self, task: Any,
                   context: Optional[PlanningContext] = None) -> Dict[str, Any]:
        """将任务分解为执行计划

        Args:
            task: 任务对象（dict 或其他类型）
            context: 可选规划上下文

        Returns:
            Dict: 包含 status / strategy / task / dependencies / steps 字段
        """
```

## 依赖关系

- **核心依赖**: AgentRT orchestration Core
- **Python**: >= 3.10, typing, dataclasses

## 使用示例

```python
from orchestration.strategies.planning import PlanningStrategy, PlanStep

planner = PlanningStrategy()

plan = await planner.plan(
    task={
        "description": "开发一个 REST API 服务",
        "constraints": {
            "deadline": "2024-02-01",
            "team_size": 3,
            "tech_stack": ["Python", "FastAPI"]
        }
    }
)

for step in plan["steps"]:
    print(f"{step.step_id}: {step.description}")
```

---

© 2025-2026 SPHARX Ltd. All Rights Reserved.
