# Strategies — 调度与规划策略

**模块路径**: `ecosystem/agents/orchestration/strategies/`
**版本**: v0.1.3

> **Status**: 本模块作为 AgentRT 平台生态的组成部分，API 持续演进中。

## 概述

Strategies 是 Airymax 的调度与规划策略集合，定义 Agent 协作的模式和任务编排方式。该模块包含两大策略：Dispatching（任务调度）和 Planning（任务规划），分别解决"任务分发给谁"和"任务如何分解"两个核心问题，共同构建多智能体协作的编排基础。

## 架构定位

```
+-------------------------------------------------------------------+
|                         Airymax 开放生态                            |
+-------------------------------------------------------------------+
|                     Contributions (官方 + 社区)                       |
|  +------------------+  +------------------+  +------------------+ |
|  |     Skills       |  |    Strategies    |  |     Agents       | |
|  |  能力单元         |  |  协作模式        |  |  角色化智能体     | |
|  |                  |  |                  |  |                  | |
|  | • Browser        |  | • Dispatching    |  | • Architect      | |
|  | • Database       |  | • Planning       |  | • Backend        | |
|  | • GitHub         |  |                  |  | • Frontend       | |
|  +------------------+  +------------------+  | • ...            | |
|                                              +------------------+ |
+-------------------------------------------------------------------+
|                     orchestration Core (orchestration/core/)        |
|  Agent Registry | Task Scheduler | Tool Executor | Storage         |
+-------------------------------------------------------------------+
```

Strategies 作为 Contributions 三大子模块之一，是 Agent 协作的编排层。Dispatching 策略决定任务如何分配给合适的 Agent，Planning 策略决定复杂任务如何分解为可执行步骤，两者协同工作实现高效的多智能体编排。

## 目录结构

```
strategies/
├── __init__.py                     # 策略包导出
├── dispatching/                    # 任务调度策略
│   ├── __init__.py                 # 导出 DispatchingStrategy 及具体策略
│   ├── dispatching.py              # DispatchingStrategy 与 4 个具体策略实现
│   └── README.md                   # Dispatching 详细文档
├── planning/                       # 任务规划策略
│   ├── __init__.py                 # 导出 PlanningStrategy / Planner 系列 / DAG
│   ├── planning.py                 # 规划策略与 TaskDAG 核心实现
│   └── README.md                   # Planning 详细文档
└── README.md                       # 本文件
```

## 策略列表

| Strategy | 路径 | 核心问题 | 核心类 |
|----------|------|----------|--------|
| **Dispatching** | `dispatching/` | 任务分发给谁 | `DispatchingStrategy`、`WeightedRoundRobinStrategy`、`PriorityBasedStrategy`、`LeastLoadedStrategy`、`AdaptiveMLStrategy` |
| **Planning** | `planning/` | 任务如何分解 | `PlanningStrategy`、`HierarchicalPlanner`、`ReactivePlanner`、`ReflectivePlanner`、`TaskDAG`、`PlanStep` |

## 策略体系

### Dispatching — 任务调度策略

负责任务到 Agent 的分配决策，根据不同策略将任务分发给最合适的执行者。

#### 核心接口

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

#### 内置调度策略

| 策略类 | 说明 | 适用场景 |
|--------|------|----------|
| `WeightedRoundRobinStrategy` | 加权轮询，按权重比例依次分配 | 异构 Agent 集群 |
| `PriorityBasedStrategy` | 优先级优先，优先分配给高优先级 Agent | 分级执行体 |
| `LeastLoadedStrategy` | 最小负载，分配给当前负载最低的 Agent | 高负载场景 |
| `AdaptiveMLStrategy` | 自适应评分，综合响应时间 / 成功率等指标动态打分 | 生产环境长期运行 |

> 详细接口和数据结构（`AgentMetrics` / `TaskContext`）请参阅 `dispatching/README.md`。

### Planning — 任务规划策略

负责将复杂目标分解为可执行的步骤序列，支持目标分解、依赖分析和计划生成。

#### 核心接口

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

#### 规划流程

```
目标输入 → 目标分解 → 依赖分析 → 资源评估 → 计划生成 → 计划执行
    ↓          ↓          ↓          ↓          ↓          ↓
 用户需求   子任务列表   DAG 图    资源分配   时间线    执行引擎
```

#### 规划器与基础设施

| 类 | 说明 |
|----|------|
| `PlanningStrategy` | 启发式任务分解规划器（默认实现） |
| `HierarchicalPlanner` | 分层规划器，可接入 OpenAI 兼容 LLM 客户端，调用失败时退化为启发式分解 |
| `ReactivePlanner` | 反应式规划器，验证目标可解性 |
| `ReflectivePlanner` | 反思式规划器，基于执行反馈改进计划 |
| `TaskDAG` | 任务依赖图，支持拓扑分层（`get_execution_order`）、就绪任务获取（`get_ready_tasks`）与环检测（`validate`） |
| `TaskNode` / `PlanningContext` | DAG 节点与规划上下文数据类 |

#### PlanStep 数据结构

| 字段 | 类型 | 说明 |
|------|------|------|
| `step_id` | `str` | 步骤唯一标识 |
| `description` | `str` | 步骤描述 |
| `dependencies` | `List[str]` | 依赖步骤 ID 列表 |
| `assigned_agent` | `Optional[str]` | 分配的 Agent 标识 |

> 详细接口和配置请参阅 `planning/README.md`。

## 使用指南

### 调度与规划协同

Dispatching 和 Planning 策略通常协同使用：先通过 Planning 将复杂任务分解为步骤，再通过 Dispatching 将每个步骤分配给合适的 Agent。

```python
from orchestration.strategies.planning import PlanningStrategy
from orchestration.strategies.dispatching import DispatchingStrategy

# 1. 规划：将复杂任务分解为步骤
planner = PlanningStrategy()
plan = await planner.plan(task={
    "description": "开发一个 REST API 服务",
    "constraints": {
        "team_size": 3,
        "tech_stack": ["Python", "FastAPI"]
    }
})

# 2. 调度：将每个步骤分配给合适的 Agent
dispatcher = DispatchingStrategy()
for step in plan["steps"]:
    result = await dispatcher.dispatch(step, agents=available_agents)
    print(f"Step {step.step_id} → {result['assigned_agents']}")
```

### 单独使用 Dispatching

```python
from orchestration.strategies.dispatching import WeightedRoundRobinStrategy

dispatcher = WeightedRoundRobinStrategy()

task = {
    "id": "task-001",
    "type": "code_review",
    "priority": "high",
    "payload": {"repo": "airymax", "branch": "main"}
}

result = await dispatcher.dispatch(task, agents=available_agents)
print(f"Task dispatched to: {result['assigned_agents']}")
```

### 单独使用 Planning

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

## 开发指南

### 创建新的策略

1. 在 `orchestration/strategies/` 下创建新目录（使用小写字母）
2. 实现策略核心类，提供标准化的异步接口
3. 创建 `__init__.py` 导出策略类
4. 编写 `README.md`，包含概述、核心组件、接口说明和使用示例
5. 在 `strategies/__init__.py` 中注册新策略

### 策略接口规范

所有策略应遵循以下接口规范：

- 构造函数接受 `config: Optional[Dict[str, Any]]` 配置参数
- 核心方法为异步方法，返回 `Dict[str, Any]` 结果
- 结果中包含 `status` 和 `strategy` 字段标识执行状态和策略类型

## 与其他模块的关系

| 模块 | 关系 |
|------|------|
| **ecosystem/agents/** | Dispatching 策略将任务分配给 Agent；Planning 策略在 PlanStep 中指定 `assigned_agent` |
| **ecosystem/skills/** | Strategies 根据 Agent 拥有的 Skills 进行能力匹配调度 |
| **orchestration/core/** | Strategies 与 Core 的 TaskScheduler 协同工作，实现任务的调度和编排 |
| **Agent 应用** | 应用层使用 Strategies 编排多 Agent 协作流程 |

## 依赖关系

- **核心依赖**: Python >= 3.10, orchestration.core, typing, dataclasses
- **无外部运行时依赖**: 核心实现仅使用 Python 标准库

---

© 2025-2026 SPHARX Ltd. All Rights Reserved.
