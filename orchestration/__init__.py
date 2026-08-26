"""
Copyright (c) 2026 SPHARX. All Rights Reserved.
"From data intelligence emerges."

orchestration - Agent Operating System
=================================

orchestration is the multi-agent orchestration kernel of Airymax Agents.

Core Modules:
    - core: Agent, Task, Tool, Storage 与 LLM 客户端管理

Example:
    import asyncio
    from orchestration import AgentRegistry, TaskScheduler, make_llm_client

    async def main():
        registry = AgentRegistry()
        scheduler = TaskScheduler()
        llm = make_llm_client()
        print(registry, scheduler, llm)
        await asyncio.sleep(0)

    asyncio.run(main())

Author: SPHARX Ltd. - Airymax Team
Version: 0.1.3
"""

from .core import (
    Agent,
    AgentCapability,
    AgentContext,
    AgentRegistry,
    AgentStatus,
    TaskResult,
    Message,
    TaskStatus,
    TaskCategory,
    TaskDefinition,
    TaskState,
    ExecutionPlan,
    TaskScheduler,
    Tool,
    ToolCapability,
    ToolCategory,
    ToolContext,
    ToolResult,
    ToolRegistry,
    ToolExecutor,
    Storage,
    StorageType,
    DataCategory,
    StorageRecord,
    QueryResult,
    MemoryStorage,
    SQLiteStorage,
    LLMClient,
    MockLLMClient,
    make_llm_client,
)

# NOTE: 历史 ArchitectAgent（orchestration/agents/architect/）已删除
# （签名不兼容、被 try/except 包裹），按 RESTRUCTURE_PLAN.md 阶段 1 移除。
# 现可用 ArchitectAgent 在 airymax_agents.architect.agent.ArchitectAgent。

__version__ = "0.1.3"

__all__ = [
    # Version
    "__version__",
    # Core - Agent
    "Agent",
    "AgentCapability",
    "AgentContext",
    "AgentRegistry",
    "AgentStatus",
    "TaskResult",
    "Message",
    # Core - Task
    "TaskStatus",
    "TaskCategory",
    "TaskDefinition",
    "TaskState",
    "ExecutionPlan",
    "TaskScheduler",
    # Core - Tool
    "Tool",
    "ToolCapability",
    "ToolCategory",
    "ToolContext",
    "ToolResult",
    "ToolRegistry",
    "ToolExecutor",
    # Core - Storage
    "Storage",
    "StorageType",
    "DataCategory",
    "StorageRecord",
    "QueryResult",
    "MemoryStorage",
    "SQLiteStorage",
    # Core - LLM
    "LLMClient",
    "MockLLMClient",
    "make_llm_client",
]
