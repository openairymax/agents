"""端到端示例 — 调用 Product Manager Agent 生成 PRD

验证 airymax_agents 包 + openlab LLMAgent 执行链路完整可用。

运行方式::

    cd airymaxhub/ecosystem/agents
    python3 examples/run_pm.py

无需任何配置即可跑通 (Mock 模式自动启用)。
设置 OPENAI_API_KEY (及可选 OPENAI_BASE_URL) 即切真实 LLM。
"""

import asyncio
import sys
from pathlib import Path

# 把 agents/ 自身 + ../openlab/ 加到 sys.path
# agents/ 根目录使 `import airymax_agents` 可用
# ../openlab/ 使 `import openlab` 可用
HERE = Path(__file__).resolve().parent
AGENTS_ROOT = HERE.parent
OPENLAB_ROOT = AGENTS_ROOT.parent / "openlab"
sys.path.insert(0, str(AGENTS_ROOT))
sys.path.insert(0, str(OPENLAB_ROOT))

from airymax_agents import get_agent, list_agents
from openlab.core.agent import AgentContext


async def main() -> None:
    print("=" * 60)
    print("Airymax Agents — 端到端验证")
    print("available roles:", list_agents())
    print("=" * 60)

    # ── 1. 创建 PM Agent ─────────────────────────────────
    agent = get_agent("product_manager")
    await agent.initialize()
    print(f"\n[agent] id={agent.agent_id} role={agent.contract.get('role')}")
    print(f"[agent] models={agent.contract.get('models')}")

    # ── 2. 分配任务 ─────────────────────────────────────
    task = "为待办事项应用写一份简短 PRD，含核心功能、非功能需求、验收标准。"
    ctx = AgentContext(agent_id=agent.agent_id, task_id="demo-001")
    print(f"\n[task] {task}")

    # ── 3. 执行并打印结果 ───────────────────────────────
    result = await agent.execute(task, ctx)

    print("\n--- TaskResult ---")
    print("success:", result.success)
    if result.error:
        print("error:", result.error)
    print("\n[output]")
    print(result.output)
    print("\n[metrics]")
    for k, v in (result.metrics or {}).items():
        print(f"  {k}: {v}")

    await agent.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
