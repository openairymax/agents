"""端到端示例 — 调用 Product Manager Agent 生成 PRD

验证 airymax_agents 包 + openlab LLMAgent 执行链路完整可用，
并演示 agentrt 系统调用接入（FFI/IPC 双后端）。

运行方式::

    cd airymaxhub/ecosystem/agents
    python3 examples/run_pm.py

无需任何配置即可跑通 (Mock 模式自动启用)。
设置 OPENAI_API_KEY (及可选 OPENAI_BASE_URL) 即切真实 LLM。

agentrt 接入（可选）：
- 默认不注入 SyscallProxy（纯 Python LLM 模式，向后兼容）
- 设置环境变量 AIRY_USE_IPC=1 时注入 SyscallProxy(backend='ipc')，
  Agent execute() 会在 LLM 推理前后将上下文/结果持久化到 mem_d 守护进程
  （需先启动 daemons：/tmp/agentrt-run-build/bin/{mem_d,agent_d}）
"""

import asyncio
import os
import sys
from pathlib import Path

# 把 agents/ 自身 + ../openlab/ + ../../sdk/sdk-python/ 加到 sys.path
# agents/ 根目录使 `import airymax_agents` 可用
# ../openlab/ 使 `import openlab` 可用
# ../../sdk/sdk-python/ 使 `import agentrt` 可用
HERE = Path(__file__).resolve().parent
AGENTS_ROOT = HERE.parent
OPENLAB_ROOT = AGENTS_ROOT.parent / "openlab"
SDK_PYTHON_ROOT = AGENTS_ROOT.parent.parent / "sdk" / "sdk-python"
sys.path.insert(0, str(AGENTS_ROOT))
sys.path.insert(0, str(OPENLAB_ROOT))
sys.path.insert(0, str(SDK_PYTHON_ROOT))

from airymax_agents import get_agent, list_agents
from openlab.core.agent import AgentContext


def _maybe_build_syscall_proxy():
    """根据环境变量决定是否注入 agentrt SyscallProxy。

    - AIRY_USE_IPC=1：注入 IPC 后端（直连 mem_d / agent_d 守护进程）
    - AIRY_USE_FFI=1：注入 FFI 后端（需 libagentrt.so 可加载）
    - 默认（两者均未设）：返回 None（纯 Python LLM 模式）
    """
    use_ipc = os.environ.get("AIRY_USE_IPC", "").lower() in ("1", "true", "yes")
    use_ffi = os.environ.get("AIRY_USE_FFI", "").lower() in ("1", "true", "yes")
    if not (use_ipc or use_ffi):
        return None

    from agentrt.syscall import SyscallProxy  # noqa: WPS433 (动态导入避免硬依赖)

    if use_ipc:
        proxy = SyscallProxy(backend="ipc")
    else:
        proxy = SyscallProxy(backend="ffi")
    print(f"[syscall] backend={proxy._backend}")
    return proxy


async def main() -> None:
    print("=" * 60)
    print("Airymax Agents — 端到端验证")
    print("available roles:", list_agents())
    print("=" * 60)

    # ── 1. 创建 PM Agent（可选注入 syscall_proxy） ─────────
    syscall_proxy = _maybe_build_syscall_proxy()
    agent = get_agent("product_manager", syscall_proxy=syscall_proxy)
    await agent.initialize()
    print(f"\n[agent] id={agent.agent_id} role={agent.contract.get('role')}")
    print(f"[agent] models={agent.contract.get('models')}")
    print(f"[agent] syscall_proxy={'on' if syscall_proxy is not None else 'off'}")

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

    # ── 4. 若启用 IPC：用 SyscallProxy 检索 mem_d 中持久化的记录 ──
    # mem_d 当前为子串匹配检索（无分词/embedding），故使用精确子串
    if syscall_proxy is not None and getattr(syscall_proxy, "_backend", "") == "ipc":
        print("\n--- agentrt mem_d 检索 ---")
        try:
            hits = syscall_proxy.memory_search("待办", limit=5)
            print(f"memory_search('待办') hits: {len(hits)}")
            for rid, score in hits:
                data = syscall_proxy.memory_get(rid)
                preview = data[:100].decode("utf-8", errors="replace")
                print(f"  [{score:.3f}] {rid} -> {preview}...")
        except Exception as e:
            print(f"memory_search failed (non-fatal): {e}")

    await agent.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
