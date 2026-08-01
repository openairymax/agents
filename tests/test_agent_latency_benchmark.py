#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025-2026 SPHARX Ltd.
# SPDX-License-Identifier: AGPL-3.0-or-later OR Apache-2.0

# -*- coding: utf-8 -*-
"""Python vs Rust Agent 延迟对比基准测试。

通过 agent_d 的 Unix socket（JSON-RPC 2.0）分别 spawn / invoke / terminate
``coding`` 角色的 Python 与 Rust Agent，测量端到端延迟：

  - spawn   : 从发请求到拿到 agent_id（含 fork + exec + runner 初始化）
  - invoke  : 从发请求到拿到 LLM 输出（含 mock LLM 50ms + 框架开销）
  - total   : spawn + invoke + terminate 一轮总耗时

对比前提（公平性）：
  - 同一角色 ``coding``、同一契约/提示词风格（Python 与 Rust 侧对称）
  - 双端均使用 MockLLMClient（无 OPENAI_API_KEY 时），LLM 侧开销一致
  - 差异即框架开销：进程启动、解释器/运行时、JSON 序列化、协议往返

前置条件（环境）：
  - agent_d / mem_d 守护进程运行于 ${AIRY_RUNTIME_DIR:-/tmp/agentrt}
  - agent_d 启动时注入 PYTHONPATH 指向 ``ecosystem/agents`` 与
    ``ecosystem/openlab``（Python runner 子进程 import 依赖）
  - Rust binary 已构建于 ``/tmp/agentrt-rs-build/release/coding_agent``
    （或通过环境变量 ``AIRY_RUST_AGENT_BIN`` 覆盖）

运行方式::

    pytest tests/test_agent_latency_benchmark.py -s -v          # 默认 8 轮
    AIRY_BENCH_ROUNDS=30 pytest tests/test_agent_latency_benchmark.py -s
"""

import json
import os
import socket
import statistics
import time

import pytest

_RUNTIME_DIR = os.environ.get("AIRY_RUNTIME_DIR", "/tmp/agentrt")
_AGENT_SOCKET = os.environ.get("AIRY_AGENT_D_SOCKET", f"{_RUNTIME_DIR}/agent.sock")
_RUST_BIN = os.environ.get(
    "AIRY_RUST_AGENT_BIN", "/tmp/agentrt-rs-build/release/coding_agent"
)
_BENCH_ROUNDS = int(os.environ.get("AIRY_BENCH_ROUNDS", "8"))
_IPC_TIMEOUT_S = 60.0
# 双端 mock LLM 共享 50ms 延迟；断言用宽松上界避免 CI 抖动
_RUST_INVOKE_MAX_MULTIPLIER = 2.0
_RUST_INVOKE_MAX_SLACK_MS = 150.0


def _agent_d_available() -> bool:
    return os.path.exists(_AGENT_SOCKET)


def _rpc(method: str, params: dict) -> dict:
    """向 agent_d 发起一次 JSON-RPC 调用，返回 result dict。

    对偶发 BrokenPipe / ConnectionReset 做指数退避重试（与 sdk
    ``_IpcBackend._call`` 一致），规避 daemon EPOLLET + MSG_DONTWAIT
    在 accept 后 recv 暂未读到数据的时序窗口。
    """
    req = json.dumps(
        {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}
    ).encode("utf-8")
    last_exc: Exception | None = None
    for attempt in range(4):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(_IPC_TIMEOUT_S)
        try:
            sock.connect(_AGENT_SOCKET)
            sock.sendall(req)
            buf = b""
            while True:
                chunk = sock.recv(8192)
                if not chunk:
                    break
                buf += chunk
                try:
                    return json.loads(buf.decode("utf-8")).get("result", {})
                except json.JSONDecodeError:
                    continue
        except (BrokenPipeError, ConnectionResetError, socket.timeout) as e:
            last_exc = e
            time.sleep(0.05 * (attempt + 1))
        finally:
            sock.close()
    raise RuntimeError(f"IPC {method} failed after retries: {last_exc}")


def _spawn(spec: dict) -> str:
    """spawn 一个 agent，返回 agent_id。"""
    result = _rpc("spawn", {"agent_spec": json.dumps(spec)})
    agent_id = result.get("agent_id")
    if not isinstance(agent_id, str):
        raise RuntimeError(f"spawn failed: {result!r}")
    return agent_id


def _invoke(agent_id: str, user_input: str) -> str:
    """invoke 一个 agent，返回输出文本。"""
    result = _rpc("invoke", {"agent_id": agent_id, "input": user_input})
    output = result.get("output")
    if not isinstance(output, str):
        raise RuntimeError(f"invoke failed: {result!r}")
    return output


def _terminate(agent_id: str) -> None:
    _rpc("terminate", {"agent_id": agent_id})


def _spec_for(language: str) -> dict:
    spec = {"language": language, "role": "coding"}
    if language == "rust":
        spec["binary_path"] = _RUST_BIN
    return spec


def _bench_one(language: str, rounds: int) -> dict:
    """对单个语言跑 N 轮 spawn+invoke+terminate，返回统计 dict。"""
    spawns: list[float] = []
    invokes: list[float] = []
    totals: list[float] = []
    outputs: list[str] = []

    # warmup：加载缓存（Python bytecode、Rust 动态库、tokio runtime 等）
    warm = _spawn(_spec_for(language))
    _invoke(warm, "warmup")
    _terminate(warm)

    for _ in range(rounds):
        t0 = time.perf_counter()
        agent_id = _spawn(_spec_for(language))
        t1 = time.perf_counter()
        output = _invoke(agent_id, "写一个斐波那契函数")
        t2 = time.perf_counter()
        _terminate(agent_id)
        t3 = time.perf_counter()

        spawns.append((t1 - t0) * 1000.0)
        invokes.append((t2 - t1) * 1000.0)
        totals.append((t3 - t0) * 1000.0)
        outputs.append(output)

    return {
        "spawn": _summarize(spawns),
        "invoke": _summarize(invokes),
        "total": _summarize(totals),
        "outputs": outputs,
    }


def _summarize(samples: list[float]) -> dict:
    return {
        "min": min(samples),
        "median": statistics.median(samples),
        "mean": statistics.mean(samples),
        "p95": sorted(samples)[max(0, int(len(samples) * 0.95) - 1)],
        "max": max(samples),
    }


def _fmt(s: dict) -> str:
    return (
        f"min={s['min']:.1f}ms median={s['median']:.1f}ms "
        f"mean={s['mean']:.1f}ms p95={s['p95']:.1f}ms max={s['max']:.1f}ms"
    )


pytestmark = pytest.mark.skipif(
    not _agent_d_available(),
    reason=f"agent_d 不可达（{_AGENT_SOCKET} 不存在）— 请先启动 daemon",
)


def test_python_vs_rust_agent_latency(capsys):
    """对比 Python 与 Rust coding_agent 的 spawn/invoke 延迟并打印报告。"""
    py = _bench_one("python", _BENCH_ROUNDS)
    rs = _bench_one("rust", _BENCH_ROUNDS)

    # 双端必须走真实子进程路径（LLM mock 输出），而非 daemon stub 回退
    for lang, bench in (("python", py), ("rust", rs)):
        for out in bench["outputs"]:
            assert "Mock" in out, (
                f"{lang} invoke 未走真实子进程路径（疑似 stub 回退）: {out[:80]!r}"
            )

    # 断言：Rust invoke 不得显著慢于 Python（框架开销可控）
    rust_invoke = rs["invoke"]["median"]
    py_invoke = py["invoke"]["median"]
    upper = py_invoke * _RUST_INVOKE_MAX_MULTIPLIER + _RUST_INVOKE_MAX_SLACK_MS
    assert rust_invoke <= upper, (
        f"Rust invoke median={rust_invoke:.1f}ms 超出上界 {upper:.1f}ms"
        f"（Python median={py_invoke:.1f}ms）"
    )

    # 报告
    rows = [
        ("python", py["spawn"], py["invoke"], py["total"]),
        ("rust", rs["spawn"], rs["invoke"], rs["total"]),
    ]
    report = "\n" + "=" * 78 + "\n"
    report += "Agent 延迟对比基准（coding 角色, MockLLM, 50ms 模拟延迟）\n"
    report += f"轮次: {_BENCH_ROUNDS}  运行目录: {_RUNTIME_DIR}\n"
    report += "-" * 78 + "\n"
    for name, spawn, invoke, total in rows:
        report += f"[{name}]\n  spawn : {_fmt(spawn)}\n"
        report += f"  invoke: {_fmt(invoke)}\n"
        report += f"  total : {_fmt(total)}\n"
    report += "-" * 78 + "\n"
    report += (
        f"invoke 差异: Rust median={rs['invoke']['median']:.1f}ms vs "
        f"Python median={py_invoke:.1f}ms\n"
    )
    report += "=" * 78 + "\n"
    with capsys.disabled():
        print(report)
