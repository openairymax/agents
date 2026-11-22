"""子代理编排：父任务派生隔离子进程（0.1.21 S-P2-01 / §25.6）。

子代理以独立 Python 进程运行本仓 runner，进程边界同时给出三重隔离：

  - 故障隔离：子进程崩溃/退出只影响本次派生，父进程继续汇总；
  - 资源隔离：独立 workspace（``<parent_ws>/.subagents/<child_id>``），
    独立会话标识即独立记忆命名空间；默认只读，写能力需父级显式授权；
  - 取消级联：父任务终态、父进程退出（SIGTERM/atexit）时终止存活子进程。

父代理 LLM 可见三个工具：``agent_spawn`` / ``agent_wait`` / ``agent_cancel``。
预算与上限全部来自环境变量，缺省取保守值；超限一律显式拒绝，不静默
串行化、不丢任务。
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

ENV_DEPTH = "AIRY_SUBAGENT_DEPTH"
ENV_MAX_DEPTH = "AIRY_SUBAGENT_MAX_DEPTH"
ENV_MAX_CONCURRENCY = "AIRY_SUBAGENT_MAX_CONCURRENCY"
ENV_MAX_TOTAL = "AIRY_SUBAGENT_MAX_TOTAL"
ENV_BUDGET_S = "AIRY_SUBAGENT_BUDGET_S"
ENV_WAIT_SLICE_S = "AIRY_SUBAGENT_WAIT_SLICE_S"
ENV_START_S = "AIRY_SUBAGENT_START_S"

DEFAULT_MAX_DEPTH = 1
DEFAULT_MAX_CONCURRENCY = 4
DEFAULT_MAX_TOTAL = 16
DEFAULT_BUDGET_S = 300.0
DEFAULT_WAIT_SLICE_S = 45.0
DEFAULT_START_S = 20.0
STDERR_KEEP_LINES = 40

SPAWN_SCHEMA = {
    "description": (
        "派生一个隔离子代理执行独立子任务，立即返回 child_id。"
        "子代理在独立 workspace 中运行，默认只读；仅当父代理本身具备"
        "写工具且显式传 writable=true 时子代理才可写。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "role": {"type": "string", "description": "子代理角色（如 coding/tester/reviewer）"},
            "input": {"type": "string", "description": "交给子代理的完整任务描述"},
            "writable": {"type": "boolean", "description": "是否授权子代理写文件（默认 false）"},
        },
        "required": ["role", "input"],
    },
}

WAIT_SCHEMA = {
    "description": (
        "等待子代理终态，最多阻塞一个切片；返回 status=running 时表示"
        "子任务仍在执行，可稍后再次调用。终态含 output/metrics/error。"
    ),
    "parameters": {
        "type": "object",
        "properties": {"child_id": {"type": "string", "description": "agent_spawn 返回的标识"}},
        "required": ["child_id"],
    },
}

CANCEL_SCHEMA = {
    "description": "取消一个仍在执行的子代理（SIGTERM 后必要时强制终止）。",
    "parameters": {
        "type": "object",
        "properties": {"child_id": {"type": "string", "description": "agent_spawn 返回的标识"}},
        "required": ["child_id"],
    },
}


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value >= 0 else default


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


def _agents_root() -> str:
    """返回 ``agents`` 包根目录（子进程 PYTHONPATH 需要）。"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SubagentManager:
    """一次父任务内的子代理生命周期（spawn/wait/cancel/shutdown）。"""

    def __init__(self, *, parent_agent_id: str, parent_task_id: str,
                 role_resolver: Callable[[str], Optional[str]],
                 parent_session_id: Optional[str] = None,
                 workspace_dir: Optional[str] = None,
                 readonly: bool = True, depth: int = 0,
                 child_argv: Optional[List[str]] = None) -> None:
        self._parent_agent_id = parent_agent_id
        self._parent_task_id = parent_task_id
        self._parent_session_id = parent_session_id
        self._workspace_dir = workspace_dir
        self._readonly = readonly
        self.depth = depth
        self._role_resolver = role_resolver
        self._child_argv = child_argv

        self._max_depth = _env_int(ENV_MAX_DEPTH, DEFAULT_MAX_DEPTH)
        self._max_concurrency = _env_int(ENV_MAX_CONCURRENCY, DEFAULT_MAX_CONCURRENCY)
        self._max_total = _env_int(ENV_MAX_TOTAL, DEFAULT_MAX_TOTAL)
        self._budget_s = _env_float(ENV_BUDGET_S, DEFAULT_BUDGET_S)
        self._wait_slice_s = _env_float(ENV_WAIT_SLICE_S, DEFAULT_WAIT_SLICE_S)
        self._start_s = _env_float(ENV_START_S, DEFAULT_START_S)

        self._children: Dict[str, dict] = {}
        self._total = 0
        self._lock = threading.Lock()
        self._closed = False
        atexit.register(self.shutdown)

    # ---- 工具入口 -----------------------------------------------------

    def spawn(self, params: dict) -> dict:
        role = str(params.get("role") or "").strip()
        task = params.get("input")
        if not role or task is None or not str(task).strip():
            return self._reject("role and input are required")
        canon = self._role_resolver(role)
        if canon is None:
            return self._reject(f"unknown agent role {role!r}")
        if self.depth + 1 > self._max_depth:
            return self._reject(
                f"subagent depth limit reached (depth={self.depth}, max={self._max_depth})"
            )

        writable = bool(params.get("writable")) and not self._readonly
        child_readonly = not writable
        child_id = f"sub_{uuid.uuid4().hex[:12]}"

        with self._lock:
            if self._closed:
                return self._reject("subagent manager is shut down")
            if self._total >= self._max_total:
                return self._reject(f"subagent total limit reached ({self._max_total})")
            live = sum(1 for c in self._children.values() if c["result"] is None)
            if live >= self._max_concurrency:
                return self._reject(
                    f"subagent concurrency limit reached ({self._max_concurrency})"
                )
            self._total += 1
            record = {
                "id": child_id,
                "role": canon,
                "input": str(task),
                "readonly": child_readonly,
                "started": time.monotonic(),
                "proc": None,
                "queue": None,
                "stderr": [],
                "result": None,
                "workspace": "",
            }
            self._children[child_id] = record

        started = self._start(record)
        if started is not None:
            return started
        return {
            "child_id": child_id,
            "status": "running",
            "role": canon,
            "readonly": child_readonly,
            "workspace": record["workspace"],
        }

    def wait(self, params: dict) -> dict:
        child_id = str(params.get("child_id") or "")
        record = self._children.get(child_id)
        if record is None:
            return self._reject(f"unknown child_id {child_id!r}")
        if record["result"] is not None:
            return record["result"]

        elapsed = time.monotonic() - record["started"]
        remaining = self._budget_s - elapsed
        if remaining <= 0:
            return self._finish(record, "timeout", error="subagent budget exhausted")
        slice_s = min(self._wait_slice_s, remaining)
        try:
            line = record["queue"].get(timeout=slice_s)
        except queue.Empty:
            return {
                "child_id": child_id,
                "status": "running",
                "elapsed_ms": int((time.monotonic() - record["started"]) * 1000),
                "remaining_ms": int(max(remaining - slice_s, 0.0) * 1000),
            }
        if line is None:
            return self._finish(
                record, "failed",
                error=self._exit_error(record, "subagent exited without a response"),
            )
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return self._finish(record, "failed", error="subagent sent malformed JSON")
        if payload.get("success"):
            return self._finish(record, "completed", output=str(payload.get("output", "")),
                                metrics=payload.get("metrics"))
        return self._finish(record, "failed", error=str(payload.get("error", "subagent failed")))

    def cancel(self, params: dict) -> dict:
        child_id = str(params.get("child_id") or "")
        record = self._children.get(child_id)
        if record is None:
            return self._reject(f"unknown child_id {child_id!r}")
        if record["result"] is not None:
            return record["result"]
        return self._finish(record, "cancelled", error="cancelled by parent")

    def shutdown(self) -> None:
        """终止所有存活子进程（父终态、SIGTERM、atexit 共用，幂等）。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            records = [c for c in self._children.values() if c["result"] is None]
        for record in records:
            self._finish(record, "cancelled", error="parent task ended")
        try:
            atexit.unregister(self.shutdown)
        except Exception:  # noqa: BLE001 - 注销失败不影响已终止的子进程
            pass

    # ---- 进程与协议 ---------------------------------------------------

    def _child_workspace(self, child_id: str) -> str:
        if self._workspace_dir:
            root = os.path.join(self._workspace_dir, ".subagents")
        else:
            root = os.path.join(tempfile.gettempdir(), "airy-subagents")
        path = os.path.join(root, child_id)
        os.makedirs(path, exist_ok=True)
        return path

    def _start(self, record: dict) -> Optional[dict]:
        child_id = record["id"]
        workspace = self._child_workspace(child_id)
        record["workspace"] = workspace
        spec = {"role": record["role"], "readonly": record["readonly"]}
        request = {
            "agent_id": f"{self._parent_agent_id}.sub",
            "input": record["input"],
            "session_id": f"{self._parent_session_id or self._parent_task_id}:{child_id}",
            "parent_task_id": self._parent_task_id,
            "workspace_dir": workspace,
        }
        env = os.environ.copy()
        env[ENV_DEPTH] = str(self.depth + 1)
        env["AIRY_SUBAGENT_PARENT_TASK"] = self._parent_task_id
        python_path = _agents_root()
        if env.get("PYTHONPATH"):
            python_path = python_path + os.pathsep + env["PYTHONPATH"]
        env["PYTHONPATH"] = python_path

        argv = self._child_argv or [
            sys.executable, "-m", "airymax_agents.runner",
            "--spec", json.dumps(spec, ensure_ascii=False),
        ]
        try:
            proc = subprocess.Popen(
                argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1,
                cwd=workspace, env=env,
            )
        except OSError as exc:
            return self._finish(record, "failed", error=f"spawn failed: {exc}")

        line_queue: "queue.Queue[Optional[str]]" = queue.Queue()
        record["proc"] = proc
        record["queue"] = line_queue
        threading.Thread(target=self._pump_stdout, args=(proc, line_queue), daemon=True).start()
        threading.Thread(target=self._pump_stderr, args=(proc, record["stderr"]), daemon=True).start()

        try:
            ready_line = line_queue.get(timeout=self._start_s)
        except queue.Empty:
            return self._finish(record, "failed", error="subagent did not become ready")
        if ready_line is None:
            return self._finish(
                record, "failed", error=self._exit_error(record, "subagent exited during startup")
            )
        try:
            ready = json.loads(ready_line)
        except json.JSONDecodeError:
            return self._finish(record, "failed", error="subagent ready frame is not JSON")
        if not ready.get("ready"):
            return self._finish(
                record, "failed",
                error=f"subagent not ready: {str(ready.get('reason') or ready)[:200]}",
            )

        try:
            proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            proc.stdin.flush()
        except (OSError, ValueError) as exc:
            return self._finish(record, "failed", error=f"subagent request write failed: {exc}")
        return None

    @staticmethod
    def _pump_stdout(proc: subprocess.Popen, sink: "queue.Queue[Optional[str]]") -> None:
        try:
            for line in proc.stdout:
                sink.put(line)
        except (OSError, ValueError):
            pass
        finally:
            sink.put(None)

    @staticmethod
    def _pump_stderr(proc: subprocess.Popen, sink: List[str]) -> None:
        try:
            for line in proc.stderr:
                sink.append(line.rstrip("\n"))
                if len(sink) > STDERR_KEEP_LINES:
                    del sink[0]
        except (OSError, ValueError):
            pass

    def _exit_error(self, record: dict, prefix: str) -> str:
        proc = record["proc"]
        code = proc.poll() if proc is not None else None
        tail = " | ".join(record["stderr"][-5:])
        message = f"{prefix} (exit_code={code})"
        return f"{message}: {tail}" if tail else message

    def _finish(self, record: dict, status: str, output: str = "",
                error: str = "", metrics: Optional[dict] = None) -> dict:
        """固化终态并回收子进程（只生效一次）。

        终态先在锁内落定再回收进程：回收最多等待数秒，不能持有锁，
        否则并发 spawn/wait 会被单个慢子进程阻塞。
        """
        with self._lock:
            if record["result"] is not None:
                return record["result"]
            elapsed_ms = int((time.monotonic() - record["started"]) * 1000)
            metrics = metrics if isinstance(metrics, dict) else {}
            record["result"] = {
                "child_id": record["id"],
                "status": status,
                "role": record["role"],
                "output": output if status == "completed" else "",
                "error": error,
                "elapsed_ms": elapsed_ms,
                "tokens": metrics.get("tokens") or 0,
                "rounds": metrics.get("rounds") or 0,
                "model": metrics.get("model", ""),
            }
            result = record["result"]
        self._stop(record.get("proc"))
        logger.info(
            "subagent %s %s (role=%s, %d ms, tokens=%s)",
            record["id"], status, record["role"], elapsed_ms, result["tokens"],
        )
        return result

    @staticmethod
    def _stop(proc: Optional[subprocess.Popen]) -> None:
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            proc.wait(timeout=2)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            try:
                proc.kill()
                proc.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                pass

    @staticmethod
    def _reject(error: str) -> dict:
        return {"child_id": None, "status": "rejected", "error": error}
