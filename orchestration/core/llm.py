"""orchestration Core — LLM 客户端基础设施

OpenAI Chat Completions 兼容协议 (https://platform.openai.com/docs/api-reference/chat)。

与 ``core/agent.py`` / ``core/tool.py`` 并列，作为 Agent 执行体的 LLM 调用基础设施。

- :class:`LLMClient`     — 真实异步客户端 (aiohttp)，支持 function calling
- :class:`MockLLMClient`  — 离线确定性 mock，无需 API key 即可跑通
- :func:`make_llm_client` — 环境感知工厂

统一客户端设计（对齐 LiteLLM / Claude Code / AtomCode 的 OpenAI 兼容模式）：
- 单一协议：所有厂商一律走 OpenAI Chat Completions 格式
- 三要素配置：base_url + api_key + model
- 厂商即配置：新增厂商只需在 ``OPENAI_COMPAT_PROVIDERS`` 加一行
  （默认端点 + key 环境变量），无需改协议代码
- 用户侧配置：``$AIRY_LLM_PROVIDER`` 选厂商（deepseek/glm/qwen/moonshot/...）；
  ``$AIRY_LLM_BASE_URL`` 可完全自定义任意 OpenAI 兼容端点；
  key 统一写在 ``$AIRY_HOME/config/secrets.env``
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import uuid
from typing import Any, Dict, List, Optional, Tuple

# OpenAI 兼容厂商目录：名称 → (默认端点, key 环境变量)
# 覆盖国内外主流厂商：DeepSeek / 智谱 GLM / 通义千问 Qwen / Moonshot Kimi /
# 硅基流动（聚合开源模型）/ 讯飞星火 / MiniMax / 本地 Ollama / OpenAI。
# 自定义厂商：填入未知名称 + 设置 $AIRY_LLM_BASE_URL 即可（走 OpenAI 兼容协议）。
OPENAI_COMPAT_PROVIDERS: Dict[str, Tuple[str, str]] = {
    "openai":      ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "deepseek":    ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "glm":         ("https://open.bigmodel.cn/api/paas/v4", "GLM_API_KEY"),
    "qwen":        ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "moonshot":    ("https://api.moonshot.cn/v1", "MOONSHOT_API_KEY"),
    "siliconflow": ("https://api.siliconflow.cn/v1", "SILICONFLOW_API_KEY"),
    "spark":       ("https://spark-api-open.xf-yun.com/v1", "SPARK_API_KEY"),
    "minimax":     ("https://api.minimax.chat/v1", "MINIMAX_API_KEY"),
    "ollama":      ("http://localhost:11434/v1", ""),
}

# 自动探测时的厂商优先级（DeepSeek 居首，与 model.yaml global.default_provider 对齐）
PROVIDER_PROBE_ORDER: Tuple[str, ...] = (
    "deepseek", "glm", "qwen", "moonshot", "siliconflow", "spark", "minimax", "openai",
)


def get_supported_providers() -> List[str]:
    """返回支持的厂商清单（含自定义未知名称则列出）。"""
    return sorted(OPENAI_COMPAT_PROVIDERS.keys())


def provider_key_env(provider: str) -> str:
    """厂商对应的 key 环境变量名；未知厂商返回空串。"""
    info = OPENAI_COMPAT_PROVIDERS.get(provider)
    return info[1] if info else ""


def provider_default_base_url(provider: str) -> str:
    """厂商默认端点；未知厂商返回空串（此时须用 $AIRY_LLM_BASE_URL）。"""
    info = OPENAI_COMPAT_PROVIDERS.get(provider)
    return info[0] if info else ""


def _model_yaml_candidates() -> List[str]:
    """model.yaml 候选路径（与 _provider_default_model 同源，SSoT）。"""
    here = os.path.dirname(os.path.abspath(__file__))  # .../orchestration/orchestration/core
    candidates = [
        os.path.join(here, "..", "..", "..", "manager", "model", "model.yaml"),
    ]
    env_cfg = os.environ.get("AIRY_MODEL_CONFIG", "").strip()
    if env_cfg:
        candidates.append(env_cfg)
    root = os.environ.get("AIRYMAXHUB_ROOT", "").strip()
    if root:
        candidates.append(os.path.join(root, "ecosystem", "manager", "model", "model.yaml"))
    candidates.append("ecosystem/manager/model/model.yaml")
    candidates.append("model.yaml")
    return candidates


def _load_models_table() -> List[Dict[str, str]]:
    """读取 model.yaml v2 模型连接表（最多 3 行，表格格式）。

    轻量文本解析（无 YAML 依赖）：匹配顶层 ``models:`` 下每个 ``- name:``
    条目，提取 name / model_id / api_key_env / base_url。解析失败或文件
    缺失返回空表；调用方按需回退到旧 provider 目录。
    """
    entries: List[Dict[str, str]] = []
    for path in _model_yaml_candidates():
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError:
            continue
        cur: Dict[str, str] = {}
        in_models = False
        for raw in content.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("models:"):
                in_models = True
                continue
            if not in_models:
                continue
            if line.startswith("- name:"):
                if cur:
                    entries.append(cur)
                cur = {}
                cur["name"] = _yaml_scalar(line[len("- name:"):])
                continue
            if line.startswith("#") or line.startswith("think") or line.startswith("- "):
                continue
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key = key.strip()
            if key in ("model_id", "api_key_env", "base_url", "mode", "api_format"):
                cur[key] = _yaml_scalar(val)
        if cur:
            entries.append(cur)
        if entries:
            break
    return entries


def _yaml_scalar(raw: str) -> str:
    """去引号/去注释的 YAML 标量提取。"""
    val = raw.strip()
    if val.startswith('"') and val.endswith('"'):
        return val[1:-1]
    if val.startswith("'") and val.endswith("'"):
        return val[1:-1]
    return val.split(" #")[0].strip()


def resolve_provider(explicit: Optional[str] = None) -> str:
    """选定厂商：显式参数 > $AIRY_LLM_PROVIDER > 探测（首个有 key 的）> 模型表 > deepseek。"""
    name = (explicit or os.environ.get("AIRY_LLM_PROVIDER", "")).strip().lower()
    if name:
        return name
    for p in PROVIDER_PROBE_ORDER:
        key_env = provider_key_env(p)
        if key_env and os.environ.get(key_env, "").strip():
            return p
    # v2 表格格式（2026-08-26）：首个 api_key_env 有 key 的表条目
    for entry in _load_models_table():
        key_env = (entry.get("api_key_env") or "").strip()
        if key_env and os.environ.get(key_env, "").strip():
            return (entry.get("name") or "").strip() or "custom"
    return "deepseek"


def _provider_default_model(provider: Optional[str] = None) -> str:
    """从 manager/model/model.yaml（SSoT）读取厂商默认模型名。

    返回空串表示不可用（文件缺失/未匹配），由调用方继续回退。
    轻量文本解析（无 YAML 依赖）：匹配 ``- name: "<provider>"`` 块中的
    ``default_model`` 或 v2 表格的 ``model_id`` 行。路径：$AIRY_MODEL_CONFIG
    > $AIRYMAXHUB_ROOT/ecosystem/manager/model/model.yaml > 相对路径回退。
    """
    if not provider:
        return ""
    candidates = _model_yaml_candidates()

    pattern = re.compile(
        r'-\s*name\s*:\s*["\']?' + re.escape(provider) + r'["\']?[^\n]*\n(?:[^\n]*\n)*?'
        r'[^\S\n]*default_model\s*:\s*["\']?([^\s"\'\n]+)["\']?'
    )
    pattern_v2 = re.compile(
        r'-\s*name\s*:\s*["\']?' + re.escape(provider) + r'["\']?[^\n]*\n(?:[^\n]*\n)*?'
        r'[^\S\n]*model_id\s*:\s*["\']?([^\s"\'\n]+)["\']?'
    )
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError:
            continue
        m = pattern.search(content) or pattern_v2.search(content)
        if m:
            return m.group(1).strip()
        # v2：provider 名也可能匹配表条目的 model_id（模型名即厂商名）
        for entry in _load_models_table():
            if (entry.get("model_id") or "").strip() == provider:
                return (entry.get("model_id") or "").strip()
    return ""


def _resolve_api_key(explicit: Optional[str] = None, provider: Optional[str] = None) -> str:
    """统一 API key 解析（变量名映射 SSoT）。

    优先级：显式参数 → provider 对应的 key 环境变量 → 全局 fallback 链
    （OPENAI → DEEPSEEK → ANTHROPIC）→ 厂商目录探测（首个有 key 者）。
    """
    if explicit:
        return explicit.strip()
    if provider:
        key_env = provider_key_env(provider)
        if key_env:
            val = os.environ.get(key_env, "").strip()
            if val:
                return val
    for var in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY"):
        val = os.environ.get(var, "").strip()
        if val:
            return val
    for p in PROVIDER_PROBE_ORDER:
        key_env = provider_key_env(p)
        if key_env:
            val = os.environ.get(key_env, "").strip()
            if val:
                return val
    # v2 表格格式（2026-08-26）：按表条目 api_key_env（MODEL_1/2/3_API_KEY）
    for entry in _load_models_table():
        key_env = (entry.get("api_key_env") or "").strip()
        if key_env:
            val = os.environ.get(key_env, "").strip()
            if val:
                return val
    return ""


def _default_base_url(provider: Optional[str] = None) -> str:
    """按 provider 或可用 key 推导默认端点（与 model.yaml default_provider 对齐）。"""
    if provider:
        url = provider_default_base_url(provider)
        if url:
            return url
    for p in PROVIDER_PROBE_ORDER:
        key_env = provider_key_env(p)
        if key_env and os.environ.get(key_env, "").strip():
            return provider_default_base_url(p)
    # v2 表格格式（2026-08-26）：首个 api_key_env 有 key 的表条目端点
    for entry in _load_models_table():
        key_env = (entry.get("api_key_env") or "").strip()
        if key_env and os.environ.get(key_env, "").strip():
            url = (entry.get("base_url") or "").strip()
            if url:
                return url
    return "https://api.openai.com/v1"


def _resolve_max_tokens(explicit: Optional[int] = None) -> Optional[int]:
    """统一输出上限解析：显式参数 > ``$AIRY_LLM_MAX_TOKENS``（正整数）> None。

    None 表示不发送 max_tokens（沿用厂商默认）。厂商默认输出上限会截断
    长生成（大文件 fs_write 的 tool_call arguments JSON 半截失败），
    部署方经该变量显式抬高上限；截断兜底由 LLMAgent 的 finish_reason
    检测 fast-fail 承担（见 orchestration/agents/llm.py）。
    """
    if explicit and explicit > 0:
        return explicit
    env = os.environ.get("AIRY_LLM_MAX_TOKENS", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    return None


def _resolve_base_url(explicit: Optional[str] = None, provider: Optional[str] = None) -> str:
    """统一 base_url 解析。

    优先级：显式参数 → ``$AIRY_LLM_BASE_URL`` → ``$OPENAI_BASE_URL`` →
    provider 默认端点 → 按可用 key 推导（deepseek 优先）→ OpenAI 官方。
    ``$AIRY_LLM_BASE_URL`` 可指向任一 OpenAI 兼容端点（含自定义厂商/本地 vLLM）。
    """
    if explicit:
        return explicit.rstrip("/")
    env = (
        os.environ.get("AIRY_LLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
    )
    if env:
        return env.rstrip("/")
    return _default_base_url(provider)


class LLMClient:
    """OpenAI 兼容的异步 LLM 客户端。

    支持任意 OpenAI-compatible endpoint (OpenAI / DeepSeek / 智谱 / Moonshot / 本地 vLLM …)。

    配置优先级：显式参数 > 环境变量 > 默认值。

    - ``api_key``  : 见 :func:`_resolve_api_key`（OPENAI → DEEPSEEK → ANTHROPIC fallback）
    - ``base_url`` : 见 :func:`_resolve_base_url`（AIRY_LLM_BASE_URL 优先于 OPENAI_BASE_URL）
    - ``model``    : 默认调用时传入，未传则用构造时 default_model
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: str = "gpt-4o-mini",
        timeout: float = 60.0,
    ) -> None:
        self.api_key = _resolve_api_key(api_key)
        self.base_url = _resolve_base_url(base_url)
        self.default_model = default_model
        self.timeout = timeout

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """调用 ``POST {base_url}/chat/completions``，返回 OpenAI 响应 dict。

        返回结构 (与 OpenAI 官方一致)::

            {
              "choices": [{
                "message": {
                  "role": "assistant",
                  "content": "..." | None,
                  "tool_calls": [{"id","type":"function",
                                  "function":{"name","arguments"}}] | None
                },
                "finish_reason": "stop" | "tool_calls" | "length"
              }],
              "usage": {"prompt_tokens","completion_tokens","total_tokens"},
              "model": "..."
            }

        ``max_tokens``：输出上限（显式参数 > ``$AIRY_LLM_MAX_TOKENS`` > 不发送）。
        不发送时沿用厂商默认，长生成可能被截断（finish_reason=length），
        调用方必须检查 finish_reason 而非假定响应完整。
        """
        payload: Dict[str, Any] = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature,
        }
        resolved_max_tokens = _resolve_max_tokens(max_tokens)
        if resolved_max_tokens:
            payload["max_tokens"] = resolved_max_tokens
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # aiohttp 懒导入：mock 模式或未装 aiohttp 时也能 import 本模块
        try:
            import aiohttp
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "aiohttp is required for real LLM calls. "
                "Install with: pip install aiohttp  (or use MockLLMClient)"
            ) from e

        url = f"{self.base_url}/chat/completions"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                text = await resp.text()
                if resp.status != 200:
                    raise RuntimeError(
                        f"LLM request failed (HTTP {resp.status}): {text[:500]}"
                    )
                return json.loads(text)


class MockLLMClient:
    """离线确定性 mock LLM 客户端。

    无需 API key、无需网络。根据 messages 内容生成上下文相关响应：
    - 若声明了 tools 且 user 提到时间相关词 → 触发一次 tool_call 演示工具回路
    - 优先按 system message 判定角色 (架构师/产品经理)，让多 Agent 输出差异化
    - 否则返回一段基于 user 输入的模板化文本

    保证端到端流程可跑通，便于在没有 LLM 配额时验证 Agent 逻辑。
    """

    def __init__(self, default_model: str = "mock-model") -> None:
        self.default_model = default_model

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        # 模拟网络延迟，让 async 流程更真实
        await asyncio.sleep(0.05)

        # 取 system 与最后一条 user 消息作为上下文
        system_text = ""
        last_user = ""
        for m in messages:
            if m.get("role") == "system":
                system_text = str(m.get("content", ""))
            if m.get("role") == "user":
                last_user = str(m.get("content", ""))

        # 若有 tool 角色 message (工具结果回流)，则生成最终总结
        has_tool_result = any(m.get("role") == "tool" for m in messages)

        # 工具回路演示触发词：几点/时间/time/日期/date
        if tools and not has_tool_result:
            tool_trigger = any(
                kw in last_user for kw in ("几点", "时间", "time", "日期", "date")
            )
            if tool_trigger:
                first_tool = tools[0]["function"]
                return self._wrap_tool_call(
                    first_tool["name"],
                    "{}",
                    model or self.default_model,
                )

        # 优先按 system message 首行判定角色，让多 Agent 输出差异化。
        # 用首行而非全文，避免 PM 提示词中 "不做架构决策"/"architect Agent"
        # 等协作边界表述被误判为 architect 角色。
        first_line = system_text.split("\n", 1)[0] if system_text else ""
        fl_lower = first_line.lower()
        is_architect = "架构师" in first_line or "architect" in fl_lower
        is_pm = "产品经理" in first_line or "product manager" in fl_lower
        is_coding = "编码" in first_line or "coding" in fl_lower

        # 普通文本响应
        if has_tool_result:
            content = (
                "【Mock 总结】已根据工具返回结果完成分析。\n"
                "基于上述工具数据，给出最终结论：流程正常，工具回路闭合。"
            )
        elif is_architect:
            prd_preview = last_user[:60].replace("\n", " ")
            content = (
                "【Mock 架构设计】\n"
                "## 系统架构\n"
                "- 前端：React + TypeScript\n"
                "- 后端：FastAPI + SQLAlchemy\n"
                "- 数据库：PostgreSQL\n"
                "- 缓存：Redis\n"
                "## 模块划分\n1. 认证模块\n2. 任务模块\n3. 通知模块\n"
                "## 关键决策\n- 采用 RESTful API\n- 异步任务用 Celery\n"
                f"## 依据 PRD\n{prd_preview}\n"
                "（Mock 生成，仅用于验证 Agent 执行链路）"
            )
        elif is_pm or any(kw in last_user for kw in ("PRD", "产品", "需求", "应用", "app", "待办")):
            content = (
                "【Mock PRD】\n"
                "## 产品需求文档\n"
                f"### 用户目标\n{last_user[:80]}\n"
                "### 核心功能\n1. 任务创建与编辑\n2. 列表视图\n3. 完成状态管理\n"
                "### 非功能需求\n- 响应时间 < 200ms\n- 支持 1000 并发\n"
                "### 验收标准\n- 用户可在 3 步内完成创建\n"
                "（Mock 生成，仅用于验证 Agent 执行链路）"
            )
        elif is_coding:
            preview = last_user[:60].replace("\n", " ")
            content = (
                "【Mock 代码实现】\n"
                "```python\n"
                "def solve():\n"
                f"    \"\"\"针对需求：{preview}\"\"\"\n"
                "    # TODO: 由真实 LLM 生成实现\n"
                "    return None\n"
                "```\n"
                "（Mock 生成，仅用于验证 Agent 执行链路）"
            )
        else:
            preview = last_user[:60].replace("\n", " ")
            content = (
                "【Mock 响应】已收到你的输入并完成处理。\n"
                f"输入摘要: {preview}\n"
                "这是一个离线 mock 响应，用于在无 LLM API key 时验证 Agent 流程。\n"
                "设置 OPENAI_API_KEY 后将自动切换到真实 LLM。"
            )

        return self._wrap_text(content, model or self.default_model)

    # ── 响应封装辅助 ────────────────────────────────────

    @staticmethod
    def _wrap_text(content: str, model: str) -> Dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": content, "tool_calls": None},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": len(content) // 3,
                "total_tokens": 50 + len(content) // 3,
            },
            "model": model,
        }

    @staticmethod
    def _wrap_tool_call(name: str, arguments: str, model: str) -> Dict[str, Any]:
        call_id = f"call_{uuid.uuid4().hex[:8]}"
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": arguments},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50},
            "model": model,
        }


def _load_secrets_env() -> None:
    """加载 ``$AIRY_HOME/config/secrets.env``（LLM key 唯一落点）到环境变量。

    幂等（模块级标志，每进程仅一次）。不覆盖已存在的环境变量，
    使 daemon 无论经 bootstrap 还是手动 nohup 启动，agent 子进程都能
    读到 LLM 凭据（对齐 ``agentrt-bootstrap.sh`` 的 set -a 加载语义）。
    """
    if getattr(_load_secrets_env, "_done", False):
        return
    _load_secrets_env._done = True  # type: ignore[attr-defined]
    candidates = []
    home = os.environ.get("AIRY_HOME", "").strip()
    if home:
        candidates.append(os.path.join(home, "config", "secrets.env"))
    user_home = os.path.expanduser("~")
    candidates.append(os.path.join(user_home, ".airymaxrt", "config", "secrets.env"))
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            # 仅当环境变量缺失或为空串时才从 secrets.env 覆盖：daemon 经
            # bootstrap set -a 加载时若 secrets.env 尚为空模板，环境里会残留
            # 空串（如 DEEPSEEK_API_KEY=），后续文件填入真实 key 后必须能覆盖，
            # 否则 agent 子进程永远读到空值而走 MockLLMClient（历史 P1-3 类问题）。
            if key and not os.environ.get(key, "").strip():
                os.environ[key] = value.strip()
        return


def make_llm_client(
    default_model: Optional[str] = None,
    force_mock: bool = False,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Any:
    """环境感知工厂。

    - ``force_mock=True``                  → 始终返回 MockLLMClient
    - ``provider``                         → 指定厂商（deepseek/glm/qwen/moonshot/
      siliconflow/spark/minimax/openai/ollama/自定义名）；未指定时按
      ``$AIRY_LLM_PROVIDER`` > 探测（首个有 key 者）> deepseek
    - ``base_url`` / ``api_key``           → 显式覆盖（任选其一，其余自动解析）
    - 任一 OpenAI 兼容 key 存在            → LLMClient (真实)
    - 否则                                  → MockLLMClient (离线)

    真实模式下解析优先级：
    - ``base_url``：显式 > ``$AIRY_LLM_BASE_URL`` > ``$OPENAI_BASE_URL`` >
      provider 默认端点（如 glm → open.bigmodel.cn）
    - ``model``   ：``default_model`` > ``$AIRY_AGENT_MODEL`` > ``$OPENAI_MODEL``
      > ``gpt-4o-mini``

    因此只需在 secrets.env 配置任意厂商的 API key（+ 可选 ``AIRY_LLM_PROVIDER`` /
    ``AIRY_AGENT_MODEL``），即可接入任一 OpenAI 兼容厂商；``AIRY_LLM_BASE_URL``
    可完全自定义任意端点（本地 vLLM / 内网代理等）。
    """
    # 先加载 secrets.env（幂等），使 daemon 无论何种方式启动都能拿到 LLM 凭据
    _load_secrets_env()

    if force_mock:
        return MockLLMClient(default_model=default_model or "mock-model")

    provider = resolve_provider(provider)
    resolved_key = _resolve_api_key(api_key, provider)
    if resolved_key:
        model = (
            default_model
            or os.environ.get("AIRY_AGENT_MODEL", "")
            or os.environ.get("OPENAI_MODEL", "")
            or _provider_default_model(provider)  # SSoT: model.yaml default_model
            or "gpt-4o-mini"
        )
        resolved_url = _resolve_base_url(base_url, provider)
        return LLMClient(default_model=model, base_url=resolved_url,
                         api_key=resolved_key)

    # 无 key：mock，但打印一次提示 (仅首次)。
    # 输出到 stderr：避免污染 agent_d 子进程的 stdout 行分隔 JSON 协议。
    if not getattr(make_llm_client, "_warned", False):
        print(
            "[orchestration] 未检测到任何 OpenAI 兼容厂商的 API key"
            "（deepseek/glm/qwen/moonshot/openai/...）→ 使用 MockLLMClient。"
            "请将 key 写入 $AIRY_HOME/config/secrets.env（模板见"
            " devtools/scripts/ops/templates/secrets.env.example）。",
            file=sys.stderr,
        )
        make_llm_client._warned = True  # type: ignore[attr-defined]
    return MockLLMClient(default_model=default_model)


__all__ = [
    "LLMClient",
    "MockLLMClient",
    "make_llm_client",
    "OPENAI_COMPAT_PROVIDERS",
    "get_supported_providers",
    "resolve_provider",
    "provider_key_env",
    "provider_default_base_url",
]
