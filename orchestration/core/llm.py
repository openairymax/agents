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
- 用户侧配置：``$AIRY_HOME/config/model.yaml`` 为模型连接唯一权威源
  （model / base_url / key 变量名）；``$AIRY_LLM_PROVIDER`` 可显式选厂商；
  key 统一写在 ``$AIRY_HOME/config/secrets.env``
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional

from .llm_cfg import (
    OPENAI_COMPAT_PROVIDERS,
    PROVIDER_PROBE_ORDER,
    _default_base_url,
    _entry_by_model_id,
    _is_local,
    _load_models_table,
    _load_secrets_env,
    _local_pick,
    _match_model_entry,
    _model_yaml_candidates,
    _resolve_api_key,
    _resolve_max_tokens,
    _yaml_scalar,
    _yaml_top_scalar,
    get_supported_providers,
    provider_default_base_url,
    provider_key_env,
    resolve_base_url,
    resolve_default_model,
    resolve_provider,
)


def _build_local(entry: Dict[str, str]) -> "LLMClient":
    """按本地条目构造免 key 客户端（本地运行时忽略鉴权头）。"""
    model_id = (entry.get("model_id") or "").strip() or "local-model"
    base_url = (entry.get("base_url") or "").strip().rstrip("/")
    if not getattr(_build_local, "_logged", False):
        print(
            f"[orchestration] local 直连 {base_url}（model={model_id}，免 key）",
            file=sys.stderr,
        )
        _build_local._logged = True  # type: ignore[attr-defined]
    return LLMClient(default_model=model_id, base_url=base_url, local=True)


class LLMClient:
    """OpenAI 兼容的异步 LLM 客户端。

    支持任意 OpenAI-compatible endpoint (OpenAI / DeepSeek / 智谱 / Moonshot / 本地 vLLM …)。

    配置解析统一走 model.yaml SSoT 单一入口（构造/工厂路径行为一致）：

    - ``api_key``  : 见 :func:`_resolve_api_key`（OPENAI → DEEPSEEK → ANTHROPIC fallback）
    - ``model``    : 见 :func:`resolve_default_model`（model.yaml 优先于环境变量）
    - ``base_url`` : 见 :func:`resolve_base_url`（跟随 model 所在条目端点）
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        timeout: float = 60.0,
        local: bool = False,
    ) -> None:
        # 本地直连（llama-server 等）不接受鉴权：显式跳过 key 解析，
        # 避免把云端凭据发给本机端点，也不因「无 key」被误判为离线。
        self.local = local
        self.api_key = "" if local else _resolve_api_key(api_key)
        self.default_model = resolve_default_model(default_model)
        self.base_url = resolve_base_url(base_url, model=self.default_model)
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

        headers = {"Content-Type": "application/json"}
        if not self.local:
            headers["Authorization"] = f"Bearer {self.api_key}"

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

    真实模式下解析统一走 model.yaml SSoT 单一入口：
    - ``model``   ：``default_model`` 显式参数 > model.yaml 顶层
      ``default_model`` > 厂商条目 ``model_id`` > ``$AIRY_AGENT_MODEL`` >
      ``$OPENAI_MODEL`` > 内置兜底
    - ``base_url``：显式 > model.yaml 条目端点（跟随 model 所在行）>
      ``$AIRY_LLM_BASE_URL`` > ``$OPENAI_BASE_URL`` > 厂商目录默认端点

    因此接入厂商只需两步：model.yaml 表填一行连接（model/base_url），
    secrets.env 配对应 key。修改 model.yaml 后新会话即生效，无需改环境
    变量；环境变量仅作 model.yaml 缺失时的兜底。
    """
    # 先加载 secrets.env（幂等），使 daemon 无论何种方式启动都能拿到 LLM 凭据
    _load_secrets_env()

    if force_mock:
        return MockLLMClient(default_model=default_model or "mock-model")

    # 显式厂商（参数或环境变量）决定端点归属：此时不得把请求改投本机
    pinned = (provider or "").strip() or os.environ.get("AIRY_LLM_PROVIDER", "").strip()
    provider = resolve_provider(provider)
    resolved_key = _resolve_api_key(api_key, provider)
    model = resolve_default_model(default_model, provider)

    # 当前模型条目本身是本地运行时：本地优先且免 key（云端凭据不得发给本机端点）。
    entry = _match_model_entry(pinned) if pinned else _entry_by_model_id(model)
    if _is_local(entry):
        return _build_local(entry)

    if resolved_key:
        resolved_url = resolve_base_url(base_url, provider, model)
        return LLMClient(default_model=model, base_url=resolved_url,
                         api_key=resolved_key)

    # 无 key：若表中声明了本地条目，则免 key 直连本地运行时，
    # 纯净离线环境也必须走真实推理而非模板文本。
    local_entry = _local_pick(model, pinned or None)
    if local_entry is not None:
        return _build_local(local_entry)

    # 无 key 但厂商端点本身免鉴权（ollama / 自定义本地网关）：直连而非回落 mock。
    if not provider_key_env(provider):
        resolved_url = resolve_base_url(base_url, provider, model)
        print(
            f"[orchestration] 端点免鉴权，直连 {resolved_url}（provider={provider}）",
            file=sys.stderr,
        )
        return LLMClient(default_model=model, base_url=resolved_url,
                         api_key="", local=True)

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
    "resolve_default_model",
    "resolve_base_url",
    "OPENAI_COMPAT_PROVIDERS",
    "get_supported_providers",
    "resolve_provider",
    "provider_key_env",
    "provider_default_base_url",
]
