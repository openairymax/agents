# Copyright (c) 2026 SPHARX. All Rights Reserved.
"""orchestration Core — 模型连接配置解析（SSoT）

单一权威源 ``$AIRY_HOME/config/model.yaml``：厂商目录、模型表解析、
model/base_url/key/max_tokens 解析与 secrets.env 载入。客户端
(:mod:`orchestration.core.llm`) 只消费本模块结果，不重复实现解析。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from .endpoints import OLLAMA_URL

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
    "ollama":      (OLLAMA_URL, ""),
}

# 自动探测时的厂商优先级（DeepSeek 居首，与 model.yaml global.default_provider 对齐）
PROVIDER_PROBE_ORDER: Tuple[str, ...] = (
    "deepseek", "glm", "qwen", "moonshot", "siliconflow", "spark", "minimax", "openai",
)

# 全局 key fallback 链（SSoT：顺序即优先级）。
# 配置一致性测试直接读取本常量，不再从函数体内正则抓取。
KEY_FALLBACK_CHAIN: Tuple[str, ...] = (
    "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY",
)

# 采样默认值（S-P1-12）：模型条目未声明时的唯一数值来源。
# top_p 缺省不下发，保持厂商默认；条目显式声明才随请求发送。
SAMPLING_DEFAULT_TEMPERATURE: float = 0.7
SAMPLING_DEFAULT_TOP_P: Optional[float] = None


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
    """model.yaml 候选路径（SSoT：$AIRY_HOME 安装面 > 源码树）。"""
    candidates: List[str] = []
    home = os.environ.get("AIRY_HOME", "").strip()
    if home:
        candidates.append(os.path.join(home, "config", "model.yaml"))
    here = os.path.dirname(os.path.abspath(__file__))  # .../orchestration/orchestration/core
    candidates.append(os.path.join(here, "..", "..", "..", "manager", "model", "model.yaml"))
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
    """读取 model.yaml 模型连接表（行数不限，逐行解析）。

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
            if key in ("model_id", "api_key_env", "base_url", "mode",
                       "api_format", "context_window", "tool_rounds",
                       "temperature", "top_p"):
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


def _match_model_entry(provider: Optional[str]) -> Optional[Dict[str, str]]:
    """按厂商名（大小写不敏感）匹配 models 表条目。"""
    if not provider:
        return None
    prov = provider.strip().casefold()
    for entry in _load_models_table():
        if (entry.get("name") or "").strip().casefold() == prov:
            return entry
    return None


def _entry_by_model_id(model_id: str) -> Optional[Dict[str, str]]:
    """按 model_id（大小写不敏感）反查 models 表条目。"""
    mid = model_id.strip().casefold()
    for entry in _load_models_table():
        if (entry.get("model_id") or "").strip().casefold() == mid:
            return entry
    return None


def _entry_float(entry: Optional[Dict[str, str]], key: str) -> Optional[float]:
    """取模型条目的可选浮点字段；缺省/非法返回 None。"""
    if not entry:
        return None
    raw = (entry.get(key) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def resolve_sampling(
    model: Optional[str] = None, provider: Optional[str] = None
) -> Tuple[float, Optional[float]]:
    """解析采样参数：模型条目显式值 > 全局默认（S-P1-12 单一来源）。"""
    entry = _entry_by_model_id(model) if model else None
    if entry is None:
        entry = _match_model_entry(provider or resolve_provider())
    temp = _entry_float(entry, "temperature")
    top_p = _entry_float(entry, "top_p")
    return (
        SAMPLING_DEFAULT_TEMPERATURE if temp is None else temp,
        SAMPLING_DEFAULT_TOP_P if top_p is None else top_p,
    )


def _is_local(entry: Optional[Dict[str, str]]) -> bool:
    """判定 models 表条目是否为本地运行时（免 key 直连）。"""
    if not entry:
        return False
    if (entry.get("mode") or "").strip().lower() == "local":
        return True
    return (entry.get("api_format") or "").strip().lower() == "local"


def _local_pick(model: str, provider: Optional[str]) -> Optional[Dict[str, str]]:
    """选出免 key 直连的本地条目。

    显式指定厂商时只认该厂商条目（不得把请求改投本机端点）；
    否则优先当前模型所在条目，再退回表中首个本地条目。
    """
    if provider:
        entry = _match_model_entry(provider)
        return entry if _is_local(entry) else None
    entry = _entry_by_model_id(model) if model else None
    if _is_local(entry):
        return entry
    for candidate in _load_models_table():
        if _is_local(candidate) and (candidate.get("base_url") or "").strip():
            return candidate
    return None


def model_tool_rounds(model: str) -> int:
    """model.yaml 中该模型条目声明的工具轮次（0 = 未声明/非法）。

    执行体侧的策略消费点：与 ``TOOL_ROUNDS_MAX`` 机制硬界配合，
    使配置项与运行期预算一一映射。
    """
    entry = _entry_by_model_id(model) if model else None
    try:
        return int((entry or {}).get("tool_rounds") or 0)
    except (TypeError, ValueError):
        return 0


def _yaml_top_scalar(key: str) -> str:
    """读取 model.yaml 顶层无缩进标量（如 ``default_model:``）。"""
    for path in _model_yaml_candidates():
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError:
            continue
        for raw in content.splitlines():
            if not raw or raw[0].isspace() or raw.lstrip().startswith("#"):
                continue
            k, sep, v = raw.partition(":")
            if sep and k.strip() == key:
                return _yaml_scalar(v)
    return ""


def resolve_default_model(
    explicit: Optional[str] = None, provider: Optional[str] = None
) -> str:
    """统一 model 解析（SSoT 单一入口）。

    优先级：显式参数 → model.yaml 顶层 ``default_model`` → 厂商条目
    ``model_id`` → ``$AIRY_AGENT_MODEL`` → ``$OPENAI_MODEL`` → 内置兜底。
    model.yaml 为第一配置权威源：修改后新会话即生效，不受残留环境
    变量遮蔽。
    """
    if explicit and explicit.strip():
        return explicit.strip()
    top = _yaml_top_scalar("default_model")
    if top:
        return top
    entry = _match_model_entry(provider or resolve_provider())
    if entry:
        mid = (entry.get("model_id") or "").strip()
        if mid:
            return mid
    for var in ("AIRY_AGENT_MODEL", "OPENAI_MODEL"):
        env = os.environ.get(var, "").strip()
        if env:
            return env
    return "deepseek-flash"


def resolve_base_url(
    explicit: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """统一 base_url 解析（与 model 同源一致）。

    优先级：显式参数 → model.yaml 条目端点（优先跟随 ``model`` 所在
    条目，否则按厂商匹配）→ ``$AIRY_LLM_BASE_URL`` → ``$OPENAI_BASE_URL``
    → 厂商目录默认端点 → OpenAI 官方。环境变量仅作 model.yaml 缺失时
    的兜底，不再遮蔽配置文件。
    """
    if explicit and explicit.strip():
        return explicit.strip().rstrip("/")
    entry = _entry_by_model_id(model) if model else None
    if entry is None:
        entry = _match_model_entry(provider or resolve_provider())
    if entry:
        url = (entry.get("base_url") or "").strip()
        if url:
            return url.rstrip("/")
    for var in ("AIRY_LLM_BASE_URL", "OPENAI_BASE_URL"):
        env = os.environ.get(var, "").strip()
        if env:
            return env.rstrip("/")
    return _default_base_url(provider)


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
    for var in KEY_FALLBACK_CHAIN:
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


