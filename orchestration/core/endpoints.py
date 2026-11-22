# Copyright (c) 2026 SPHARX. All Rights Reserved.
"""orchestration Core — 本地推理端点目录（SSoT）

与 ``agentrt/commons/include/airy_defaults.h`` 同值：三处「本地」语义
不同，禁止互相顶替。providers 适配器缺省走 vLLM / OpenAI 兼容口 8080；
platform_gpu 探测平台加速服务口 8000；自托管 llama-server 用 11435，
Ollama 保留 11434。CI 值一致性门禁逐项比对两侧字面量，本模块是
Python 侧唯一权威源。
"""

from __future__ import annotations

PROVIDERS_PORT = 8080
PLATFORM_PORT = 8000
SELFHOST_PORT = 11435
OLLAMA_PORT = 11434

OLLAMA_URL = f"http://127.0.0.1:{OLLAMA_PORT}/v1"
SELFHOST_URL = f"http://127.0.0.1:{SELFHOST_PORT}/v1"
PROVIDERS_URL = f"http://127.0.0.1:{PROVIDERS_PORT}/v1"
