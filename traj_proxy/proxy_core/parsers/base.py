# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Parser 基础数据结构

从 vllm 兼容层重新导出核心数据结构，方便其他模块使用。
"""
import sys
from pathlib import Path

# 确保 vllm_compat 目录在 sys.path 中
_vllm_compat_dir = Path(__file__).parent / "vllm_compat"
_vllm_compat_path = str(_vllm_compat_dir)
if _vllm_compat_path not in sys.path:
    sys.path.insert(0, _vllm_compat_path)

# 从 vllm 兼容层导入核心数据结构
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    DeltaToolCall,
    DeltaFunctionCall,
    ToolCall,
    FunctionCall,
    ExtractedToolCallInformation,
)

# 别名，保持与现有代码的兼容性
ExtractedToolCallInfo = ExtractedToolCallInformation

__all__ = [
    # Delta 消息
    "DeltaMessage",
    "DeltaToolCall",
    "DeltaFunctionCall",

    # 工具调用
    "ToolCall",
    "FunctionCall",
    "ExtractedToolCallInformation",
    "ExtractedToolCallInfo",  # 别名
]
