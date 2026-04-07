# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Parser 模块

提供统一的 Parser 管理接口。

使用方式：
    from traj_proxy.proxy_core.parsers import ParserManager

    # 获取 parser
    tool_parser = ParserManager.get_tool_parser("qwen3_coder")
    reasoning_parser = ParserManager.get_reasoning_parser("qwen3")

    # 创建 parser 实例
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser = tool_parser(tokenizer)

    # 解析工具调用
    result = parser.extract_tool_calls(model_output, request)
"""
import sys
from pathlib import Path

# 确保 vllm_compat 目录在 sys.path 中（在其他导入之前）
_vllm_compat_dir = Path(__file__).parent / "vllm_compat"
_vllm_compat_path = str(_vllm_compat_dir)
if _vllm_compat_path not in sys.path:
    sys.path.insert(0, _vllm_compat_path)

from traj_proxy.proxy_core.parsers.parser_manager import ParserManager

# 确保 vllm 兼容层已初始化
from traj_proxy.proxy_core.parsers.vllm_compat import ensure_initialized

__all__ = [
    "ParserManager",
    "ensure_initialized",
]
