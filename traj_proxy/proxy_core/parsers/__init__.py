"""
Parser 模块入口

注册所有内置 Parser 和模型配置
"""

from .parser_manager import ParserManager
from .base import (
    ToolCall,
    DeltaToolCall,
    ExtractedToolCallInfo,
    DeltaMessage,
    BaseToolParser,
    BaseReasoningParser,
)

# 导入 Parser 实现（延迟导入避免循环依赖）
def _register_parsers():
    """注册所有内置 Parser"""
    from .tool_parsers.qwen_xml_parser import QwenXMLToolParser
    from .tool_parsers.deepseek_v3_parser import DeepSeekV3ToolParser
    from .tool_parsers.llama3_json_parser import Llama3JSONToolParser

    from .reasoning_parsers.qwen3_reasoning_parser import Qwen3ReasoningParser
    from .reasoning_parsers.deepseek_reasoning_parser import DeepSeekReasoningParser

    # 注册 Tool Parser
    ParserManager.register_tool_parser("qwen_xml", QwenXMLToolParser)
    ParserManager.register_tool_parser("deepseek_v3", DeepSeekV3ToolParser)
    ParserManager.register_tool_parser("llama3_json", Llama3JSONToolParser)

    # 注册 Reasoning Parser
    ParserManager.register_reasoning_parser("qwen3", Qwen3ReasoningParser)
    ParserManager.register_reasoning_parser("deepseek", DeepSeekReasoningParser)

    # ==================== 注册模型配置 ====================

    # Qwen 系列
    ParserManager.register_model("qwen3-coder", {
        "tool_parser": "qwen_xml",
        "reasoning_parser": "qwen3",
    })

    ParserManager.register_model("qwen3.5", {
        "tool_parser": "qwen_xml",
        "reasoning_parser": "qwen3",
    })

    ParserManager.register_model("qwen3", {
        "tool_parser": "qwen_xml",
        "reasoning_parser": "qwen3",
    })

    # DeepSeek 系列
    ParserManager.register_model("deepseek-v3", {
        "tool_parser": "deepseek_v3",
        "reasoning_parser": "deepseek",
    })

    ParserManager.register_model("deepseek-coder", {
        "tool_parser": "deepseek_v3",
        "reasoning_parser": None,
    })

    # Llama 系列
    ParserManager.register_model("llama-3.1", {
        "tool_parser": "llama3_json",
        "reasoning_parser": None,
    })

    ParserManager.register_model("llama-3.2", {
        "tool_parser": "llama3_json",
        "reasoning_parser": None,
    })


# 模块加载时注册
_register_parsers()

# 导出
__all__ = [
    "ParserManager",
    "ToolCall",
    "DeltaToolCall",
    "ExtractedToolCallInfo",
    "DeltaMessage",
    "BaseToolParser",
    "BaseReasoningParser",
]
