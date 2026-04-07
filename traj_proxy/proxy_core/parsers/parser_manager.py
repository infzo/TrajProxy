# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Parser 管理器

统一管理 Tool Parser 和 Reasoning Parser 的创建和获取。
"""
from typing import Optional, Tuple, List, Any

# 确保 vllm 兼容层初始化
from traj_proxy.proxy_core.parsers.vllm_compat import ensure_initialized

# 导入 vllm 兼容的管理器
from vllm.tool_parsers.abstract_tool_parser import (
    ToolParser,
    ToolParserManager,
)
from vllm.reasoning.abs_reasoning_parsers import (
    ReasoningParser,
    ReasoningParserManager,
)


class ParserManager:
    """统一的 Parser 管理器

    提供一站式接口获取和创建 Parser 实例。

    使用示例：
        # 获取 parser 类
        tool_parser_cls = ParserManager.get_tool_parser_cls("qwen3_coder")

        # 创建 parser 实例
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Coder-30B-A3B-Instruct")
        tool_parser = tool_parser_cls(tokenizer)

        # 使用 parser
        result = tool_parser.extract_tool_calls(model_output, request)
    """

    @classmethod
    def create_parsers(
        cls,
        tool_parser_name: Optional[str],
        reasoning_parser_name: Optional[str],
        tokenizer: Any,
    ) -> Tuple[Optional[ToolParser], Optional[ReasoningParser]]:
        """创建 Parser 实例

        Args:
            tool_parser_name: Tool Parser 名称（None 或空字符串表示不使用）
            reasoning_parser_name: Reasoning Parser 名称（None 或空字符串表示不使用）
            tokenizer: Tokenizer 实例（transformers.PreTrainedTokenizerBase）

        Returns:
            (tool_parser, reasoning_parser) 元组
        """
        # 确保兼容层已初始化（自动发现和注册）
        ensure_initialized()

        tool_parser = None
        reasoning_parser = None

        if tool_parser_name:
            parser_cls = ToolParserManager.get_tool_parser(tool_parser_name)
            if parser_cls:
                tool_parser = parser_cls(tokenizer=tokenizer)

        if reasoning_parser_name:
            parser_cls = ReasoningParserManager.get_reasoning_parser(reasoning_parser_name)
            if parser_cls:
                reasoning_parser = parser_cls(tokenizer=tokenizer)

        return tool_parser, reasoning_parser

    @classmethod
    def get_tool_parser_cls(cls, name: str) -> Optional[type]:
        """获取 Tool Parser 类

        Args:
            name: Parser 名称（如 "qwen3_coder"）

        Returns:
            Parser 类，如果未找到则返回 None
        """
        ensure_initialized()
        try:
            return ToolParserManager.get_tool_parser(name)
        except KeyError:
            return None

    @classmethod
    def get_reasoning_parser_cls(cls, name: str) -> Optional[type]:
        """获取 Reasoning Parser 类

        Args:
            name: Parser 名称（如 "qwen3"）

        Returns:
            Parser 类，如果未找到则返回 None
        """
        ensure_initialized()
        try:
            return ReasoningParserManager.get_reasoning_parser(name)
        except KeyError:
            return None

    @classmethod
    def list_tool_parsers(cls) -> List[str]:
        """列出所有已注册的 Tool Parser 名称"""
        ensure_initialized()
        return ToolParserManager.list_registered()

    @classmethod
    def list_reasoning_parsers(cls) -> List[str]:
        """列出所有已注册的 Reasoning Parser 名称"""
        ensure_initialized()
        return ReasoningParserManager.list_registered()

    @classmethod
    def create_tool_parser(cls, name: str, tokenizer: Any) -> Optional[ToolParser]:
        """创建 Tool Parser 实例

        Args:
            name: Parser 名称
            tokenizer: Tokenizer 实例

        Returns:
            Parser 实例，如果未找到则返回 None
        """
        parser_cls = cls.get_tool_parser_cls(name)
        if parser_cls:
            return parser_cls(tokenizer=tokenizer)
        return None

    @classmethod
    def create_reasoning_parser(cls, name: str, tokenizer: Any) -> Optional[ReasoningParser]:
        """创建 Reasoning Parser 实例

        Args:
            name: Parser 名称
            tokenizer: Tokenizer 实例

        Returns:
            Parser 实例，如果未找到则返回 None
        """
        parser_cls = cls.get_reasoning_parser_cls(name)
        if parser_cls:
            return parser_cls(tokenizer=tokenizer)
        return None


# 便捷函数
get_tool_parser = ParserManager.get_tool_parser_cls
get_reasoning_parser = ParserManager.get_reasoning_parser_cls
create_tool_parser = ParserManager.create_tool_parser
create_reasoning_parser = ParserManager.create_reasoning_parser
list_tool_parsers = ParserManager.list_tool_parsers
list_reasoning_parsers = ParserManager.list_reasoning_parsers


__all__ = [
    "ParserManager",
    "ToolParser",
    "ReasoningParser",
    "ToolParserManager",
    "ReasoningParserManager",
    "get_tool_parser",
    "get_reasoning_parser",
    "create_tool_parser",
    "create_reasoning_parser",
    "list_tool_parsers",
    "list_reasoning_parsers",
]
