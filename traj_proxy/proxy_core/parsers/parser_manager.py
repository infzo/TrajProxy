"""
Parser 管理器
负责注册、获取和创建 Parser 实例
"""

from typing import Dict, Type, Optional, Any, List

from .base import BaseToolParser, BaseReasoningParser


class ParserManager:
    """Parser 管理器"""

    # 工具解析器注册表
    _tool_parsers: Dict[str, Type[BaseToolParser]] = {}

    # 推理解析器注册表
    _reasoning_parsers: Dict[str, Type[BaseReasoningParser]] = {}

    # 模型配置映射
    _model_configs: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register_tool_parser(cls, name: str, parser_cls: Type[BaseToolParser]):
        """注册工具解析器"""
        cls._tool_parsers[name] = parser_cls

    @classmethod
    def register_reasoning_parser(cls, name: str, parser_cls: Type[BaseReasoningParser]):
        """注册推理解析器"""
        cls._reasoning_parsers[name] = parser_cls

    @classmethod
    def register_model(cls, model_name: str, config: Dict[str, Any]):
        """
        注册模型配置

        Args:
            model_name: 模型名称
            config: {
                "tool_parser": "qwen_xml",
                "reasoning_parser": "qwen3",
            }
        """
        cls._model_configs[model_name] = config

    @classmethod
    def get_tool_parser_cls(cls, name: str) -> Optional[Type[BaseToolParser]]:
        """获取工具解析器类"""
        return cls._tool_parsers.get(name)

    @classmethod
    def get_reasoning_parser_cls(cls, name: str) -> Optional[Type[BaseReasoningParser]]:
        """获取推理解析器类"""
        return cls._reasoning_parsers.get(name)

    @classmethod
    def get_model_config(cls, model_name: str) -> Dict[str, Any]:
        """获取模型配置，支持模糊匹配"""
        # 精确匹配
        if model_name in cls._model_configs:
            return cls._model_configs[model_name]

        # 模糊匹配（模型名称可能包含版本号等）
        model_lower = model_name.lower()
        for key in cls._model_configs:
            key_lower = key.lower()
            if key_lower in model_lower or model_lower in key_lower:
                return cls._model_configs[key]

        return {}

    @classmethod
    def create_parsers_for_model(
        cls,
        model_name: str,
        tokenizer=None
    ) -> tuple[Optional[BaseToolParser], Optional[BaseReasoningParser]]:
        """为模型创建 Parser 实例"""
        config = cls.get_model_config(model_name)

        tool_parser = None
        reasoning_parser = None

        if config.get("tool_parser"):
            parser_cls = cls.get_tool_parser_cls(config["tool_parser"])
            if parser_cls:
                tool_parser = parser_cls(tokenizer)

        if config.get("reasoning_parser"):
            parser_cls = cls.get_reasoning_parser_cls(config["reasoning_parser"])
            if parser_cls:
                reasoning_parser = parser_cls(tokenizer)

        return tool_parser, reasoning_parser

    @classmethod
    def list_tool_parsers(cls) -> List[str]:
        """列出所有注册的工具解析器"""
        return list(cls._tool_parsers.keys())

    @classmethod
    def list_reasoning_parsers(cls) -> List[str]:
        """列出所有注册的推理解析器"""
        return list(cls._reasoning_parsers.keys())

    @classmethod
    def list_models(cls) -> List[str]:
        """列出所有注册的模型"""
        return list(cls._model_configs.keys())

    @classmethod
    def clear(cls):
        """清空所有注册（用于测试）"""
        cls._tool_parsers.clear()
        cls._reasoning_parsers.clear()
        cls._model_configs.clear()
