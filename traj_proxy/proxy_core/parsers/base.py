"""
基础数据结构和抽象类
不依赖任何外部库（除 typing）
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Any, Sequence


# ==================== 数据结构 ====================

@dataclass
class ToolCall:
    """工具调用"""
    id: str
    type: str = "function"
    name: str = ""
    arguments: str = ""


@dataclass
class DeltaToolCall:
    """流式工具调用增量"""
    id: Optional[str] = None
    type: Optional[str] = None
    index: int = 0
    name: Optional[str] = None
    arguments: Optional[str] = None


@dataclass
class ExtractedToolCallInfo:
    """提取的工具调用信息"""
    tools_called: bool
    tool_calls: List[ToolCall]
    content: Optional[str] = None


@dataclass
class DeltaMessage:
    """流式增量消息"""
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning: Optional[str] = None
    tool_calls: List[DeltaToolCall] = field(default_factory=list)

    def __post_init__(self):
        """确保 tool_calls 不为 None"""
        if self.tool_calls is None:
            self.tool_calls = []


# ==================== 抽象基类 ====================

class BaseToolParser(ABC):
    """工具解析器基类"""

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    @abstractmethod
    def extract_tool_calls(
        self,
        model_output: str,
        tools: Optional[List[dict]] = None
    ) -> ExtractedToolCallInfo:
        """
        非流式解析工具调用

        Args:
            model_output: 模型输出的完整文本
            tools: 工具定义列表（用于参数类型转换）

        Returns:
            ExtractedToolCallInfo
        """
        pass

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        tools: Optional[List[dict]] = None
    ) -> Optional[DeltaMessage]:
        """
        流式解析工具调用（可选实现）

        Args:
            previous_text: 之前的累积文本
            current_text: 当前的累积文本
            delta_text: 本次增量文本
            previous_token_ids: 之前的 token IDs
            current_token_ids: 当前的 token IDs
            delta_token_ids: 本次增量的 token IDs
            tools: 工具定义列表

        Returns:
            DeltaMessage 或 None
        """
        return None

    def reset_streaming_state(self):
        """重置流式状态（每个新请求开始时调用）"""
        pass


class BaseReasoningParser(ABC):
    """推理内容解析器基类"""

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    @property
    @abstractmethod
    def start_token(self) -> str:
        """推理开始标记"""
        pass

    @property
    @abstractmethod
    def end_token(self) -> str:
        """推理结束标记"""
        pass

    @abstractmethod
    def extract_reasoning(
        self,
        model_output: str
    ) -> tuple[Optional[str], Optional[str]]:
        """
        非流式解析推理内容

        Args:
            model_output: 模型输出的完整文本

        Returns:
            (reasoning, content): 推理内容和剩余内容
        """
        pass

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int]
    ) -> Optional[DeltaMessage]:
        """
        流式解析推理内容（可选实现）
        """
        return None

    def reset_streaming_state(self):
        """重置流式状态"""
        pass
