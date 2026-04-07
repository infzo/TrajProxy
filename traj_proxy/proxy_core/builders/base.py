"""
BaseResponseBuilder - 响应构建器基类
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from traj_proxy.proxy_core.context import ProcessContext
    from traj_proxy.proxy_core.parsers.parser_manager import Parser


class BaseResponseBuilder(ABC):
    """响应构建器抽象基类

    定义构建 OpenAI 格式响应的通用接口。
    """

    @abstractmethod
    def build(
        self,
        content: str,
        context: "ProcessContext"
    ) -> Dict[str, Any]:
        """构建完整响应

        Args:
            content: 响应内容
            context: 处理上下文

        Returns:
            OpenAI 格式的响应字典
        """
        pass

    @abstractmethod
    def build_chunk(
        self,
        content: str,
        context: "ProcessContext",
        finish_reason: Optional[str] = None,
        tool_calls_delta: Optional[List[Dict[str, Any]]] = None,
        reasoning_delta: Optional[str] = None
    ) -> Dict[str, Any]:
        """构建流式响应块

        Args:
            content: 本次输出的内容片段
            context: 处理上下文
            finish_reason: 结束原因（仅最后一个 chunk 有值）
            tool_calls_delta: 工具调用的增量数据
            reasoning_delta: 推理内容的增量

        Returns:
            OpenAI 格式的 chunk 字典
        """
        pass
