"""
BaseConverter - 数据转换器基类
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseConverter(ABC):
    """数据转换器抽象基类"""

    @abstractmethod
    async def convert(self, data: Any, context: "ProcessContext") -> Any:
        """转换数据

        Args:
            data: 输入数据
            context: 处理上下文

        Returns:
            转换后的数据
        """
        pass
