"""
BaseCacheStrategy - 缓存策略基类
"""

from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from traj_proxy.proxy_core.context import ProcessContext


class BaseCacheStrategy(ABC):
    """缓存策略抽象基类

    定义 Token 编码缓存的通用接口。
    """

    @abstractmethod
    async def encode_with_cache(
        self,
        text: str,
        context: "ProcessContext",
        tokenizer
    ) -> List[int]:
        """使用缓存优化文本编码

        Args:
            text: 待编码的文本
            context: 处理上下文
            tokenizer: tokenizer 实例

        Returns:
            token ID 列表
        """
        pass
