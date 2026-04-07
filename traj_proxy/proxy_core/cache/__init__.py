"""
Cache - 缓存策略模块

提供 Token 编码的缓存优化策略。
"""

from traj_proxy.proxy_core.cache.base import BaseCacheStrategy
from traj_proxy.proxy_core.cache.prefix_cache import PrefixMatchCache

__all__ = [
    "BaseCacheStrategy",
    "PrefixMatchCache",
]
