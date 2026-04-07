"""
ProxyCore 模块 - LLM 代理核心处理模块

该模块提供以下主要组件：
- Processor: 统一请求处理器，根据配置选择 Pipeline
- ProcessorManager: 多模型处理器管理器
- InferClient: Infer 服务客户端
- ProcessContext: 处理上下文数据类
"""

# 仅导出核心组件，避免循环导入
# ProcessorManager 在 serve/schemas 中被使用，需要延迟导入
from traj_proxy.proxy_core.context import ProcessContext
from traj_proxy.proxy_core.infer_client import InferClient

# Pipeline 模块
from traj_proxy.proxy_core.pipeline import BasePipeline, DirectPipeline, TokenPipeline

# Converters 模块
from traj_proxy.proxy_core.converters import MessageConverter, TokenConverter

# Builders 模块
from traj_proxy.proxy_core.builders import OpenAIResponseBuilder, StreamChunkBuilder

# Cache 模块
from traj_proxy.proxy_core.cache import PrefixMatchCache

import traj_proxy.exceptions as exceptions_module

__all__ = [
    # 核心组件
    "ProcessContext",
    "InferClient",
    # Pipeline
    "BasePipeline",
    "DirectPipeline",
    "TokenPipeline",
    # Converters
    "MessageConverter",
    "TokenConverter",
    # Builders
    "OpenAIResponseBuilder",
    "StreamChunkBuilder",
    # Cache
    "PrefixMatchCache",
    # 异常
    "ProxyCoreError",
    "TokenizerNotFoundError",
    "CacheError",
    "InferServiceError",
    "DatabaseError",
    "SessionIdError",
]

# 从上层模块导入异常
ProxyCoreError = exceptions_module.ProxyCoreError
TokenizerNotFoundError = exceptions_module.TokenizerNotFoundError
CacheError = exceptions_module.CacheError
InferServiceError = exceptions_module.InferServiceError
DatabaseError = exceptions_module.DatabaseError
SessionIdError = exceptions_module.SessionIdError


def __getattr__(name: str):
    """延迟导入 Processor 和 ProcessorManager，避免循环导入"""
    if name == "Processor":
        from traj_proxy.proxy_core.processor import Processor
        return Processor
    elif name == "ProcessorManager":
        from traj_proxy.proxy_core.processor_manager import ProcessorManager
        return ProcessorManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
