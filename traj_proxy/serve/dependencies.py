"""
FastAPI 依赖注入

提供路由处理器中使用的依赖注入函数。
"""

from typing import TYPE_CHECKING
from fastapi import Request, HTTPException
from traj_proxy.utils.logger import get_logger

if TYPE_CHECKING:
    from traj_proxy.proxy_core.processor_manager import ProcessorManager

logger = get_logger(__name__)


def get_processor_manager(request: Request) -> "ProcessorManager":
    """
    从请求上下文获取 ProcessorManager

    Args:
        request: FastAPI Request 对象

    Returns:
        ProcessorManager 实例

    Raises:
        HTTPException: 如果 ProcessorManager 未初始化
    """
    pm = getattr(request.app.state, "processor_manager", None)
    if pm is None:
        logger.error("ProcessorManager 未初始化")
        raise HTTPException(status_code=500, detail="服务未初始化")
    return pm
