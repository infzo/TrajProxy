"""
TranscriptProvider FastAPI路由

处理轨迹记录查询相关路由
"""

from fastapi import APIRouter, HTTPException, Request
from typing import Dict, Any
import traceback

from traj_proxy.workers.worker import get_transcript_provider as get_provider
from traj_proxy.utils.validators import validate_session_id


transcript_router = APIRouter()


@transcript_router.get("/trajectory")
async def get_trajectory(
    request: Request,
    session_id: str,
    limit: int = 10000
) -> Dict[str, Any]:
    """
    根据 session_id 获取所有轨迹记录

    参数:
        request: FastAPI Request 对象
        session_id: 会话ID (格式: app_id,sample_id,task_id)
        limit: 最多返回的记录数，默认为10000

    返回:
        包含session_id、记录数量和记录列表的字典

    Raises:
        HTTPException: 当查询失败时抛出
    """
    # 校验 session_id
    valid, msg = validate_session_id(session_id)
    if not valid:
        raise HTTPException(status_code=422, detail=msg)

    try:
        provider = get_provider(request)
        return await provider.get_trajectory(session_id, limit)
    except Exception as e:
        # 记录详细错误到日志
        import logging
        logging.getLogger(__name__).exception(f"轨迹查询失败: {str(e)}")
        # 返回通用错误信息
        raise HTTPException(status_code=500, detail="轨迹查询失败，请稍后重试")
