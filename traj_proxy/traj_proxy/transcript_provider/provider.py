"""
TranscriptProvider - 转录提供者

负责处理轨迹记录查询的业务逻辑
"""

from typing import List, Dict, Any
from traj_proxy.store.database import DatabaseManager


class TranscriptProvider:
    """转录提供者 - 处理轨迹记录查询业务逻辑

    封装数据库访问逻辑，为 routes 提供业务接口
    """

    def __init__(self, db_manager: DatabaseManager):
        """初始化 TranscriptProvider

        Args:
            db_manager: 数据库管理器
        """
        self.db_manager = db_manager

    async def get_trajectory(
        self,
        session_id: str,
        limit: int = 10000
    ) -> Dict[str, Any]:
        """根据 session_id 获取所有轨迹记录

        Args:
            session_id: 会话ID (格式: app_id#sample_id#task_id)
            limit: 最多返回的记录数，默认为100

        Returns:
            包含session_id、记录数量和记录列表的字典
        """
        records = await self.db_manager.get_request_records_by_session(session_id, limit)
        return {
            "session_id": session_id,
            "count": len(records),
            "records": records
        }
