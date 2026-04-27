"""
DatabaseManager - 数据库管理器

负责连接池管理。

注意：数据库表和索引的创建已迁移到 scripts/init_db.py（一次性脚本），
集群部署前请先执行该脚本初始化数据库。
"""

import traceback
from typing import Optional

from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

from traj_proxy.exceptions import DatabaseError
from traj_proxy.utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """数据库管理器 - 处理连接池

    使用 PostgreSQL 存储数据，管理三个表：
    - request_metadata: 请求轨迹元数据（长期保留）
    - request_details_active: 请求轨迹详情（近期大字段）
    - model_registry: 模型配置注册表
    """

    def __init__(self, db_url: str, pool_config: Optional[dict] = None):
        """初始化 DatabaseManager

        Args:
            db_url: 数据库连接 URL（如 postgresql://user:pass@host:port/dbname）
            pool_config: 连接池配置，包含 min_size, max_size, timeout
        """
        self.db_url = db_url
        self.pool: Optional[AsyncConnectionPool] = None
        # 使用传入的配置或默认值
        self.pool_config = pool_config or {
            "min_size": 2,
            "max_size": 20,
            "timeout": 30
        }

    async def initialize(self):
        """初始化连接池

        注意：不再自动创建数据库表，请确保部署前已执行 scripts/init_db.py
        """
        logger.info("DatabaseManager: 开始初始化连接池")
        try:
            self.pool = AsyncConnectionPool(
                conninfo=self.db_url,
                min_size=self.pool_config["min_size"],
                max_size=self.pool_config["max_size"],
                timeout=self.pool_config["timeout"],
                kwargs={"row_factory": dict_row}
            )
            # 显式打开连接池
            logger.info("DatabaseManager: 正在打开连接池...")
            await self.pool.open()
            logger.info("DatabaseManager: 连接池已打开，初始化完成")
        except Exception as e:
            logger.error(f"DatabaseManager: 初始化失败: {e}\n{traceback.format_exc()}")
            raise

    async def close(self):
        """关闭连接池"""
        if self.pool:
            await self.pool.close()
