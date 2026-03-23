"""
ModelRegistry - 模型配置持久化管理器

负责模型配置的 CRUD 操作
"""

from typing import List, Optional
from datetime import datetime
from dataclasses import dataclass
from psycopg import errors as pg_errors

from traj_proxy.exceptions import DatabaseError
from traj_proxy.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelConfig:
    """模型配置数据类"""
    model_name: str
    url: str
    api_key: str
    tokenizer_path: str
    token_in_token_out: bool = False
    updated_at: datetime = None


class ModelRegistry:
    """模型注册表管理器

    提供模型配置的持久化和查询功能
    """

    def __init__(self, pool):
        """初始化 ModelRegistry

        Args:
            pool: PostgreSQL 连接池
        """
        self.pool = pool

    async def register_model(
        self,
        model_name: str,
        url: str,
        api_key: str,
        tokenizer_path: str,
        token_in_token_out: bool = False
    ) -> ModelConfig:
        """注册新模型到数据库（使用 UPSERT）

        Args:
            model_name: 模型名称
            url: Infer 服务 URL
            api_key: API 密钥
            tokenizer_path: Tokenizer 路径
            token_in_token_out: 是否使用 Token-in-Token-out 模式

        Returns:
            ModelConfig 实例

        Raises:
            DatabaseError: 数据库操作失败
        """
        try:
            async with self.pool.connection() as conn:
                now = datetime.now()
                await conn.execute("""
                    INSERT INTO model_registry
                    (model_name, url, api_key, tokenizer_path, token_in_token_out, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (model_name)
                    DO UPDATE SET
                        url = EXCLUDED.url,
                        api_key = EXCLUDED.api_key,
                        tokenizer_path = EXCLUDED.tokenizer_path,
                        token_in_token_out = EXCLUDED.token_in_token_out,
                        updated_at = EXCLUDED.updated_at
                """, (model_name, url, api_key, tokenizer_path, token_in_token_out, now))

                return ModelConfig(
                    model_name=model_name,
                    url=url,
                    api_key=api_key,
                    tokenizer_path=tokenizer_path,
                    token_in_token_out=token_in_token_out,
                    updated_at=now
                )
        except Exception as e:
            raise DatabaseError(f"注册模型到数据库失败: {str(e)}")

    async def unregister_model(self, model_name: str) -> bool:
        """从数据库删除模型

        Args:
            model_name: 模型名称

        Returns:
            是否成功删除

        Raises:
            DatabaseError: 数据库操作失败
        """
        try:
            async with self.pool.connection() as conn:
                result = await conn.execute("""
                    DELETE FROM model_registry
                    WHERE model_name = %s
                """, (model_name,))

                return result.rowcount > 0
        except Exception as e:
            raise DatabaseError(f"从数据库删除模型失败: {str(e)}")

    async def get_all_models(self) -> List[ModelConfig]:
        """获取所有模型配置

        Returns:
            ModelConfig 列表

        Raises:
            DatabaseError: 数据库操作失败（表不存在除外）
        """
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("""
                        SELECT model_name, url, api_key, tokenizer_path, token_in_token_out, updated_at
                        FROM model_registry
                        ORDER BY model_name
                    """)
                    rows = await cur.fetchall()

                    return [
                        ModelConfig(
                            model_name=row[0],
                            url=row[1],
                            api_key=row[2],
                            tokenizer_path=row[3],
                            token_in_token_out=row[4],
                            updated_at=row[5]
                        )
                        for row in rows
                    ]
        except pg_errors.UndefinedTable:
            # 表不存在，返回空列表（首次启动时可能出现）
            logger.info("model_registry 表不存在，返回空列表")
            return []
        except Exception as e:
            raise DatabaseError(f"从数据库获取模型列表失败: {str(e)}")
