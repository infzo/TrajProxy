"""
ProcessorManager - 多模型处理器管理器

管理多个 Processor 实例，支持动态注册、删除和查询。
"""

from typing import Dict, Optional, List
from threading import Lock
from pydantic import BaseModel, Field
import asyncio

from traj_proxy.proxy_core.processor import Processor
from traj_proxy.proxy_core.infer_client import InferClient
from traj_proxy.store.database import DatabaseManager
from traj_proxy.store.model_registry import ModelRegistry, ModelConfig
from traj_proxy.exceptions import DatabaseError
from traj_proxy.utils.logger import get_logger

logger = get_logger(__name__)


# ========== Pydantic 数据模型 ==========

class RegisterModelRequest(BaseModel):
    """注册模型请求"""
    model_name: str = Field(..., description="模型名称")
    url: str = Field(..., description="Infer 服务 URL")
    api_key: str = Field(..., description="API 密钥")
    tokenizer_path: str = Field(..., description="Tokenizer 路径")
    token_in_token_out: bool = Field(default=False, description="是否使用 Token-in-Token-out 模式")


class RegisterModelResponse(BaseModel):
    """注册模型响应"""
    status: str
    model_name: str
    detail: dict


class DeleteModelResponse(BaseModel):
    """删除模型响应"""
    status: str
    model_name: str
    deleted: bool


class ModelInfo(BaseModel):
    """单个模型信息"""
    id: str = Field(..., description="模型 ID")
    object: str = Field(default="model", description="对象类型")
    created: int = Field(default=1677610602, description="创建时间戳")
    owned_by: str = Field(default="organization-owner", description="所有者")


class ListModelsResponse(BaseModel):
    """列出模型响应"""
    object: str = "list"
    data: List[ModelInfo]


# ========== ProcessorManager 类 ==========

class ProcessorManager:
    """多模型处理器管理器

    管理 model_name 到 Processor 的映射，支持：
    - 动态注册新模型
    - 删除已注册模型
    - 根据 model_name 获取 Processor
    - 列出所有已注册模型

    线程安全：使用 Lock 保护并发访问
    """

    def __init__(self, db_manager: DatabaseManager):
        """初始化 ProcessorManager

        Args:
            db_manager: 数据库管理器（所有 Processor 共享）
        """
        self.db_manager = db_manager
        self.processors: Dict[str, Processor] = {}
        self._lock = Lock()

        # 新增：模型注册表和同步控制
        self.model_registry = ModelRegistry(db_manager.pool)
        self._sync_task: Optional[asyncio.Task] = None
        self._sync_interval = 30  # 轮询间隔（秒）

        logger.info("ProcessorManager 初始化完成")

    async def start_sync(self):
        """启动模型同步（定时轮询）"""
        self._sync_task = asyncio.create_task(self._periodic_sync())
        logger.info(f"模型同步已启动，轮询间隔: {self._sync_interval}秒")

    async def stop_sync(self):
        """停止模型同步"""
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        logger.info("模型同步已停止")

    async def _periodic_sync(self):
        """定期从数据库同步模型配置（快速失败策略）"""
        while True:
            try:
                await asyncio.sleep(self._sync_interval)
                await self._sync_from_db()
                logger.debug("模型同步完成")
            except DatabaseError as e:
                # 快速失败：记录错误并抛出异常
                logger.error(f"模型同步失败（数据库错误）: {e}")
                raise  # 向上传播错误
            except Exception as e:
                logger.error(f"模型同步失败: {e}")
                raise

    async def _sync_from_db(self):
        """从数据库同步模型配置到内存"""
        try:
            db_models = await self.model_registry.get_all_models()
            db_model_names = {m.model_name for m in db_models}

            with self._lock:
                local_model_names = set(self.processors.keys())

                # 添加新模型或更新已有模型
                for config in db_models:
                    if config.model_name not in self.processors:
                        # 新增模型
                        self._register_from_config(config)
                    else:
                        # 检查是否需要更新
                        existing = self.processors[config.model_name]
                        if (existing.tokenizer_path != config.tokenizer_path or
                            existing.token_in_token_out != config.token_in_token_out):
                            # 配置变化，重新注册
                            self._register_from_config(config)

                # 删除数据库中不存在的模型
                to_remove = local_model_names - db_model_names
                for model_name in to_remove:
                    del self.processors[model_name]
                    logger.info(f"同步删除模型: {model_name}")

        except Exception as e:
            logger.error(f"从数据库同步模型失败: {e}")
            raise

    def _register_from_config(self, config: ModelConfig):
        """从配置创建 Processor（内部方法，不持久化到数据库）"""
        infer_client = InferClient(
            base_url=config.url,
            api_key=config.api_key
        )

        processor_config = {
            "token_in_token_out": config.token_in_token_out
        }

        processor = Processor(
            model=config.model_name,
            tokenizer_path=config.tokenizer_path,
            db_manager=self.db_manager,
            infer_client=infer_client,
            config=processor_config
        )

        self.processors[config.model_name] = processor
        logger.info(f"同步注册模型: {config.model_name}")

    async def register_processor(
        self,
        model_name: str,
        url: str,
        api_key: str,
        tokenizer_path: str,
        token_in_token_out: bool = False,
        persist_to_db: bool = True
    ) -> Processor:
        """注册新的 Processor（改为 async）

        Args:
            model_name: 模型名称
            url: Infer 服务 URL
            api_key: API 密钥
            tokenizer_path: Tokenizer 路径
            token_in_token_out: 是否使用 Token-in-Token-out 模式
            persist_to_db: 是否持久化到数据库（默认 True）

        Returns:
            新创建的 Processor 实例

        Raises:
            ValueError: 如果 model_name 已存在
            DatabaseError: 数据库操作失败
        """
        with self._lock:
            if model_name in self.processors:
                raise ValueError(f"模型 '{model_name}' 已存在")

            # 创建 InferClient
            infer_client = InferClient(
                base_url=url,
                api_key=api_key
            )

            # 创建配置字典
            config = {
                "token_in_token_out": token_in_token_out
            }

            # 创建 Processor
            processor = Processor(
                model=model_name,
                tokenizer_path=tokenizer_path,
                db_manager=self.db_manager,
                infer_client=infer_client,
                config=config
            )

            self.processors[model_name] = processor
            logger.info(f"注册模型成功: {model_name}, url={url}, tokenizer={tokenizer_path}, token_in_token_out={token_in_token_out}")

        # 持久化到数据库（同步，快速失败）
        if persist_to_db:
            try:
                await self.model_registry.register_model(
                    model_name=model_name,
                    url=url,
                    api_key=api_key,
                    tokenizer_path=tokenizer_path,
                    token_in_token_out=token_in_token_out
                )
            except Exception as e:
                # 数据库失败时，回滚本地注册
                with self._lock:
                    if model_name in self.processors:
                        del self.processors[model_name]
                logger.error(f"持久化模型到数据库失败: {e}")
                raise DatabaseError(f"注册模型失败（数据库错误）: {str(e)}")

        return processor

    async def unregister_processor(self, model_name: str, persist_to_db: bool = True) -> bool:
        """删除已注册的 Processor（改为 async）

        Args:
            model_name: 模型名称
            persist_to_db: 是否从数据库删除（默认 True）

        Returns:
            是否成功删除（False 表示模型不存在）

        Raises:
            DatabaseError: 数据库操作失败
        """
        deleted = False
        with self._lock:
            if model_name not in self.processors:
                logger.warning(f"尝试删除不存在的模型: {model_name}")
                return False

            del self.processors[model_name]
            logger.info(f"删除模型成功: {model_name}")
            deleted = True

        # 从数据库删除（同步，快速失败）
        if persist_to_db and deleted:
            try:
                success = await self.model_registry.unregister_model(model_name)
                if not success:
                    # 数据库中不存在，记录警告但不抛出异常
                    logger.warning(f"数据库中未找到模型: {model_name}")
            except Exception as e:
                logger.error(f"从数据库删除模型失败: {e}")
                raise DatabaseError(f"删除模型失败（数据库错误）: {str(e)}")

        return deleted

    def get_processor(self, model_name: str) -> Optional[Processor]:
        """根据 model_name 获取 Processor

        Args:
            model_name: 模型名称

        Returns:
            Processor 实例，如果不存在则返回 None
        """
        with self._lock:
            return self.processors.get(model_name)

    def get_processor_or_raise(self, model_name: str) -> Processor:
        """根据 model_name 获取 Processor，不存在时抛出异常

        Args:
            model_name: 模型名称

        Returns:
            Processor 实例

        Raises:
            ValueError: 如果模型不存在
        """
        processor = self.get_processor(model_name)
        if processor is None:
            raise ValueError(f"模型 '{model_name}' 未注册")
        return processor

    def list_models(self) -> List[str]:
        """列出所有已注册的模型名称

        Returns:
            模型名称列表
        """
        return list(self.processors.keys())

    def get_processor_info(self, model_name: str) -> Optional[Dict]:
        """获取 Processor 的详细信息

        Args:
            model_name: 模型名称

        Returns:
            包含模型信息的字典，如果不存在则返回 None
        """
        processor = self.get_processor(model_name)
        if processor is None:
            return None

        return {
            "model_name": processor.model,
            "tokenizer_path": processor.tokenizer_path,
            "token_in_token_out": processor.token_in_token_out,
            "infer_client_url": processor.infer_client.base_url if processor.infer_client else None
        }

    def get_all_processors_info(self) -> List[Dict]:
        """获取所有 Processor 的详细信息

        Returns:
            包含所有模型信息的字典列表
        """
        return [
            self.get_processor_info(model_name)
            for model_name in self.processors.keys()
        ]
