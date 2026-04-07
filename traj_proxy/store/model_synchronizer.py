"""
ModelSynchronizer - 模型同步器

负责从数据库同步动态模型配置到内存。
支持 LISTEN/NOTIFY 实时同步 + 定期兜底同步。
"""

from typing import Optional, Callable, List, Awaitable, Tuple
import asyncio
import traceback

from traj_proxy.store.model_repository import ModelRepository
from traj_proxy.store.models import ModelConfig
from traj_proxy.exceptions import DatabaseError
from traj_proxy.utils.logger import get_logger
from traj_proxy.utils.config import get_sync_max_retries, get_sync_retry_delay, get_sync_fallback_interval

logger = get_logger(__name__)


class ModelSynchronizer:
    """模型同步器

    负责从数据库同步动态模型配置，通过回调通知 ProcessorManager 进行注册/删除。

    使用示例：
        synchronizer = ModelSynchronizer(
            model_registry=registry,
            db_url=db_url,
            on_model_register=processor_manager.register_from_config,
            on_model_unregister=processor_manager.unregister_by_key,
            on_full_sync=processor_manager.full_sync,
        )
        await synchronizer.start()
    """

    def __init__(
        self,
        model_registry: ModelRepository,
        db_url: str,
        on_model_register: Callable[[ModelConfig], Awaitable[None]],
        on_model_unregister: Callable[[Tuple[str, str]], Awaitable[None]],
        on_full_sync: Callable[[List[ModelConfig]], Awaitable[None]],
        sync_max_retries: Optional[int] = None,
        sync_retry_delay: Optional[float] = None,
        fallback_interval: Optional[int] = None,
    ):
        """初始化 ModelSynchronizer

        Args:
            model_registry: 模型配置仓库
            db_url: 数据库连接 URL（用于 LISTEN/NOTIFY 专用连接）
            on_model_register: 单个模型注册回调
            on_model_unregister: 单个模型删除回调
            on_full_sync: 全量同步回调
            sync_max_retries: 同步最大重试次数
            sync_retry_delay: 重试延迟（秒）
            fallback_interval: 兜底同步间隔（秒）
        """
        self._model_registry = model_registry
        self._db_url = db_url
        self._on_model_register = on_model_register
        self._on_model_unregister = on_model_unregister
        self._on_full_sync = on_full_sync

        # 同步参数
        self._sync_max_retries = sync_max_retries or get_sync_max_retries()
        self._sync_retry_delay = sync_retry_delay or get_sync_retry_delay()
        self._fallback_interval = fallback_interval or get_sync_fallback_interval()

        # 同步任务
        self._notification_listener = None
        self._fallback_sync_task: Optional[asyncio.Task] = None

        logger.info(f"ModelSynchronizer 初始化完成，兜底同步间隔: {self._fallback_interval}秒")

    async def start(self):
        """启动模型同步：LISTEN/NOTIFY（主）+ 定期兜底"""
        # 1. 首先执行一次全量同步（初始加载）
        try:
            await self._full_sync_from_db()
            logger.info("初始全量模型同步完成")
        except Exception as e:
            logger.error(f"初始全量同步失败: {e}")

        # 2. 启动 LISTEN/NOTIFY 监听器（如果配置了 db_url）
        if self._db_url:
            from traj_proxy.store.notification_listener import NotificationListener
            self._notification_listener = NotificationListener(
                db_url=self._db_url,
                on_notification=self._handle_notification,
                reconnect_delay=self._sync_retry_delay,
            )
            await self._notification_listener.start()
            logger.info("LISTEN/NOTIFY 实时同步已激活")
        else:
            logger.warning("未配置 db_url，LISTEN/NOTIFY 已禁用，仅依赖轮询同步")

        # 3. 启动兜底定期全量同步（间隔较长）
        self._fallback_sync_task = asyncio.create_task(
            self._periodic_sync(interval=self._fallback_interval)
        )
        logger.info(f"兜底定期同步已启动，间隔: {self._fallback_interval}秒")

    async def stop(self):
        """停止所有同步任务"""
        if self._notification_listener:
            await self._notification_listener.stop()
            self._notification_listener = None
        if self._fallback_sync_task:
            self._fallback_sync_task.cancel()
            try:
                await self._fallback_sync_task
            except asyncio.CancelledError:
                pass
            self._fallback_sync_task = None
        logger.info("模型同步已停止")

    async def _periodic_sync(self, interval: int = None):
        """定期全量同步（兜底机制，带重试）"""
        interval = interval or self._fallback_interval
        retry_count = 0
        current_retry_delay = self._sync_retry_delay

        while True:
            try:
                await asyncio.sleep(interval)
                await self._full_sync_from_db()
                logger.debug("兜底同步完成")
                retry_count = 0
                current_retry_delay = self._sync_retry_delay
            except DatabaseError as e:
                retry_count += 1
                if retry_count >= self._sync_max_retries:
                    logger.error(f"兜底同步失败（达到最大重试次数 {self._sync_max_retries}）: {e}")
                    retry_count = 0
                    current_retry_delay = self._sync_retry_delay
                else:
                    delay = current_retry_delay * (2 ** (retry_count - 1))
                    logger.warning(f"兜底同步失败（第 {retry_count}/{self._sync_max_retries} 次），{delay}秒后重试: {e}")
                    await asyncio.sleep(delay)
            except Exception as e:
                logger.error(f"兜底同步出现非数据库错误: {e}", exc_info=True)
                await asyncio.sleep(interval)

    async def _handle_notification(self, payload: dict):
        """处理 LISTEN/NOTIFY 通知，执行增量同步

        对于 register：从数据库获取单个模型并更新内存
        对于 unregister：直接从内存移除

        Args:
            payload: 通知内容，包含 action, run_id, model_name, timestamp
        """
        action = payload.get("action")
        run_id = payload.get("run_id", "")
        model_name = payload.get("model_name", "")
        key = (run_id, model_name)

        try:
            if action == "register":
                # 增量查询：仅获取变更的单个模型
                config = await self._model_registry.get_by_key(run_id, model_name)
                if config:
                    await self._on_model_register(config)
                    logger.info(f"通知同步: 注册模型 {model_name} (run_id={run_id})")
                else:
                    # 模型未找到，降级到全量同步
                    logger.warning(
                        f"通知同步: register 事件但模型未在 DB 中找到: "
                        f"{model_name} (run_id={run_id})，降级到全量同步"
                    )
                    await self._full_sync_from_db()

            elif action == "unregister":
                await self._on_model_unregister(key)
                logger.info(f"通知同步: 删除模型 {model_name} (run_id={run_id})")

            else:
                logger.warning(f"通知同步: 未知 action '{action}'，忽略")

        except Exception as e:
            logger.error(
                f"通知同步处理失败 (action={action}, model={model_name}, "
                f"run_id={run_id})，降级到全量同步: {e}",
                exc_info=True
            )
            await self._full_sync_from_db()

    async def _full_sync_from_db(self):
        """从数据库全量同步，通过回调通知调用方"""
        try:
            db_models = await self._model_registry.get_all()
            await self._on_full_sync(db_models)
        except Exception as e:
            logger.error(f"从数据库同步动态模型失败: {e}", exc_info=True)
            raise
