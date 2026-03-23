"""
ProxyCore Worker实现

处理LLM请求转发
"""

from traj_proxy.workers.base import Worker
from traj_proxy.proxy_core.processor import Processor
from traj_proxy.proxy_core.infer_client import InferClient
from traj_proxy.proxy_core.processor_manager import ProcessorManager
from traj_proxy.store.database import DatabaseManager
from traj_proxy.utils.logger import get_logger
import yaml
import os

logger = get_logger(__name__)


# 全局 ProcessorManager 实例，用于依赖注入
_processor_manager: ProcessorManager = None


def get_processor_manager() -> ProcessorManager:
    """获取 ProcessorManager 实例，用于依赖注入"""
    global _processor_manager
    if _processor_manager is None:
        raise RuntimeError("ProcessorManager 未初始化")
    return _processor_manager


def load_config() -> dict:
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_model_config(proxy_core_config: dict) -> list:
    """
    解析模型配置，返回 models 列表

    参数:
        proxy_core_config: proxy_core_workers 配置字典

    返回:
        模型配置列表，每个元素包含: model_name, url, api_key, tokenizer_path, token_in_token_out
    """
    return proxy_core_config.get("models", [])


class ProxyCoreWorker(Worker):
    """代理核心Worker，处理LLM请求转发"""

    def __init__(self, worker_id: int, port: int, db_url: str):
        """
        初始化 ProxyCoreWorker

        参数:
            worker_id: Worker唯一标识
            port: 监听端口
            db_url: 数据库连接URL
        """
        self.db_url = db_url
        self.db_manager = None
        self.processor_manager = None
        super().__init__(worker_id, port)

    async def initialize(self):
        """
        初始化数据库连接池、ProcessorManager 和默认模型
        """
        global _processor_manager

        # 加载配置（统一在worker中加载）
        config = load_config()
        proxy_core_config = config.get("proxy_core_workers", {})

        # 初始化数据库管理器
        self.db_manager = DatabaseManager(self.db_url)
        await self.db_manager.initialize()

        # 创建 ProcessorManager（共享 db_manager）
        self.processor_manager = ProcessorManager(self.db_manager)
        _processor_manager = self.processor_manager

        # 优先从数据库加载模型
        try:
            await self.processor_manager._sync_from_db()
            logger.info("从数据库加载模型完成")
        except Exception as e:
            logger.error(f"从数据库加载模型失败: {e}")
            # 快速失败：如果数据库加载失败，不继续
            raise

        # 启动定时同步
        await self.processor_manager.start_sync()

        # 解析模型配置
        models_config = parse_model_config(proxy_core_config)

        # 配置文件中的模型作为默认值，如果数据库为空则初始化
        if models_config:
            # 检查数据库是否为空
            db_models = await self.processor_manager.model_registry.get_all_models()
            if not db_models:
                logger.info("数据库为空，初始化配置文件中的模型")
                registered_count = 0
                for model_config in models_config:
                    try:
                        await self.processor_manager.register_processor(
                            model_name=model_config.get("model_name"),
                            url=model_config.get("url"),
                            api_key=model_config.get("api_key"),
                            tokenizer_path=model_config.get("tokenizer_path"),
                            token_in_token_out=model_config.get("token_in_token_out", False)
                        )
                        registered_count += 1
                        logger.info(f"模型注册成功: {model_config.get('model_name')}")
                    except ValueError as e:
                        # 模型已存在，跳过
                        logger.warning(f"模型已存在，跳过注册: {model_config.get('model_name')}")
                    except Exception as e:
                        logger.error(f"模型注册失败: {model_config.get('model_name')}, 错误: {str(e)}")
                        raise

                logger.info(f"配置文件模型初始化完成: 成功注册 {registered_count}/{len(models_config)} 个模型")

        logger.info(f"ProxyCoreWorker 初始化完成，当前模型数: {len(self.processor_manager.processors)}")

    async def shutdown(self):
        """
        关闭资源
        """
        # 停止同步
        if self.processor_manager:
            try:
                await self.processor_manager.stop_sync()
            except Exception as e:
                logger.error(f"停止同步失败: {e}")

        if self.db_manager:
            await self.db_manager.close()

    def get_worker_name(self) -> str:
        """
        返回Worker名称

        返回:
            Worker的名称字符串
        """
        return f"ProxyCoreWorker-{self.worker_id}"

    def _setup_routes(self):
        """
        设置路由

        包含健康检查和聊天补全路由
        """
        from traj_proxy.proxy_core.routes import router, admin_router

        # OpenAI Chat 相关路由使用 /proxy/v1 前缀
        self.app.include_router(router, prefix="/proxy/v1", tags=["OpenAI Chat"])

        # 管理接口使用 /proxy/models 前缀
        self.app.include_router(admin_router, prefix="/proxy/models", tags=["Admin"])

        # 健康检查接口单独使用 /proxy 前缀
        @self.app.get("/proxy/health", tags=["Health"])
        async def health():
            return {"status": "ok"}
