"""
TranscriptProvider Worker实现

处理文本转录相关请求
"""

from traj_proxy.workers.base import Worker
from traj_proxy.store.database import DatabaseManager
from traj_proxy.transcript_provider.provider import TranscriptProvider


# 全局TranscriptProvider实例，用于依赖注入
_provider: TranscriptProvider = None


def get_provider() -> TranscriptProvider:
    """获取TranscriptProvider实例，用于依赖注入"""
    global _provider
    if _provider is None:
        raise RuntimeError("TranscriptProvider未初始化")
    return _provider


class TranscriptProviderWorker(Worker):
    """转录提供者Worker，处理文本转录相关请求"""

    def __init__(self, worker_id: int, port: int, db_url: str):
        """
        初始化TranscriptProviderWorker

        参数:
            worker_id: Worker唯一标识
            port: 监听端口
            db_url: 数据库连接URL
        """
        self.db_url = db_url
        self.db_manager = None
        self.provider = None
        super().__init__(worker_id, port)

    def get_worker_name(self) -> str:
        """
        返回Worker名称

        返回:
            Worker的名称字符串
        """
        return f"TranscriptProviderWorker-{self.worker_id}"

    async def initialize(self):
        """
        初始化数据库连接池和TranscriptProvider
        """
        global _provider

        # 初始化数据库管理器
        self.db_manager = DatabaseManager(self.db_url)
        await self.db_manager.initialize()

        # 初始化TranscriptProvider
        self.provider = TranscriptProvider(self.db_manager)
        _provider = self.provider

    async def shutdown(self):
        """
        关闭资源
        """
        if self.db_manager:
            await self.db_manager.close()

    def _setup_routes(self):
        """
        设置路由

        包含健康检查和转录处理路由
        """
        from traj_proxy.transcript_provider.routes import router
        self.app.include_router(router, prefix="/transcript", tags=["Transcript"])
