"""
路由注册器

管理 Worker 的路由注册，提供解耦的路由组合方式
"""
from fastapi import FastAPI


class RouteRegistrar:
    """路由注册器 - 管理多个路由模块的注册"""

    def __init__(self, app: FastAPI):
        """
        初始化路由注册器

        参数:
            app: FastAPI 应用实例
        """
        self.app = app

    def register_proxy_routes(self):
        """注册 ProxyCore 相关路由"""
        from traj_proxy.proxy_core.routes import router as proxy_router, admin_router as admin_router

        # /v1/chat/completions - 无session_id的聊天
        self.app.include_router(proxy_router, prefix="/v1", tags=["OpenAI Chat"])
        # /s/{session_id}/v1/chat/completions - 带session_id的聊天
        self.app.include_router(proxy_router, prefix="/s/{session_id}/v1", tags=["OpenAI Chat (Path-based)"])
        # /models/* - 模型管理
        self.app.include_router(admin_router, prefix="/models", tags=["Admin"])

    def register_transcript_routes(self):
        """注册 TranscriptProvider 相关路由（只注册 /trajectory）"""
        from traj_proxy.transcript_provider.routes import get_trajectory

        # 直接注册 /trajectory 路由，不使用 /transcript 前缀
        @self.app.get("/trajectory", tags=["Transcript"])
        async def trajectory_endpoint(session_id: str, limit: int = 10000):
            return await get_trajectory(session_id, limit)

    def register_health_route(self):
        """注册健康检查路由（只保留一个）"""
        @self.app.get("/health", tags=["Health"])
        async def health():
            return {"status": "ok"}

    def register_all(self):
        """注册所有路由"""
        self.register_proxy_routes()
        self.register_transcript_routes()
        self.register_health_route()
