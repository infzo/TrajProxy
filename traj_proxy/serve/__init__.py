"""
Serve 模块 - HTTP 路由和依赖注入

提供所有 FastAPI 路由定义和依赖注入。

注意：为避免循环导入，路由器应直接从 traj_proxy.serve.routes 导入，
而不是从此模块导入。
"""

# 不在此处导入 routes，避免循环导入
# from traj_proxy.serve.routes import chat_router, model_router, transcript_router

__all__ = [
    "chat_router",
    "model_router",
    "transcript_router",
]


def __getattr__(name: str):
    """延迟导入路由器，避免循环导入"""
    if name == "chat_router":
        from traj_proxy.serve.routes import chat_router
        return chat_router
    elif name == "model_router":
        from traj_proxy.serve.routes import model_router
        return model_router
    elif name == "transcript_router":
        from traj_proxy.serve.routes import transcript_router
        return transcript_router
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
