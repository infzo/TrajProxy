"""
ProxyCore FastAPI路由

处理LLM请求转发相关路由
"""

from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from typing import Dict, Any, Optional
from traj_proxy.utils.logger import get_logger
from traj_proxy.proxy_core.worker import get_processor_manager
from traj_proxy.proxy_core.processor_manager import (
    RegisterModelRequest,
    RegisterModelResponse,
    DeleteModelResponse,
    ListModelsResponse,
    ModelInfo
)
from traj_proxy.exceptions import DatabaseError
import uuid

router = APIRouter()
admin_router = APIRouter()
logger = get_logger(__name__)


@router.post("/chat/completions")
async def chat_completions(request: Request, background_tasks: BackgroundTasks):
    """
    处理聊天补全请求

    参数:
        request: FastAPI请求对象
        background_tasks: 后台任务

    返回:
        处理后的响应
    """
    try:
        # 获取请求体
        body = await request.json()

        # 提取请求参数
        messages = body.get("messages", [])
        model = body.get("model")
        session_id = request.headers.get("x-session-id")

        # 其他请求参数
        request_params = {}
        for key in ["max_tokens", "temperature", "top_p", "presence_penalty", "frequency_penalty"]:
            if key in body:
                request_params[key] = body[key]

        # 生成 request_id
        request_id = str(uuid.uuid4())

        logger.info(f"处理聊天补全请求: model={model}, messages={len(messages)}, session_id={session_id}, headers={request.headers}")

        # 获取 ProcessorManager 实例
        processor_manager = get_processor_manager()

        # 根据 model 获取对应的 processor
        try:
            processor = processor_manager.get_processor_or_raise(model)
        except ValueError as e:
            logger.warning(f"模型未注册: {model}, 错误: {str(e)}")
            raise HTTPException(
                status_code=404,
                detail=f"模型 '{model}' 未注册"
            )

        # 处理请求
        context = await processor.process_request(
            messages=messages,
            request_id=request_id,
            session_id=session_id,
            **request_params
        )

        # 返回 OpenAI 格式响应
        return context.response

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"聊天补全请求处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_models():
    """
    列出可用模型

    返回:
        可用模型列表
    """
    # 获取 ProcessorManager 实例
    processor_manager = get_processor_manager()
    model_names = processor_manager.list_models()

    # 构建 OpenAI 格式的响应
    data = [
        ModelInfo(id=model_name)
        for model_name in model_names
    ]

    return ListModelsResponse(data=data)


# ========== 管理接口 ==========

@admin_router.post("/register", response_model=RegisterModelResponse)
async def register_model(request: RegisterModelRequest):
    """
    注册新模型（模型会自动同步到所有 Worker）

    参数:
        request: 注册模型请求

    返回:
        注册结果
    """
    try:
        processor_manager = get_processor_manager()

        # 注册模型（会同步持久化到数据库）
        processor = await processor_manager.register_processor(
            model_name=request.model_name,
            url=request.url,
            api_key=request.api_key,
            tokenizer_path=request.tokenizer_path,
            token_in_token_out=request.token_in_token_out,
            persist_to_db=True
        )

        logger.info(f"注册模型成功: {request.model_name}")

        return RegisterModelResponse(
            status="success",
            model_name=request.model_name,
            detail={
                "model": processor.model,
                "tokenizer_path": processor.tokenizer_path,
                "token_in_token_out": processor.token_in_token_out,
                "sync_info": "模型已持久化到数据库，其他 Worker 将在 30 秒内自动同步"
            }
        )

    except ValueError as e:
        # 模型已存在
        logger.warning(f"注册模型失败: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except DatabaseError as e:
        logger.error(f"数据库错误: {str(e)}")
        raise HTTPException(status_code=503, detail=f"数据库不可用: {str(e)}")
    except Exception as e:
        logger.error(f"注册模型异常: {str(e)}")
        raise HTTPException(status_code=500, detail=f"注册模型失败: {str(e)}")


@admin_router.delete("/{model_name}", response_model=DeleteModelResponse)
async def delete_model(model_name: str):
    """
    删除已注册的模型（会自动从所有 Worker 中删除）

    参数:
        model_name: 模型名称

    返回:
        删除结果
    """
    try:
        processor_manager = get_processor_manager()

        deleted = await processor_manager.unregister_processor(model_name, persist_to_db=True)

        if not deleted:
            raise HTTPException(status_code=404, detail=f"模型 '{model_name}' 不存在")

        logger.info(f"删除模型成功: {model_name}")

        return DeleteModelResponse(
            status="success",
            model_name=model_name,
            deleted=True
        )

    except HTTPException:
        raise
    except DatabaseError as e:
        logger.error(f"数据库错误: {str(e)}")
        raise HTTPException(status_code=503, detail=f"数据库不可用: {str(e)}")
    except Exception as e:
        logger.error(f"删除模型异常: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除模型失败: {str(e)}")


@admin_router.get("")
async def list_admin_models():
    """
    列出所有已注册模型（包含详细信息）

    返回:
        所有模型的详细信息列表
    """
    try:
        processor_manager = get_processor_manager()
        models_info = processor_manager.get_all_processors_info()

        return {
            "status": "success",
            "count": len(models_info),
            "models": models_info
        }

    except Exception as e:
        logger.error(f"列出模型异常: {str(e)}")
        raise HTTPException(status_code=500, detail=f"列出模型失败: {str(e)}")
