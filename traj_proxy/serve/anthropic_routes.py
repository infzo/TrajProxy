"""
Anthropic /v1/messages 路由定义

临时特性：直接转发 Anthropic /v1/messages 请求。
核心原则：不修改 OpenAI 路径代码，所有 Anthropic 逻辑隔离在本模块。

对外接口：
- POST /v1/messages（及 /s/{run_id}/{session_id}/v1/messages 等路径变体）

错误响应格式遵循 Anthropic 规范：
{"type": "error", "error": {"type": <error_type>, "message": <message>}}
"""

import asyncio
import json
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from traj_proxy.serve.dependencies import get_processor_manager
from traj_proxy.utils.config import get_semaphore_acquire_timeout
from traj_proxy.utils.logger import get_logger
from traj_proxy.utils.validators import normalize_run_id, validate_model_for_inference

# 复用 OpenAI 路由的 header 提取逻辑（HEADER_BLACKLIST 已包含 x-api-key 放行规则）
from traj_proxy.serve.routes import _extract_forward_headers, _extract_actual_model, _extract_run_id

logger = get_logger(__name__)

# 路由定义
anthropic_router = APIRouter()


def build_anthropic_error_body(request_id: str, error: Exception) -> dict:
    """构建 Anthropic 格式错误响应体

    将内部异常转换为 Anthropic 规范的错误结构：
    {"type": "error", "error": {"type": <error_type>, "message": <message>}}

    Args:
        request_id: 请求 ID（用于日志关联，不写入响应体）
        error: 捕获的异常

    Returns:
        Anthropic 格式错误响应字典
    """
    if isinstance(error, HTTPException):
        type_map = {
            400: "invalid_request_error",
            401: "authentication_error",
            403: "permission_error",
            404: "not_found_error",
            409: "invalid_request_error",
            422: "invalid_request_error",
            429: "rate_limit_error",
            500: "api_error",
            502: "api_error",
            503: "overloaded_error",
            504: "api_error",
            529: "overloaded_error",
        }
        error_type = type_map.get(error.status_code, "api_error")
        detail = error.detail
        if isinstance(detail, dict) and detail.get("error"):
            # 已是 Anthropic 格式（流式路径透传）
            inner = detail.get("error", {})
            return {
                "type": "error",
                "error": {
                    "type": inner.get("type", error_type),
                    "message": inner.get("message", str(detail)),
                },
            }
        message = str(detail)
    else:
        error_type = "api_error"
        message = str(error)

    return {"type": "error", "error": {"type": error_type, "message": message}}


@anthropic_router.post("/messages")
async def anthropic_messages(
    request: Request,
    background_tasks: BackgroundTasks,
    run_id: Optional[str] = None,
    session_id: Optional[str] = None
):
    """处理 Anthropic /v1/messages 请求（支持流式和非流式）

    流程与 chat_completions 一致，但适配 Anthropic 差异：
    - 认证透传 x-api-key（HEADER_BLACKLIST 已放行）
    - max_tokens 必填校验（缺失 → 400 Anthropic 格式）
    - 流式响应不附加 data: [DONE]，直接透传原始 SSE
    - 所有错误响应使用 Anthropic 格式

    参数:
        request: FastAPI 请求对象
        background_tasks: 后台任务
        run_id: 路径参数中的运行ID（可选）
        session_id: 路径参数中的会话ID（可选）

    返回:
        Anthropic Message 响应（JSON 或 SSE 流）
    """
    request_id = getattr(request.state, "request_id", None) or str(uuid.uuid4())

    # 并发限流：获取信号量，超时返回 429（Anthropic 格式）
    semaphore = getattr(request.app.state, "request_semaphore", None)
    acquired = False
    streaming_handled = False  # 标记流式路径是否已接管信号量管理
    if semaphore:
        t_wait = time.perf_counter()
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=get_semaphore_acquire_timeout())
            acquired = True
        except asyncio.TimeoutError:
            # 信号量获取超时 → 并发限流拒绝
            wait_ms = (time.perf_counter() - t_wait) * 1000
            rejected_run_id = run_id or request.headers.get("x-run-id") or ""
            from traj_proxy.observability.event_bus import emit
            from traj_proxy.observability.events import EVENT_CONCURRENCY_REJECTED
            emit(EVENT_CONCURRENCY_REJECTED, model="unknown",
                 run_id=rejected_run_id, wait_duration_ms=wait_ms)
            max_conc = getattr(request.app.state, "max_concurrent_requests", "?")
            logger.warning(
                f"并发限流拒绝: max_concurrent={max_conc} 已达上限"
            )
            error_body = build_anthropic_error_body(
                request_id, HTTPException(status_code=429, detail="服务繁忙，请稍后重试")
            )
            return JSONResponse(
                status_code=429,
                content=error_body,
                headers={"Retry-After": "3"},
            )

    try:
        # 获取请求体
        body = await request.json()

        # 提取请求参数
        messages = body.get("messages", [])
        model = body.get("model")

        # 从 header 获取 run_id 和 session_id
        x_run_id = request.headers.get("x-run-id")
        x_session_id = request.headers.get("x-session-id")
        x_sandbox_traj_id = request.headers.get("x-sandbox-traj-id")

        # session_id 优先级：路径参数 > x-session-id > x-sandbox-traj-id
        final_session_id = session_id or x_session_id or x_sandbox_traj_id
        if final_session_id == "":
            final_session_id = None

        stream = body.get("stream", False)

        # 校验 model 参数格式
        valid, msg, _ = validate_model_for_inference(model or "")
        if not valid:
            return JSONResponse(
                status_code=422,
                content=build_anthropic_error_body(
                    request_id, HTTPException(status_code=422, detail=msg)
                ),
            )

        # 提取 run_id（优先级：路径参数 > x-run-id header > model 参数）
        final_run_id = _extract_run_id(model, x_run_id, run_id)

        # 设置 ContextVar，使后续日志自动携带 run_id
        from traj_proxy.observability.request_context import set_run_id
        set_run_id(final_run_id or "")

        # 提取实际 model_name
        actual_model = _extract_actual_model(model)

        # 可观测性：信号量等待结果 emit
        if semaphore and acquired:
            wait_ms = (time.perf_counter() - t_wait) * 1000
            from traj_proxy.observability.event_bus import emit
            from traj_proxy.observability.events import EVENT_SEMAPHORE_ACQUIRED
            emit(EVENT_SEMAPHORE_ACQUIRED, wait_duration_ms=wait_ms, model=actual_model)

        # 提取需要转发到推理服务的 header（黑名单模式，x-api-key 已放行）
        # Anthropic 路径透传完整 body，不需要单独拆解 request_params
        forward_headers = _extract_forward_headers(request)

        logger.info(
            f"处理 Anthropic messages 请求: model={actual_model}, "
            f"run_id={final_run_id}, session_id={final_session_id}, "
            f"stream={stream}, messages={len(messages)}"
        )

        # 校验 max_tokens 必填（Anthropic 协议要求）
        if body.get("max_tokens") is None:
            return JSONResponse(
                status_code=400,
                content=build_anthropic_error_body(
                    request_id,
                    HTTPException(
                        status_code=400,
                        detail="max_tokens is required for /v1/messages"
                    ),
                ),
            )

        # 获取 ProcessorManager 实例
        processor_manager = get_processor_manager(request)

        # 根据 run_id 和 model_name 获取对应的 processor（懒加载）
        processor = await processor_manager.get_processor_async(final_run_id, actual_model)

        if processor is None:
            # 本地未找到模型，尝试从数据库查询（回退机制）
            logger.info(
                f"本地未找到模型，尝试 DB 回退查询: "
                f"model={actual_model}, run_id={final_run_id}"
            )
            processor = await processor_manager.try_get_or_sync_from_db(
                final_run_id, actual_model
            )

        if processor is None:
            logger.warning(f"模型未注册: model={actual_model}, run_id={final_run_id}")
            return JSONResponse(
                status_code=404,
                content=build_anthropic_error_body(
                    request_id,
                    HTTPException(
                        status_code=404,
                        detail=f"模型 '{actual_model}' 未注册 (run_id={final_run_id})"
                    ),
                ),
            )

        # 根据是否流式选择处理方式
        if stream:
            # 流式处理 - 使用 Processor.process_anthropic_stream
            # 上下文容器，流式完成后用于后台存储
            context_holder: Dict[str, Any] = {}
            streaming_handled = True  # 流式路径接管信号量管理

            async def generate_anthropic_stream():
                """Anthropic 流式生成器：透传原始 SSE，异常时发送 error 事件；信号量在流结束后释放"""
                try:
                    async for raw_sse in processor.process_anthropic_stream(
                        body=body,
                        request_id=request_id,
                        session_id=final_session_id,
                        run_id=final_run_id,
                        context_holder=context_holder,
                        forward_headers=forward_headers,
                    ):
                        yield raw_sse
                    # 不附加 data: [DONE]（Anthropic 协议以 message_stop 事件结束）
                except Exception as stream_err:
                    logger.exception(f"Anthropic 流式处理异常: {str(stream_err)}")
                    error_body = build_anthropic_error_body(request_id, stream_err)
                    yield f"event: error\ndata: {json.dumps(error_body, ensure_ascii=False)}\n\n"
                finally:
                    # 流式信号量在此处释放，确保在整个 SSE 流生命周期内持有
                    if acquired:
                        semaphore.release()

            return StreamingResponse(
                generate_anthropic_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
                }
            )
        else:
            # 非流式处理
            context = await processor.process_anthropic_request(
                body=body,
                request_id=request_id,
                session_id=final_session_id,
                run_id=final_run_id,
                forward_headers=forward_headers,
            )

            # 返回 Anthropic Message 响应（原样透传 raw_response）
            return context.raw_response

    except Exception as e:
        logger.exception(f"Anthropic messages 请求处理失败: {str(e)}")
        error_body = build_anthropic_error_body(request_id, e)
        status = getattr(e, 'status_code', 500)
        return JSONResponse(status_code=status, content=error_body)
    finally:
        # 仅在非流式路径（或流式创建前异常）释放信号量
        if acquired and not streaming_handled:
            semaphore.release()
