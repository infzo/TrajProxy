"""
AnthropicDirectPipeline - Anthropic /v1/messages 直接转发管道

将 Anthropic /v1/messages 格式请求直接转发到推理服务，
不经过 token 编码/解码流程，也不经过 OpenAI 格式转换。

功能边界：
- 非流式：转发完整 body → 推理服务 → 原样返回 raw_response（Anthropic Message）
- 流式：转发完整 body → 推理服务 → 逐行透传 raw SSE，旁路解析累积用于轨迹存储

对外接口：
- process(messages, context) -> ProcessContext：非流式处理
- process_stream(messages, context) -> AsyncIterator[str]：流式处理（yield 原始 SSE 字符串）

依赖关系：
- InferClient（send_anthropic_messages / send_anthropic_messages_stream_raw）
- RequestRepository（轨迹存储，可选）
- ProcessContext（复用现有字段 + 动态属性累积 Anthropic 特有的 content blocks）
"""

import json
import time
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, TYPE_CHECKING

from traj_proxy.proxy_core.pipeline.base import BasePipeline
from traj_proxy.proxy_core.context import ProcessContext
from traj_proxy.utils.logger import get_logger

if TYPE_CHECKING:
    from traj_proxy.proxy_core.infer_client import InferClient
    from traj_proxy.store.request_repository import RequestRepository

logger = get_logger(__name__)


class AnthropicDirectPipeline(BasePipeline):
    """Anthropic /v1/messages 直接转发管道

    处理流程：
    raw_request(Anthropic) → infer_client.send_anthropic_messages → raw_response(Anthropic)

    流式处理：
    raw_request(Anthropic) → infer_client.send_anthropic_messages_stream_raw
    → 逐行透传 raw SSE + 旁路解析累积 → _finalize_anthropic_stream 构建 raw_response

    注意：
    Anthropic 特有的 content blocks 累积通过 context 动态属性存储
    （_anthropic_blocks / _anthropic_thinking），而非 pipeline 实例属性。
    因为 pipeline 实例在 Processor 中被多个并发请求共享，
    实例属性会导致并发污染；context 是 per-request 的，安全。
    """

    def _create_context(self, *args, **kwargs) -> ProcessContext:
        """创建处理上下文，标记管道模式与 API 协议格式"""
        ctx = super()._create_context(*args, **kwargs)
        ctx.pipeline_mode = "direct"
        ctx.api_format = "anthropic"
        return ctx

    # ==================== 非流式处理 ====================

    async def process(
        self,
        messages: list,
        context: ProcessContext
    ) -> ProcessContext:
        """处理 Anthropic /v1/messages 非流式请求

        Args:
            messages: Anthropic 格式的消息列表（仅用于日志计数）
            context: 处理上下文，raw_request 为完整 Anthropic 请求体

        Returns:
            处理后的上下文
        """
        logger.info(
            f"开始处理 Anthropic 请求（直接转发模式）: "
            f"model={self.model}, messages_count={len(messages)}"
        )

        try:
            body = context.raw_request or {}
            t0 = time.perf_counter()
            raw_response = await self.infer_client.send_anthropic_messages(
                body=body,
                model=self.model,
                extra_headers=context.forward_headers,
            )
            context.raw_response = raw_response
            context.inference_duration_ms = (time.perf_counter() - t0) * 1000

            # 提取响应文本
            context.response_text = self._extract_response_text(raw_response)

            # 提取 usage 信息（Anthropic 字段名：input_tokens / output_tokens）
            usage = raw_response.get("usage", {}) or {}
            context.prompt_tokens = usage.get("input_tokens", 0)
            context.completion_tokens = usage.get("output_tokens", 0)
            context.total_tokens = (
                (context.prompt_tokens or 0) + (context.completion_tokens or 0)
            )

            self._update_timing(context)
            logger.info(
                f"Anthropic 请求完成: "
                f"duration_ms={context.processing_duration_ms:.2f}, "
                f"inference_ms={context.inference_duration_ms:.2f}"
            )

            # 存储到数据库
            await self._store_trajectory(context, run_id=context.run_id)

            return context

        except Exception as e:
            self._handle_error(context, e)
            # 即使出错也尝试存储轨迹
            try:
                await self._store_trajectory(context, run_id=context.run_id)
            except Exception as store_err:
                from traj_proxy.observability.event_bus import emit
                from traj_proxy.observability.events import EVENT_TRAJECTORY_STORE_ERROR
                emit(
                    EVENT_TRAJECTORY_STORE_ERROR,
                    model=context.model,
                    error_type=type(store_err).__name__,
                    error_message=str(store_err)[:200],
                    run_id=context.run_id or "",
                )
            raise

    # ==================== 流式处理 ====================

    async def process_stream(
        self,
        messages: list,
        context: ProcessContext
    ) -> AsyncIterator[str]:
        """处理 Anthropic /v1/messages 流式请求

        逐行透传推理服务返回的原始 SSE 字符串，同时旁路解析累积用于轨迹存储。
        不向客户端附加 data: [DONE]（Anthropic 协议以 message_stop 事件结束）。

        Args:
            messages: Anthropic 格式的消息列表（仅用于日志计数）
            context: 处理上下文

        Yields:
            原始 SSE 字符串（原样透传给客户端）
        """
        logger.info(
            f"开始流式处理 Anthropic 请求（直接转发模式）: "
            f"model={self.model}, messages_count={len(messages)}"
        )

        try:
            body = context.raw_request or {}
            first_chunk_received = False
            infer_start_time = time.perf_counter()

            async for raw_sse in self.infer_client.send_anthropic_messages_stream_raw(
                body=body,
                model=self.model,
                extra_headers=context.forward_headers,
            ):
                # 记录 TTFT（首块时间）
                if not first_chunk_received:
                    context.ttft_ms = (time.perf_counter() - infer_start_time) * 1000
                    first_chunk_received = True

                # 旁路解析累积（不修改 raw_sse，原样透传）
                self._accumulate_anthropic_sse(context, raw_sse)

                context.stream_chunk_count += 1
                yield raw_sse

            # 记录推理总耗时
            context.inference_duration_ms = (time.perf_counter() - infer_start_time) * 1000

            # 流式结束后构建完整 raw_response 并存储轨迹
            await self._finalize_anthropic_stream(context)

        except Exception as e:
            self._handle_error(context, e)
            from traj_proxy.observability.event_bus import emit
            from traj_proxy.observability.events import EVENT_STREAM_CLIENT_DISCONNECT
            emit(
                EVENT_STREAM_CLIENT_DISCONNECT,
                model=context.model,
                chunk_count=context.stream_chunk_count,
                duration_ms=(
                    (datetime.now(timezone.utc) - context.start_time).total_seconds() * 1000
                    if context.start_time else 0
                ),
            )
            raise

    # ==================== SSE 旁路解析 ====================

    def _init_anthropic_accumulators(self, context: ProcessContext) -> None:
        """初始化 Anthropic 流式累积所需的动态属性（per-request，挂在 context 上）

        使用 context 动态属性而非 pipeline 实例属性，因为 pipeline 实例
        在 Processor 中被多个并发请求共享，实例属性会导致并发污染。
        """
        if not hasattr(context, "_anthropic_blocks"):
            setattr(context, "_anthropic_blocks", [])        # content blocks 累积列表
            setattr(context, "_anthropic_thinking", "")      # thinking 文本累积

    def _accumulate_anthropic_sse(
        self,
        context: ProcessContext,
        raw_sse: str
    ) -> None:
        """旁路解析单个 raw SSE 字符串，累积 Anthropic 响应数据

        Anthropic SSE 每个 data: 行的 JSON 都包含 type 字段，
        可独立解析，不依赖前序 event: 行，因此可逐行无状态处理。

        Args:
            context: 处理上下文
            raw_sse: 原始 SSE 字符串（单行，含末尾换行）
        """
        self._init_anthropic_accumulators(context)

        raw = raw_sse.rstrip("\n").rstrip("\r")
        if not raw.startswith("data:"):
            # 跳过 event: 行和空行（SSE 结构性内容）
            return

        # 提取 data: 后的 JSON 字符串
        json_str = raw[5:].lstrip()
        if not json_str:
            return
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return

        event_type = data.get("type")
        blocks: List[Optional[Dict[str, Any]]] = getattr(context, "_anthropic_blocks")

        if event_type == "message_start":
            # 提取 message 级元数据
            msg = data.get("message", {}) or {}
            if context.stream_response_metadata is None:
                context.stream_response_metadata = {}
            context.stream_response_metadata["id"] = msg.get("id")
            context.stream_response_metadata["model"] = msg.get("model")
            context.stream_role = msg.get("role", "assistant")
            usage = msg.get("usage", {}) or {}
            if usage.get("input_tokens") is not None:
                context.prompt_tokens = usage.get("input_tokens")

        elif event_type == "content_block_start":
            # 记录新 block 的 index 和 type
            idx = data.get("index", 0)
            block = data.get("content_block", {}) or {}
            while len(blocks) <= idx:
                blocks.append(None)
            blocks[idx] = {
                "type": block.get("type", "text"),
                "text": block.get("text", "") or "",
                "id": block.get("id"),
                "name": block.get("name"),
                "_partial_json": "",          # tool_use 的 input 增量累积
                "thinking": block.get("thinking", "") or "",
            }

        elif event_type == "content_block_delta":
            idx = data.get("index", 0)
            delta = data.get("delta", {}) or {}
            delta_type = delta.get("type")
            while len(blocks) <= idx:
                blocks.append({"type": "text", "text": "", "_partial_json": "", "thinking": ""})
            block = blocks[idx] or {}
            blocks[idx] = block

            if delta_type == "text_delta":
                text = delta.get("text", "") or ""
                block["text"] = (block.get("text") or "") + text
                context.stream_buffer_text += text
            elif delta_type == "input_json_delta":
                block["_partial_json"] = (
                    (block.get("_partial_json") or "") + delta.get("partial_json", "")
                )
            elif delta_type == "thinking_delta":
                thinking = delta.get("thinking", "") or ""
                block["thinking"] = (block.get("thinking") or "") + thinking
                setattr(
                    context, "_anthropic_thinking",
                    (getattr(context, "_anthropic_thinking", "") or "") + thinking,
                )

        elif event_type == "message_delta":
            # 结束原因与 output_tokens
            delta = data.get("delta", {}) or {}
            if delta.get("stop_reason") is not None:
                context.stream_finish_reason = delta.get("stop_reason")
            usage = data.get("usage", {}) or {}
            if usage.get("output_tokens") is not None:
                context.completion_tokens = usage.get("output_tokens")

        elif event_type == "message_stop":
            context.stream_finished = True

    async def _finalize_anthropic_stream(self, context: ProcessContext) -> None:
        """流式结束后构建完整 Anthropic Message 对象并存储轨迹

        Args:
            context: 处理上下文
        """
        self._init_anthropic_accumulators(context)

        # 响应文本（已由 text_delta 累积）
        context.response_text = context.stream_buffer_text

        # 构建 content blocks
        content_blocks: List[Dict[str, Any]] = []
        blocks: List[Optional[Dict[str, Any]]] = getattr(context, "_anthropic_blocks")
        for block in blocks:
            if not block:
                continue
            block_type = block.get("type", "text")
            if block_type == "text":
                content_blocks.append({
                    "type": "text",
                    "text": block.get("text", "") or "",
                })
            elif block_type == "tool_use":
                # 解析累积的 partial_json 为 input 对象
                partial = block.get("_partial_json", "") or ""
                input_obj: Any = {}
                if partial:
                    try:
                        input_obj = json.loads(partial)
                    except json.JSONDecodeError:
                        # 解析失败则保留原始字符串
                        input_obj = partial
                content_blocks.append({
                    "type": "tool_use",
                    "id": block.get("id", "") or "",
                    "name": block.get("name", "") or "",
                    "input": input_obj,
                })
            elif block_type == "thinking":
                content_blocks.append({
                    "type": "thinking",
                    "thinking": block.get("thinking", "") or "",
                })

        # 若未收到任何 content_block 事件，则用累积文本兜底
        if not content_blocks and context.response_text:
            content_blocks.append({
                "type": "text",
                "text": context.response_text,
            })

        # usage 映射（同非流式：input_tokens / output_tokens）
        prompt_tokens = context.prompt_tokens or 0
        completion_tokens = context.completion_tokens or 0
        # 若后端未返回 output_tokens，按字符数粗略估算
        if not completion_tokens and context.response_text:
            completion_tokens = len(context.response_text) // 4
        total_tokens = prompt_tokens + completion_tokens
        context.completion_tokens = completion_tokens
        context.total_tokens = total_tokens

        # 构建 message 级元数据
        meta = context.stream_response_metadata or {}
        context.raw_response = {
            "id": meta.get("id") or f"msg_{context.request_id}",
            "type": "message",
            "role": context.stream_role or "assistant",
            "content": content_blocks,
            "model": meta.get("model") or self.model,
            "stop_reason": context.stream_finish_reason or "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
            },
        }

        self._update_timing(context)

        ttft_str = f"{context.ttft_ms:.2f}" if context.ttft_ms else "N/A"
        inference_str = f"{context.inference_duration_ms:.2f}" if context.inference_duration_ms else "N/A"
        logger.info(
            f"Anthropic 流式处理完成: "
            f"chunks={context.stream_chunk_count}, "
            f"duration_ms={context.processing_duration_ms:.2f}, "
            f"ttft_ms={ttft_str}, inference_ms={inference_str}"
        )

        # 存储到数据库
        await self._store_trajectory(context, run_id=context.run_id)

    @staticmethod
    def _extract_response_text(raw_response: Dict[str, Any]) -> str:
        """从 Anthropic 非流式响应中提取纯文本

        拼接所有 type == "text" 的 content block 的 text 字段。

        Args:
            raw_response: Anthropic Message 响应体

        Returns:
            拼接后的文本
        """
        content = raw_response.get("content", []) or []
        return "".join(
            b.get("text", "") for b in content if b.get("type") == "text"
        )
