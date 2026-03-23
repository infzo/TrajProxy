"""
Processor - 主处理器

协调整个 LLM 请求处理流程的核心类。
"""

from typing import Optional, Dict, Any
from datetime import datetime
import traceback

from traj_proxy.utils.logger import get_logger

# 配置日志
logger = get_logger(__name__)

from traj_proxy.proxy_core.context import ProcessContext
from traj_proxy.proxy_core.prompt_builder import PromptBuilder
from traj_proxy.proxy_core.token_builder import TokenBuilder
from traj_proxy.exceptions import DatabaseError


class Processor:
    """主处理器 - 协调整个 LLM 请求处理流程

    处理流程：
    1. Message → PromptText 转换（使用 tokenizer.apply_chat_template）
    2. (可选) PromptText → TokenIds 转换（token_in_token_out=True 时）
    3. 向 Infer 发送请求
    4. Response → ResponseText
    5. 构建完整对话（请求+响应）用于前缀匹配
    6. 统计信息
    7. 构建 OpenAI Response 响应（支持 tool_calls）
    8. 存储完整请求到数据库
    """

    def __init__(
        self,
        model: str,
        tokenizer_path: str,
        db_manager=None,
        infer_client=None,
        config: Optional[Dict[str, Any]] = None
    ):
        """初始化 Processor

        Args:
            model: 模型名称
            tokenizer_path: Tokenizer 路径
            db_manager: 数据库管理器，用于存储轨迹记录
            infer_client: Infer 服务客户端
            config: 完整配置字典（由worker传入）
        """
        self.model = model
        self.tokenizer_path = tokenizer_path
        self.db_manager = db_manager
        self.infer_client = infer_client

        # 从传入的配置读取 token_in_token_out
        if config:
            self.token_in_token_out = config.get("token_in_token_out", False)
        else:
            self.token_in_token_out = False

        # 初始化构建器
        self.prompt_builder = PromptBuilder(model, tokenizer_path)
        # 只有在 token_in_token_out 模式下才需要 TokenBuilder
        if self.token_in_token_out:
            self.token_builder = TokenBuilder(model, tokenizer_path, db_manager)
        else:
            self.token_builder = None

    async def _process_with_tokens(
        self,
        context: ProcessContext
    ) -> ProcessContext:
        """使用 Token-in-Token-out 模式处理请求

        包含 TokenBuilder 的完整处理流程。

        Args:
            context: 处理上下文

        Returns:
            处理后的上下文
        """
        # 2. PromptText → TokenIds 转换（使用前缀匹配完整对话）
        context.token_ids = await self.token_builder.encode_text(
            context.prompt_text, context
        )
        context.prompt_tokens = len(context.token_ids)
        logger.info(f"[{context.unique_id}] TokenIds 转换完成: prompt_tokens={context.prompt_tokens}")

        # 3. 向 Infer 发送 token ids
        context.infer_request_body = {
            "prompt": context.token_ids,
            "model": self.model,
            **context.request_params
        }
        logger.info(f"[{context.unique_id}] 发送 Infer 请求（Token-in-Token-out 模式）")
        context.infer_response = await self.infer_client.send_completion(
            prompt=context.token_ids,
            model=self.model,
            **context.request_params
        )

        # 4. Response → ResponseText（从 token ids 解码）
        if "token_ids" in context.infer_response:
            # 扩展格式：infer 服务直接返回 token_ids
            response_token_ids = context.infer_response["token_ids"]
            context.response_ids = response_token_ids
            context.response_text = await self.token_builder.decode_tokens(
                response_token_ids, context
            )
        elif "choices" in context.infer_response and context.infer_response["choices"]:
            # 标准格式：choices[0].text
            # 在 token-in-token-out 模式下，text 是 token ID 字符串，需要解析
            choice = context.infer_response["choices"][0]
            raw_text = choice.get("text", "")
            # 尝试解析为 token ID 列表
            try:
                # 尝试按空格分割并转换为整数
                if raw_text.strip():
                    context.response_ids = [int(tid) for tid in raw_text.strip().split()]
                    # 使用 tokenizer 解码为文本
                    context.response_text = await self.token_builder.decode_tokens(
                        context.response_ids, context
                    )
                else:
                    context.response_text = ""
                    context.response_ids = []
            except (ValueError, AttributeError):
                # 解析失败，说明返回的是普通文本
                context.response_text = raw_text
                context.response_ids = None

        logger.info(f"[{context.unique_id}] Infer 请求完成")
        logger.info(f"[{context.unique_id}] ResponseText 转换完成: response_length={len(context.response_text)}")

        return context

    async def _process_with_text(
        self,
        context: ProcessContext
    ) -> ProcessContext:
        """使用 Text 模式处理请求

        不经过 TokenBuilder，直接使用 prompt text。

        Args:
            context: 处理上下文

        Returns:
            处理后的上下文
        """
        # 3. 向 Infer 发送 prompt text
        context.infer_request_body = {
            "prompt": context.prompt_text,
            "model": self.model,
            **context.request_params
        }
        logger.info(f"[{context.unique_id}] 发送 Infer 请求（Text 模式）")
        context.infer_response = await self.infer_client.send_completion(
            prompt=context.prompt_text,
            model=self.model,
            **context.request_params
        )

        # 4. Response → ResponseText
        if "choices" in context.infer_response and context.infer_response["choices"]:
            choice = context.infer_response["choices"][0]
            context.response_text = choice.get("text", "")
        else:
            context.response_text = ""

        logger.info(f"[{context.unique_id}] Infer 请求完成")
        logger.info(f"[{context.unique_id}] ResponseText 转换完成: response_length={len(context.response_text)}")

        return context

    async def process_request(
        self,
        messages: list,
        request_id: str,
        session_id: Optional[str] = None,
        **request_params
    ) -> ProcessContext:
        """处理完整的 LLM 请求

        Args:
            messages: OpenAI 格式的消息列表
            request_id: 请求 ID
            session_id: 会话 ID（格式: app_id#sample_id#task_id）
            **request_params: 请求参数（如 max_tokens, temperature 等）

        Returns:
            处理上下文，包含完整的处理结果和响应

        Raises:
            各种异常，包括 DatabaseError、InferServiceError 等
        """
        # 构建 unique_id
        unique_id = f"{session_id}#{request_id}" if session_id else request_id

        # 初始化上下文
        context = ProcessContext(
            request_id=request_id,
            model=self.model,
            messages=messages,
            request_params=request_params,
            session_id=session_id,
            unique_id=unique_id
        )
        context.start_time = datetime.now()

        logger.info(f"[{unique_id}] 开始处理请求: model={self.model}, messages_count={len(messages)}")

        try:
            # 1. Message → PromptText 转换（使用 tokenizer.apply_chat_template）
            context.prompt_text = await self.prompt_builder.build_prompt_text(
                messages, context
            )
            logger.info(f"[{unique_id}] PromptText 转换完成: prompt_length={len(context.prompt_text)}")

            # 2+3+4. 根据 token_in_token_out 模式选择处理方式
            if self.token_in_token_out:
                context = await self._process_with_tokens(context)
            else:
                context = await self._process_with_text(context)

            # 5. 构建完整对话（请求+响应）用于前缀匹配
            context.full_conversation_text = context.prompt_text + context.response_text

            # 构建完整对话 token_ids（仅 token_in_token_out 模式）
            if self.token_in_token_out:
                # Token-in-Token-out 模式：response_ids 已经从 infer_response 获取
                if context.response_ids:
                    context.full_conversation_token_ids = context.token_ids + context.response_ids
                else:
                    context.full_conversation_token_ids = context.token_ids

            logger.info(f"[{unique_id}] 完整对话构建完成: total_tokens={len(context.full_conversation_token_ids) if context.full_conversation_token_ids else len(context.full_conversation_text)}")

            # 6. 统计信息
            if "usage" in context.infer_response:
                usage = context.infer_response["usage"]
                context.completion_tokens = usage.get("completion_tokens", 0)
                context.total_tokens = usage.get("total_tokens", 0)
            else:
                # 从响应估算
                if self.token_in_token_out:
                    context.completion_tokens = len(context.response_ids or [])
                    context.prompt_tokens = len(context.token_ids or [])
                else:
                    context.completion_tokens = 0
                    context.prompt_tokens = 0
                context.total_tokens = context.prompt_tokens + context.completion_tokens

            logger.info(f"[{unique_id}] 统计信息: completion_tokens={context.completion_tokens}, total_tokens={context.total_tokens}")

            context.end_time = datetime.now()
            context.processing_duration_ms = (
                context.end_time - context.start_time
            ).total_seconds() * 1000

            # 7. 构建 OpenAI Response 响应（支持 tool_calls）
            context.response = self.prompt_builder.build_openai_response(
                context.response_text, context
            )
            logger.info(f"[{unique_id}] OpenAI Response 构建完成: has_response={context.response is not None}")

            # 8. 存储完整请求到数据库
            try:
                await self.db_manager.insert_trajectory(context, self.tokenizer_path)
            except DatabaseError as e:
                # 存储失败不影响主流程，记录错误即可
                context.error = f"存储轨迹失败: {str(e)}"
                logger.error(f"[{unique_id}] 存储轨迹失败: {str(e)}")
            else:
                logger.info(f"[{unique_id}] 轨迹存储成功")

            logger.info(f"[{unique_id}] 请求处理完成: duration_ms={context.processing_duration_ms:.2f}, context={context}")
            return context

        except Exception as e:
            context.error = str(e)
            context.error_traceback = traceback.format_exc()
            context.end_time = datetime.now()
            logger.error(f"[{unique_id}] 处理请求时发生异常: {str(e)}\n{traceback.format_exc()}")

            # 即使出错也尝试存储到数据库
            try:
                await self.db_manager.insert_trajectory(context, self.tokenizer_path)
            except DatabaseError:
                # 忽略存储错误，避免掩盖原始错误
                pass
            raise
