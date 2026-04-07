"""
MessageConverter - 消息转换器

负责将 OpenAI Messages 转换为模型特定的 PromptText。
"""

from typing import List, Dict, Any
import json
import copy

from traj_proxy.proxy_core.converters.base import BaseConverter
from traj_proxy.proxy_core.context import ProcessContext
from traj_proxy.utils.logger import get_logger

logger = get_logger(__name__)


class MessageConverter(BaseConverter):
    """消息转换器 - 将 OpenAI Messages 转换为 PromptText

    使用 transformers 的 AutoTokenizer.apply_chat_template() 方法
    将 OpenAI 格式的 messages 转换为模型特定的 prompt 格式。
    """

    def __init__(self, tokenizer):
        """初始化 MessageConverter

        Args:
            tokenizer: transformers.PreTrainedTokenizerBase 实例
        """
        self.tokenizer = tokenizer

    async def convert(
        self,
        messages: List[Dict[str, Any]],
        context: ProcessContext
    ) -> str:
        """将 OpenAI Messages 转换为 PromptText

        使用 tokenizer.apply_chat_template() 方法，
        将 OpenAI 格式的 messages 转换为模型特定的 prompt 格式。

        支持 tools、documents、tool_choice 等参数传递给 chat template。

        Args:
            messages: OpenAI Message 格式的消息列表
            context: 处理上下文

        Returns:
            模型特定的 prompt 文本
        """
        # 预处理消息，解决格式兼容性问题
        processed_messages = self._preprocess_messages(messages)

        # 构建 apply_chat_template 参数
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True
        }

        # 从 request_params 提取 tools 相关参数
        tools = context.request_params.get("tools")
        if tools:
            template_kwargs["tools"] = tools

        # 从 request_params 提取 documents 参数（RAG 场景）
        documents = context.request_params.get("documents")
        if documents:
            template_kwargs["documents"] = documents

        # 从 request_params 提取 tool_choice 参数
        # 简单传递策略：依赖模型的 chat template 处理
        tool_choice = context.request_params.get("tool_choice")
        if tool_choice:
            template_kwargs["tool_choice"] = tool_choice

        return self.tokenizer.apply_chat_template(
            processed_messages,
            **template_kwargs
        )

    def _preprocess_messages(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """预处理消息列表

        解决 OpenAI 格式与 tokenizer chat template 的兼容性问题：
        1. 将 tool_call.function.arguments 从 JSON 字符串解析为字典
           OpenAI API 中 arguments 是 JSON 字符串，但某些模型的 chat template
           期望它已经是字典对象（如 Qwen 模板会调用 |items 过滤器）

        Args:
            messages: 原始消息列表

        Returns:
            预处理后的消息列表
        """
        processed_messages = []

        for message in messages:
            msg = copy.deepcopy(message)

            # 处理 assistant 消息中的 tool_calls
            if msg.get('role') == 'assistant' and msg.get('tool_calls'):
                for tool_call in msg['tool_calls']:
                    if 'function' in tool_call and 'arguments' in tool_call['function']:
                        args = tool_call['function']['arguments']
                        # 如果 arguments 是 JSON 字符串，解析为字典
                        if isinstance(args, str):
                            try:
                                tool_call['function']['arguments'] = json.loads(args)
                            except json.JSONDecodeError:
                                # 解析失败则保持原样
                                pass

            processed_messages.append(msg)

        return processed_messages
