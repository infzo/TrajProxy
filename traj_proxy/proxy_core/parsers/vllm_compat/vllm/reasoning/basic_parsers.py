# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm/reasoning/basic_parsers.py

"""
基础推理解析器

提供 BaseThinkingReasoningParser 基类，用于使用思考标记的模型。
"""
from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.tokenizers import TokenizerLike

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import (
        ResponsesRequest,
    )
else:
    ChatCompletionRequest = Any
    ResponsesRequest = Any


class BaseThinkingReasoningParser(ReasoningParser):
    """
    使用思考标记的推理解析器的基类。

    此类为使用开始和结束标记来界定推理内容的解析器提供通用功能
    （例如 思索...最终答案, <seed:think>...</seed:think>）。

    子类必须通过抽象属性实现 start_token 和 end_token。
    """

    @property
    @abstractmethod
    def start_token(self) -> str:
        """开始推理内容的标记"""
        raise NotImplementedError

    @property
    @abstractmethod
    def end_token(self) -> str:
        """结束推理内容的标记"""
        raise NotImplementedError

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction."
            )

        if not self.start_token or not self.end_token:
            raise ValueError("start_token and end_token must be defined in subclasses")

        self.start_token_id = self.vocab.get(self.start_token)
        self.end_token_id = self.vocab.get(self.end_token)
        if self.start_token_id is None or self.end_token_id is None:
            raise RuntimeError(
                f"{self.__class__.__name__} reasoning parser could not locate "
                "think start/end tokens in the tokenizer!"
            )

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        start_token_id = self.start_token_id
        end_token_id = self.end_token_id

        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == start_token_id:
                return False
            if input_ids[i] == end_token_id:
                return True
        return False

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Sequence[int]
    ) -> bool:
        end_token_id = self.end_token_id
        return end_token_id in delta_ids

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        提取结束标记之后的内容
        """
        if self.end_token_id not in input_ids[:-1]:
            return []
        else:
            return input_ids[input_ids.index(self.end_token_id) + 1 :]

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """
        从增量消息中提取推理内容。
        处理流式输出，其中 previous + delta = current。
        使用 token IDs 进行更快的处理。
        """
        # 跳过单个特殊标记
        if len(delta_token_ids) == 1 and (
            delta_token_ids[0] in [self.start_token_id, self.end_token_id]
        ):
            return None

        # 检查开始标记是否存在于 previous 或 delta 中
        # 保持与不生成开始标记的模型的兼容性
        if self.start_token_id in previous_token_ids:
            if self.end_token_id in delta_token_ids:
                # 开始标记在 previous 中，结束标记在 delta 中
                # 提取推理内容
                end_index = delta_text.find(self.end_token)
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token) :]
                return DeltaMessage(
                    reasoning=reasoning, content=content if content else None
                )
            elif self.end_token_id in previous_token_ids:
                # 开始标记在 previous 中，结束标记在 previous 中
                # 推理内容继续
                return DeltaMessage(content=delta_text)
            else:
                # 开始标记在 previous 中，previous 或 delta 中没有结束标记
                # 推理内容继续
                return DeltaMessage(reasoning=delta_text)
        elif self.start_token_id in delta_token_ids:
            if self.end_token_id in delta_token_ids:
                # 开始标记在 delta 中，结束标记在 delta 中
                # 提取推理内容
                start_index = delta_text.find(self.start_token)
                end_index = delta_text.find(self.end_token)
                reasoning = delta_text[start_index + len(self.start_token) : end_index]
                content = delta_text[end_index + len(self.end_token) :]
                return DeltaMessage(
                    reasoning=reasoning, content=content if content else None
                )
            else:
                # 开始标记在 delta 中，delta 中没有结束标记
                # 推理内容继续
                return DeltaMessage(reasoning=delta_text)
        else:
            # 未找到思考开始标记
            return DeltaMessage(content=delta_text)

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]:
        """
        从模型输出中提取推理内容。

        这是适用于大多数模型的基础实现。
        子类可以覆盖此方法以实现特定行为。
        """
        # 检查开始标记是否存在于模型输出中，如果存在则移除
        model_output_parts = model_output.partition(self.start_token)
        model_output = (
            model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
        )

        # 对于可能不生成开始标记的模型
        # 假设推理内容总是在开头
        if self.end_token not in model_output:
            return model_output, None
        else:
            reasoning, _, content = model_output.partition(self.end_token)
            # 如果生成在思考结束后立即停止，返回空内容
            final_content = content or None
            return reasoning, final_content


__all__ = ["BaseThinkingReasoningParser"]
