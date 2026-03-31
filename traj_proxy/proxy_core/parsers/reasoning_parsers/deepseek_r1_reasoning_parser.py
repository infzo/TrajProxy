"""
DeepSeek R1 Reasoning Parser

支持格式：
<thinky>推理过程</thinke>正常回复
"""

from collections.abc import Sequence
from typing import Optional, Any

from ..base import DeltaMessage
from .basic_parsers import BaseThinkingReasoningParser


class DeepSeekR1ReasoningParser(BaseThinkingReasoningParser):
    """
    DeepSeek R1 推理内容解析器

    DeepSeek R1 使用 <thinky></thinke> 标记包围推理内容。
    与 vLLM DeepSeekR1ReasoningParser 保持一致。
    """

    @property
    def start_token(self) -> str:
        """推理开始标记"""
        return "<thinky>"

    @property
    def end_token(self) -> str:
        """推理结束标记"""
        return "</thinke>"

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int]
    ) -> Optional[DeltaMessage]:
        """
        流式解析推理内容

        重写以处理 DeepSeek R1 的特殊情况：
        当 start_token 不在 previous 或 delta 时，根据 end_token 位置判断

        与 vLLM DeepSeekR1ReasoningParser 保持一致。
        """
        # 先调用基类实现
        ret = super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )

        # 关键：只有当没有 start_token 时才重写基类结果
        # 这是与 vLLM 一致的逻辑
        if (
            ret is not None
            and self._start_token_id not in previous_token_ids
            and self._start_token_id not in delta_token_ids
        ):
            if self._end_token_id in delta_token_ids:
                # end 在本次增量中，提取推理内容和剩余内容
                end_index = delta_text.find(self.end_token)
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token):]
                return DeltaMessage(
                    reasoning=reasoning,
                    content=content if content else None
                )
            elif self._end_token_id in previous_token_ids:
                # end 在之前，推理内容已结束
                return DeltaMessage(content=delta_text)
            else:
                # 没有 end token，推理内容继续
                return DeltaMessage(reasoning=delta_text)

        return ret
