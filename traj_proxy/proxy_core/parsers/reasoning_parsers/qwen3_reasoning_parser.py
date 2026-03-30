"""
Qwen3 推理内容解析器

支持格式:
<thinky>这里是推理过程</thinke>这是正常回复

注意：实际 token 可能是 <thinky></thinke> 或类似变体
"""

from typing import Optional, Sequence

from ..base import BaseReasoningParser, DeltaMessage


class Qwen3ReasoningParser(BaseReasoningParser):
    """Qwen3 推理内容解析器"""

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)

        # Token IDs（如果可用）
        if tokenizer:
            vocab = tokenizer.get_vocab() if hasattr(tokenizer, 'get_vocab') else {}
            self._start_token_id = vocab.get(self.start_token)
            self._end_token_id = vocab.get(self.end_token)
        else:
            self._start_token_id = None
            self._end_token_id = None

        # 流式状态
        self._in_reasoning = False
        self._start_token_sent = False
        self._end_token_sent = False

    @property
    def start_token(self) -> str:
        """推理开始标记"""
        return "<thinky>"

    @property
    def end_token(self) -> str:
        """推理结束标记"""
        return "</thinke>"

    def reset_streaming_state(self):
        """重置流式状态"""
        self._in_reasoning = False
        self._start_token_sent = False
        self._end_token_sent = False

    def extract_reasoning(
        self,
        model_output: str
    ) -> tuple[Optional[str], Optional[str]]:
        """非流式解析推理内容"""

        if self.start_token not in model_output:
            return None, model_output

        # 分割推理内容
        parts = model_output.split(self.start_token, 1)
        before_think = parts[0]

        if len(parts) > 1:
            after_start = parts[1]
            if self.end_token in after_start:
                end_idx = after_start.find(self.end_token)
                reasoning = after_start[:end_idx]
                content = after_start[end_idx + len(self.end_token):]

                # 合并 thinky 之前的内容
                if before_think:
                    content = before_think + content

                return reasoning, content if content else None
            else:
                # 没有结束标记，全部作为推理内容
                return after_start, before_think if before_think else None

        return None, model_output

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int]
    ) -> Optional[DeltaMessage]:
        """流式解析推理内容"""

        # 使用 token ID 精确判断
        if self._start_token_id is not None and self._end_token_id is not None:
            return self._extract_with_token_ids(
                previous_token_ids, current_token_ids, delta_token_ids, delta_text
            )

        # 文本回退
        return self._extract_with_text(delta_text)

    def _extract_with_token_ids(
        self,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        delta_text: str
    ) -> Optional[DeltaMessage]:
        """使用 token ID 精确解析"""

        # 跳过单独的特殊 token
        if len(delta_token_ids) == 1 and delta_token_ids[0] in [self._start_token_id, self._end_token_id]:
            return None

        # 检查当前状态
        start_in_previous = self._start_token_id in previous_token_ids
        start_in_delta = self._start_token_id in delta_token_ids
        end_in_delta = self._end_token_id in delta_token_ids
        end_in_previous = self._end_token_id in previous_token_ids

        if start_in_previous or start_in_delta:
            if end_in_delta:
                # start 在之前或 delta，end 在 delta
                end_idx = delta_text.find(self.end_token)
                if start_in_delta:
                    start_idx = delta_text.find(self.start_token)
                    reasoning = delta_text[start_idx + len(self.start_token):end_idx]
                else:
                    reasoning = delta_text[:end_idx]
                content = delta_text[end_idx + len(self.end_token):]
                self._in_reasoning = False
                return DeltaMessage(reasoning=reasoning, content=content if content else None)
            elif end_in_previous:
                # end 已在之前，当前是普通内容
                return DeltaMessage(content=delta_text)
            else:
                # 还在推理内容中
                self._in_reasoning = True
                return DeltaMessage(reasoning=delta_text)
        else:
            # 未进入推理区域
            return DeltaMessage(content=delta_text)

    def _extract_with_text(self, delta_text: str) -> Optional[DeltaMessage]:
        """文本回退解析"""

        # 检查是否包含开始标记
        if self.start_token in delta_text:
            parts = delta_text.split(self.start_token, 1)
            self._in_reasoning = True
            if len(parts) > 1 and parts[1]:
                return DeltaMessage(
                    content=parts[0] if parts[0] else None,
                    reasoning=parts[1]
                )
            return DeltaMessage(content=parts[0] if parts[0] else None)

        # 检查是否包含结束标记
        if self.end_token in delta_text:
            parts = delta_text.split(self.end_token, 1)
            self._in_reasoning = False
            return DeltaMessage(
                reasoning=parts[0] if parts[0] else None,
                content=parts[1] if len(parts) > 1 and parts[1] else None
            )

        # 正常内容
        if self._in_reasoning:
            return DeltaMessage(reasoning=delta_text)
        else:
            return DeltaMessage(content=delta_text)
