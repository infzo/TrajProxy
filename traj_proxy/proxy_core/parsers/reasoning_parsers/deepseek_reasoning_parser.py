"""
DeepSeek 推理内容解析器

支持格式:
<｜begin▁of▁think｜>推理过程<｜end▁of▁think｜>正常回复

或者使用 <thinky></thinke> 格式（某些版本）
"""

from typing import Optional, Sequence

from ..base import BaseReasoningParser, DeltaMessage


class DeepSeekReasoningParser(BaseReasoningParser):
    """DeepSeek 推理内容解析器"""

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)

        # 备用标记符
        self._alt_start_token = "<｜begin▁of▁think｜>"
        self._alt_end_token = "<｜end▁of▁think｜>"

        # ASCII 变体
        self._ascii_start_token = "<|begin_of_think|>"
        self._ascii_end_token = "<|end_of_think|>"

        # Token IDs（如果可用）
        if tokenizer:
            vocab = tokenizer.get_vocab() if hasattr(tokenizer, 'get_vocab') else {}
            self._start_token_id = vocab.get(self.start_token) or vocab.get(self._alt_start_token)
            self._end_token_id = vocab.get(self.end_token) or vocab.get(self._alt_end_token)
        else:
            self._start_token_id = None
            self._end_token_id = None

        # 流式状态
        self._in_reasoning = False
        self._detected_format = None  # 'unicode' or 'ascii'

    @property
    def start_token(self) -> str:
        """推理开始标记（Unicode 版本）"""
        return "<｜begin▁of▁think｜>"

    @property
    def end_token(self) -> str:
        """推理结束标记（Unicode 版本）"""
        return "<｜end▁of▁think｜>"

    def reset_streaming_state(self):
        """重置流式状态"""
        self._in_reasoning = False
        self._detected_format = None

    def _detect_format(self, text: str) -> tuple[str, str]:
        """检测使用的标记符格式"""
        if self.start_token in text or self.end_token in text:
            self._detected_format = 'unicode'
            return self.start_token, self.end_token
        elif self._ascii_start_token in text or self._ascii_end_token in text:
            self._detected_format = 'ascii'
            return self._ascii_start_token, self._ascii_end_token
        else:
            # 默认使用 Unicode
            return self.start_token, self.end_token

    def extract_reasoning(
        self,
        model_output: str
    ) -> tuple[Optional[str], Optional[str]]:
        """非流式解析推理内容"""

        # 检测格式
        start_tok, end_tok = self._detect_format(model_output)

        if start_tok not in model_output:
            return None, model_output

        parts = model_output.split(start_tok, 1)
        before_think = parts[0]

        if len(parts) > 1:
            after_start = parts[1]
            if end_tok in after_start:
                end_idx = after_start.find(end_tok)
                reasoning = after_start[:end_idx]
                content = after_start[end_idx + len(end_tok):]

                # 合并 thinky 之前的内容
                if before_think:
                    content = before_think + content

                return reasoning, content if content else None

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

        # 检测格式
        start_tok, end_tok = self._get_current_format(current_text)

        # 使用 token ID 精确判断（如果可用）
        if self._start_token_id is not None and self._end_token_id is not None:
            return self._extract_with_token_ids(
                previous_token_ids, current_token_ids, delta_token_ids, delta_text, start_tok, end_tok
            )

        # 文本回退
        return self._extract_with_text(delta_text, start_tok, end_tok)

    def _get_current_format(self, text: str) -> tuple[str, str]:
        """获取当前使用的格式"""
        if self._detected_format == 'unicode':
            return self.start_token, self.end_token
        elif self._detected_format == 'ascii':
            return self._ascii_start_token, self._ascii_end_token
        else:
            # 自动检测
            return self._detect_format(text)

    def _extract_with_token_ids(
        self,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        delta_text: str,
        start_tok: str,
        end_tok: str
    ) -> Optional[DeltaMessage]:
        """使用 token ID 精确解析"""

        # 跳过单独的特殊 token
        if len(delta_token_ids) == 1 and delta_token_ids[0] in [self._start_token_id, self._end_token_id]:
            return None

        start_in_previous = self._start_token_id in previous_token_ids
        start_in_delta = self._start_token_id in delta_token_ids
        end_in_delta = self._end_token_id in delta_token_ids
        end_in_previous = self._end_token_id in previous_token_ids

        if start_in_previous or start_in_delta:
            if end_in_delta:
                end_idx = delta_text.find(end_tok)
                if start_in_delta:
                    start_idx = delta_text.find(start_tok)
                    reasoning = delta_text[start_idx + len(start_tok):end_idx]
                else:
                    reasoning = delta_text[:end_idx]
                content = delta_text[end_idx + len(end_tok):]
                self._in_reasoning = False
                return DeltaMessage(reasoning=reasoning, content=content if content else None)
            elif end_in_previous:
                return DeltaMessage(content=delta_text)
            else:
                self._in_reasoning = True
                return DeltaMessage(reasoning=delta_text)
        else:
            return DeltaMessage(content=delta_text)

    def _extract_with_text(
        self,
        delta_text: str,
        start_tok: str,
        end_tok: str
    ) -> Optional[DeltaMessage]:
        """文本回退解析"""

        # 检查是否包含开始标记
        if start_tok in delta_text:
            parts = delta_text.split(start_tok, 1)
            self._in_reasoning = True
            if len(parts) > 1 and parts[1]:
                return DeltaMessage(
                    content=parts[0] if parts[0] else None,
                    reasoning=parts[1]
                )
            return DeltaMessage(content=parts[0] if parts[0] else None)

        # 检查是否包含结束标记
        if end_tok in delta_text:
            parts = delta_text.split(end_tok, 1)
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
