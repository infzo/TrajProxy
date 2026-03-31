"""
Infer 响应解析工具

解析 Infer 服务返回的响应（流式和非流式）
"""
from typing import Dict, Any, List, Tuple, Optional


class InferResponseParser:
    """Infer 响应解析器

    提供 Infer 服务响应的统一解析方法，支持流式和非流式两种模式。
    """

    @staticmethod
    def parse_text_response(response: Dict[str, Any]) -> Tuple[str, Optional[List[int]]]:
        """解析非流式文本响应

        Args:
            response: Infer 服务返回的响应字典

        Returns:
            (text, token_ids) 元组，token_ids 可能为 None
        """
        text = ""
        token_ids = None

        if "choices" in response and response["choices"]:
            choice = response["choices"][0]
            text = choice.get("text", "")

            # 扩展格式：直接返回 token_ids
            if "token_ids" in choice:
                token_ids = choice["token_ids"]

        return text, token_ids

    @staticmethod
    def parse_token_ids_from_text(text: str) -> Optional[List[int]]:
        """从文本解析 token IDs（Token-in-Token-out 模式）

        在 Token-in-Token-out 模式下，Infer 服务可能返回空格分隔的 token ID 字符串。

        Args:
            text: 可能包含 token IDs 的文本

        Returns:
            token ID 列表，解析失败返回 None
        """
        if not text or not text.strip():
            return []

        try:
            return [int(tid) for tid in text.strip().split()]
        except ValueError:
            # 不是 token IDs，是普通文本
            return None

    @staticmethod
    def parse_stream_chunk(
        chunk: Dict[str, Any],
        is_token_mode: bool
    ) -> Tuple[str, List[int], Optional[List[Dict]]]:
        """解析流式响应块

        Args:
            chunk: Infer 服务返回的流式响应块
            is_token_mode: 是否为 Token-in-Token-out 模式

        Returns:
            (content, token_ids, tool_calls_delta) 元组
        """
        content = ""
        token_ids = []
        tool_calls_delta = None

        if "choices" in chunk and chunk["choices"]:
            choice = chunk["choices"][0]

            # 标准格式：choices[0].text
            if "text" in choice:
                text = choice["text"]

                if is_token_mode:
                    # Token-in-Token-out 模式：尝试解析为 token IDs
                    try:
                        if text.strip():
                            token_ids = [int(tid) for tid in text.strip().split()]
                    except ValueError:
                        # 不是 token IDs，直接作为文本
                        content = text
                else:
                    content = text

            # 扩展格式：直接返回 token_ids
            if "token_ids" in choice:
                token_ids = choice["token_ids"]

            # 处理 tool_calls
            if "tool_calls" in choice:
                tool_calls_delta = choice["tool_calls"]

        return content, token_ids, tool_calls_delta

    @staticmethod
    def is_stream_finished(chunk: Dict[str, Any]) -> bool:
        """判断流式响应是否结束

        Args:
            chunk: Infer 响应块

        Returns:
            是否结束
        """
        if "choices" in chunk and chunk["choices"]:
            finish_reason = chunk["choices"][0].get("finish_reason")
            return finish_reason is not None
        return False

    @staticmethod
    def get_finish_reason(chunk: Dict[str, Any]) -> str:
        """获取结束原因

        Args:
            chunk: Infer 响应块

        Returns:
            结束原因，默认为 "stop"
        """
        if "choices" in chunk and chunk["choices"]:
            return chunk["choices"][0].get("finish_reason", "stop")
        return "stop"

    @staticmethod
    def extract_usage(response: Dict[str, Any]) -> Dict[str, int]:
        """提取使用统计信息

        Args:
            response: Infer 服务返回的响应字典

        Returns:
            包含 prompt_tokens, completion_tokens, total_tokens 的字典
        """
        if "usage" in response:
            usage = response["usage"]
            return {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            }
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
