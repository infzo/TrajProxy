"""
Llama 3.1+ JSON 工具解析器

支持格式:
<|python_tag|>{"name": "get_weather", "parameters": {"city": "北京"}}

或直接的 JSON 格式:
{"name": "get_weather", "parameters": {"city": "北京"}}
"""

import re
import json
import uuid
from typing import Optional, List, Sequence

from ..base import (
    BaseToolParser,
    ExtractedToolCallInfo,
    ToolCall,
    FunctionCall,
    DeltaMessage,
    DeltaToolCall,
    DeltaFunctionCall,
)


class Llama3JSONToolParser(BaseToolParser):
    """Llama 3.1+ JSON 工具解析器"""

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)

        self.python_tag = "<|python_tag|>"

        # 流式状态
        self._reset_streaming_state()

    def _generate_tool_call_id(self) -> str:
        return f"call_{uuid.uuid4().hex[:24]}"

    def _reset_streaming_state(self):
        """重置流式状态"""
        self.current_tool_index = 0
        self.in_json = False
        self.json_buffer = ""
        self.brace_count = 0
        self.in_string = False
        self.escape_next = False
        self.header_sent = False
        self.current_tool_id = None
        self.current_function_name = None
        self.streaming_tools = None

    def extract_tool_calls(
        self,
        model_output: str,
        tools: Optional[List[dict]] = None
    ) -> ExtractedToolCallInfo:
        """非流式解析工具调用"""

        # 查找 python_tag 或直接查找 JSON 对象
        if self.python_tag in model_output:
            start_idx = model_output.find(self.python_tag) + len(self.python_tag)
            json_start = model_output.find("{", start_idx)
        else:
            # 尝试直接查找包含 name 和 parameters 的 JSON
            json_start = self._find_tool_json_start(model_output)

        if json_start == -1:
            return ExtractedToolCallInfo(
                tools_called=False,
                tool_calls=[],
                content=model_output
            )

        try:
            # 提取并解析 JSON
            json_str, json_end = self._extract_json_object(model_output, json_start)
            tool_data = json.loads(json_str)

            tool_calls = []

            if isinstance(tool_data, dict) and "name" in tool_data:
                params = tool_data.get("parameters", tool_data.get("arguments", {}))
                tool_calls.append(ToolCall(
                    id=self._generate_tool_call_id(),
                    type="function",
                    function=FunctionCall(
                        name=tool_data["name"],
                        arguments=json.dumps(params, ensure_ascii=False)
                    )
                ))

            # 处理数组格式
            elif isinstance(tool_data, list):
                for item in tool_data:
                    if isinstance(item, dict) and "name" in item:
                        params = item.get("parameters", item.get("arguments", {}))
                        tool_calls.append(ToolCall(
                            id=self._generate_tool_call_id(),
                            type="function",
                            function=FunctionCall(
                                name=item["name"],
                                arguments=json.dumps(params, ensure_ascii=False)
                            )
                        ))

            content = model_output[:json_start] if json_start > 0 else None

            return ExtractedToolCallInfo(
                tools_called=len(tool_calls) > 0,
                tool_calls=tool_calls,
                content=content
            )

        except json.JSONDecodeError:
            return ExtractedToolCallInfo(
                tools_called=False,
                tool_calls=[],
                content=model_output
            )

    def _find_tool_json_start(self, text: str) -> int:
        """查找工具调用 JSON 的起始位置"""
        # 查找包含 name 字段的 JSON 对象
        pattern = r'\{[^{}]*"name"\s*:'
        match = re.search(pattern, text)
        if match:
            # 找到匹配位置后，回溯到该 JSON 对象的起始位置
            pos = match.start()
            while pos >= 0 and text[pos] != '{':
                pos -= 1
            return pos
        return -1

    def _extract_json_object(self, text: str, start: int) -> tuple[str, int]:
        """提取完整的 JSON 对象"""
        brace_count = 0
        in_string = False
        escape_next = False
        end = start

        for i, char in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue

            if char == '\\' and in_string:
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break

        return text[start:end], end

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        tools: Optional[List[dict]] = None
    ) -> Optional[DeltaMessage]:
        """流式解析工具调用"""

        # 首次调用时重置状态
        if not previous_text:
            self._reset_streaming_state()
            self.streaming_tools = tools

        # 检测 JSON 开始
        if not self.in_json:
            # 查找 python_tag
            if self.python_tag in current_text:
                tag_idx = current_text.find(self.python_tag)
                json_idx = current_text.find("{", tag_idx)
                if json_idx != -1:
                    self.in_json = True
                    # 返回 tag 之前的内容
                    if tag_idx > 0:
                        return DeltaMessage(content=delta_text[:tag_idx])
                    return None

            # 直接查找 JSON 对象
            json_start = self._find_tool_json_start(current_text)
            if json_start != -1:
                self.in_json = True
                # 返回 JSON 开始之前的内容
                if json_start > 0:
                    content_before = current_text[:json_start]
                    if content_before and not previous_text:
                        return DeltaMessage(content=content_before)
                return self._start_new_tool_call(current_text, json_start)

            return DeltaMessage(content=delta_text)

        # 在 JSON 解析中
        return self._parse_json_streaming(delta_text)

    def _start_new_tool_call(
        self,
        current_text: str,
        json_start: int
    ) -> Optional[DeltaMessage]:
        """开始新的工具调用解析"""
        # 提取 JSON 开始部分
        after_json_start = current_text[json_start:]

        # 尝试解析部分 JSON 获取函数名
        name_match = re.search(r'"name"\s*:\s*"([^"]+)"', after_json_start)
        if name_match:
            func_name = name_match.group(1)
            self.current_function_name = func_name
            self.current_tool_id = self._generate_tool_call_id()

            if not self.header_sent:
                self.header_sent = True

                # 找到 name 之后的内容
                name_end = name_match.end()
                remaining = after_json_start[name_end:]

                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_index,
                            id=self.current_tool_id,
                            type="function",
                            function=DeltaFunctionCall(
                                name=func_name,
                                arguments=""
                            )
                        )
                    ]
                )

        return None

    def _parse_json_streaming(self, delta_text: str) -> Optional[DeltaMessage]:
        """解析 JSON 流式增量"""

        if not delta_text:
            return None

        # 更新括号计数
        for char in delta_text:
            if self.escape_next:
                self.escape_next = False
                continue

            if char == '\\' and self.in_string:
                self.escape_next = True
                continue

            if char == '"' and not self.escape_next:
                self.in_string = not self.in_string
                continue

            if not self.in_string:
                if char == '{':
                    self.brace_count += 1
                elif char == '}':
                    self.brace_count -= 1

        self.json_buffer += delta_text

        # 检查是否需要发送函数名（如果还没发送）
        if not self.header_sent and '"name"' in self.json_buffer:
            name_match = re.search(r'"name"\s*:\s*"([^"]+)"', self.json_buffer)
            if name_match:
                self.current_function_name = name_match.group(1)
                self.current_tool_id = self._generate_tool_call_id()
                self.header_sent = True

                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_index,
                            id=self.current_tool_id,
                            type="function",
                            function=DeltaFunctionCall(
                                name=self.current_function_name,
                                arguments=""
                            )
                        )
                    ]
                )

        # 发送参数增量
        if self.header_sent and delta_text:
            # 提取 parameters 部分
            params_match = re.search(r'"parameters"\s*:\s*', self.json_buffer)
            if params_match:
                # 找到 parameters 的起始位置
                params_start = params_match.end()
                # 提取增量部分
                current_params = self.json_buffer[params_start:]

                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_index,
                            function=DeltaFunctionCall(
                                arguments=delta_text
                            )
                        )
                    ]
                )

        # JSON 完成
        if self.brace_count == 0 and self.json_buffer:
            try:
                # 解析完整 JSON
                tool_data = json.loads(self.json_buffer)
                if isinstance(tool_data, dict):
                    params = tool_data.get("parameters", tool_data.get("arguments", {}))
                    # 准备下一个工具调用
                    self.current_tool_index += 1
                    self._reset_for_next_tool()
            except json.JSONDecodeError:
                pass

        return None

    def _reset_for_next_tool(self):
        """重置状态以准备下一个工具调用"""
        self.json_buffer = ""
        self.brace_count = 0
        self.in_json = False
        self.in_string = False
        self.escape_next = False
        self.header_sent = False
        self.current_function_name = None
        self.current_tool_id = None
