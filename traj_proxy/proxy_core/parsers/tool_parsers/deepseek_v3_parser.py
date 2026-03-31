"""
DeepSeek V3 工具解析器

支持格式:
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
```json
{"city": "北京"}
```
<｜tool▁call▁end｜><｜tool▁calls▁end｜>
"""

import re
import json
import uuid
from typing import Optional, List, Any, Sequence

from ..base import (
    BaseToolParser,
    ExtractedToolCallInfo,
    ToolCall,
    FunctionCall,
    DeltaMessage,
    DeltaToolCall,
    DeltaFunctionCall,
)


class DeepSeekV3ToolParser(BaseToolParser):
    """DeepSeek V3 工具解析器"""

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)

        # 标记符（使用常见的 Unicode 变体）
        self.tool_calls_begin = "<｜tool▁calls▁begin｜>"
        self.tool_calls_end = "<｜tool▁calls▁end｜>"
        self.tool_call_begin = "<｜tool▁call▁begin｜>"
        self.tool_call_end = "<｜tool▁call▁end｜>"
        self.tool_sep = "<｜tool▁sep｜>"

        # 备用 ASCII 标记符
        self.alt_tool_calls_begin = "<|tool_calls_begin|>"
        self.alt_tool_calls_end = "<|tool_calls_end|>"
        self.alt_tool_call_begin = "<|tool_call_begin|>"
        self.alt_tool_call_end = "<|tool_call_end|>"
        self.alt_tool_sep = "<|tool_sep|>"

        # 流式状态
        self._reset_streaming_state()

    def _generate_tool_call_id(self) -> str:
        return f"call_{uuid.uuid4().hex[:24]}"

    def _reset_streaming_state(self):
        """重置流式状态"""
        self.current_tool_index = 0
        self.in_tool_calls = False
        self.current_tool_id = None
        self.current_function_name = None
        self.current_arguments = ""
        self.in_arguments = False
        self.arguments_buffer = ""
        self.header_sent = False
        self.brace_count = 0
        self.in_json = False
        self.streaming_tools = None

    def _normalize_tokens(self, text: str) -> str:
        """将文本中的标记符标准化为 Unicode 版本"""
        text = text.replace(self.alt_tool_calls_begin, self.tool_calls_begin)
        text = text.replace(self.alt_tool_calls_end, self.tool_calls_end)
        text = text.replace(self.alt_tool_call_begin, self.tool_call_begin)
        text = text.replace(self.alt_tool_call_end, self.tool_call_end)
        text = text.replace(self.alt_tool_sep, self.tool_sep)
        return text

    def _detect_format(self, model_output: str) -> tuple[str, str, str, str, str]:
        """检测使用的标记符格式"""
        if self.tool_calls_begin in model_output or self.tool_call_begin in model_output:
            return (self.tool_calls_begin, self.tool_calls_end,
                    self.tool_call_begin, self.tool_call_end, self.tool_sep)
        else:
            return (self.alt_tool_calls_begin, self.alt_tool_calls_end,
                    self.alt_tool_call_begin, self.alt_tool_call_end, self.alt_tool_sep)

    def extract_tool_calls(
        self,
        model_output: str,
        tools: Optional[List[dict]] = None
    ) -> ExtractedToolCallInfo:
        """非流式解析工具调用"""

        # 标准化标记符
        normalized = self._normalize_tokens(model_output)

        if self.tool_calls_begin not in normalized:
            return ExtractedToolCallInfo(
                tools_called=False,
                tool_calls=[],
                content=model_output
            )

        try:
            tool_calls = []

            # 提取工具调用区域
            start_idx = normalized.find(self.tool_calls_begin)
            end_idx = normalized.find(self.tool_calls_end)

            if start_idx == -1:
                return ExtractedToolCallInfo(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output
                )

            if end_idx > start_idx:
                tool_calls_region = normalized[start_idx:end_idx + len(self.tool_calls_end)]
            else:
                tool_calls_region = normalized[start_idx:]

            # 解析每个工具调用
            call_pattern = rf'{re.escape(self.tool_call_begin)}(.*?)(?:{re.escape(self.tool_call_end)}|$)'
            for match in re.finditer(call_pattern, tool_calls_region, re.DOTALL):
                call_content = match.group(1)
                tool_call = self._parse_single_call(call_content, tools)
                if tool_call:
                    tool_calls.append(tool_call)

            # 提取工具调用前的内容
            content = model_output[:start_idx] if start_idx > 0 else None

            return ExtractedToolCallInfo(
                tools_called=len(tool_calls) > 0,
                tool_calls=tool_calls,
                content=content
            )

        except Exception:
            return ExtractedToolCallInfo(
                tools_called=False,
                tool_calls=[],
                content=model_output
            )

    def _parse_single_call(
        self,
        call_content: str,
        tools: Optional[List[dict]] = None
    ) -> Optional[ToolCall]:
        """解析单个工具调用"""
        # 格式: function<｜tool▁sep｜>name\n```json\n{...}\n```
        parts = call_content.split(self.tool_sep)
        if len(parts) < 2:
            # 尝试备用分隔符
            parts = call_content.split(self.alt_tool_sep)
            if len(parts) < 2:
                return None

        func_type = parts[0].strip()  # "function"
        remaining = parts[1]

        # 提取函数名（到换行或 ``` 之前）
        lines = remaining.split('\n', 1)
        func_name = lines[0].strip()

        # 提取 JSON 参数
        if len(lines) > 1:
            json_content = lines[1]
            # 提取 ```json ... ``` 中的内容
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', json_content)
            if json_match:
                arguments = json_match.group(1).strip()
            else:
                # 尝试直接解析为 JSON
                arguments = json_content.strip()
        else:
            arguments = "{}"

        # 验证 JSON 有效性
        try:
            json.loads(arguments)
        except json.JSONDecodeError:
            # 尝试修复常见的 JSON 格式问题
            arguments = self._fix_json(arguments)

        return ToolCall(
            id=self._generate_tool_call_id(),
            type="function",
            function=FunctionCall(
                name=func_name,
                arguments=arguments
            )
        )

    def _fix_json(self, json_str: str) -> str:
        """尝试修复 JSON 格式问题"""
        # 尝试提取 JSON 对象
        brace_start = json_str.find('{')
        brace_end = json_str.rfind('}')
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            json_str = json_str[brace_start:brace_end + 1]

        # 尝试解析
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            return "{}"

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

        # 标准化文本
        normalized_current = self._normalize_tokens(current_text)
        normalized_delta = self._normalize_tokens(delta_text)

        # 检测是否进入工具调用区域
        if not self.in_tool_calls:
            if self.tool_calls_begin in normalized_current:
                self.in_tool_calls = True
                # 返回工具调用开始前的内容
                start_idx = normalized_current.find(self.tool_calls_begin)
                if start_idx > 0:
                    return DeltaMessage(content=delta_text[:start_idx])
                return None

        if not self.in_tool_calls:
            return DeltaMessage(content=delta_text)

        # 在工具调用区域内
        return self._parse_streaming_in_tool_calls(
            normalized_current, normalized_delta, tools
        )

    def _parse_streaming_in_tool_calls(
        self,
        current_text: str,
        delta_text: str,
        tools: Optional[List[dict]] = None
    ) -> Optional[DeltaMessage]:
        """在工具调用区域内解析"""

        # 检测新的工具调用开始
        if self.tool_call_begin in delta_text or (
            self.tool_call_begin in current_text and not self.current_function_name
        ):
            # 解析新的工具调用
            return self._parse_new_tool_call_streaming(current_text, delta_text, tools)

        # 解析 JSON 参数
        if self.current_function_name:
            return self._parse_arguments_streaming(delta_text)

        return None

    def _parse_new_tool_call_streaming(
        self,
        current_text: str,
        delta_text: str,
        tools: Optional[List[dict]] = None
    ) -> Optional[DeltaMessage]:
        """解析新工具调用的开始"""

        # 查找 tool_sep
        if self.tool_sep not in current_text:
            return None

        # 提取函数名
        sep_idx = current_text.find(self.tool_sep)
        after_sep = current_text[sep_idx + len(self.tool_sep):]

        # 函数名到换行或 ``` 之前
        func_name_end = len(after_sep)
        for delimiter in ['\n', '`', '{']:
            idx = after_sep.find(delimiter)
            if idx != -1 and idx < func_name_end:
                func_name_end = idx

        func_name = after_sep[:func_name_end].strip()

        if not func_name:
            return None

        # 检查是否已经发送过 header
        if self.header_sent and self.current_function_name == func_name:
            return None

        self.current_function_name = func_name
        self.current_tool_id = self._generate_tool_call_id()
        self.header_sent = True

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

    def _parse_arguments_streaming(self, delta_text: str) -> Optional[DeltaMessage]:
        """解析参数的流式增量"""

        # 查找 JSON 开始
        if '{' in delta_text and not self.in_json:
            self.in_json = True
            brace_idx = delta_text.find('{')
            json_start = delta_text[brace_idx:]

            # 发送 JSON 开始
            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_index,
                        function=DeltaFunctionCall(
                            arguments=json_start
                        )
                    )
                ]
            )

        # 继续解析 JSON
        if self.in_json:
            # 计算括号数量
            for char in delta_text:
                if char == '{':
                    self.brace_count += 1
                elif char == '}':
                    self.brace_count -= 1

            self.arguments_buffer += delta_text

            # 检查 JSON 是否完成
            if self.brace_count == 0:
                # JSON 完成
                self.current_tool_index += 1
                self._reset_for_next_tool()

            # 发送增量
            if delta_text:
                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_index - 1 if self.brace_count == 0 else self.current_tool_index,
                            function=DeltaFunctionCall(
                                arguments=delta_text
                            )
                        )
                    ]
                )

        return None

    def _reset_for_next_tool(self):
        """重置状态以准备下一个工具调用"""
        self.current_function_name = None
        self.current_tool_id = None
        self.in_json = False
        self.arguments_buffer = ""
        self.brace_count = 0
        self.header_sent = False
