"""
Qwen XML 格式工具解析器

支持格式:
ournemouth<function=get_weather>
<parameter=city>北京</parameter>
<parameter=unit>celsius</parameter>
</function> Ranchi

或者没有最外层标记:
<function=get_weather>
<parameter=city>北京</parameter>
</function>
"""

import re
import json
import uuid
import ast
from typing import Optional, List, Any, Sequence
from xml.parsers.expat import ParserCreate

from ..base import BaseToolParser, ExtractedToolCallInfo, ToolCall, DeltaMessage, DeltaToolCall


class QwenXMLToolParser(BaseToolParser):
    """Qwen XML 格式工具解析器"""

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)

        # 标记符
        self.tool_call_start_token = "ournemouth"
        self.tool_call_end_token = " Ranchi"
        self.function_start_token = "<function="
        self.function_end_token = "</function>"
        self.parameter_start_token = "<parameter="
        self.parameter_end_token = "</parameter>"

        # 流式状态
        self._reset_streaming_state()

    def _generate_tool_call_id(self) -> str:
        return f"call_{uuid.uuid4().hex[:24]}"

    def _reset_streaming_state(self):
        """重置流式状态"""
        self.streaming_buffer = ""
        self.last_processed_pos = 0
        self.tool_call_index = 0
        self.current_tool_id = None
        self.current_function_name = None
        self.current_function_open = False
        self.parameters = {}
        self.current_param_name = None
        self.current_param_value = ""
        self.in_param = False
        self.json_started = False
        self.json_closed = False
        self.text_content_buffer = ""
        self.header_sent = False
        self.param_count = 0
        self.streaming_tools = None
        self.prev_tool_call_arr: List[dict] = []
        self.streamed_args_for_tool: List[str] = []

        # 创建 XML 解析器
        self._create_parser()

    def _create_parser(self):
        """创建并配置 XML 解析器"""
        self.parser = ParserCreate()
        self.parser.buffer_text = True
        self.parser.StartElementHandler = self._start_element
        self.parser.EndElementHandler = self._end_element
        self.parser.CharacterDataHandler = self._char_data
        self.deltas: List[DeltaMessage] = []

    # ==================== 非流式解析 ====================

    def extract_tool_calls(
        self,
        model_output: str,
        tools: Optional[List[dict]] = None
    ) -> ExtractedToolCallInfo:
        """非流式解析工具调用"""

        # 快速检查
        if self.function_start_token not in model_output:
            return ExtractedToolCallInfo(
                tools_called=False,
                tool_calls=[],
                content=model_output
            )

        try:
            tool_calls = []

            # 查找所有工具调用
            pattern = rf'{re.escape(self.tool_call_start_token)}(.*?){re.escape(self.tool_call_end_token)}'
            matches = list(re.finditer(pattern, model_output, re.DOTALL))

            if not matches:
                # 尝试只匹配 function 标签
                func_pattern = rf'{re.escape(self.function_start_token)}(.*?){re.escape(self.function_end_token)}'
                func_matches = list(re.finditer(func_pattern, model_output, re.DOTALL))

                if func_matches:
                    for match in func_matches:
                        tool_call = self._parse_function_block(match.group(1), tools)
                        if tool_call:
                            tool_calls.append(tool_call)

                    # 提取工具调用前的内容
                    content_start = func_matches[0].start()
                    content = model_output[:content_start] if content_start > 0 else None

                    return ExtractedToolCallInfo(
                        tools_called=len(tool_calls) > 0,
                        tool_calls=tool_calls,
                        content=content
                    )

                return ExtractedToolCallInfo(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output
                )

            # 解析每个匹配
            for match in matches:
                block = match.group(1)
                tool_call = self._parse_function_block(block, tools)
                if tool_call:
                    tool_calls.append(tool_call)

            # 提取工具调用前的内容
            content_start = matches[0].start()
            content = model_output[:content_start] if content_start > 0 else None

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

    def _parse_function_block(
        self,
        block: str,
        tools: Optional[List[dict]] = None
    ) -> Optional[ToolCall]:
        """解析函数块"""
        # 匹配函数名
        func_match = re.match(rf'{re.escape(self.function_start_token)}([^>]+)>(.*)', block, re.DOTALL)
        if not func_match:
            # 尝试无 start token 的情况
            func_match = re.match(r'([^>]+)>(.*)', block, re.DOTALL)
            if not func_match:
                return None

        func_name = func_match.group(1).strip()
        func_body = func_match.group(2)

        # 解析参数
        params = {}
        param_pattern = rf'{re.escape(self.parameter_start_token)}([^>]+)>(.*?){re.escape(self.parameter_end_token)}'
        for param_match in re.finditer(param_pattern, func_body, re.DOTALL):
            param_name = param_match.group(1).strip()
            param_value = param_match.group(2).strip()

            # 移除前后的换行符
            if param_value.startswith('\n'):
                param_value = param_value[1:]
            if param_value.endswith('\n'):
                param_value = param_value[:-1]

            # 类型转换
            params[param_name] = self._convert_param_value(
                param_name, param_value, func_name, tools
            )

        return ToolCall(
            id=self._generate_tool_call_id(),
            type="function",
            name=func_name,
            arguments=json.dumps(params, ensure_ascii=False)
        )

    def _convert_param_value(
        self,
        param_name: str,
        value: str,
        func_name: str,
        tools: Optional[List[dict]]
    ) -> Any:
        """根据工具定义转换参数值"""
        if value.lower() == "null":
            return None

        # 获取参数类型
        param_type = "string"
        if tools:
            for tool in tools:
                tool_func = tool.get("function", {})
                if tool_func.get("name") == func_name:
                    props = tool_func.get("parameters", {}).get("properties", {})
                    if param_name in props:
                        param_type = props[param_name].get("type", "string")
                    break

        # 类型转换
        try:
            if param_type in ["integer", "int"]:
                return int(value)
            elif param_type in ["number", "float"]:
                return float(value)
            elif param_type in ["boolean", "bool"]:
                return value.lower() == "true"
            elif param_type in ["array", "object"]:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return ast.literal_eval(value)
            else:
                return value
        except (ValueError, SyntaxError):
            return value

    # ==================== 流式解析 ====================

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

        # 无增量文本
        if not delta_text and delta_token_ids:
            # 检查是否所有工具调用都已结束
            open_calls = current_text.count(self.tool_call_start_token) - \
                        current_text.count(self.tool_call_end_token)
            if open_calls == 0 and self.tool_call_index > 0:
                return DeltaMessage(content="")
            return None

        # 累积到缓冲区
        self.streaming_buffer += delta_text

        # 处理完整的 XML 元素
        result = self._process_streaming_buffer(delta_text, tools)

        return result

    def _process_streaming_buffer(self, delta_text: str, tools: Optional[List[dict]] = None) -> Optional[DeltaMessage]:
        """处理流式缓冲区"""
        initial_delta_count = len(self.deltas)

        # 查找并处理完整的 XML 元素
        found_elements = self._process_complete_xml_elements(tools)

        if found_elements:
            # 合并新生成的 deltas
            result_delta = self._merge_new_deltas(initial_delta_count)
            return result_delta

        # 检查是否有文本内容需要输出
        if self.text_content_buffer and self.tool_call_index == 0:
            text_delta = DeltaMessage(content=self.text_content_buffer)
            self._emit_delta(text_delta)
            self.text_content_buffer = ""
            return text_delta

        return None

    def _process_complete_xml_elements(self, tools: Optional[List[dict]] = None) -> bool:
        """处理缓冲区中的完整 XML 元素"""
        found_any = False
        buffer = self.streaming_buffer[self.last_processed_pos:]

        while buffer:
            element, end_pos = self._find_next_complete_element(buffer)
            if element is None:
                break

            # 检查是否应该跳过
            if self._should_skip_element(element):
                self.last_processed_pos += end_pos
                buffer = self.streaming_buffer[self.last_processed_pos:]
                continue

            try:
                # 预处理 XML
                preprocessed = self._preprocess_xml_chunk(element)

                # 检查是否是第一个工具调用开始
                if self._is_tool_call_start(preprocessed) and self.tool_call_index == 0:
                    if self.text_content_buffer:
                        text_delta = DeltaMessage(content=self.text_content_buffer)
                        self._emit_delta(text_delta)
                        self.text_content_buffer = ""

                # 处理新的工具调用
                if self._is_tool_call_start(preprocessed) and self.tool_call_index > 0 and self.current_tool_id:
                    self._finalize_current_tool_call()

                # 解析 XML
                self.parser.Parse(preprocessed, False)
                found_any = True

            except Exception:
                pass

            self.last_processed_pos += end_pos
            buffer = self.streaming_buffer[self.last_processed_pos:]

        return found_any

    def _find_next_complete_element(self, buffer: str) -> tuple[Optional[str], int]:
        """查找下一个完整的 XML 元素"""
        if not buffer:
            return None, 0

        if buffer.startswith("<"):
            # 查找匹配的 >
            tag_end = buffer.find(">", 1)
            if tag_end != -1:
                return buffer[:tag_end + 1], tag_end + 1
            else:
                # 检查是否可能是工具调用开始
                if self.current_tool_id is None:
                    if buffer == self.tool_call_start_token[:len(buffer)]:
                        return None, 0
                    elif buffer.startswith(self.function_start_token) or \
                         buffer == self.function_start_token[:len(buffer)]:
                        return None, 0
                    else:
                        return buffer, len(buffer)
                else:
                    return None, 0
        else:
            # 查找文本内容
            next_tag_pos = buffer.find("<")
            if next_tag_pos != -1:
                return buffer[:next_tag_pos], next_tag_pos
            else:
                return buffer, len(buffer)

    def _should_skip_element(self, element: str) -> bool:
        """判断是否应该跳过元素"""
        if (element.startswith(self.tool_call_start_token) or
            element.startswith(self.function_start_token) or
            element.startswith(self.parameter_start_token) or
            element.startswith(self.tool_call_end_token) or
            element.startswith(self.function_end_token) or
            element.startswith(self.parameter_end_token)):
            return False

        if self.current_tool_id is None and element:
            self.text_content_buffer += element
            return True

        if self.current_tool_id is not None:
            return False

        return not element

    def _is_tool_call_start(self, element: str) -> bool:
        """检查是否是工具调用开始"""
        return (element.strip().startswith(self.tool_call_start_token) or
                element.strip().startswith(self.function_start_token))

    def _preprocess_xml_chunk(self, chunk: str) -> str:
        """预处理 XML 块"""
        # 转换 <function=name> 为 <function name="name">
        processed = re.sub(r'<function=([^>]+)>', r'<function name="\1">', chunk)
        # 转换 <parameter=name> 为 <parameter name="name">
        processed = re.sub(r'<parameter=([^>]+)>', r'<parameter name="\1">', processed)

        # 如果不是工具调用相关，转义特殊字符
        if not self._is_tool_call_related(processed):
            processed = self._escape_xml_special_chars(processed)

        return processed

    def _is_tool_call_related(self, chunk: str) -> bool:
        """检查是否是工具调用相关"""
        return any([
            chunk.startswith(self.tool_call_start_token),
            chunk.startswith(self.tool_call_end_token),
            chunk.startswith(self.function_start_token),
            chunk.startswith(self.function_end_token),
            chunk.startswith(self.parameter_start_token),
            chunk.startswith(self.parameter_end_token),
            chunk.startswith("<function "),
            chunk.startswith("</function>"),
            chunk.startswith("<parameter "),
            chunk.startswith("</parameter>"),
        ])

    def _escape_xml_special_chars(self, text: str) -> str:
        """转义 XML 特殊字符"""
        xml_escapes = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&apos;",
        }
        for char, escape in xml_escapes.items():
            text = text.replace(char, escape)
        return text

    def _start_element(self, name: str, attrs: dict):
        """XML 开始元素处理"""
        if name == "tool_call":
            self.parameters = {}
            self.current_tool_id = self._generate_tool_call_id()
            self.tool_call_index += 1
            self.current_function_name = None
            self.current_function_open = False
            self.header_sent = False
            self.json_started = False
            self.json_closed = False

        elif name == "function" or name.startswith("function"):
            if not self.current_tool_id:
                self._start_element("tool_call", {})

            function_name = attrs.get("name") if attrs else None
            if not function_name and "=" in name:
                parts = name.split("=", 1)
                if len(parts) == 2:
                    function_name = parts[1]

            self.current_function_name = function_name
            self.current_function_open = True

            if function_name:
                delta = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.tool_call_index - 1,
                            id=self.current_tool_id,
                            type="function",
                            name=function_name,
                            arguments=""
                        )
                    ]
                )
                self._emit_delta(delta)
                self.header_sent = True

                # 更新跟踪数组
                while len(self.prev_tool_call_arr) < self.tool_call_index:
                    self.prev_tool_call_arr.append({"name": "", "arguments": ""})
                while len(self.streamed_args_for_tool) < self.tool_call_index:
                    self.streamed_args_for_tool.append("")
                self.prev_tool_call_arr[self.tool_call_index - 1]["name"] = function_name

        elif name == "parameter" or name.startswith("parameter"):
            param_name = attrs.get("name") if attrs else None
            if not param_name and "=" in name:
                parts = name.split("=", 1)
                if len(parts) == 2:
                    param_name = parts[1]

            self.current_param_name = param_name
            self.current_param_value = ""

            if param_name:
                if not self.parameters:
                    # 第一个参数
                    json_start = f'{{"{param_name}": '
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.tool_call_index - 1,
                                id=self.current_tool_id,
                                type="function",
                                arguments=json_start
                            )
                        ]
                    )
                    self.json_started = True
                else:
                    # 后续参数
                    json_continue = f', "{param_name}": '
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.tool_call_index - 1,
                                id=self.current_tool_id,
                                type="function",
                                arguments=json_continue
                            )
                        ]
                    )
                self._emit_delta(delta)

    def _char_data(self, data: str):
        """XML 字符数据处理"""
        if data and self.current_param_name and self.current_function_name:
            # 获取参数类型
            param_type = self._get_param_type(self.current_param_name)

            # 处理换行
            if not self.current_param_value and data.startswith('\n'):
                data = data[1:]

            self.current_param_value += data

            # 转换并输出
            converted_value = self._convert_param_value(
                self.current_param_name,
                self.current_param_value,
                self.current_function_name,
                self.streaming_tools
            )

            # 生成 JSON 输出
            if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
                output_data = json.dumps(converted_value, ensure_ascii=False)[1:-1]
            else:
                output_data = json.dumps(converted_value, ensure_ascii=False)

            # 计算增量
            prev_output = getattr(self, '_prev_param_output', '')
            delta_data = output_data[len(prev_output):]
            self._prev_param_output = output_data

            if delta_data:
                delta = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.tool_call_index - 1,
                            id=self.current_tool_id,
                            type="function",
                            arguments=delta_data
                        )
                    ]
                )
                self._emit_delta(delta)

    def _end_element(self, name: str):
        """XML 结束元素处理"""
        if name == "parameter" or name.startswith("parameter"):
            if self.current_param_name:
                # 完成参数
                param_type = self._get_param_type(self.current_param_name)

                if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
                    # 添加结束引号
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.tool_call_index - 1,
                                id=self.current_tool_id,
                                type="function",
                                arguments='"'
                            )
                        ]
                    )
                    self._emit_delta(delta)

                self.parameters[self.current_param_name] = self.current_param_value
                self.current_param_name = None
                self.current_param_value = ""
                self._prev_param_output = ''

        elif name == "function" or name.startswith("function"):
            # 关闭 JSON 对象
            if self.parameters:
                delta = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.tool_call_index - 1,
                            id=self.current_tool_id,
                            type="function",
                            arguments="}"
                        )
                    ]
                )
            else:
                delta = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.tool_call_index - 1,
                            id=self.current_tool_id,
                            type="function",
                            arguments="{}"
                        )
                    ]
                )
            self._emit_delta(delta)
            self.current_function_open = False

        elif name == "tool_call":
            # 最终 delta
            delta = DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=self.tool_call_index - 1,
                        id=self.current_tool_id,
                        type="function",
                        arguments=""
                    )
                ]
            )
            self._emit_delta(delta)
            self._reset_tool_call_state()

    def _finalize_current_tool_call(self):
        """完成当前工具调用"""
        if self.current_param_name:
            self._end_element("parameter")
        if self.current_function_open:
            self._end_element("function")
        self._end_element("tool_call")

    def _reset_tool_call_state(self):
        """重置单个工具调用的状态"""
        self.current_tool_id = None
        self.current_function_name = None
        self.current_function_open = False
        self.parameters = {}
        self.current_param_name = None
        self.current_param_value = ""
        self.header_sent = False
        self.json_started = False
        self.json_closed = False
        self.text_content_buffer = ""
        self._prev_param_output = ''

        # 重新创建解析器
        self._create_parser()

    def _get_param_type(self, param_name: str) -> str:
        """获取参数类型"""
        if not self.streaming_tools or not self.current_function_name:
            return "string"

        for tool in self.streaming_tools:
            tool_func = tool.get("function", {})
            if tool_func.get("name") == self.current_function_name:
                props = tool_func.get("parameters", {}).get("properties", {})
                if param_name in props:
                    return props[param_name].get("type", "string")
                break

        return "string"

    def _emit_delta(self, delta: DeltaMessage):
        """发送 delta"""
        self.deltas.append(delta)

        # 更新跟踪数组
        if delta.tool_calls:
            for tc in delta.tool_calls:
                if tc.function:
                    idx = tc.index if tc.index is not None else self.tool_call_index - 1
                    while len(self.prev_tool_call_arr) <= idx:
                        self.prev_tool_call_arr.append({"name": "", "arguments": ""})
                    while len(self.streamed_args_for_tool) <= idx:
                        self.streamed_args_for_tool.append("")

                    if tc.function.name:
                        self.prev_tool_call_arr[idx]["name"] = tc.function.name
                    if tc.function.arguments is not None:
                        self.streamed_args_for_tool[idx] += tc.function.arguments
                        self.prev_tool_call_arr[idx]["arguments"] = self.streamed_args_for_tool[idx]

    def _merge_new_deltas(self, initial_count: int) -> Optional[DeltaMessage]:
        """合并新生成的 deltas"""
        if len(self.deltas) <= initial_count:
            return None

        new_deltas = self.deltas[initial_count:]
        if len(new_deltas) == 1:
            return new_deltas[0]

        # 合并多个 deltas
        merged_content = ""
        merged_reasoning = ""
        merged_tool_calls: List[DeltaToolCall] = []

        for delta in new_deltas:
            if delta.content:
                merged_content += delta.content
            if delta.reasoning:
                merged_reasoning += delta.reasoning
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    # 查找是否已有相同 id 的 tool_call
                    existing = None
                    for etc in merged_tool_calls:
                        if etc.id == tc.id:
                            existing = etc
                            break

                    if existing:
                        if tc.function:
                            if tc.function.name:
                                existing.function.name = tc.function.name
                            if tc.function.arguments is not None:
                                if existing.function.arguments is None:
                                    existing.function.arguments = ""
                                existing.function.arguments += tc.function.arguments
                            if tc.type:
                                existing.type = tc.type
                    else:
                        merged_tool_calls.append(tc)

        return DeltaMessage(
            content=merged_content if merged_content else None,
            reasoning=merged_reasoning if merged_reasoning else None,
            tool_calls=merged_tool_calls if merged_tool_calls else None
        )
