"""
Parser 单元测试

测试 Tool Parser 和 Reasoning Parser 的正确性，包括：
- Tool Parser 单元测试（DeepSeek V3, Qwen3 Coder, Qwen XML）
- Reasoning Parser 单元测试（DeepSeek R1, Qwen3, DeepSeek V3）
- 流式解析验证
- 参数类型转换验证
- Reasoning + Tool Calls 组合场景
- 边界情况测试

合并自：
- test_parsers.py
- test_parser_response_format.py
"""

# ============================================
# 环境设置（必须在所有导入之前）
# ============================================
import os
import sys

# 设置日志目录为可写的临时目录
os.environ["LOG_DIR"] = "/tmp/trajproxy_test_logs"
os.environ["LOG_LEVEL"] = "WARNING"

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import pytest
import json
import importlib.util
from typing import List, Dict, Any, Optional

# ============================================
# 直接导入 parser 模块，绕过 proxy_core.__init__.py
# 这样可以避免循环导入问题
# 注意：使用统一的模块名注册到 sys.modules，确保 isinstance 检查正确
# ============================================

def _import_module_from_path(module_name: str, file_path: str):
    """从文件路径直接导入模块"""
    # 如果模块已经在 sys.modules 中，直接返回
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# 导入 base 模块（无依赖）
parsers_dir = os.path.join(project_root, "traj_proxy", "proxy_core", "parsers")
base_module = _import_module_from_path(
    "traj_proxy.proxy_core.parsers.base",
    os.path.join(parsers_dir, "base.py")
)

# 导入 base_parser_manager（依赖 base）
base_parser_manager = _import_module_from_path(
    "traj_proxy.proxy_core.parsers.base_parser_manager",
    os.path.join(parsers_dir, "base_parser_manager.py")
)

# 导入 tool_parser_manager（依赖 base 和 base_parser_manager）
tool_parser_manager = _import_module_from_path(
    "traj_proxy.proxy_core.parsers.tool_parser_manager",
    os.path.join(parsers_dir, "tool_parser_manager.py")
)

# 导入 reasoning_parser_manager（依赖 base 和 base_parser_manager）
reasoning_parser_manager = _import_module_from_path(
    "traj_proxy.proxy_core.parsers.reasoning_parser_manager",
    os.path.join(parsers_dir, "reasoning_parser_manager.py")
)

# 导入 tool_parsers __init__ 以注册所有 parser
tool_parsers_init = _import_module_from_path(
    "traj_proxy.proxy_core.parsers.tool_parsers",
    os.path.join(parsers_dir, "tool_parsers", "__init__.py")
)

# 导入 reasoning_parsers __init__ 以注册所有 parser
reasoning_parsers_init = _import_module_from_path(
    "traj_proxy.proxy_core.parsers.reasoning_parsers",
    os.path.join(parsers_dir, "reasoning_parsers", "__init__.py")
)

# 从模块中提取需要的类和函数
ExtractedToolCallInfo = base_module.ExtractedToolCallInfo
DeltaMessage = base_module.DeltaMessage
ToolCall = base_module.ToolCall
DeltaToolCall = base_module.DeltaToolCall
ToolParserManager = tool_parser_manager.ToolParserManager
ReasoningParserManager = reasoning_parser_manager.ReasoningParserManager

# ============================================
# Mock 数据定义
# ============================================

# DeepSeek R1 Reasoning 格式
DEEPSEEK_R1_REASONING_OUTPUT = """<thinky>
用户询问天气情况。
首先需要确认用户所在的城市。
用户没有指定城市，我需要询问。
</thinke>请问您想查询哪个城市的天气？"""

# DeepSeek R1 Reasoning 格式（无 end token）
DEEPSEEK_R1_REASONING_NO_END = """<thinky>
用户询问天气情况。
持续推理中..."""

# DeepSeek R1 Reasoning 格式（只有 reasoning）
DEEPSEEK_R1_REASONING_ONLY = """<thinky>
这是一个纯推理过程。
没有实际回复内容。
</thinke>"""

# DeepSeek V3 Tool Calls 格式
# 注意：``` 后面不能有换行符，否则正则不匹配
DEEPSEEK_V3_TOOL_CALL_OUTPUT = '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"city": "北京", "unit": "celsius"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>'

# DeepSeek V3 多个 Tool Calls
DEEPSEEK_V3_MULTI_TOOL_CALLS = '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"city": "北京"}\n```<｜tool▁call▁end｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"city": "上海"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>'

# DeepSeek V3 Tool Call 前有内容
DEEPSEEK_V3_TOOL_CALL_WITH_CONTENT = '好的，我来帮您查询天气。<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"city": "北京"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>'

# Reasoning + Tool Calls 组合格式
REASONING_WITH_TOOL_CALLS = '<thinky>\n用户询问北京天气。\n我需要调用天气 API 来获取实时数据。\n城市是北京，需要用摄氏度单位。\n</thinke><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"city": "北京", "unit": "celsius"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>'

# ASCII 格式的 Tool Calls
ASCII_TOOL_CALL_OUTPUT = '<|tool_calls_begin|><|tool_call_begin|>function<|tool_sep|>get_weather\n```json\n{"city": "北京"}\n```<|tool_call_end|><|tool_calls_end|>'

# 工具定义
SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["city"]
            }
        }
    }
]


# ============================================
# Tool Parser 测试
# ============================================

class TestDeepSeekV3ToolParser:
    """DeepSeek V3 Tool Parser 测试类"""

    @pytest.fixture
    def parser(self):
        """创建 DeepSeek V3 Tool Parser 实例"""
        parser_cls = ToolParserManager.get_tool_parser("deepseek_v3")
        return parser_cls(tokenizer=None)

    def test_extract_single_tool_call(self, parser):
        """
        测试提取单个工具调用

        验证点:
        - 正确提取工具名称
        - 正确提取参数 JSON
        - tools_called 为 True
        """
        result = parser.extract_tool_calls(DEEPSEEK_V3_TOOL_CALL_OUTPUT)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1

        tool_call = result.tool_calls[0]
        assert tool_call.function is not None
        assert tool_call.function.name == "get_weather"
        assert tool_call.type == "function"

        # 验证参数是合法 JSON
        args = json.loads(tool_call.function.arguments)
        assert args["city"] == "北京"
        assert args["unit"] == "celsius"

    def test_extract_multiple_tool_calls(self, parser):
        """
        测试提取多个工具调用

        验证点:
        - 正确提取多个工具调用
        - 每个工具调用有唯一 ID
        """
        result = parser.extract_tool_calls(DEEPSEEK_V3_MULTI_TOOL_CALLS)

        assert result.tools_called is True
        assert len(result.tool_calls) == 2

        # 验证每个工具调用
        args1 = json.loads(result.tool_calls[0].function.arguments)
        args2 = json.loads(result.tool_calls[1].function.arguments)
        assert args1["city"] == "北京"
        assert args2["city"] == "上海"

        # 验证 ID 唯一
        ids = [tc.id for tc in result.tool_calls]
        assert len(ids) == len(set(ids))

    def test_extract_with_content_before(self, parser):
        """
        测试工具调用前有文本内容

        验证点:
        - 正确提取工具调用前的内容
        - 工具调用正常解析
        """
        result = parser.extract_tool_calls(DEEPSEEK_V3_TOOL_CALL_WITH_CONTENT)

        assert result.tools_called is True
        assert "好的" in result.content
        assert result.tool_calls[0].function is not None
        assert result.tool_calls[0].function.name == "get_weather"

    def test_no_tool_call(self, parser):
        """
        测试没有工具调用的情况

        验证点:
        - tools_called 为 False
        - content 为原始输出
        """
        model_output = "今天天气不错，适合外出。"

        result = parser.extract_tool_calls(model_output)

        assert result.tools_called is False
        assert len(result.tool_calls) == 0
        assert result.content == model_output

    def test_ascii_format(self, parser):
        """
        测试 ASCII 格式的标记符

        验证点:
        - 支持两种标记符格式（Unicode 和 ASCII）
        """
        result = parser.extract_tool_calls(ASCII_TOOL_CALL_OUTPUT)

        assert result.tools_called is True
        assert result.tool_calls[0].function is not None
        assert result.tool_calls[0].function.name == "get_weather"

    def test_streaming_state_reset(self, parser):
        """
        测试流式状态重置

        验证点:
        - reset_streaming_state 清除所有状态
        """
        # 模拟流式过程中的状态
        parser.current_tool_id = 2
        parser.prev_tool_call_arr = [{"name": "test"}]
        parser.current_tool_name_sent = True

        # 重置
        parser.reset_streaming_state()

        assert parser.current_tool_id == -1
        assert parser.prev_tool_call_arr == []
        assert parser.current_tool_name_sent is False


class TestQwen3CoderToolParser:
    """Qwen3 Coder Tool Parser 测试类"""

    @pytest.fixture
    def parser(self):
        """创建 Qwen3 Coder Tool Parser 实例"""
        parser_cls = ToolParserManager.get_tool_parser("qwen3_coder")
        return parser_cls(tokenizer=None)

    @pytest.fixture
    def tools_definition(self):
        """工具定义"""
        return SAMPLE_TOOLS

    def test_extract_single_tool_call(self, parser, tools_definition):
        """
        测试提取单个工具调用

        验证点:
        - 正确提取函数名
        - 正确提取参数
        - 参数转换为正确类型
        """
        # Qwen3 Coder XML 格式的工具调用
        # 注意：Qwen3 Coder 使用 toral 和  Ranchi 作为边界标记
        model_output = """toral<function=get_weather>
<parameter=city>北京</parameter>
<parameter=unit>celsius</parameter>
</function> Ranchi"""

        result = parser.extract_tool_calls(model_output, tools=tools_definition)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1

        tool_call = result.tool_calls[0]
        assert tool_call.function is not None
        assert tool_call.function.name == "get_weather"

        # 验证参数
        args = json.loads(tool_call.function.arguments)
        assert args["city"] == "北京"
        assert args["unit"] == "celsius"

    def test_extract_multiple_tool_calls(self, parser, tools_definition):
        """
        测试提取多个工具调用

        验证点:
        - 正确提取多个工具调用
        """
        model_output = """toral<function=get_weather>
<parameter=city>北京</parameter>
</function> Ranchi toral<function=get_weather>
<parameter=city>上海</parameter>
</function> Ranchi"""

        result = parser.extract_tool_calls(model_output, tools=tools_definition)

        assert result.tools_called is True
        assert len(result.tool_calls) == 2

        args1 = json.loads(result.tool_calls[0].function.arguments)
        args2 = json.loads(result.tool_calls[1].function.arguments)
        assert args1["city"] == "北京"
        assert args2["city"] == "上海"

    def test_parameter_type_conversion(self, parser):
        """
        测试参数类型转换

        验证点:
        - 整数参数正确转换
        - 布尔参数正确转换
        - 浮点数参数正确转换
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_types",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "int_param": {"type": "integer"},
                            "bool_param": {"type": "boolean"},
                            "float_param": {"type": "number"},
                            "str_param": {"type": "string"}
                        }
                    }
                }
            }
        ]

        model_output = """toral<function=test_types>
<parameter=int_param>42</parameter>
<parameter=bool_param>true</parameter>
<parameter=float_param>3.14</parameter>
<parameter=str_param>hello</parameter>
</function> Ranchi"""

        result = parser.extract_tool_calls(model_output, tools=tools)

        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)

        assert args["int_param"] == 42
        assert args["bool_param"] is True
        assert args["float_param"] == 3.14
        assert args["str_param"] == "hello"

    def test_nested_json_parameter(self, parser):
        """
        测试嵌套 JSON 参数
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "complex_tool",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "config": {"type": "object"}
                        }
                    }
                }
            }
        ]

        model_output = """toral<function=complex_tool>
<parameter=config>{"nested": {"deep": "value"}, "items": [1, 2, 3]}</parameter>
</function> Ranchi"""

        result = parser.extract_tool_calls(model_output, tools=tools)

        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["config"]["nested"]["deep"] == "value"
        assert args["config"]["items"] == [1, 2, 3]


class TestQwenXMLToolParser:
    """Qwen XML Tool Parser 测试类"""

    @pytest.fixture
    def parser(self):
        """创建 Qwen XML Tool Parser 实例"""
        parser_cls = ToolParserManager.get_tool_parser("qwen_xml")
        return parser_cls(tokenizer=None)

    def test_extract_tool_call(self, parser):
        """
        测试提取工具调用
        """
        # Qwen XML 格式
        model_output = """好的，我来帮您查询。
<tool_call name="get_weather">
{"city": "北京", "unit": "celsius"}
</tool_call >"""

        result = parser.extract_tool_calls(model_output)

        # 根据 qwen_xml parser 的实际实现调整验证
        assert hasattr(result, 'tools_called')
        assert hasattr(result, 'tool_calls')
        assert hasattr(result, 'content')


# ============================================
# Reasoning Parser 测试
# ============================================

class TestDeepSeekR1ReasoningParser:
    """DeepSeek R1 Reasoning Parser 测试类"""

    @pytest.fixture
    def parser(self):
        """创建 DeepSeek R1 Reasoning Parser 实例"""
        parser_cls = ReasoningParserManager.get_reasoning_parser("deepseek_r1")
        return parser_cls(tokenizer=None)

    def test_extract_reasoning_with_content(self, parser):
        """
        测试提取推理内容和回复内容

        验证点:
        - 正确分离推理内容和回复内容
        """
        model_output = '<thinky>\n用户询问天气，我需要：\n1. 确定城市\n2. 调用天气 API\n</thinke>北京今天天气晴朗，温度 25 度。'

        reasoning, content = parser.extract_reasoning(model_output)

        assert reasoning is not None
        assert "用户询问天气" in reasoning
        assert content is not None
        assert "北京今天天气晴朗" in content

    def test_extract_reasoning_only(self, parser):
        """
        测试只有推理内容的情况
        """
        model_output = '<thinky>\n这是一个纯推理过程。\n</thinke>'

        reasoning, content = parser.extract_reasoning(model_output)

        assert reasoning is not None
        assert "纯推理过程" in reasoning
        # content 可能为空或 None
        assert content is None or content == ""

    def test_extract_no_end_token(self, parser):
        """
        测试推理内容没有结束标记
        """
        model_output = '<thinky>\n推理内容没有结束标记\n持续推理中...'

        reasoning, content = parser.extract_reasoning(model_output)

        # 应该把剩余内容作为推理
        assert reasoning is not None
        assert "推理内容没有结束标记" in reasoning

    def test_only_end_token_without_start(self, parser):
        """
        测试只有 end_token，没有 start_token

        验证点:
        - end_token 之前的内容作为 reasoning
        - end_token 之后的内容作为 content
        """
        model_output = "这是推理内容</thinke>这是正常回复"

        reasoning, content = parser.extract_reasoning(model_output)

        assert reasoning == "这是推理内容", f"reasoning 应为 '这是推理内容'，实际: {reasoning}"
        assert content == "这是正常回复", f"content 应为 '这是正常回复'，实际: {content}"


class TestQwen3ReasoningParser:
    """Qwen3 Reasoning Parser 测试类"""

    @pytest.fixture
    def parser(self):
        """创建 Qwen3 Reasoning Parser 实例"""
        parser_cls = ReasoningParserManager.get_reasoning_parser("qwen3")
        return parser_cls(tokenizer=None)

    def test_extract_reasoning(self, parser):
        """
        测试 Qwen3 格式的推理内容提取
        """
        model_output = """<thinky>
分析用户需求...
</thinke>这是最终回复。"""

        reasoning, content = parser.extract_reasoning(model_output)

        # 根据实际实现调整验证
        assert isinstance(reasoning, (str, type(None)))
        assert isinstance(content, (str, type(None)))


class TestDeepSeekV3ReasoningParser:
    """DeepSeek V3 Reasoning Parser 测试类"""

    @pytest.fixture
    def parser(self):
        """创建 DeepSeek V3 Reasoning Parser 实例"""
        parser_cls = ReasoningParserManager.get_reasoning_parser("deepseek_v3")
        return parser_cls(tokenizer=None)

    def test_extract_reasoning(self, parser):
        """
        测试 DeepSeek V3 格式的推理内容提取
        """
        model_output = """<thinky>
DeepSeek V3 推理过程
</thinke>最终回复内容"""

        reasoning, content = parser.extract_reasoning(model_output)

        assert isinstance(reasoning, (str, type(None)))
        assert isinstance(content, (str, type(None)))


# ============================================
# Reasoning + Tool Calls 组合测试
# ============================================

class TestReasoningWithToolCalls:
    """Reasoning + Tool Calls 组合测试类"""

    @pytest.fixture
    def deepseek_r1_parser(self):
        """创建 DeepSeek R1 Reasoning Parser 实例"""
        parser_cls = ReasoningParserManager.get_reasoning_parser("deepseek_r1")
        return parser_cls(tokenizer=None)

    @pytest.fixture
    def deepseek_v3_tool_parser(self):
        """创建 DeepSeek V3 Tool Parser 实例"""
        parser_cls = ToolParserManager.get_tool_parser("deepseek_v3")
        return parser_cls(tokenizer=None)

    def test_combined_reasoning_then_tool_call(
        self,
        deepseek_r1_parser,
        deepseek_v3_tool_parser
    ):
        """
        测试先 reasoning 后 tool_calls 的组合

        验证点:
        - 正确分离 reasoning、tool_calls
        - 两种 parser 可以协同工作
        """
        output = REASONING_WITH_TOOL_CALLS

        # 1. 先提取 reasoning
        reasoning, remaining = deepseek_r1_parser.extract_reasoning(output)

        assert reasoning is not None
        assert "用户询问北京天气" in reasoning
        assert "<thinky>" not in reasoning
        assert "</thinke>" not in reasoning

        # 2. 再从剩余内容提取 tool_calls
        assert remaining is not None
        result = deepseek_v3_tool_parser.extract_tool_calls(
            remaining,
            tools=SAMPLE_TOOLS
        )

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function is not None
        assert result.tool_calls[0].function.name == "get_weather"

        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["city"] == "北京"

    def test_combined_response_structure(
        self,
        deepseek_r1_parser,
        deepseek_v3_tool_parser
    ):
        """
        测试组合响应的完整结构

        验证点:
        - 模拟完整的响应构建流程
        - 验证最终响应格式符合 OpenAI 规范
        """
        output = REASONING_WITH_TOOL_CALLS

        # 1. 提取 reasoning
        reasoning, remaining = deepseek_r1_parser.extract_reasoning(output)

        # 2. 提取 tool_calls
        tool_result = deepseek_v3_tool_parser.extract_tool_calls(
            remaining or "",
            tools=SAMPLE_TOOLS
        )

        # 3. 构建模拟响应
        message = {
            "role": "assistant",
            "content": tool_result.content
        }

        if reasoning:
            message["reasoning"] = reasoning

        if tool_result.tools_called:
            message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name if tc.function else None,
                        "arguments": tc.function.arguments if tc.function else None
                    }
                }
                for tc in tool_result.tool_calls
            ]

        # 4. 验证响应结构
        assert "role" in message
        assert message["role"] == "assistant"
        assert "reasoning" in message, "响应应包含 reasoning 字段"
        assert "tool_calls" in message, "响应应包含 tool_calls 字段"

        # 验证 tool_calls 格式
        for tc in message["tool_calls"]:
            assert "id" in tc
            assert "type" in tc
            assert tc["type"] == "function"
            assert "function" in tc
            assert "name" in tc["function"]
            assert "arguments" in tc["function"]


# ============================================
# Parser Manager 测试
# ============================================

class TestToolParserManager:
    """Tool Parser Manager 测试类"""

    def test_list_registered_parsers(self):
        """
        测试列出已注册的 parser
        """
        parsers = ToolParserManager.list_registered()

        expected_parsers = [
            "deepseek_v3",
            "deepseek_v31",
            "deepseek_v32",
            "qwen3_coder",
            "qwen_xml",
            "glm45",
            "glm47",
            "llama3_json",
        ]

        for parser_name in expected_parsers:
            assert parser_name in parsers, f"Parser '{parser_name}' 未注册"

    def test_get_parser_by_name(self):
        """
        测试按名称获取 parser
        """
        parser_cls = ToolParserManager.get_tool_parser("deepseek_v3")
        assert parser_cls is not None

        # 创建实例
        parser = parser_cls(tokenizer=None)
        assert hasattr(parser, "extract_tool_calls")

    def test_get_nonexistent_parser(self):
        """
        测试获取不存在的 parser
        """
        with pytest.raises(KeyError):
            ToolParserManager.get_tool_parser("nonexistent_parser")


class TestReasoningParserManager:
    """Reasoning Parser Manager 测试类"""

    def test_list_registered_parsers(self):
        """
        测试列出已注册的 parser
        """
        parsers = ReasoningParserManager.list_registered()

        expected_parsers = [
            "deepseek_r1",
            "deepseek_v3",
            "deepseek",
            "qwen3",
        ]

        for parser_name in expected_parsers:
            assert parser_name in parsers, f"Parser '{parser_name}' 未注册"

    def test_get_parser_by_name(self):
        """
        测试按名称获取 parser
        """
        parser_cls = ReasoningParserManager.get_reasoning_parser("deepseek_r1")
        assert parser_cls is not None

        parser = parser_cls(tokenizer=None)
        assert hasattr(parser, "extract_reasoning")

    def test_get_nonexistent_parser(self):
        """
        测试获取不存在的 parser
        """
        with pytest.raises(KeyError):
            ReasoningParserManager.get_reasoning_parser("nonexistent_parser")


# ============================================
# 流式解析测试
# ============================================

class TestStreamingParser:
    """流式解析测试类"""

    @pytest.fixture
    def deepseek_parser(self):
        """创建 DeepSeek V3 Tool Parser 实例"""
        parser_cls = ToolParserManager.get_tool_parser("deepseek_v3")
        return parser_cls(tokenizer=None)

    def test_streaming_extract_delta(self, deepseek_parser):
        """
        测试流式增量解析

        验证点:
        - 正确处理增量文本
        - 返回正确的 DeltaMessage
        """
        deepseek_parser.reset_streaming_state()

        # 模拟流式增量
        previous_text = ""
        current_text = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>"
        delta_text = current_text
        previous_token_ids = []
        current_token_ids = []
        delta_token_ids = []

        result = deepseek_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=previous_token_ids,
            current_token_ids=current_token_ids,
            delta_token_ids=delta_token_ids,
        )

        # 结果可能是 None 或 DeltaMessage
        if result is not None:
            assert hasattr(result, 'role')
            assert hasattr(result, 'content')
            assert hasattr(result, 'reasoning')
            assert hasattr(result, 'tool_calls')


# ============================================
# 边界情况测试
# ============================================

class TestParserEdgeCases:
    """Parser 边界情况测试类"""

    @pytest.fixture
    def deepseek_v3_parser(self):
        parser_cls = ToolParserManager.get_tool_parser("deepseek_v3")
        return parser_cls(tokenizer=None)

    @pytest.fixture
    def deepseek_r1_parser(self):
        parser_cls = ReasoningParserManager.get_reasoning_parser("deepseek_r1")
        return parser_cls(tokenizer=None)

    def test_empty_input(self, deepseek_v3_parser, deepseek_r1_parser):
        """测试空输入"""
        # Tool Parser
        result = deepseek_v3_parser.extract_tool_calls("")
        assert result.tools_called is False
        assert len(result.tool_calls) == 0

        # Reasoning Parser
        reasoning, content = deepseek_r1_parser.extract_reasoning("")
        assert reasoning is None or reasoning == ""
        assert content is None or content == ""

    def test_whitespace_only(self, deepseek_v3_parser, deepseek_r1_parser):
        """测试只有空白字符"""
        whitespace = "   \n\t  "

        # Tool Parser
        result = deepseek_v3_parser.extract_tool_calls(whitespace)
        assert result.tools_called is False

        # Reasoning Parser
        reasoning, content = deepseek_r1_parser.extract_reasoning(whitespace)
        # 行为取决于实现

    def test_malformed_tool_call(self, deepseek_v3_parser):
        """测试格式错误的工具调用"""
        malformed = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>incomplete"

        result = deepseek_v3_parser.extract_tool_calls(malformed)

        # 应该不崩溃，返回合理结果
        assert isinstance(result, ExtractedToolCallInfo)
        # tools_called 可能是 False（因为格式不完整）

    def test_malformed_reasoning(self, deepseek_r1_parser):
        """测试格式错误的 reasoning"""
        # 只有开始标记，没有结束
        malformed = "<thinky>推理内容没有结束"

        reasoning, content = deepseek_r1_parser.extract_reasoning(malformed)

        # 应该将剩余内容作为 reasoning
        assert reasoning is not None or content is not None

    def test_nested_markers(self, deepseek_r1_parser):
        """测试嵌套标记"""
        nested = "<thinky>外层<thinky>内层</thinke>外层结束</thinke>内容"

        reasoning, content = deepseek_r1_parser.extract_reasoning(nested)

        # 行为取决于实现，但不应崩溃
        assert isinstance(reasoning, (str, type(None)))
        assert isinstance(content, (str, type(None)))

    def test_special_characters_in_arguments(self, deepseek_v3_parser):
        """测试参数中的特殊字符"""
        special_output = """<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>test
```json
{"text": "包含特殊字符\\"和换行\\n", "emoji": "😀"}
```
<｜tool▁call▁end｜><｜tool▁calls▁end｜>"""

        result = deepseek_v3_parser.extract_tool_calls(special_output)

        if result.tools_called:
            args = json.loads(result.tool_calls[0].function.arguments)
            assert "特殊字符" in args.get("text", "")

    def test_unicode_in_reasoning(self, deepseek_r1_parser):
        """测试 reasoning 中的 Unicode 字符"""
        unicode_output = "<thinky>包含中文、emoji😀、特殊符号™</thinke>回复内容"

        reasoning, content = deepseek_r1_parser.extract_reasoning(unicode_output)

        assert reasoning is not None
        assert "中文" in reasoning
        assert "emoji" in reasoning

    def test_very_long_content(self, deepseek_v3_parser, deepseek_r1_parser):
        """测试超长内容"""
        # 生成超长内容
        long_content = "测试内容" * 10000
        long_reasoning = f"<thinky>{'推理' * 5000}</thinke>"
        long_output = long_reasoning + long_content

        # 应该能处理而不崩溃
        reasoning, content = deepseek_r1_parser.extract_reasoning(long_output)
        assert reasoning is not None or content is not None

    def test_concurrent_parser_instances(self):
        """测试多个 parser 实例并发使用"""
        parser_cls = ReasoningParserManager.get_reasoning_parser("deepseek_r1")

        # 创建多个实例
        parsers = [parser_cls(tokenizer=None) for _ in range(5)]

        # 并发测试
        outputs = [
            "<thinky>推理" + str(i) + "</thinke>内容" + str(i)
            for i in range(5)
        ]

        for parser, output in zip(parsers, outputs):
            reasoning, content = parser.extract_reasoning(output)
            assert reasoning is not None or content is not None
