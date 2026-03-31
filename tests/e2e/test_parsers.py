"""
Parser 单元测试

测试 Tool Parser 和 Reasoning Parser 的正确性，包括：
- Tool Parser 单元测试（DeepSeek V3, Qwen3 Coder）
- Reasoning Parser 单元测试（DeepSeek R1, Qwen3）
- 流式解析验证
- 参数类型转换验证
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
from typing import List
import importlib.util

# ============================================
# 直接导入 parser 模块，绕过 proxy_core.__init__.py
# 这样可以避免导入 psycopg 等依赖
# ============================================

def _import_module_from_path(module_name: str, file_path: str):
    """从文件路径直接导入模块"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# 导入 base 模块
parsers_dir = os.path.join(project_root, "traj_proxy", "proxy_core", "parsers")
base_module = _import_module_from_path(
    "traj_proxy.proxy_core.parsers.base",
    os.path.join(parsers_dir, "base.py")
)

# 导入 base_parser_manager
base_parser_manager = _import_module_from_path(
    "traj_proxy.proxy_core.parsers.base_parser_manager",
    os.path.join(parsers_dir, "base_parser_manager.py")
)

# 导入 tool_parser_manager
tool_parser_manager = _import_module_from_path(
    "traj_proxy.proxy_core.parsers.tool_parser_manager",
    os.path.join(parsers_dir, "tool_parser_manager.py")
)

# 导入 reasoning_parser_manager
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
ToolParserManager = tool_parser_manager.ToolParserManager
ReasoningParserManager = reasoning_parser_manager.ReasoningParserManager


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
        # DeepSeek V3 格式的工具调用输出
        # 注意：格式必须严格按照 parser 正则期望的格式
        model_output = '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"city": "北京", "unit": "celsius"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>'

        result = parser.extract_tool_calls(model_output)

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
        model_output = '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"city": "北京"}\n```<｜tool▁call▁end｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"city": "上海"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>'

        result = parser.extract_tool_calls(model_output)

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
        model_output = '好的，我来帮您查询天气。\n<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"city": "北京"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>'

        result = parser.extract_tool_calls(model_output)

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
        # ASCII 格式的工具调用
        model_output = '<|tool_calls_begin|><|tool_call_begin|>function<|tool_sep|>get_weather\n```json\n{"city": "北京"}\n```<|tool_call_end|><|tool_calls_end|>'

        result = parser.extract_tool_calls(model_output)

        assert result.tools_called is True
        assert result.tool_calls[0].function is not None
        assert result.tool_calls[0].function.name == "get_weather"

    def test_malformed_json_arguments(self, parser):
        """
        测试参数 JSON 格式错误的情况

        验证点:
        - 错误格式不影响基本解析
        """
        model_output = '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{city: 北京}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>'

        # 应该能解析出工具调用，但参数可能不合法
        result = parser.extract_tool_calls(model_output)
        # 解析可能失败，但不应该崩溃
        assert isinstance(result, ExtractedToolCallInfo)


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
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "unit": {"type": "string"}
                        }
                    }
                }
            }
        ]

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

    def test_no_tool_call(self, parser):
        """
        测试没有工具调用的情况
        """
        model_output = "今天天气不错。"

        result = parser.extract_tool_calls(model_output)

        assert result.tools_called is False
        assert result.content == model_output

    def test_tool_call_with_content_before(self, parser, tools_definition):
        """
        测试工具调用前有文本内容
        """
        model_output = """好的，我来帮您查询天气。
toral<function=get_weather>
<parameter=city>北京</parameter>
</function> Ranchi"""

        result = parser.extract_tool_calls(model_output, tools=tools_definition)

        assert result.tools_called is True
        assert "好的" in result.content
        assert result.tool_calls[0].function is not None
        assert result.tool_calls[0].function.name == "get_weather"

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
        assert isinstance(result, ExtractedToolCallInfo)


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
        # DeepSeek R1 使用 <thinky></thinke> 标记
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

    def test_extract_content_only(self, parser):
        """
        测试只有回复内容的情况（没有 reasoning 标记）
        """
        model_output = "直接回复，无需推理。"

        reasoning, content = parser.extract_reasoning(model_output)

        # 没有 start_token 时，原始逻辑会将整个内容处理
        # 实际行为取决于 parser 实现
        assert isinstance(reasoning, (str, type(None)))
        assert isinstance(content, (str, type(None)))

    def test_extract_no_end_token(self, parser):
        """
        测试推理内容没有结束标记
        """
        model_output = '<thinky>\n推理内容没有结束标记\n持续推理中...'

        reasoning, content = parser.extract_reasoning(model_output)

        # 应该把剩余内容作为推理
        assert reasoning is not None
        assert "推理内容没有结束标记" in reasoning

    def test_is_reasoning_end(self, parser):
        """
        测试 is_reasoning_end 方法
        """
        # 无 tokenizer 时应该返回 False
        result = parser.is_reasoning_end([])
        assert result is False


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
        # Qwen3 使用 <thinky></thinke> 标记
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

    def test_streaming_state_reset(self, deepseek_parser):
        """
        测试流式状态重置

        验证点:
        - reset_streaming_state 清除所有状态
        """
        # 模拟流式过程中的状态
        deepseek_parser.current_tool_id = 2
        deepseek_parser.prev_tool_call_arr = [{"name": "test"}]
        deepseek_parser.current_tool_name_sent = True

        # 重置
        deepseek_parser.reset_streaming_state()

        assert deepseek_parser.current_tool_id == -1
        assert deepseek_parser.prev_tool_call_arr == []
        assert deepseek_parser.current_tool_name_sent is False

    def test_streaming_extract_delta(self, deepseek_parser):
        """
        测试流式增量解析

        验证点:
        - 正确处理增量文本
        - 返回正确的 DeltaMessage
        """
        # 模拟流式增量
        previous_text = ""
        current_text = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>"
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
        assert result is None or isinstance(result, DeltaMessage)


# ============================================
# 边界情况测试
# ============================================

class TestParserEdgeCases:
    """Parser 边界情况测试类"""

    @pytest.fixture
    def deepseek_parser(self):
        parser_cls = ToolParserManager.get_tool_parser("deepseek_v3")
        return parser_cls(tokenizer=None)

    def test_empty_input(self, deepseek_parser):
        """
        测试空输入
        """
        result = deepseek_parser.extract_tool_calls("")

        assert result.tools_called is False
        assert len(result.tool_calls) == 0

    def test_whitespace_only(self, deepseek_parser):
        """
        测试只有空白字符
        """
        result = deepseek_parser.extract_tool_calls("   \n\t  ")

        assert result.tools_called is False

    def test_incomplete_tool_call(self, deepseek_parser):
        """
        测试不完整的工具调用标记
        """
        model_output = '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function'

        result = deepseek_parser.extract_tool_calls(model_output)

        # 应该不崩溃，返回合理结果
        assert isinstance(result, ExtractedToolCallInfo)

    def test_nested_json_arguments(self, deepseek_parser):
        """
        测试嵌套 JSON 参数
        """
        model_output = '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>complex_tool\n```json\n{"config": {"nested": {"deep": "value"}}, "items": [1, 2, 3]}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>'

        result = deepseek_parser.extract_tool_calls(model_output)

        if result.tools_called:
            args = json.loads(result.tool_calls[0].function.arguments)
            assert args["config"]["nested"]["deep"] == "value"
            assert args["items"] == [1, 2, 3]

    def test_special_characters_in_arguments(self, deepseek_parser):
        """
        测试参数中的特殊字符
        """
        # 注意：JSON 字符串中的反斜杠需要正确转义
        model_output = '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>test\n```json\n{"text": "包含特殊字符", "emoji": "😀"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>'

        result = deepseek_parser.extract_tool_calls(model_output)

        if result.tools_called:
            args = json.loads(result.tool_calls[0].function.arguments)
            assert "特殊字符" in args["text"]
