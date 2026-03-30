"""
Parser 单元测试
"""

import json
import pytest

from traj_proxy.proxy_core.parsers import ParserManager
from traj_proxy.proxy_core.parsers.base import (
    ToolCall, DeltaToolCall, ExtractedToolCallInfo, DeltaMessage
)
from traj_proxy.proxy_core.parsers.tool_parsers.qwen_xml_parser import QwenXMLToolParser
from traj_proxy.proxy_core.parsers.tool_parsers.deepseek_v3_parser import DeepSeekV3ToolParser
from traj_proxy.proxy_core.parsers.tool_parsers.llama3_json_parser import Llama3JSONToolParser
from traj_proxy.proxy_core.parsers.reasoning_parsers.qwen3_reasoning_parser import Qwen3ReasoningParser
from traj_proxy.proxy_core.parsers.reasoning_parsers.deepseek_reasoning_parser import DeepSeekReasoningParser


class TestQwenXMLToolParser:
    """Qwen XML 工具解析器测试"""

    def setup_method(self):
        self.parser = QwenXMLToolParser()

    def test_extract_tool_calls_single(self):
        """测试单个工具调用解析"""
        output = '''ournemouth<function=get_weather>
<parameter=city>北京</parameter>
<parameter=unit>celsius</parameter>
</function> Ranchi'''

        result = self.parser.extract_tool_calls(output)

        assert result.tools_called == True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_weather"

        args = json.loads(result.tool_calls[0].arguments)
        assert args["city"] == "北京"
        assert args["unit"] == "celsius"

    def test_extract_tool_calls_without_outer_tokens(self):
        """测试没有最外层标记的工具调用解析"""
        output = '''这是之前的文本<function=get_weather>
<parameter=city>北京</parameter>
</function>'''

        result = self.parser.extract_tool_calls(output)

        assert result.tools_called == True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_weather"
        assert result.content == "这是之前的文本"

    def test_extract_tool_calls_no_tools(self):
        """测试无工具调用的情况"""
        output = "这是一个普通的回复，没有工具调用。"

        result = self.parser.extract_tool_calls(output)

        assert result.tools_called == False
        assert len(result.tool_calls) == 0
        assert result.content == output

    def test_extract_tool_calls_with_type_conversion(self):
        """测试参数类型转换"""
        output = '''<function=calculate>
<parameter=num>42</parameter>
<parameter>3.14</parameter>
<parameter=flag>true</parameter>
</function>'''

        tools = [{
            "type": "function",
            "function": {
                "name": "calculate",
                "parameters": {
                    "properties": {
                        "num": {"type": "integer"},
                        "pi": {"type": "number"},
                        "flag": {"type": "boolean"}
                    }
                }
            }
        }]

        result = self.parser.extract_tool_calls(output, tools=tools)

        assert result.tools_called == True
        args = json.loads(result.tool_calls[0].arguments)
        assert args["num"] == 42
        assert args.get("pi", 3.14) == 3.14  # 默认值


class TestDeepSeekV3ToolParser:
    """DeepSeek V3 工具解析器测试"""

    def setup_method(self):
        self.parser = DeepSeekV3ToolParser()

    def test_extract_tool_calls_single(self):
        """测试单个工具调用解析"""
        output = '''<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
```json
{"city": "北京"}
```
<｜tool▁call▁end｜><｜tool▁calls▁end｜>'''

        result = self.parser.extract_tool_calls(output)

        assert result.tools_called == True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_weather"

        args = json.loads(result.tool_calls[0].arguments)
        assert args["city"] == "北京"

    def test_extract_tool_calls_ascii_variant(self):
        """测试 ASCII 变体标记"""
        output = '''<|tool_calls_begin|><|tool_call_begin|>function<|tool_sep|>get_weather
```json
{"city": "上海"}
```
<|tool_call_end|><|tool_calls_end|>'''

        result = self.parser.extract_tool_calls(output)

        assert result.tools_called == True
        assert result.tool_calls[0].name == "get_weather"

    def test_extract_tool_calls_no_tools(self):
        """测试无工具调用的情况"""
        output = "这是一个普通的回复"

        result = self.parser.extract_tool_calls(output)

        assert result.tools_called == False
        assert result.content == output


class TestLlama3JSONToolParser:
    """Llama3 JSON 工具解析器测试"""

    def setup_method(self):
        self.parser = Llama3JSONToolParser()

    def test_extract_tool_calls_with_python_tag(self):
        """测试带 python_tag 的工具调用"""
        output = '<|python_tag|>{"name": "get_weather", "parameters": {"city": "北京"}}'

        result = self.parser.extract_tool_calls(output)

        assert result.tools_called == True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_weather"

        args = json.loads(result.tool_calls[0].arguments)
        assert args["city"] == "北京"

    def test_extract_tool_calls_json_only(self):
        """测试仅 JSON 格式的工具调用"""
        output = '这是之前的文本{"name": "get_weather", "parameters": {"city": "上海"}}'

        result = self.parser.extract_tool_calls(output)

        assert result.tools_called == True
        assert result.tool_calls[0].name == "get_weather"
        assert result.content == "这是之前的文本"

    def test_extract_tool_calls_no_tools(self):
        """测试无工具调用的情况"""
        output = "这是一个普通的回复"

        result = self.parser.extract_tool_calls(output)

        assert result.tools_called == False


class TestQwen3ReasoningParser:
    """Qwen3 推理解析器测试"""

    def setup_method(self):
        self.parser = Qwen3ReasoningParser()

    def test_extract_reasoning(self):
        """测试推理内容解析"""
        output = '<thinky>这是推理过程</thinke>这是正常回复'

        reasoning, content = self.parser.extract_reasoning(output)

        assert reasoning == "这是推理过程"
        assert content == "这是正常回复"

    def test_extract_reasoning_no_end_token(self):
        """测试没有结束标记的情况"""
        output = '<thinky>这是推理过程'

        reasoning, content = self.parser.extract_reasoning(output)

        assert reasoning == "这是推理过程"

    def test_extract_reasoning_no_tokens(self):
        """测试没有推理标记的情况"""
        output = '这是普通回复'

        reasoning, content = self.parser.extract_reasoning(output)

        assert reasoning is None
        assert content == "这是普通回复"


class TestDeepSeekReasoningParser:
    """DeepSeek 推理解析器测试"""

    def setup_method(self):
        self.parser = DeepSeekReasoningParser()

    def test_extract_reasoning_unicode(self):
        """测试 Unicode 标记格式"""
        output = '<｜begin▁of▁think｜>这是推理过程<｜end▁of▁think｜>这是正常回复'

        reasoning, content = self.parser.extract_reasoning(output)

        assert reasoning == "这是推理过程"
        assert content == "这是正常回复"

    def test_extract_reasoning_ascii(self):
        """测试 ASCII 标记格式"""
        output = '<|begin_of_think|>这是推理过程<|end_of_think|>这是正常回复'

        reasoning, content = self.parser.extract_reasoning(output)

        assert reasoning == "这是推理过程"
        assert content == "这是正常回复"


class TestParserManager:
    """Parser 管理器测试"""

    def test_get_model_config_exact(self):
        """测试精确匹配模型配置"""
        config = ParserManager.get_model_config("qwen3-coder")

        assert config.get("tool_parser") == "qwen_xml"
        assert config.get("reasoning_parser") == "qwen3"

    def test_get_model_config_fuzzy(self):
        """测试模糊匹配模型配置"""
        config = ParserManager.get_model_config("Qwen3-Coder-30B-A3B-Instruct")

        assert config.get("tool_parser") == "qwen_xml"

    def test_create_parsers_for_model(self):
        """测试为模型创建 Parser"""
        tool_parser, reasoning_parser = ParserManager.create_parsers_for_model("qwen3-coder")

        assert tool_parser is not None
        assert isinstance(tool_parser, QwenXMLToolParser)
        assert reasoning_parser is not None
        assert isinstance(reasoning_parser, Qwen3ReasoningParser)

    def test_list_registered(self):
        """测试列出已注册项"""
        tool_parsers = ParserManager.list_tool_parsers()
        reasoning_parsers = ParserManager.list_reasoning_parsers()
        models = ParserManager.list_models()

        assert "qwen_xml" in tool_parsers
        assert "deepseek_v3" in tool_parsers
        assert "llama3_json" in tool_parsers

        assert "qwen3" in reasoning_parsers
        assert "deepseek" in reasoning_parsers

        assert "qwen3-coder" in models


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
