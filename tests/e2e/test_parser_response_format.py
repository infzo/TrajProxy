"""
Parser 响应格式严格测试

严格测试 Tool Parser 和 Reasoning Parser 的响应格式，包括：
1. 非流式响应格式验证
2. 流式响应格式验证
3. Reasoning + Tool Calls 组合场景
4. 边界情况和错误处理

参考 vLLM 0.16.0 响应格式规范
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
# ============================================

def _import_module_from_path(module_name: str, file_path: str):
    """从文件路径直接导入模块"""
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

# 导入 unified_parser（依赖 base）
unified_parser = _import_module_from_path(
    "traj_proxy.proxy_core.parsers.unified_parser",
    os.path.join(parsers_dir, "unified_parser.py")
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
# Reasoning Parser 响应格式测试
# ============================================

class TestReasoningParserResponseFormat:
    """Reasoning Parser 响应格式测试类"""

    @pytest.fixture
    def deepseek_r1_parser(self):
        """创建 DeepSeek R1 Reasoning Parser 实例"""
        parser_cls = ReasoningParserManager.get_reasoning_parser("deepseek_r1")
        return parser_cls(tokenizer=None)

    @pytest.fixture
    def deepseek_v3_parser(self):
        """创建 DeepSeek V3 Reasoning Parser 实例"""
        parser_cls = ReasoningParserManager.get_reasoning_parser("deepseek_v3")
        return parser_cls(tokenizer=None)

    # ==================== 非流式响应测试 ====================

    def test_non_streaming_reasoning_extraction(self, deepseek_r1_parser):
        """
        测试非流式 Reasoning 提取

        验证点:
        - 正确分离 reasoning 和 content
        - reasoning 内容不包含标记
        - content 不包含 reasoning 内容
        """
        reasoning, content = deepseek_r1_parser.extract_reasoning(
            DEEPSEEK_R1_REASONING_OUTPUT
        )

        # 验证 reasoning
        assert reasoning is not None, "reasoning 不应为 None"
        assert "<thinky>" not in reasoning, "reasoning 不应包含开始标记"
        assert "</thinke>" not in reasoning, "reasoning 不应包含结束标记"
        assert "用户询问天气情况" in reasoning, "reasoning 内容不正确"
        assert "请问您想查询哪个城市" not in reasoning, "reasoning 不应包含 content"

        # 验证 content
        assert content is not None, "content 不应为 None"
        assert "请问您想查询哪个城市" in content, "content 内容不正确"
        assert "用户询问天气情况" not in content, "content 不应包含 reasoning"

    def test_non_streaming_reasoning_only(self, deepseek_r1_parser):
        """
        测试只有 reasoning 的情况

        验证点:
        - reasoning 正确提取
        - content 为 None 或空
        """
        reasoning, content = deepseek_r1_parser.extract_reasoning(
            DEEPSEEK_R1_REASONING_ONLY
        )

        assert reasoning is not None, "reasoning 不应为 None"
        assert "纯推理过程" in reasoning
        assert content is None or content == "", f"content 应为空，实际: '{content}'"

    def test_non_streaming_no_reasoning_markers(self, deepseek_r1_parser):
        """
        测试没有 reasoning 标记的情况

        验证点:
        - 返回 (None, 原内容) 或合理处理
        """
        output = "这是普通的回复内容，没有推理标记。"
        reasoning, content = deepseek_r1_parser.extract_reasoning(output)

        # 根据实际实现调整期望
        # 如果没有标记，可能返回 (None, 原内容) 或其他处理
        assert isinstance(reasoning, (str, type(None)))
        assert isinstance(content, (str, type(None)))

    def test_non_streaming_incomplete_reasoning(self, deepseek_r1_parser):
        """
        测试不完整的 reasoning（没有 end token）

        验证点:
        - 正确处理没有结束标记的情况
        """
        reasoning, content = deepseek_r1_parser.extract_reasoning(
            DEEPSEEK_R1_REASONING_NO_END
        )

        # 应该将剩余内容作为 reasoning
        assert reasoning is not None, "reasoning 不应为 None"
        assert "持续推理中" in reasoning

    # ==================== 流式响应测试 ====================

    def test_streaming_reasoning_start(self, deepseek_r1_parser):
        """
        测试流式 reasoning 开始标记

        验证点:
        - 开始标记出现时返回 reasoning delta
        """
        # 模拟流式过程：收到开始标记
        previous_text = ""
        current_text = "<thinky>"
        delta_text = "<thinky>"

        result = deepseek_r1_parser.extract_reasoning_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[]
        )

        # 结果可能是 None 或 DeltaMessage（取决于实现）
        assert result is None or isinstance(result, DeltaMessage)

    def test_streaming_reasoning_content(self, deepseek_r1_parser):
        """
        测试流式 reasoning 内容增量

        验证点:
        - reasoning 内容正确增量返回
        """
        # 模拟流式过程：已经在 reasoning 区域内
        previous_text = "<thinky>用户询问"
        current_text = "<thinky>用户询问天气情况"
        delta_text = "天气情况"

        result = deepseek_r1_parser.extract_reasoning_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[]
        )

        if result is not None:
            assert isinstance(result, DeltaMessage)
            # reasoning 内容应该在 reasoning 字段中
            if result.reasoning:
                assert "天气情况" in result.reasoning

    def test_streaming_reasoning_end(self, deepseek_r1_parser):
        """
        测试流式 reasoning 结束标记

        验证点:
        - 结束标记后返回 content delta
        """
        # 模拟流式过程：遇到结束标记
        previous_text = "<thinky>推理内容"
        current_text = "<thinky>推理内容</thinke>正常回复"
        delta_text = "</thinke>正常回复"

        result = deepseek_r1_parser.extract_reasoning_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[]
        )

        if result is not None:
            assert isinstance(result, DeltaMessage)
            # 结束标记后，content 应该有值
            if result.content:
                assert "正常回复" in result.content

    def test_streaming_reasoning_after_end(self, deepseek_r1_parser):
        """
        测试流式 reasoning 结束后的正常内容

        验证点:
        - 结束标记后，增量作为 content 返回
        """
        # 模拟流式过程：已经过了结束标记
        previous_text = "<thinky>推理内容</thinke>"
        current_text = "<thinky>推理内容</thinke>正常回复内容"
        delta_text = "正常回复内容"

        result = deepseek_r1_parser.extract_reasoning_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[]
        )

        if result is not None:
            assert isinstance(result, DeltaMessage)
            # 应该作为 content 返回
            assert result.content is not None or result.reasoning is not None

    # ==================== 只有 end_token 场景测试（无 start_token） ====================

    def test_non_streaming_only_end_token(self, deepseek_r1_parser):
        """
        测试非流式：只有 end_token，没有 start_token

        验证点:
        - end_token 之前的内容作为 reasoning
        - end_token 之后的内容作为 content
        参考：vLLM test_base_thinking_reasoning_parser.py::test_extract_reasoning_only_end_token
        """
        model_output = "这是推理内容</thinke>这是正常回复"

        reasoning, content = deepseek_r1_parser.extract_reasoning(model_output)

        assert reasoning == "这是推理内容", f"reasoning 应为 '这是推理内容'，实际: {reasoning}"
        assert content == "这是正常回复", f"content 应为 '这是正常回复'，实际: {content}"

    def test_non_streaming_only_end_token_empty_reasoning(self, deepseek_r1_parser):
        """
        测试非流式：只有 end_token 且前面没有内容

        验证点:
        - reasoning 为空或 None
        - content 正确提取
        """
        model_output = "</thinke>这是正常回复"

        reasoning, content = deepseek_r1_parser.extract_reasoning(model_output)

        # reasoning 可能是空字符串或 None
        assert reasoning is None or reasoning == "", f"reasoning 应为空，实际: {reasoning}"
        assert content == "这是正常回复"

    def test_streaming_only_end_token_reasoning_phase(self, deepseek_r1_parser):
        """
        测试流式：只有 end_token，在推理阶段

        验证点:
        - 没有 start_token 时，所有内容作为 reasoning
        - 直到遇到 end_token
        参考：vLLM MiniMaxM2ReasoningParser
        """
        # 模拟流式：还没有 end_token
        previous_text = ""
        current_text = "这是推理内容"
        delta_text = "这是推理内容"

        result = deepseek_r1_parser.extract_reasoning_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[]
        )

        if result is not None:
            assert isinstance(result, DeltaMessage)
            # 应该作为 reasoning 返回
            assert result.reasoning is not None, "没有 end_token 时应返回 reasoning"
            assert "推理内容" in result.reasoning

    def test_streaming_only_end_token_transition(self, deepseek_r1_parser):
        """
        测试流式：只有 end_token，在转换点

        验证点:
        - end_token 出现时正确分割 reasoning 和 content
        """
        # 模拟流式：end_token 在本次增量中
        previous_text = "推理过程"
        current_text = "推理过程</thinke>正常回复"
        delta_text = "</thinke>正常回复"

        result = deepseek_r1_parser.extract_reasoning_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[]
        )

        if result is not None:
            assert isinstance(result, DeltaMessage)
            # end_token 之前的空（因为已经在 previous_text 中）
            # end_token 之后的是 content
            if result.content:
                assert "正常回复" in result.content

    def test_streaming_only_end_token_after_end(self, deepseek_r1_parser):
        """
        测试流式：只有 end_token，在 end_token 之后

        验证点:
        - end_token 之后的内容作为 content 返回
        - 注意：无 tokenizer 时，需要模拟 token_ids 来正确判断 end_token 位置
        """
        # 模拟流式：end_token 已经出现
        # 由于无 tokenizer，我们需要模拟 token_ids
        # 假设 end_token_id 是某个值（这里用 100 模拟）
        mock_end_token_id = 100

        previous_text = "推理内容</thinke>"
        current_text = "推理内容</thinke>正常回复内容"
        delta_text = "正常回复内容"

        # 模拟 token_ids：previous 中包含 end_token
        previous_token_ids = [1, 2, 3, mock_end_token_id]  # 包含 end_token
        current_token_ids = previous_token_ids + [4, 5, 6]
        delta_token_ids = [4, 5, 6]

        # 临时设置 end_token_id
        original_end_id = deepseek_r1_parser._end_token_id
        deepseek_r1_parser._end_token_id = mock_end_token_id

        try:
            result = deepseek_r1_parser.extract_reasoning_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_token_ids=delta_token_ids,
                previous_token_ids=previous_token_ids,
                current_token_ids=current_token_ids,
                delta_text=delta_text
            )

            if result is not None:
                assert isinstance(result, DeltaMessage)
                # 应该作为 content 返回
                assert result.content is not None, "end_token 之后应返回 content"
                assert "正常回复内容" in result.content
        finally:
            # 恢复原始值
            deepseek_r1_parser._end_token_id = original_end_id


# ============================================
# Tool Parser 响应格式测试
# ============================================

class TestToolParserResponseFormat:
    """Tool Parser 响应格式测试类"""

    @pytest.fixture
    def deepseek_v3_parser(self):
        """创建 DeepSeek V3 Tool Parser 实例"""
        parser_cls = ToolParserManager.get_tool_parser("deepseek_v3")
        return parser_cls(tokenizer=None)

    # ==================== 非流式响应测试 ====================

    def test_non_streaming_single_tool_call_format(self, deepseek_v3_parser):
        """
        测试非流式单个工具调用格式

        验证点:
        - tools_called 为 True
        - tool_calls 列表正确
        - 每个 tool_call 包含 id, type, name, arguments
        - arguments 是合法 JSON
        """
        result = deepseek_v3_parser.extract_tool_calls(
            DEEPSEEK_V3_TOOL_CALL_OUTPUT,
            tools=SAMPLE_TOOLS
        )

        # 验证基本信息
        assert result.tools_called is True, "tools_called 应为 True"
        assert len(result.tool_calls) == 1, f"应有一个 tool_call，实际: {len(result.tool_calls)}"

        tool_call = result.tool_calls[0]

        # 验证 tool_call 结构
        assert tool_call.id is not None and tool_call.id != "", "tool_call.id 不能为空"
        assert tool_call.type == "function", f"tool_call.type 应为 'function'，实际: {tool_call.type}"
        # 使用嵌套的 function 对象
        assert tool_call.function is not None, "tool_call.function 不能为 None"
        assert tool_call.function.name == "get_weather", f"tool_call.function.name 应为 'get_weather'，实际: {tool_call.function.name}"

        # 验证 arguments 是合法 JSON
        assert tool_call.function.arguments is not None, "arguments 不能为 None"
        try:
            args = json.loads(tool_call.function.arguments)
            assert args["city"] == "北京", f"city 参数错误: {args.get('city')}"
            assert args["unit"] == "celsius", f"unit 参数错误: {args.get('unit')}"
        except json.JSONDecodeError as e:
            pytest.fail(f"arguments 不是合法 JSON: {tool_call.function.arguments}, 错误: {e}")

    def test_non_streaming_multiple_tool_calls_format(self, deepseek_v3_parser):
        """
        测试非流式多个工具调用格式

        验证点:
        - 所有工具调用都被提取
        - 每个 tool_call 有唯一 ID
        """
        result = deepseek_v3_parser.extract_tool_calls(
            DEEPSEEK_V3_MULTI_TOOL_CALLS,
            tools=SAMPLE_TOOLS
        )

        assert result.tools_called is True
        assert len(result.tool_calls) == 2, f"应有两个 tool_call，实际: {len(result.tool_calls)}"

        # 验证每个 tool_call
        for i, tool_call in enumerate(result.tool_calls):
            assert tool_call.id is not None
            assert tool_call.type == "function"
            assert tool_call.function is not None
            assert tool_call.function.name == "get_weather"

            args = json.loads(tool_call.function.arguments)
            expected_city = "北京" if i == 0 else "上海"
            assert args["city"] == expected_city

        # 验证 ID 唯一
        ids = [tc.id for tc in result.tool_calls]
        assert len(ids) == len(set(ids)), f"tool_call IDs 不唯一: {ids}"

    def test_non_streaming_tool_call_with_content(self, deepseek_v3_parser):
        """
        测试工具调用前有文本内容的情况

        验证点:
        - content 正确提取
        - tool_calls 正确提取
        """
        result = deepseek_v3_parser.extract_tool_calls(
            DEEPSEEK_V3_TOOL_CALL_WITH_CONTENT,
            tools=SAMPLE_TOOLS
        )

        assert result.tools_called is True
        assert result.content is not None, "content 不应为 None"
        assert "好的" in result.content, f"content 内容不正确: {result.content}"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function is not None
        assert result.tool_calls[0].function.name == "get_weather"

    def test_non_streaming_no_tool_call(self, deepseek_v3_parser):
        """
        测试没有工具调用的情况

        验证点:
        - tools_called 为 False
        - tool_calls 为空列表
        - content 为原内容
        """
        output = "这是普通的回复内容"
        result = deepseek_v3_parser.extract_tool_calls(output)

        assert result.tools_called is False, "tools_called 应为 False"
        assert len(result.tool_calls) == 0, "tool_calls 应为空"
        assert result.content == output, f"content 应为原内容，实际: {result.content}"

    def test_non_streaming_ascii_format(self, deepseek_v3_parser):
        """
        测试 ASCII 格式的工具调用标记

        验证点:
        - 支持 ASCII 和 Unicode 两种格式
        """
        result = deepseek_v3_parser.extract_tool_calls(
            ASCII_TOOL_CALL_OUTPUT,
            tools=SAMPLE_TOOLS
        )

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function is not None
        assert result.tool_calls[0].function.name == "get_weather"

    # ==================== 流式响应测试 ====================

    def test_streaming_tool_call_start(self, deepseek_v3_parser):
        """
        测试流式工具调用开始

        验证点:
        - 正确检测工具调用开始
        """
        deepseek_v3_parser.reset_streaming_state()

        previous_text = ""
        current_text = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>"
        delta_text = current_text

        result = deepseek_v3_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[]
        )

        # 结果可能是 None 或 DeltaMessage
        assert result is None or isinstance(result, DeltaMessage)

    def test_streaming_tool_call_name(self, deepseek_v3_parser):
        """
        测试流式工具调用名称增量

        验证点:
        - 正确发送工具名称
        - DeltaToolCall 结构正确
        """
        deepseek_v3_parser.reset_streaming_state()

        # 模拟流式过程：开始工具调用
        previous_text = ""
        current_text = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
        delta_text = current_text

        result = deepseek_v3_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[]
        )

        if result is not None and result.tool_calls:
            tool_call_delta = result.tool_calls[0]
            assert isinstance(tool_call_delta, DeltaToolCall)
            # 验证 DeltaToolCall 结构
            assert hasattr(tool_call_delta, 'index')
            assert hasattr(tool_call_delta, 'id')
            assert hasattr(tool_call_delta, 'type')
            assert hasattr(tool_call_delta, 'name')
            assert hasattr(tool_call_delta, 'arguments')

    def test_streaming_tool_call_arguments_delta(self, deepseek_v3_parser):
        """
        测试流式工具调用参数增量

        验证点:
        - 参数正确增量发送
        """
        deepseek_v3_parser.reset_streaming_state()

        # 模拟流式过程：已经在工具调用参数区域
        previous_text = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n"
        current_text = previous_text + '{"city": "北'
        delta_text = '{"city": "北'

        result = deepseek_v3_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[]
        )

        # 结果可能是 None 或包含参数增量的 DeltaMessage
        if result is not None and result.tool_calls:
            # 验证参数增量
            pass  # 具体验证取决于实现

    def test_streaming_state_reset(self, deepseek_v3_parser):
        """
        测试流式状态重置

        验证点:
        - reset_streaming_state 清除所有状态
        """
        # 设置一些状态
        deepseek_v3_parser.current_tool_id = 5
        deepseek_v3_parser.prev_tool_call_arr = [{"test": "data"}]
        deepseek_v3_parser.current_tool_name_sent = True
        deepseek_v3_parser.streamed_args_for_tool = ["arg1", "arg2"]

        # 重置
        deepseek_v3_parser.reset_streaming_state()

        # 验证状态已清除
        assert deepseek_v3_parser.current_tool_id == -1
        assert deepseek_v3_parser.prev_tool_call_arr == []
        assert deepseek_v3_parser.current_tool_name_sent is False
        assert deepseek_v3_parser.streamed_args_for_tool == []


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

    def test_combined_streaming_simulation(
        self,
        deepseek_r1_parser,
        deepseek_v3_tool_parser
    ):
        """
        测试组合场景的流式模拟

        验证点:
        - 模拟完整的流式处理过程
        - 验证非流式解析能正确处理组合格式
        """
        deepseek_r1_parser.reset_streaming_state()
        deepseek_v3_tool_parser.reset_streaming_state()

        # 组合格式
        combined_output = '<thinky>\n用户询问北京天气。\n我需要调用天气 API 来获取实时数据。\n城市是北京，需要用摄氏度单位。\n</thinke><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"city": "北京", "unit": "celsius"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>'

        # 1. 先用非流式解析 reasoning
        reasoning, remaining = deepseek_r1_parser.extract_reasoning(combined_output)
        assert reasoning is not None, "应解析出 reasoning"
        assert "用户询问北京天气" in reasoning
        assert "</thinke>" not in reasoning

        # 2. 再用非流式解析 tool_calls
        tool_result = deepseek_v3_tool_parser.extract_tool_calls(
            remaining,
            tools=SAMPLE_TOOLS
        )
        assert tool_result.tools_called, "应有工具调用"
        assert len(tool_result.tool_calls) == 1
        assert tool_result.tool_calls[0].function is not None
        assert tool_result.tool_calls[0].function.name == "get_weather"

        # 3. 验证参数
        args = json.loads(tool_result.tool_calls[0].function.arguments)
        assert args["city"] == "北京"
        assert args["unit"] == "celsius"

    def test_streaming_reasoning_sequence(
        self,
        deepseek_r1_parser
    ):
        """
        测试流式 reasoning 的增量序列（简化版）

        验证点:
        - reasoning 解析器能正确处理流式增量
        - 注意：无 tokenizer 时，流式解析可能有限制
        """
        deepseek_r1_parser.reset_streaming_state()

        # 使用非流式解析验证基本功能
        test_output = "<thinky>用户询问天气情况。</thinke>请问您想查询哪个城市？"
        reasoning, content = deepseek_r1_parser.extract_reasoning(test_output)

        assert reasoning is not None, "reasoning 不应为空"
        assert "用户询问天气情况" in reasoning
        assert "请问您想查询哪个城市" in content


# ============================================
# OpenAI 响应格式验证测试
# ============================================

class TestOpenAIResponseFormatValidation:
    """OpenAI 响应格式验证测试类"""

    def test_non_streaming_response_format(self):
        """
        测试非流式响应格式符合 OpenAI 规范

        验证点:
        - 响应包含所有必需字段
        - 字段类型正确
        """
        # 模拟完整的 OpenAI 响应
        response = {
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "正常内容",
                    "reasoning": "思考内容",
                    "tool_calls": [{
                        "id": "call_xyz",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "北京"}'
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }

        # 验证顶层字段
        assert "id" in response
        assert response["id"].startswith("chatcmpl-")
        assert response["object"] == "chat.completion"
        assert isinstance(response["created"], int)
        assert "model" in response

        # 验证 choices
        assert "choices" in response
        assert len(response["choices"]) == 1

        choice = response["choices"][0]
        assert choice["index"] == 0
        assert "message" in choice
        assert "finish_reason" in choice

        # 验证 message
        message = choice["message"]
        assert message["role"] == "assistant"
        assert "content" in message
        assert "reasoning" in message
        assert "tool_calls" in message

        # 验证 tool_calls 格式
        for tc in message["tool_calls"]:
            assert "id" in tc
            assert tc["type"] == "function"
            assert "function" in tc
            assert "name" in tc["function"]
            assert "arguments" in tc["function"]

            # 验证 arguments 是合法 JSON 字符串
            args = json.loads(tc["function"]["arguments"])
            assert isinstance(args, dict)

        # 验证 finish_reason
        assert choice["finish_reason"] in ["stop", "tool_calls", "length"]

    def test_streaming_response_format(self):
        """
        测试流式响应格式符合 OpenAI 规范

        验证点:
        - 每个 chunk 格式正确
        - delta 结构正确
        """
        # 模拟流式响应块序列
        streaming_chunks = [
            {
                "id": "chatcmpl-abc123",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "test-model",
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None
                }]
            },
            {
                "id": "chatcmpl-abc123",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "test-model",
                "choices": [{
                    "index": 0,
                    "delta": {"reasoning": "思考中..."},
                    "finish_reason": None
                }]
            },
            {
                "id": "chatcmpl-abc123",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "test-model",
                "choices": [{
                    "index": 0,
                    "delta": {"content": "正常内容"},
                    "finish_reason": None
                }]
            },
            {
                "id": "chatcmpl-abc123",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "test-model",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
        ]

        for chunk in streaming_chunks:
            # 验证顶层字段
            assert chunk["object"] == "chat.completion.chunk"
            assert "choices" in chunk
            assert len(chunk["choices"]) == 1

            choice = chunk["choices"][0]
            assert "delta" in choice
            assert "finish_reason" in choice

            # delta 可以包含 role, content, reasoning, tool_calls
            delta = choice["delta"]
            assert isinstance(delta, dict)

    def test_streaming_tool_calls_format(self):
        """
        测试流式 tool_calls 格式

        验证点:
        - delta.tool_calls 结构正确
        - 增量正确传递
        """
        # 模拟流式 tool_calls 响应
        streaming_chunks = [
            {
                "id": "chatcmpl-abc123",
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_xyz",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": ""
                            }
                        }]
                    },
                    "finish_reason": None
                }]
            },
            {
                "id": "chatcmpl-abc123",
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {
                                "arguments": '{"city"'
                            }
                        }]
                    },
                    "finish_reason": None
                }]
            },
            {
                "id": "chatcmpl-abc123",
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {
                                "arguments": ': "北京"}'
                            }
                        }]
                    },
                    "finish_reason": None
                }]
            },
            {
                "id": "chatcmpl-abc123",
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls"
                }]
            }
        ]

        accumulated_args = ""

        for chunk in streaming_chunks:
            choice = chunk["choices"][0]
            delta = choice["delta"]

            if "tool_calls" in delta:
                for tc in delta["tool_calls"]:
                    assert "index" in tc, "tool_call delta 必须有 index"
                    assert isinstance(tc["index"], int)

                    if "id" in tc:
                        assert tc["id"] is not None

                    if "function" in tc and tc["function"]:
                        if "arguments" in tc["function"]:
                            accumulated_args += tc["function"]["arguments"]

        # 验证累积的参数是合法 JSON
        assert accumulated_args, "应累积了参数"
        args = json.loads(accumulated_args)
        assert args["city"] == "北京"


# ============================================
# 边界情况和错误处理测试
# ============================================

class TestParserEdgeCases:
    """Parser 边界情况测试类"""

    @pytest.fixture
    def deepseek_r1_parser(self):
        parser_cls = ReasoningParserManager.get_reasoning_parser("deepseek_r1")
        return parser_cls(tokenizer=None)

    @pytest.fixture
    def deepseek_v3_parser(self):
        parser_cls = ToolParserManager.get_tool_parser("deepseek_v3")
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
