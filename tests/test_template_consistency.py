"""
多轮一致性 Prompt Template 校验工具

验证 Jinja 模板是否满足"多轮一致性"：
- 第一轮 messages + response 拼接为字符串 a
- 第二轮 messages + response 拼接为字符串 b
- a 必须是 b 的子字符串
"""

import pytest
from jinja2 import Environment, FileSystemLoader
from pathlib import Path


# 模板路径
TEMPLATE_DIR = Path(__file__).parent.parent / "models" / "Qwen" / "Qwen3.5-2B"
CONSISTENT_TEMPLATE = "chat_template_consistency.jinja"


def load_template(template_name: str):
    """加载模板"""
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    return env.get_template(template_name)


def render_template(template, messages, tools=None, add_generation_prompt=False, **kwargs):
    """渲染模板"""
    return template.render(
        messages=messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        **kwargs
    )


def verify_consistency(template, messages_round1, messages_round2, tools=None):
    """
    验证多轮一致性
    """
    output1 = render_template(template, messages_round1, tools=tools, add_generation_prompt=False)
    output2 = render_template(template, messages_round2, tools=tools, add_generation_prompt=False)

    assert output1 in output2, (
        f"多轮一致性验证失败:\n"
        f"第一轮输出长度: {len(output1)}\n"
        f"第二轮输出长度: {len(output2)}\n"
        f"第一轮输出:\n{output1}\n\n"
        f"第二轮输出:\n{output2}"
    )
    return output1, output2


class TestTemplateConsistency:
    """多轮一致性测试"""

    @pytest.fixture(autouse=True)
    def setup_templates(self):
        """加载模板"""
        self.consistent_template = load_template(CONSISTENT_TEMPLATE)

    # ========== 测试用例 ==========

    def test_pure_dialog_consistency(self):
        """测试：纯对话 - 多轮一致性"""
        messages_round1 = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮助你的？"}
        ]
        messages_round2 = messages_round1 + [
            {"role": "user", "content": "今天天气怎么样？"},
            {"role": "assistant", "content": "抱歉，我无法获取实时天气信息。"}
        ]
        verify_consistency(self.consistent_template, messages_round1, messages_round2)

    def test_tool_calls_consistency(self):
        """测试：tool调用 - 多轮一致性"""
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取城市天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"}
                    },
                    "required": ["city"]
                }
            }
        }]

        messages_round1 = [
            {"role": "user", "content": "北京天气"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {"city": "北京"}}}
            ]},
            {"role": "tool", "content": "晴天 25度"},
            {"role": "assistant", "content": "北京今天晴天，25度"}
        ]

        messages_round2 = messages_round1 + [
            {"role": "user", "content": "上海呢"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {"city": "上海"}}}
            ]},
            {"role": "tool", "content": "多云 20度"},
            {"role": "assistant", "content": "上海多云，20度"}
        ]

        verify_consistency(self.consistent_template, messages_round1, messages_round2, tools=tools)

    def test_reasoning_content_consistency(self):
        """测试：reasoning_content - 多轮一致性"""
        messages_round1 = [
            {"role": "user", "content": "1+1=?"},
            {"role": "assistant", "reasoning_content": "这是一个简单的加法问题", "content": "答案是2"}
        ]
        messages_round2 = messages_round1 + [
            {"role": "user", "content": "那2+2呢？"},
            {"role": "assistant", "reasoning_content": "继续加法运算", "content": "答案是4"}
        ]
        verify_consistency(self.consistent_template, messages_round1, messages_round2)

    def test_multiple_tool_calls_consistency(self):
        """测试：多次tool调用 - 多轮一致性"""
        tools = [{
            "type": "function",
            "function": {
                "name": "search",
                "description": "搜索信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索关键词"}
                    },
                    "required": ["query"]
                }
            }
        }]

        messages_round1 = [
            {"role": "user", "content": "搜索Python教程"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "search", "arguments": {"query": "Python教程"}}}
            ]},
            {"role": "tool", "content": "找到多个Python教程..."},
            {"role": "assistant", "content": "找到了多个Python教程，需要详细介绍吗？"}
        ]

        messages_round2 = messages_round1 + [
            {"role": "user", "content": "搜索JavaScript教程"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "search", "arguments": {"query": "JavaScript教程"}}}
            ]},
            {"role": "tool", "content": "找到多个JavaScript教程..."},
            {"role": "assistant", "content": "找到了多个JavaScript教程。"}
        ]

        verify_consistency(self.consistent_template, messages_round1, messages_round2, tools=tools)

    def test_three_rounds_consistency(self):
        """测试：三轮对话 - 多轮一致性"""
        messages_round1 = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"}
        ]

        messages_round2 = messages_round1 + [
            {"role": "user", "content": "再见"},
            {"role": "assistant", "content": "再见！"}
        ]

        messages_round3 = messages_round2 + [
            {"role": "user", "content": "又见面了"},
            {"role": "assistant", "content": "是的！"}
        ]

        # 验证 round1 在 round2 中
        verify_consistency(self.consistent_template, messages_round1, messages_round2)
        # 验证 round2 在 round3 中
        verify_consistency(self.consistent_template, messages_round2, messages_round3)
        # 验证 round1 在 round3 中（传递性）
        verify_consistency(self.consistent_template, messages_round1, messages_round3)

    def test_tool_calls_with_reasoning_consistency(self):
        """测试：tool调用 + reasoning - 多轮一致性"""
        messages_round1 = [
            {"role": "user", "content": "帮我查天气"},
            {"role": "assistant", "reasoning_content": "用户想查天气，需要调用天气API", "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {"city": "北京"}}}
            ]},
            {"role": "tool", "content": "晴天"},
            {"role": "assistant", "content": "北京今天是晴天"}
        ]

        messages_round2 = messages_round1 + [
            {"role": "user", "content": "上海呢"},
            {"role": "assistant", "reasoning_content": "用户问上海天气", "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {"city": "上海"}}}
            ]},
            {"role": "tool", "content": "多云"},
            {"role": "assistant", "content": "上海多云"}
        ]

        verify_consistency(self.consistent_template, messages_round1, messages_round2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
