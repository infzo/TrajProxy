"""
Tool Calling 高级测试

测试 Tool 调用的详细验证，包括：
- Tool 响应内容验证（参数解析正确性）
- Tool 结果回传测试（完整的工具调用流程）
- 多轮 Tool 调用测试
- 并行 Tool 调用测试
- 异常场景测试
"""

import pytest
import requests
import json
import uuid

from tests.e2e.config import (
    PROXY_URL,
    NGINX_URL,
    REQUEST_TIMEOUT,
    STREAM_TIMEOUT,
    LITELLM_API_KEY,
    ANTHROPIC_VERSION
)


# 示例工具定义 - 天气查询
WEATHER_TOOL_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# Claude 格式的天气工具
WEATHER_TOOL_CLAUDE = [
    {
        "name": "get_weather",
        "description": "获取指定城市的天气信息",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "温度单位"
                }
            },
            "required": ["city"]
        }
    }
]

# 计算器工具
CALCULATOR_TOOL_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行数学计算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 '2 + 3 * 4'"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# 多个工具定义
MULTIPLE_TOOLS_OPENAI = WEATHER_TOOL_OPENAI + CALCULATOR_TOOL_OPENAI


class TestToolResponseValidation:
    """Tool 响应内容验证测试类"""

    @pytest.mark.integration
    def test_tool_call_arguments_json_valid(
        self,
        proxy_client: requests.Session,
        default_headers: dict,
        registered_model_name: str
    ):
        """
        测试 tool_calls 中的 arguments 是有效的 JSON

        验证点:
        - tool_calls 存在时，arguments 是合法 JSON
        - JSON 解析后包含预期字段
        """
        response = proxy_client.post(
            f"{PROXY_URL}/v1/chat/completions",
            headers=default_headers,
            json={
                "model": registered_model_name,
                "messages": [
                    {"role": "user", "content": "北京今天天气怎么样？请用摄氏度"}
                ],
                "tools": WEATHER_TOOL_OPENAI,
                "tool_choice": "auto",
                "max_tokens": 200
            },
            timeout=REQUEST_TIMEOUT
        )

        assert response.status_code == 200, f"请求失败: {response.text}"

        data = response.json()
        choice = data["choices"][0]
        message = choice["message"]

        # 如果返回了 tool_calls，验证格式
        if "tool_calls" in message and message["tool_calls"]:
            tool_call = message["tool_calls"][0]

            # 验证基本结构
            assert "id" in tool_call, "tool_call 缺少 id 字段"
            assert tool_call.get("type") == "function", f"type 应为 function: {tool_call.get('type')}"
            assert "function" in tool_call, "tool_call 缺少 function 字段"

            func = tool_call["function"]
            assert "name" in func, "function 缺少 name 字段"
            assert "arguments" in func, "function 缺少 arguments 字段"

            # 验证 arguments 是合法 JSON
            try:
                args = json.loads(func["arguments"])
                assert isinstance(args, dict), "arguments 应为 dict 类型"

                # 如果是天气工具，验证字段
                if func["name"] == "get_weather":
                    assert "city" in args, "get_weather 参数缺少 city 字段"
                    assert args["city"], "city 不能为空"

            except json.JSONDecodeError as e:
                pytest.fail(f"arguments 不是合法 JSON: {func['arguments']}, 错误: {e}")

    @pytest.mark.integration
    def test_tool_call_id_format(
        self,
        proxy_client: requests.Session,
        default_headers: dict,
        registered_model_name: str
    ):
        """
        测试 tool_call ID 格式

        验证点:
        - tool_call id 以 "call_" 开头或符合 OpenAI 格式
        - 多个 tool_calls 有不同的 id
        """
        response = proxy_client.post(
            f"{PROXY_URL}/v1/chat/completions",
            headers=default_headers,
            json={
                "model": registered_model_name,
                "messages": [
                    {"role": "user", "content": "查询北京和上海的天气"}
                ],
                "tools": WEATHER_TOOL_OPENAI,
                "max_tokens": 300
            },
            timeout=REQUEST_TIMEOUT
        )

        assert response.status_code == 200, f"请求失败: {response.text}"

        data = response.json()
        message = data["choices"][0]["message"]

        if "tool_calls" in message and len(message["tool_calls"]) > 1:
            # 验证多个 tool_calls 的 ID 唯一性
            ids = [tc["id"] for tc in message["tool_calls"]]
            assert len(ids) == len(set(ids)), f"tool_call IDs 不唯一: {ids}"

            # 验证 ID 格式
            for tc_id in ids:
                assert tc_id, "tool_call id 不能为空"
                # OpenAI 格式通常是 "call_" 开头
                assert tc_id.startswith("call_") or len(tc_id) > 10, \
                    f"tool_call id 格式异常: {tc_id}"


class TestToolResultSubmission:
    """Tool 结果回传测试类"""

    @pytest.mark.integration
    def test_openai_tool_result_continuation(
        self,
        proxy_client: requests.Session,
        default_headers: dict,
        registered_model_name: str,
        unique_session_id: str
    ):
        """
        测试 OpenAI 格式工具结果提交后继续对话

        完整流程:
        1. 用户请求 -> 模型返回 tool_calls
        2. 提交工具结果 -> 模型继续回复

        验证点:
        - tool 角色消息格式正确
        - 模型能基于工具结果生成回复
        """
        # 第一步：发送需要工具调用的请求
        response1 = proxy_client.post(
            f"{PROXY_URL}/v1/chat/completions",
            headers=default_headers,
            json={
                "model": registered_model_name,
                "messages": [
                    {"role": "user", "content": "北京今天天气怎么样？"}
                ],
                "tools": WEATHER_TOOL_OPENAI,
                "tool_choice": "auto",
                "max_tokens": 200
            },
            timeout=REQUEST_TIMEOUT
        )

        assert response1.status_code == 200, f"首次请求失败: {response1.text}"

        data1 = response1.json()
        message1 = data1["choices"][0]["message"]

        # 如果模型返回了 tool_calls
        if "tool_calls" in message1 and message1["tool_calls"]:
            tool_call = message1["tool_calls"][0]
            tool_call_id = tool_call["id"]
            func_name = tool_call["function"]["name"]

            # 构造工具结果消息
            tool_result = {
                "city": "北京",
                "temperature": 25,
                "condition": "晴天",
                "humidity": 45
            }

            # 第二步：提交工具结果
            messages = [
                {"role": "user", "content": "北京今天天气怎么样？"},
                message1,  # 助手的 tool_calls 消息
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(tool_result, ensure_ascii=False)
                }
            ]

            response2 = proxy_client.post(
                f"{PROXY_URL}/v1/chat/completions",
                headers=default_headers,
                json={
                    "model": registered_model_name,
                    "messages": messages,
                    "max_tokens": 200
                },
                timeout=REQUEST_TIMEOUT
            )

            assert response2.status_code == 200, f"工具结果提交失败: {response2.text}"

            data2 = response2.json()
            final_message = data2["choices"][0]["message"]

            # 验证最终回复
            assert final_message.get("role") == "assistant"
            # 回复应该包含天气相关信息
            if final_message.get("content"):
                content = final_message["content"]
                # 不强制要求特定内容，只验证有回复
                assert len(content) > 0, "最终回复为空"

    @pytest.mark.integration
    def test_claude_tool_result_continuation(
        self,
        nginx_client: requests.Session,
        nginx_url: str,
        registered_model_name: str,
        unique_session_id: str
    ):
        """
        测试 Claude 格式工具结果提交后继续对话

        完整流程:
        1. 用户请求 -> 模型返回 tool_use content block
        2. 提交工具结果 -> 模型继续回复

        验证点:
        - tool_result 角色消息格式正确
        - 模型能基于工具结果生成回复
        """
        claude_headers = {
            "Content-Type": "application/json",
            "x-api-key": LITELLM_API_KEY,
            "anthropic-version": ANTHROPIC_VERSION
        }

        url = f"{nginx_url}/s/{unique_session_id}/v1/messages"

        # 第一步：发送需要工具调用的请求
        response1 = nginx_client.post(
            url,
            headers=claude_headers,
            json={
                "model": registered_model_name,
                "max_tokens": 200,
                "messages": [
                    {"role": "user", "content": "北京今天天气怎么样？"}
                ],
                "tools": WEATHER_TOOL_CLAUDE
            },
            timeout=REQUEST_TIMEOUT
        )

        assert response1.status_code == 200, f"首次请求失败: {response1.text}"

        data1 = response1.json()
        content1 = data1.get("content", [])

        # 查找 tool_use content block
        tool_use_block = None
        for block in content1:
            if block.get("type") == "tool_use":
                tool_use_block = block
                break

        if tool_use_block:
            tool_use_id = tool_use_block["id"]
            tool_name = tool_use_block["name"]

            # 构造工具结果
            tool_result = {
                "city": "北京",
                "temperature": 25,
                "condition": "晴天"
            }

            # 第二步：提交工具结果
            messages = [
                {"role": "user", "content": "北京今天天气怎么样？"},
                {
                    "role": "assistant",
                    "content": content1
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": json.dumps(tool_result, ensure_ascii=False)
                        }
                    ]
                }
            ]

            response2 = nginx_client.post(
                url,
                headers=claude_headers,
                json={
                    "model": registered_model_name,
                    "max_tokens": 200,
                    "messages": messages
                },
                timeout=REQUEST_TIMEOUT
            )

            assert response2.status_code == 200, f"工具结果提交失败: {response2.text}"

            data2 = response2.json()
            assert "content" in data2, "响应缺少 content 字段"


class TestMultipleToolCalls:
    """多轮/并行 Tool 调用测试类"""

    @pytest.mark.integration
    def test_parallel_tool_calls(
        self,
        proxy_client: requests.Session,
        default_headers: dict,
        registered_model_name: str
    ):
        """
        测试并行工具调用

        验证点:
        - 模型能同时返回多个 tool_calls
        - 每个 tool_call 有唯一的 id
        - finish_reason 为 "tool_calls"
        """
        response = proxy_client.post(
            f"{PROXY_URL}/v1/chat/completions",
            headers=default_headers,
            json={
                "model": registered_model_name,
                "messages": [
                    {"role": "user", "content": "请同时查询北京和上海的天气"}
                ],
                "tools": WEATHER_TOOL_OPENAI,
                "max_tokens": 400
            },
            timeout=REQUEST_TIMEOUT
        )

        assert response.status_code == 200, f"请求失败: {response.text}"

        data = response.json()
        choice = data["choices"][0]
        message = choice["message"]

        # 如果有多个 tool_calls
        if "tool_calls" in message and len(message["tool_calls"]) > 1:
            # 验证并行调用的结构
            tool_calls = message["tool_calls"]
            assert len(tool_calls) >= 2, f"预期至少 2 个 tool_calls，实际 {len(tool_calls)}"

            # 验证每个都有唯一 ID
            ids = [tc["id"] for tc in tool_calls]
            assert len(ids) == len(set(ids)), "并行 tool_calls 的 ID 应唯一"

            # 验证 finish_reason
            if choice.get("finish_reason"):
                assert choice["finish_reason"] == "tool_calls", \
                    f"finish_reason 应为 'tool_calls': {choice['finish_reason']}"

    @pytest.mark.integration
    def test_sequential_tool_calls(
        self,
        proxy_client: requests.Session,
        default_headers: dict,
        registered_model_name: str,
        unique_session_id: str
    ):
        """
        测试顺序多轮工具调用

        验证点:
        - 第一轮工具调用正确
        - 第二轮工具调用基于第一轮结果
        """
        # 第一轮
        response1 = proxy_client.post(
            f"{PROXY_URL}/v1/chat/completions",
            headers=default_headers,
            json={
                "model": registered_model_name,
                "messages": [
                    {"role": "user", "content": "北京的天气怎么样？"}
                ],
                "tools": WEATHER_TOOL_OPENAI,
                "max_tokens": 200
            },
            timeout=REQUEST_TIMEOUT
        )

        assert response1.status_code == 200
        data1 = response1.json()
        message1 = data1["choices"][0]["message"]

        if "tool_calls" in message1 and message1["tool_calls"]:
            tool_call = message1["tool_calls"][0]

            # 提交工具结果并请求第二个查询
            messages = [
                {"role": "user", "content": "北京的天气怎么样？"},
                message1,
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps({"city": "北京", "temperature": 25})
                },
                {"role": "user", "content": "现在查一下上海的天气"}
            ]

            response2 = proxy_client.post(
                f"{PROXY_URL}/v1/chat/completions",
                headers=default_headers,
                json={
                    "model": registered_model_name,
                    "messages": messages,
                    "tools": WEATHER_TOOL_OPENAI,
                    "max_tokens": 200
                },
                timeout=REQUEST_TIMEOUT
            )

            assert response2.status_code == 200, f"第二轮请求失败: {response2.text}"


class TestToolCallingErrors:
    """Tool 调用异常场景测试类"""

    def test_tool_call_with_invalid_tool_choice(
        self,
        proxy_client: requests.Session,
        default_headers: dict,
        registered_model_name: str
    ):
        """
        测试无效的 tool_choice 参数

        验证点:
        - tool_choice 为无效值时的处理
        """
        response = proxy_client.post(
            f"{PROXY_URL}/v1/chat/completions",
            headers=default_headers,
            json={
                "model": registered_model_name,
                "messages": [
                    {"role": "user", "content": "测试"}
                ],
                "tools": WEATHER_TOOL_OPENAI,
                "tool_choice": "invalid_choice_value",
                "max_tokens": 50
            },
            timeout=REQUEST_TIMEOUT
        )

        # 可能返回 400 或忽略无效值返回 200
        assert response.status_code in [200, 400, 422], \
            f"预期返回 200/400/422，实际返回 {response.status_code}"

    def test_tool_result_without_tool_call_id(
        self,
        proxy_client: requests.Session,
        default_headers: dict,
        registered_model_name: str
    ):
        """
        测试缺少 tool_call_id 的工具结果

        验证点:
        - 缺少 tool_call_id 时返回错误或正确处理
        """
        messages = [
            {"role": "user", "content": "测试"},
            {
                "role": "tool",
                "content": "工具结果"
                # 缺少 tool_call_id
            }
        ]

        response = proxy_client.post(
            f"{PROXY_URL}/v1/chat/completions",
            headers=default_headers,
            json={
                "model": registered_model_name,
                "messages": messages,
                "max_tokens": 50
            },
            timeout=REQUEST_TIMEOUT
        )

        # 应该返回错误或能够处理
        assert response.status_code in [200, 400, 422], \
            f"预期返回 200/400/422，实际返回 {response.status_code}"

    def test_tool_result_mismatched_id(
        self,
        proxy_client: requests.Session,
        default_headers: dict,
        registered_model_name: str
    ):
        """
        测试 tool_call_id 不匹配的情况

        验证点:
        - 工具结果的 ID 与之前的 tool_call 不匹配时的处理
        """
        messages = [
            {"role": "user", "content": "测试天气"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_original_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "北京"}'
                    }
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_different_456",  # 不匹配的 ID
                "content": '{"temperature": 25}'
            }
        ]

        response = proxy_client.post(
            f"{PROXY_URL}/v1/chat/completions",
            headers=default_headers,
            json={
                "model": registered_model_name,
                "messages": messages,
                "max_tokens": 100
            },
            timeout=REQUEST_TIMEOUT
        )

        # 请求应该能处理（可能忽略或返回错误）
        assert response.status_code in [200, 400, 422], \
            f"预期返回 200/400/422，实际返回 {response.status_code}"

    def test_tools_exceed_limit(
        self,
        proxy_client: requests.Session,
        default_headers: dict,
        registered_model_name: str
    ):
        """
        测试工具数量过多的情况

        验证点:
        - 大量工具定义时的处理
        """
        # 创建大量工具定义
        many_tools = []
        for i in range(50):
            many_tools.append({
                "type": "function",
                "function": {
                    "name": f"tool_{i}",
                    "description": f"测试工具 {i}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "arg": {"type": "string"}
                        }
                    }
                }
            })

        response = proxy_client.post(
            f"{PROXY_URL}/v1/chat/completions",
            headers=default_headers,
            json={
                "model": registered_model_name,
                "messages": [
                    {"role": "user", "content": "测试"}
                ],
                "tools": many_tools,
                "max_tokens": 50
            },
            timeout=REQUEST_TIMEOUT
        )

        # 应该能处理或返回错误
        assert response.status_code in [200, 400, 413], \
            f"预期返回 200/400/413，实际返回 {response.status_code}"


class TestToolCallingStreamAdvanced:
    """Tool 调用流式高级测试类"""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_stream_tool_calls_incremental_arguments(
        self,
        proxy_client: requests.Session,
        default_headers: dict,
        registered_model_name: str
    ):
        """
        测试流式 tool_calls 参数增量传输

        验证点:
        - 流式传输中 tool_calls.arguments 正确累积
        - 最终拼接的 arguments 是合法 JSON
        """
        response = proxy_client.post(
            f"{PROXY_URL}/v1/chat/completions",
            headers=default_headers,
            json={
                "model": registered_model_name,
                "messages": [
                    {"role": "user", "content": "北京的天气怎么样？"}
                ],
                "tools": WEATHER_TOOL_OPENAI,
                "stream": True,
                "max_tokens": 200
            },
            stream=True,
            timeout=STREAM_TIMEOUT
        )

        assert response.status_code == 200

        accumulated_args = {}
        tool_call_chunks = []

        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue

            data_str = line[6:]
            if data_str == "[DONE]":
                break

            try:
                chunk = json.loads(data_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})

                if "tool_calls" in delta:
                    for tc in delta["tool_calls"]:
                        idx = tc.get("index", 0)
                        if idx not in accumulated_args:
                            accumulated_args[idx] = ""

                        # 累积 arguments
                        if tc.get("arguments"):
                            accumulated_args[idx] += tc["arguments"]

                        tool_call_chunks.append(tc)

            except json.JSONDecodeError:
                pass

        # 如果有 tool_calls，验证最终 arguments 是合法 JSON
        for idx, args_str in accumulated_args.items():
            if args_str:
                try:
                    parsed = json.loads(args_str)
                    assert isinstance(parsed, dict), \
                        f"tool_call[{idx}] arguments 应为 dict: {args_str}"
                except json.JSONDecodeError as e:
                    pytest.fail(
                        f"tool_call[{idx}] 累积的 arguments 不是合法 JSON: "
                        f"{args_str}, 错误: {e}"
                    )

    @pytest.mark.integration
    @pytest.mark.slow
    def test_stream_multiple_tool_calls_order(
        self,
        proxy_client: requests.Session,
        default_headers: dict,
        registered_model_name: str
    ):
        """
        测试流式多个 tool_calls 的顺序

        验证点:
        - 多个 tool_calls 按正确顺序传输
        - 每个 tool_call 的 index 正确
        """
        response = proxy_client.post(
            f"{PROXY_URL}/v1/chat/completions",
            headers=default_headers,
            json={
                "model": registered_model_name,
                "messages": [
                    {"role": "user", "content": "请同时查询北京、上海和广州的天气"}
                ],
                "tools": WEATHER_TOOL_OPENAI,
                "stream": True,
                "max_tokens": 400
            },
            stream=True,
            timeout=STREAM_TIMEOUT
        )

        assert response.status_code == 200

        tool_call_ids = set()
        indices_seen = set()

        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue

            data_str = line[6:]
            if data_str == "[DONE]":
                break

            try:
                chunk = json.loads(data_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})

                if "tool_calls" in delta:
                    for tc in delta["tool_calls"]:
                        if tc.get("id"):
                            tool_call_ids.add(tc["id"])
                        if "index" in tc:
                            indices_seen.add(tc["index"])

            except json.JSONDecodeError:
                pass

        # 验证 index 连续
        if indices_seen:
            expected_indices = set(range(len(indices_seen)))
            assert indices_seen == expected_indices, \
                f"tool_call indices 不连续: {indices_seen}"
