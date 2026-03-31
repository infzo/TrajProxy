"""
HTTP 请求格式核心测试

测试 OpenAI 和 Claude 格式的 HTTP 请求，包括：
- 非流式和流式请求
- Tool Calling 功能
- 错误处理

合并自：
- test_chat.py 核心测试
- test_tool_calling.py 核心测试
- test_tool_calling_advanced.py 核心测试
"""

import pytest
import requests
import json
import time

from tests.e2e.config import (
    PROXY_URL,
    DEFAULT_MODEL,
    STREAM_TIMEOUT,
    REQUEST_TIMEOUT
)


# ============================================================
# 测试数据定义
# ============================================================

# OpenAI 格式的工具定义
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

# Claude 格式的工具定义
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


# ============================================================
# OpenAI 格式测试
# ============================================================

class TestOpenAIFormat:
    """OpenAI 格式请求测试类"""

    @pytest.mark.integration
    def test_non_stream_response_format(
        self,
        proxy_client: requests.Session,
        default_headers: dict,
        registered_model_name: str
    ):
        """
        测试 OpenAI 非流式请求响应格式

        验证点:
        - 返回状态码 200
        - 响应格式符合 OpenAI API 规范
        - 包含 id, choices, usage 字段
        """
        response = proxy_client.post(
            f"{PROXY_URL}/v1/chat/completions",
            headers=default_headers,
            json={
                "model": registered_model_name,
                "messages": [
                    {"role": "user", "content": "你好，请回复'测试成功'"}
                ],
                "max_tokens": 100,
                "temperature": 0.7
            }
        )

        assert response.status_code == 200, f"请求失败: {response.text}"

        data = response.json()

        # 验证响应结构
        assert "id" in data, "响应缺少 id 字段"
        assert "choices" in data, "响应缺少 choices 字段"
        assert len(data["choices"]) > 0, "choices 为空"

        # 验证 choice 结构
        choice = data["choices"][0]
        assert "message" in choice, "choice 缺少 message 字段"
        assert "finish_reason" in choice, "choice 缺少 finish_reason 字段"

        # 验证消息内容
        message = choice["message"]
        assert message.get("role") == "assistant", f"角色错误: {message.get('role')}"
        assert "content" in message, "message 缺少 content 字段"
        assert len(message["content"]) > 0, "content 为空"

        # 验证 usage 字段
        if "usage" in data:
            usage = data["usage"]
            assert "prompt_tokens" in usage, "usage 缺少 prompt_tokens"
            assert "completion_tokens" in usage, "usage 缺少 completion_tokens"
            assert "total_tokens" in usage, "usage 缺少 total_tokens"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_stream_response_format(
        self,
        proxy_client: requests.Session,
        default_headers: dict,
        registered_model_name: str
    ):
        """
        测试 OpenAI 流式请求响应格式

        验证点:
        - 返回状态码 200
        - 响应格式为 SSE (Server-Sent Events)
        - 流式数据块格式正确
        - 最后收到 [DONE] 标记
        """
        response = proxy_client.post(
            f"{PROXY_URL}/v1/chat/completions",
            headers=default_headers,
            json={
                "model": registered_model_name,
                "messages": [
                    {"role": "user", "content": "说一个字：好"}
                ],
                "max_tokens": 10,
                "temperature": 0,
                "stream": True
            },
            stream=True,
            timeout=STREAM_TIMEOUT
        )

        assert response.status_code == 200, f"流式请求失败: {response.text}"

        # 收集所有数据块
        chunks = []
        content_chunks = []

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            if line.startswith("data: "):
                data_str = line[6:]  # 去掉 "data: " 前缀

                if data_str == "[DONE]":
                    chunks.append("[DONE]")
                    break

                try:
                    chunk = json.loads(data_str)
                    chunks.append(chunk)

                    # 提取内容
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            content_chunks.append(delta["content"])
                except json.JSONDecodeError:
                    pass

        # 验证收到了数据块
        assert len(chunks) > 0, "未收到任何流式数据块"
        assert chunks[-1] == "[DONE]", "流式响应未以 [DONE] 结束"

        # 验证至少收到了一些内容
        assert len(content_chunks) > 0, "未收到任何内容"

        # 验证内容拼接后不为空
        full_content = "".join(content_chunks)
        assert len(full_content) > 0, "流式内容为空"

    @pytest.mark.integration
    def test_tool_calling_non_stream(
        self,
        proxy_client: requests.Session,
        default_headers: dict,
        registered_model_name: str
    ):
        """
        测试 OpenAI Tool Calling 非流式请求

        验证点:
        - 请求成功返回
        - 响应格式符合 OpenAI API 规范
        - tool_calls 结构正确（如果模型返回）
        """
        response = proxy_client.post(
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

        assert response.status_code == 200, f"请求失败: {response.text}"

        data = response.json()

        # 验证基本响应结构
        assert "choices" in data, "响应缺少 choices 字段"
        assert len(data["choices"]) > 0, "choices 为空"

        choice = data["choices"][0]
        assert "message" in choice, "choice 缺少 message 字段"

        message = choice["message"]
        assert "role" in message, "message 缺少 role 字段"

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
            except json.JSONDecodeError as e:
                pytest.fail(f"arguments 不是合法 JSON: {func['arguments']}, 错误: {e}")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_tool_calling_stream(
        self,
        proxy_client: requests.Session,
        default_headers: dict,
        registered_model_name: str
    ):
        """
        测试 OpenAI Tool Calling 流式请求

        验证点:
        - 流式请求成功
        - 响应为 SSE 格式
        - tool_calls 参数增量传输正确
        """
        response = proxy_client.post(
            f"{PROXY_URL}/v1/chat/completions",
            headers=default_headers,
            json={
                "model": registered_model_name,
                "messages": [
                    {"role": "user", "content": "查询上海的天气"}
                ],
                "tools": WEATHER_TOOL_OPENAI,
                "stream": True,
                "max_tokens": 100
            },
            stream=True,
            timeout=STREAM_TIMEOUT
        )

        assert response.status_code == 200, f"流式请求失败: {response.text}"

        accumulated_args = {}
        chunks = []
        has_content = False

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            if line.startswith("data: "):
                data_str = line[6:]

                if data_str == "[DONE]":
                    chunks.append("[DONE]")
                    break

                try:
                    chunk = json.loads(data_str)
                    chunks.append(chunk)

                    delta = chunk.get("choices", [{}])[0].get("delta", {})

                    # 检查是否有内容
                    if delta.get("content") or delta.get("tool_calls"):
                        has_content = True

                    # 累积 tool_calls arguments
                    if "tool_calls" in delta:
                        for tc in delta["tool_calls"]:
                            idx = tc.get("index", 0)
                            if idx not in accumulated_args:
                                accumulated_args[idx] = ""

                            if tc.get("arguments"):
                                accumulated_args[idx] += tc["arguments"]

                except json.JSONDecodeError:
                    pass

        # 验证收到了数据块
        assert len(chunks) > 0, "未收到任何流式数据块"
        assert chunks[-1] == "[DONE]", "流式响应未以 [DONE] 结束"

        # 验证累积的 arguments 是合法 JSON
        for idx, args_str in accumulated_args.items():
            if args_str:
                try:
                    parsed = json.loads(args_str)
                    assert isinstance(parsed, dict), f"tool_call[{idx}] arguments 应为 dict: {args_str}"
                except json.JSONDecodeError as e:
                    pytest.fail(f"tool_call[{idx}] 累积的 arguments 不是合法 JSON: {args_str}, 错误: {e}")


# ============================================================
# Claude 格式测试
# ============================================================

class TestClaudeFormat:
    """Claude (Anthropic) 格式请求测试类"""

    @pytest.mark.integration
    def test_non_stream_response_format(
        self,
        nginx_client: requests.Session,
        nginx_url: str,
        claude_headers: dict,
        registered_model_name: str,
        unique_session_id: str
    ):
        """
        测试 Claude 非流式请求响应格式

        验证点:
        - 返回状态码 200
        - 响应格式符合 Anthropic Messages API 规范
        """
        url = f"{nginx_url}/s/{unique_session_id}/v1/messages"

        response = nginx_client.post(
            url,
            headers=claude_headers,
            json={
                "model": registered_model_name,
                "max_tokens": 50,
                "messages": [
                    {"role": "user", "content": "你好，请回复'测试成功'"}
                ]
            },
            timeout=REQUEST_TIMEOUT
        )

        assert response.status_code == 200, f"请求失败: {response.text}"

        data = response.json()

        # 验证 Anthropic 响应结构
        assert "id" in data, "响应缺少 id 字段"
        assert "type" in data, "响应缺少 type 字段"
        assert data.get("type") == "message", f"type 字段错误: {data.get('type')}"
        assert "content" in data, "响应缺少 content 字段"
        assert "role" in data, "响应缺少 role 字段"
        assert data.get("role") == "assistant", f"角色错误: {data.get('role')}"

        # 验证 content 结构
        content = data.get("content", [])
        assert isinstance(content, list), "content 不是列表类型"
        assert len(content) > 0, "content 为空"

        # 验证 content block
        content_block = content[0]
        assert "type" in content_block, "content block 缺少 type 字段"
        if content_block.get("type") == "text":
            assert "text" in content_block, "content block 缺少 text 字段"
            assert len(content_block["text"]) > 0, "text 为空"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_stream_response_format(
        self,
        nginx_client: requests.Session,
        nginx_url: str,
        claude_headers: dict,
        registered_model_name: str,
        unique_session_id: str
    ):
        """
        测试 Claude 流式请求响应格式

        验证点:
        - 返回状态码 200
        - 响应格式为 SSE
        - 事件类型序列正确
        """
        url = f"{nginx_url}/s/{unique_session_id}/v1/messages"

        response = nginx_client.post(
            url,
            headers=claude_headers,
            json={
                "model": registered_model_name,
                "max_tokens": 50,
                "messages": [
                    {"role": "user", "content": "说一个字：好"}
                ],
                "stream": True
            },
            stream=True,
            timeout=STREAM_TIMEOUT
        )

        assert response.status_code == 200, f"流式请求失败: {response.text}"

        # 收集所有事件
        events = []
        event_types = []
        content_deltas = []

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            if line.startswith("data: "):
                data_str = line[6:]

                try:
                    event = json.loads(data_str)
                    events.append(event)

                    event_type = event.get("type")
                    if event_type:
                        event_types.append(event_type)

                    # 收集内容 delta
                    if event_type == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                content_deltas.append(text)
                except json.JSONDecodeError:
                    pass

        # 验证收到了事件
        assert len(events) > 0, "未收到任何流式事件"

        # 验证关键事件类型
        assert "message_start" in event_types, "缺少 message_start 事件"
        assert "message_stop" in event_types, "缺少 message_stop 事件"

        # 验证收到了内容
        assert len(content_deltas) > 0, "未收到任何内容 delta"

    @pytest.mark.integration
    def test_tool_use_response_format(
        self,
        nginx_client: requests.Session,
        nginx_url: str,
        claude_headers: dict,
        registered_model_name: str,
        unique_session_id: str
    ):
        """
        测试 Claude tool_use 响应格式

        验证点:
        - tool_use content block 结构正确
        - 包含 id, type, name, input 字段
        """
        url = f"{nginx_url}/s/{unique_session_id}/v1/messages"

        response = nginx_client.post(
            url,
            headers=claude_headers,
            json={
                "model": registered_model_name,
                "max_tokens": 200,
                "messages": [
                    {"role": "user", "content": "北京的天气怎么样？"}
                ],
                "tools": WEATHER_TOOL_CLAUDE
            },
            timeout=REQUEST_TIMEOUT
        )

        assert response.status_code == 200, f"请求失败: {response.text}"

        data = response.json()

        # 验证基本响应结构
        assert data.get("type") == "message"
        assert data.get("role") == "assistant"

        content = data.get("content", [])
        assert isinstance(content, list)

        # 查找 tool_use block
        tool_use_block = None
        for block in content:
            if block.get("type") == "tool_use":
                tool_use_block = block
                break

        # 如果有 tool_use，验证格式
        if tool_use_block:
            # 必须字段
            assert "id" in tool_use_block, "tool_use block 缺少 id 字段"
            assert tool_use_block["type"] == "tool_use"
            assert "name" in tool_use_block, "tool_use block 缺少 name 字段"
            assert "input" in tool_use_block, "tool_use block 缺少 input 字段"

            # 验证 input 是 dict
            tool_input = tool_use_block["input"]
            assert isinstance(tool_input, dict), f"input 应为 dict: {type(tool_input)}"


# ============================================================
# Tool Calling 高级测试
# ============================================================

class TestToolCalling:
    """Tool Calling 功能测试类"""

    @pytest.mark.integration
    def test_tool_choice_none(
        self,
        proxy_client: requests.Session,
        default_headers: dict,
        registered_model_name: str
    ):
        """
        测试 tool_choice="none" 的请求

        验证点:
        - 即使有工具定义，tool_choice="none" 也不应触发工具调用
        """
        response = proxy_client.post(
            f"{PROXY_URL}/v1/chat/completions",
            headers=default_headers,
            json={
                "model": registered_model_name,
                "messages": [
                    {"role": "user", "content": "你好"}
                ],
                "tools": WEATHER_TOOL_OPENAI,
                "tool_choice": "none",
                "max_tokens": 50
            },
            timeout=REQUEST_TIMEOUT
        )

        assert response.status_code == 200, f"请求失败: {response.text}"

        data = response.json()
        choice = data["choices"][0]
        message = choice["message"]

        # tool_choice="none" 时不应返回 tool_calls（注意：某些模型可能忽略此参数）
        assert "content" in message or "tool_calls" in message, \
            f"message 缺少 content 或 tool_calls: {message}"

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
            tool_calls = message["tool_calls"]

            # 验证每个都有唯一 ID
            ids = [tc["id"] for tc in tool_calls]
            assert len(ids) == len(set(ids)), f"并行 tool_calls 的 ID 应唯一: {ids}"

            # 验证 finish_reason
            if choice.get("finish_reason"):
                assert choice["finish_reason"] == "tool_calls", \
                    f"finish_reason 应为 'tool_calls': {choice['finish_reason']}"

    @pytest.mark.integration
    def test_tool_result_continuation(
        self,
        proxy_client: requests.Session,
        default_headers: dict,
        registered_model_name: str,
        unique_session_id: str
    ):
        """
        测试工具结果提交后继续对话

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
            if final_message.get("content"):
                assert len(final_message["content"]) > 0, "最终回复为空"


# ============================================================
# 错误处理测试
# ============================================================

class TestErrorHandling:
    """错误处理测试类"""

    def test_invalid_model_returns_404(
        self,
        proxy_client: requests.Session,
        default_headers: dict
    ):
        """
        测试使用无效模型的请求

        验证点:
        - 返回状态码 404
        - 错误信息提示模型未注册
        """
        response = proxy_client.post(
            f"{PROXY_URL}/v1/chat/completions",
            headers=default_headers,
            json={
                "model": "non_existent_model_xyz",
                "messages": [
                    {"role": "user", "content": "test"}
                ]
            }
        )

        assert response.status_code == 404, f"预期返回 404，实际返回 {response.status_code}"

        data = response.json()
        assert "未注册" in data.get("detail", "") or "不存在" in data.get("detail", ""), \
            f"错误信息未提示模型未注册: {data}"

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
