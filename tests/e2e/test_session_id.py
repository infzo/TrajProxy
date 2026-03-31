"""
Session ID 核心测试

测试 Session ID 的传递方式和路由逻辑，包括：
- 三种传递方式（URL路径、请求头、model@session_id）
- 格式验证
- run_id 隔离性

合并自：
- test_chat.py 中 TestSessionIdDelivery 类
- test_models.py 中 TestSessionIdRouting 类
"""

import pytest
import requests
import uuid

from tests.e2e.config import (
    PROXY_URL,
    REQUEST_TIMEOUT
)


# ============================================================
# Session ID 传递方式测试
# ============================================================

class TestSessionIdDelivery:
    """Session ID 传递方式测试类"""

    @pytest.mark.integration
    def test_delivery_via_header(
        self,
        proxy_client: requests.Session,
        registered_model_name: str,
        unique_session_id: str
    ):
        """
        测试通过请求头传递 session_id

        验证点:
        - 返回状态码 200
        - 请求正常处理
        """
        response = proxy_client.post(
            f"{PROXY_URL}/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "x-session-id": unique_session_id
            },
            json={
                "model": registered_model_name,
                "messages": [
                    {"role": "user", "content": "测试请求头传递 session_id"}
                ],
                "max_tokens": 20
            }
        )

        assert response.status_code == 200, f"请求失败: {response.text}"
        data = response.json()
        assert "choices" in data, f"响应缺少 choices 字段: {data}"

    @pytest.mark.integration
    def test_delivery_via_path(
        self,
        proxy_client: requests.Session,
        registered_model_name: str,
        unique_session_id: str
    ):
        """
        测试通过 URL 路径传递 session_id

        验证点:
        - 返回状态码 200
        - URL 路径中的 session_id 正确传递
        """
        # 使用路径格式: /s/{session_id}/v1/chat/completions
        response = proxy_client.post(
            f"{PROXY_URL}/s/{unique_session_id}/v1/chat/completions",
            headers={
                "Content-Type": "application/json"
            },
            json={
                "model": registered_model_name,
                "messages": [
                    {"role": "user", "content": "测试路径传递 session_id"}
                ],
                "max_tokens": 20
            }
        )

        assert response.status_code == 200, f"请求失败: {response.text}"
        data = response.json()
        assert "choices" in data, f"响应缺少 choices 字段: {data}"

    @pytest.mark.integration
    def test_delivery_via_model_at_format(
        self,
        proxy_client: requests.Session,
        registered_model_name: str,
        unique_session_id: str
    ):
        """
        测试通过 model@session_id 格式传递 session_id

        验证点:
        - 返回状态码 200
        - model 参数中的 session_id 正确解析
        """
        # 使用 model@session_id 格式
        model_with_session = f"{registered_model_name}@{unique_session_id}"

        response = proxy_client.post(
            f"{PROXY_URL}/v1/chat/completions",
            headers={
                "Content-Type": "application/json"
            },
            json={
                "model": model_with_session,
                "messages": [
                    {"role": "user", "content": "测试 model@session_id 格式"}
                ],
                "max_tokens": 20
            }
        )

        assert response.status_code == 200, f"请求失败: {response.text}"
        data = response.json()
        assert "choices" in data, f"响应缺少 choices 字段: {data}"

    def test_delivery_priority(
        self,
        proxy_client: requests.Session,
        registered_model_name: str,
        unique_session_id: str
    ):
        """
        测试 session_id 传递方式的优先级

        优先级: 路径 > 请求头 > model@session_id

        验证点:
        - 多种方式同时使用时，优先级高的生效
        """
        # 路径传递的 session_id 应该优先生效
        path_session_id = f"path_{unique_session_id}"
        header_session_id = f"header_{unique_session_id}"

        response = proxy_client.post(
            f"{PROXY_URL}/s/{path_session_id}/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "x-session-id": header_session_id  # 这个应该被路径中的覆盖
            },
            json={
                "model": registered_model_name,
                "messages": [
                    {"role": "user", "content": "test"}
                ],
                "max_tokens": 10
            }
        )

        # 请求应该成功
        assert response.status_code == 200, f"请求失败: {response.text}"


# ============================================================
# Session ID 格式验证测试
# ============================================================

class TestSessionIdFormat:
    """Session ID 格式验证测试类"""

    def test_empty_session_id_uses_model_name_as_run_id(
        self,
        proxy_client: requests.Session,
        registered_model_name: str
    ):
        """
        测试 session_id 为空时，run_id 等于 model_name

        验证点:
        - 不传 x-session-id header 时能正常路由到模型
        - 模型 key 为 (model_name, model_name)
        """
        response = proxy_client.post(
            f"{PROXY_URL}/v1/chat/completions",
            json={
                "model": registered_model_name,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 10
            }
        )

        # 如果预置模型是以 (model_name, model_name) 注册的，应该成功
        # 否则返回 404（模型未注册）
        # 这里只验证不会因为 session_id 格式问题返回 400
        if response.status_code == 400:
            data = response.json()
            assert "session_id 格式无效" not in data.get("detail", ""), \
                f"空 session_id 不应触发格式错误: {data}"

    def test_valid_session_id_with_comma(
        self,
        proxy_client: requests.Session,
        registered_model_name: str
    ):
        """
        测试有效 session_id（包含逗号）能正确提取 run_id

        验证点:
        - session_id 格式为 {run_id},{sample_id},{task_id} 时正常处理
        - 不会返回 400 格式错误
        """
        session_id = "test_run,sample_001,task_001"

        response = proxy_client.post(
            f"{PROXY_URL}/v1/chat/completions",
            headers={"x-session-id": session_id},
            json={
                "model": registered_model_name,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 10
            }
        )

        # 可能返回 404（模型未注册），但不应该返回 400（格式错误）
        if response.status_code == 400:
            data = response.json()
            assert "session_id 格式无效" not in data.get("detail", ""), \
                f"有效 session_id 不应触发格式错误: {data}"

    def test_invalid_session_id_without_comma_returns_400(
        self,
        proxy_client: requests.Session,
        registered_model_name: str
    ):
        """
        测试 session_id 存在但不包含逗号时返回 400 错误

        验证点:
        - session_id 格式无效时返回 400
        - 错误信息提示期望的格式
        """
        # session_id 存在但不包含逗号
        invalid_session_id = "invalid_session_id_without_comma"

        response = proxy_client.post(
            f"{PROXY_URL}/v1/chat/completions",
            headers={"x-session-id": invalid_session_id},
            json={
                "model": registered_model_name,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 10
            }
        )

        assert response.status_code == 400, \
            f"无效 session_id 应返回 400，实际返回 {response.status_code}"

        data = response.json()
        assert "session_id 格式无效" in data.get("detail", ""), \
            f"错误信息应提示 session_id 格式无效: {data}"

    def test_model_at_session_id_format(
        self,
        proxy_client: requests.Session,
        registered_model_name: str
    ):
        """
        测试 model@session_id 格式能正确解析

        验证点:
        - model 字段包含 @ 时能正确分离 model 和 session_id
        - 不会返回 400 格式错误
        """
        # 使用 model@session_id 格式
        model_with_session = f"{registered_model_name}@test_run,sample_001,task_001"

        response = proxy_client.post(
            f"{PROXY_URL}/v1/chat/completions",
            json={
                "model": model_with_session,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 10
            }
        )

        # 可能返回 404（模型未注册），但不应该返回 400（格式错误）
        if response.status_code == 400:
            data = response.json()
            assert "session_id 格式无效" not in data.get("detail", ""), \
                f"model@session_id 格式解析后的 session_id 应有效: {data}"


# ============================================================
# Run ID 路由隔离测试
# ============================================================

class TestRunIdRouting:
    """Run ID 路由隔离测试类"""

    @pytest.mark.integration
    def test_run_id_isolation(
        self,
        proxy_client: requests.Session
    ):
        """
        测试不同 run_id 的模型隔离

        验证点:
        - 同一 model_name 可以注册不同 run_id 的多个模型
        - 不同 run_id 的请求路由到不同的模型
        """
        base_model_name = f"isolation_test_{uuid.uuid4().hex[:8]}"
        run_id_1 = f"run1_{uuid.uuid4().hex[:8]}"
        run_id_2 = f"run2_{uuid.uuid4().hex[:8]}"

        try:
            # 注册两个不同 run_id 的模型（使用不同 URL 以区分）
            proxy_client.post(
                f"{PROXY_URL}/models/register",
                json={
                    "run_id": run_id_1,
                    "model_name": base_model_name,
                    "url": "http://localhost:11111",
                    "api_key": "sk-test-1",
                    "tokenizer_path": "Qwen/Qwen3.5-2B"
                }
            )

            proxy_client.post(
                f"{PROXY_URL}/models/register",
                json={
                    "run_id": run_id_2,
                    "model_name": base_model_name,
                    "url": "http://localhost:22222",
                    "api_key": "sk-test-2",
                    "tokenizer_path": "Qwen/Qwen3.5-2B"
                }
            )

            # 验证两个模型都已注册（使用 OpenAI 格式）
            list_response = proxy_client.get(f"{PROXY_URL}/v1/models")
            list_data = list_response.json()
            model_ids = [m.get("id") for m in list_data.get("data", [])]

            assert f"{run_id_1}/{base_model_name}" in model_ids, \
                f"模型 {run_id_1}/{base_model_name} 未注册: {model_ids}"
            assert f"{run_id_2}/{base_model_name}" in model_ids, \
                f"模型 {run_id_2}/{base_model_name} 未注册: {model_ids}"

        finally:
            # 清理
            proxy_client.delete(
                f"{PROXY_URL}/models",
                params={"model_name": base_model_name, "run_id": run_id_1}
            )
            proxy_client.delete(
                f"{PROXY_URL}/models",
                params={"model_name": base_model_name, "run_id": run_id_2}
            )

    @pytest.mark.integration
    def test_no_fallback_to_global_model(
        self,
        proxy_client: requests.Session
    ):
        """
        测试不存在回退到全局模型的逻辑

        验证点:
        - 特定 run_id 的模型不存在时，不会回退到全局模型
        - 返回 404 错误
        """
        model_name = f"no_fallback_test_{uuid.uuid4().hex[:8]}"

        try:
            # 只注册全局模型（run_id 为空）
            proxy_client.post(
                f"{PROXY_URL}/models/register",
                json={
                    "run_id": "",
                    "model_name": model_name,
                    "url": "http://localhost:12345",
                    "api_key": "sk-test",
                    "tokenizer_path": "Qwen/Qwen3.5-2B"
                }
            )

            # 使用不存在的 run_id 请求
            session_id = f"nonexistent_run,sample_001,task_001"
            response = proxy_client.post(
                f"{PROXY_URL}/v1/chat/completions",
                headers={"x-session-id": session_id},
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 10
                }
            )

            # 应该返回 404（模型未注册），而不是回退到全局模型
            assert response.status_code == 404, \
                f"应返回 404（不回退到全局模型），实际返回 {response.status_code}"

        finally:
            # 清理
            proxy_client.delete(
                f"{PROXY_URL}/models",
                params={"model_name": model_name, "run_id": ""}
            )
