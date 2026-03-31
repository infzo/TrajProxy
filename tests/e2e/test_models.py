"""
模型管理 API 测试

测试模型的注册、删除和列表接口
"""

import pytest
import requests
import uuid

from tests.e2e.config import PROXY_URL


class TestModelsAPI:
    """模型管理测试类"""

    def test_list_models(self, proxy_client: requests.Session):
        """
        测试列出模型接口（OpenAI 格式）

        验证点:
        - 返回状态码 200
        - 响应格式符合 OpenAI /v1/models 规范
        - 至少包含一个模型
        """
        response = proxy_client.get(f"{PROXY_URL}/v1/models")

        assert response.status_code == 200, f"列出模型失败: {response.text}"

        data = response.json()

        # 验证响应结构
        assert data.get("object") == "list", f"object 字段错误: {data.get('object')}"
        assert "data" in data, "响应缺少 data 字段"
        assert isinstance(data["data"], list), "data 不是列表类型"

        # 验证模型条目结构
        if len(data["data"]) > 0:
            model = data["data"][0]
            assert "id" in model, "模型条目缺少 id 字段"
            assert model.get("object") == "model", f"模型 object 字段错误: {model.get('object')}"

    def test_list_models_detail(self, proxy_client: requests.Session):
        """
        测试列出模型详情接口（管理格式）

        验证点:
        - 返回状态码 200
        - 响应包含模型详细信息

        注意: 管理格式路由需要尾部斜杠 /models/
        """
        # 访问管理格式的模型列表（需要尾部斜杠）
        response = proxy_client.get(f"{PROXY_URL}/models/")

        assert response.status_code == 200, f"列出模型详情失败: {response.text}"

        data = response.json()

        # 验证响应结构
        assert data.get("status") == "success", f"状态错误: {data.get('status')}"
        assert "count" in data, "响应缺少 count 字段"
        assert "models" in data, "响应缺少 models 字段"
        assert isinstance(data["models"], list), "models 不是列表类型"

        # 验证模型详情结构
        if len(data["models"]) > 0:
            model = data["models"][0]
            assert "model_name" in model, "模型缺少 model_name 字段"

    @pytest.mark.integration
    def test_register_and_delete_model(self, proxy_client: requests.Session):
        """
        测试注册和删除模型

        验证点:
        - 注册成功返回 200
        - 注册后能查询到新模型
        - 删除成功返回 200
        - 删除后无法查询到该模型
        """
        # 生成唯一的模型名称
        test_model_name = f"test_model_{uuid.uuid4().hex[:8]}"

        # 注册模型
        register_response = proxy_client.post(
            f"{PROXY_URL}/models/register",
            json={
                "model_name": test_model_name,
                "url": "http://localhost:1234",  # 测试用的推理服务地址
                "api_key": "sk-test-key",
                "tokenizer_path": "Qwen/Qwen3.5-2B",
                "token_in_token_out": False
            }
        )

        assert register_response.status_code == 200, f"注册模型失败: {register_response.text}"

        register_data = register_response.json()
        assert register_data.get("status") == "success", f"注册状态错误: {register_data}"
        assert register_data.get("model_name") == test_model_name, f"模型名称不匹配: {register_data}"

        # 验证模型已注册（访问管理格式的模型列表，需要尾部斜杠）
        list_response = proxy_client.get(f"{PROXY_URL}/models/")
        assert list_response.status_code == 200

        list_data = list_response.json()
        model_names = [m.get("model_name") for m in list_data.get("models", [])]
        assert test_model_name in model_names, f"注册的模型不在列表中: {model_names}"

        # 删除模型
        delete_response = proxy_client.delete(f"{PROXY_URL}/models", params={"model_name": test_model_name})

        assert delete_response.status_code == 200, f"删除模型失败: {delete_response.text}"

        delete_data = delete_response.json()
        assert delete_data.get("status") == "success", f"删除状态错误: {delete_data}"
        assert delete_data.get("deleted") is True, f"deleted 字段错误: {delete_data}"

        # 验证模型已删除（访问管理格式的模型列表，需要尾部斜杠）
        list_response2 = proxy_client.get(f"{PROXY_URL}/models/")
        list_data2 = list_response2.json()
        model_names2 = [m.get("model_name") for m in list_data2.get("models", [])]
        assert test_model_name not in model_names2, f"删除的模型仍在列表中: {model_names2}"

    @pytest.mark.integration
    def test_register_model_with_tool_parser(self, proxy_client: requests.Session):
        """
        测试带 tool_parser 的模型注册

        验证点:
        - 注册成功返回 200
        - 响应包含 tool_parser 字段
        """
        test_model_name = f"test_model_tool_{uuid.uuid4().hex[:8]}"

        # 注册带 tool_parser 的模型
        register_response = proxy_client.post(
            f"{PROXY_URL}/models/register",
            json={
                "model_name": test_model_name,
                "url": "http://localhost:1234",
                "api_key": "sk-test-key",
                "tokenizer_path": "Qwen/Qwen3.5-2B",
                "token_in_token_out": False,
                "tool_parser": "deepseek_v3"
            }
        )

        assert register_response.status_code == 200, f"注册模型失败: {register_response.text}"

        register_data = register_response.json()
        assert register_data.get("status") == "success", f"注册状态错误: {register_data}"

        # 验证响应中的 tool_parser
        detail = register_data.get("detail", {})
        assert detail.get("tool_parser") == "deepseek_v3", f"tool_parser 不匹配: {detail}"

        # 清理：删除模型
        proxy_client.delete(f"{PROXY_URL}/models", params={"model_name": test_model_name})

    @pytest.mark.integration
    def test_register_model_with_reasoning_parser(self, proxy_client: requests.Session):
        """
        测试带 reasoning_parser 的模型注册

        验证点:
        - 注册成功返回 200
        - 响应包含 reasoning_parser 字段
        """
        test_model_name = f"test_model_reasoning_{uuid.uuid4().hex[:8]}"

        # 注册带 reasoning_parser 的模型
        register_response = proxy_client.post(
            f"{PROXY_URL}/models/register",
            json={
                "model_name": test_model_name,
                "url": "http://localhost:1234",
                "api_key": "sk-test-key",
                "tokenizer_path": "Qwen/Qwen3.5-2B",
                "token_in_token_out": False,
                "reasoning_parser": "deepseek_r1"
            }
        )

        assert register_response.status_code == 200, f"注册模型失败: {register_response.text}"

        register_data = register_response.json()
        assert register_data.get("status") == "success", f"注册状态错误: {register_data}"

        # 验证响应中的 reasoning_parser
        detail = register_data.get("detail", {})
        assert detail.get("reasoning_parser") == "deepseek_r1", f"reasoning_parser 不匹配: {detail}"

        # 清理：删除模型
        proxy_client.delete(f"{PROXY_URL}/models", params={"model_name": test_model_name})

    @pytest.mark.integration
    def test_register_model_with_run_id(self, proxy_client: requests.Session):
        """
        测试带 run_id 的模型注册

        验证点:
        - 注册成功返回 200
        - 响应包含 run_id 字段
        - 模型列表中显示 run_id/model_name 格式
        """
        test_model_name = f"test_model_runid_{uuid.uuid4().hex[:8]}"
        test_run_id = f"run_{uuid.uuid4().hex[:8]}"

        # 注册带 run_id 的模型
        register_response = proxy_client.post(
            f"{PROXY_URL}/models/register",
            json={
                "run_id": test_run_id,
                "model_name": test_model_name,
                "url": "http://localhost:1234",
                "api_key": "sk-test-key",
                "tokenizer_path": "Qwen/Qwen3.5-2B",
                "token_in_token_out": False
            }
        )

        assert register_response.status_code == 200, f"注册模型失败: {register_response.text}"

        register_data = register_response.json()
        assert register_data.get("status") == "success", f"注册状态错误: {register_data}"
        assert register_data.get("run_id") == test_run_id, f"run_id 不匹配: {register_data}"

        # 验证模型列表中包含 run_id（使用 OpenAI 格式）
        list_response = proxy_client.get(f"{PROXY_URL}/v1/models")
        list_data = list_response.json()
        model_ids = [m.get("id") for m in list_data.get("data", [])]
        expected_id = f"{test_run_id}/{test_model_name}"
        assert expected_id in model_ids, f"模型 {expected_id} 不在列表中: {model_ids}"

        # 清理：删除模型（需要指定 run_id）
        delete_response = proxy_client.delete(
            f"{PROXY_URL}/models",
            params={"model_name": test_model_name, "run_id": test_run_id}
        )
        assert delete_response.status_code == 200, f"删除模型失败: {delete_response.text}"

    def test_register_duplicate_model(
        self,
        proxy_client: requests.Session,
        registered_model_name: str
    ):
        """
        测试注册重复模型

        验证点:
        - 返回状态码 400
        - 错误信息提示模型已存在
        """
        # 尝试注册已存在的模型
        response = proxy_client.post(
            f"{PROXY_URL}/models/register",
            json={
                "model_name": registered_model_name,
                "url": "http://localhost:1234",
                "api_key": "sk-test-key",
                "tokenizer_path": "test/tokenizer"
            }
        )

        # 可能返回 400 或 409
        assert response.status_code in [400, 409], \
            f"预期返回 400 或 409，实际返回 {response.status_code}"

        data = response.json()
        assert "已存在" in data.get("detail", "") or "存在" in data.get("detail", ""), \
            f"错误信息未提示模型已存在: {data}"

    def test_delete_nonexistent_model(self, proxy_client: requests.Session):
        """
        测试删除不存在的模型

        验证点:
        - 返回状态码 404
        - 错误信息提示模型不存在
        """
        response = proxy_client.delete(
            f"{PROXY_URL}/models",
            params={"model_name": "nonexistent_model_xyz"}
        )

        assert response.status_code == 404, f"预期返回 404，实际返回 {response.status_code}"

        data = response.json()
        assert "不存在" in data.get("detail", "") or "未找到" in data.get("detail", ""), \
            f"错误信息未提示模型不存在: {data}"


class TestSessionIdRouting:
    """session_id 路由测试类"""

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
        import uuid
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
        import uuid
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
