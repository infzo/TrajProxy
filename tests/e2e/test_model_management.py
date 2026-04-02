"""
模型管理 API 测试

测试模型的注册、删除和列表接口

注意：Session ID 相关测试已移至 test_session_id.py
"""

import pytest
import requests
import uuid

from tests.e2e.config import PROXY_URL


class TestModelRegistration:
    """模型注册测试类"""

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
                "url": "http://localhost:1234/v1",  # 测试用的推理服务地址
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

        # 验证模型已删除
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
                "url": "http://localhost:1234/v1",
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
                "url": "http://localhost:1234/v1",
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
                "url": "http://localhost:1234/v1",
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


class TestModelDeletion:
    """模型删除测试类"""

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
                "url": "http://localhost:1234/v1",
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


class TestModelListing:
    """模型列表测试类"""

    def test_list_models(self, proxy_client: requests.Session):
        """
        测试列出模型接口

        验证 OpenAI 格式和管理格式的模型列表接口：
        - OpenAI 格式: /v1/models
        - 管理格式: /models/
        """
        # 测试 OpenAI 格式
        openai_response = proxy_client.get(f"{PROXY_URL}/v1/models")
        assert openai_response.status_code == 200, f"OpenAI 格式列出模型失败: {openai_response.text}"

        openai_data = openai_response.json()
        assert openai_data.get("object") == "list", f"object 字段错误: {openai_data.get('object')}"
        assert "data" in openai_data, "响应缺少 data 字段"
        assert isinstance(openai_data["data"], list), "data 不是列表类型"

        if len(openai_data["data"]) > 0:
            model = openai_data["data"][0]
            assert "id" in model, "模型条目缺少 id 字段"
            assert model.get("object") == "model", f"模型 object 字段错误: {model.get('object')}"

        # 测试管理格式
        admin_response = proxy_client.get(f"{PROXY_URL}/models/")
        assert admin_response.status_code == 200, f"管理格式列出模型失败: {admin_response.text}"

        admin_data = admin_response.json()
        assert admin_data.get("status") == "success", f"状态错误: {admin_data.get('status')}"
        assert "count" in admin_data, "响应缺少 count 字段"
        assert "models" in admin_data, "响应缺少 models 字段"

        if len(admin_data["models"]) > 0:
            model = admin_data["models"][0]
            assert "model_name" in model, "模型缺少 model_name 字段"
