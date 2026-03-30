"""
pytest 配置和 fixtures

提供测试所需的通用 fixtures
"""

import pytest
import requests
import uuid
from typing import Generator

from tests.e2e.config import (
    PROXY_URL,
    LITELLM_URL,
    NGINX_URL,
    DEFAULT_MODEL,
    LITELLM_API_KEY,
    ANTHROPIC_VERSION,
    get_session_id,
    REQUEST_TIMEOUT
)


@pytest.fixture
def proxy_client() -> requests.Session:
    """
    创建用于访问 TrajProxy 的 HTTP 客户端

    返回:
        配置好的 requests.Session 实例
    """
    session = requests.Session()
    session.timeout = REQUEST_TIMEOUT
    yield session
    session.close()


@pytest.fixture
def unique_session_id() -> str:
    """
    生成唯一的 session_id，确保测试隔离

    返回:
        格式为 {prefix}_{uuid};{sample_id};{task_id} 的唯一 session_id
    """
    unique_prefix = f"e2e_{uuid.uuid4().hex[:8]}"
    return f"{unique_prefix};sample_001;task_001"


@pytest.fixture
def sample_id() -> str:
    """
    生成唯一的 sample_id

    返回:
        格式为 sample_{uuid} 的唯一 sample_id
    """
    return f"sample_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def task_id() -> str:
    """
    生成唯一的 task_id

    返回:
        格式为 task_{uuid} 的唯一 task_id
    """
    return f"task_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def default_headers(unique_session_id: str) -> dict:
    """
    默认请求头，包含唯一的 session_id

    参数:
        unique_session_id: 唯一 session_id fixture

    返回:
        请求头字典
    """
    return {
        "Content-Type": "application/json",
        "x-session-id": unique_session_id
    }


@pytest.fixture(scope="session")
def check_service_available():
    """
    检查服务是否可用

    在所有测试开始前检查 TrajProxy 是否运行
    """
    try:
        response = requests.get(f"{PROXY_URL}/health", timeout=5)
        if response.status_code != 200:
            pytest.fail(f"TrajProxy 服务不可用: {PROXY_URL}")
    except requests.exceptions.ConnectionError:
        pytest.fail(f"无法连接到 TrajProxy 服务: {PROXY_URL}")


@pytest.fixture
def registered_model_name() -> str:
    """
    返回预置模型名称

    返回:
        config.yaml 中配置的模型名称
    """
    return DEFAULT_MODEL


# ============================================
# 完整链路测试 fixtures (Nginx -> LiteLLM -> TrajProxy)
# ============================================

@pytest.fixture
def nginx_client() -> requests.Session:
    """
    创建用于访问 Nginx 网关的 HTTP 客户端

    返回:
        配置好的 requests.Session 实例
    """
    session = requests.Session()
    session.timeout = REQUEST_TIMEOUT
    yield session
    session.close()


@pytest.fixture
def nginx_url() -> str:
    """
    返回 Nginx 网关地址

    返回:
        Nginx URL (默认 http://localhost:80)
    """
    return NGINX_URL


@pytest.fixture
def litellm_api_key() -> str:
    """
    返回 LiteLLM 认证密钥

    返回:
        LiteLLM API Key
    """
    return LITELLM_API_KEY


@pytest.fixture
def openai_headers(litellm_api_key: str) -> dict:
    """
    OpenAI 请求头

    参数:
        litellm_api_key: LiteLLM API Key fixture

    返回:
        OpenAI 格式的请求头
    """
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {litellm_api_key}"
    }


@pytest.fixture
def claude_headers(litellm_api_key: str) -> dict:
    """
    Claude (Anthropic) 请求头

    参数:
        litellm_api_key: LiteLLM API Key fixture

    返回:
        Anthropic 格式的请求头
    """
    return {
        "Content-Type": "application/json",
        "x-api-key": litellm_api_key,
        "anthropic-version": ANTHROPIC_VERSION
    }


def pytest_configure(config):
    """
    pytest 配置钩子，注册自定义标记
    """
    config.addinivalue_line(
        "markers", "slow: 标记慢速测试（如流式测试）"
    )
    config.addinivalue_line(
        "markers", "integration: 标记集成测试（需要推理服务）"
    )
