"""
InferClient - Infer 服务客户端

调用 OpenAI /v1/completions 接口。
"""

from typing import Dict, Any, AsyncIterator, List, Union, Optional
import traceback
import httpx
import json
import asyncio

from traj_proxy.exceptions import InferServiceError
from traj_proxy.utils.logger import get_logger

logger = get_logger(__name__)


# prompt 参数类型：可以是 string 或 List[int] (token ids)
PromptInput = Union[str, List[int]]


class InferClient:
    """Infer 服务客户端 - 调用 OpenAI /v1/completions 接口

    提供 send_completion (非流式) 和 send_completion_stream (流式) 两种调用方式。
    prompt 参数支持 string 或 List[int] (token ids)，兼容 OpenAI 标准。

    每个实例维护独立的 HTTP 客户端，避免多进程环境下的共享问题。
    支持配置化的连接池参数。
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 300.0,
        connect_timeout: float = 60.0,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        keepalive_expiry: float = 30.0
    ):
        """初始化 InferClient

        Args:
            base_url: Infer 服务的 base URL（如 http://localhost:8000）
            api_key: API 密钥
            timeout: 请求超时时间（秒）
            connect_timeout: 连接超时时间（秒）
            max_connections: 最大连接数
            max_keepalive_connections: 最大保活连接数
            keepalive_expiry: 保活连接过期时间（秒）
        """
        if base_url is None:
            raise ValueError("InferClient base_url 必须提供")
        if api_key is None:
            raise ValueError("InferClient api_key 必须提供")

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

        # 连接池配置
        self._timeout_config = httpx.Timeout(timeout, connect=connect_timeout)
        self._limits_config = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            keepalive_expiry=keepalive_expiry
        )

        # 实例级客户端（延迟初始化）
        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        """获取或创建 HTTP 客户端（延迟初始化）

        Returns:
            httpx.AsyncClient 实例
        """
        if self._client is None or self._client.is_closed:
            async with self._client_lock:
                # 双重检查
                if self._client is None or self._client.is_closed:
                    self._client = httpx.AsyncClient(
                        timeout=self._timeout_config,
                        limits=self._limits_config
                    )
                    logger.debug(f"[{self.base_url}] InferClient: 创建新的 HTTP 客户端")
        return self._client

    async def close(self):
        """关闭 HTTP 客户端"""
        if self._client is not None and not self._client.is_closed:
            async with self._client_lock:
                if self._client is not None and not self._client.is_closed:
                    await self._client.aclose()
                    self._client = None
                    logger.debug(f"[{self.base_url}] InferClient: 关闭 HTTP 客户端")

    def _build_request_body(
        self,
        prompt: PromptInput,
        model: str,
        stream: bool,
        **kwargs
    ) -> Dict[str, Any]:
        """构建请求体

        Args:
            prompt: 待补全的提示（string 或 List[int]）
            model: 模型名称
            stream: 是否流式
            **kwargs: 其他请求参数

        Returns:
            请求体字典
        """
        # 获取参数并确保非 None 值
        max_tokens = kwargs.get("max_tokens", 100)
        if max_tokens is None:
            max_tokens = 100

        temperature = kwargs.get("temperature", 0.7)
        if temperature is None:
            temperature = 0.7

        request_body = {
            "model": model,
            "prompt": prompt,  # OpenAI 标准：支持 string 或 List[int]
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if stream:
            request_body["stream"] = True

        # 支持额外参数（只添加非 None 的值）
        extra_params = ["top_p", "presence_penalty", "frequency_penalty", "logprobs", "n", "seed", "stop", "echo", "best_of"]
        for param in extra_params:
            if param in kwargs and kwargs[param] is not None:
                request_body[param] = kwargs[param]
        request_body['return_token_ids'] = True # 强制返回 token ids，便于后续处理
        request_body['logprobs'] = 1

        return request_body

    async def send_completion(
        self,
        prompt: PromptInput,
        model: str,
        **kwargs
    ) -> Dict[str, Any]:
        """发送补全请求（非流式）

        Args:
            prompt: 待补全的提示，支持：
                - string: 文本模式
                - List[int]: token ids 模式
            model: 模型名称
            **kwargs: 其他请求参数（如 max_tokens, temperature, logprobs, tools 等）

        Returns:
            Infer 服务返回的 JSON 响应

        Raises:
            InferServiceError: 当请求失败时抛出
        """
        url = f"{self.base_url}/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        request_body = self._build_request_body(prompt, model, stream=False, **kwargs)

        logger.debug(f"[{model}] 发送 Infer 请求: url={url}, prompt_type={type(prompt).__name__}")
        try:
            client = await self._get_client()
            response = await client.post(url, json=request_body, headers=headers)
            response.raise_for_status()
            logger.debug(f"[{model}] Infer 响应: status={response.status_code}")
            logger.error(f"request_body: {request_body}, response: {response.json()}")
            return response.json()
        except httpx.HTTPStatusError as e:
            raise InferServiceError(f"Infer 服务请求失败: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise InferServiceError(f"Infer 服务请求异常: {str(e)}\n{traceback.format_exc()}")

    async def send_completion_stream(
        self,
        prompt: PromptInput,
        model: str,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """发送补全请求（流式）

        Args:
            prompt: 待补全的提示，支持：
                - string: 文本模式
                - List[int]: token ids 模式
            model: 模型名称
            **kwargs: 其他请求参数

        Yields:
            每个流式响应的 JSON 数据

        Raises:
            InferServiceError: 当请求失败时抛出
        """
        url = f"{self.base_url}/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        request_body = self._build_request_body(prompt, model, stream=True, **kwargs)

        try:
            client = await self._get_client()
            async with client.stream("POST", url, json=request_body, headers=headers) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            yield json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
        except httpx.HTTPStatusError as e:
            raise InferServiceError(f"Infer 服务流式请求失败: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise InferServiceError(f"Infer 服务流式请求异常: {str(e)}\n{traceback.format_exc()}")

    async def send_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        **kwargs
    ) -> Dict[str, Any]:
        """发送 Chat Completions 请求（非流式）

        直接转发 OpenAI 格式的 chat completions 请求到推理服务。

        Args:
            messages: OpenAI 格式的消息列表
            model: 模型名称
            **kwargs: 其他请求参数（如 max_tokens, temperature, tools 等）

        Returns:
            推理服务返回的原始 JSON 响应

        Raises:
            InferServiceError: 当请求失败时抛出
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 构建请求体
        request_body = {
            "model": model,
            "messages": messages,
        }

        # 添加可选参数
        optional_params = [
            "max_tokens", "max_completion_tokens", "temperature", "top_p", "presence_penalty", "frequency_penalty",
            "tools", "tool_choice", "parallel_tool_calls", "stream", "stop", "n", "seed"
        ]
        for param in optional_params:
            if param in kwargs and kwargs[param] is not None:
                request_body[param] = kwargs[param]

        logger.debug(f"[{model}] 发送 Chat Completions 请求: url={url}")
        try:
            client = await self._get_client()
            response = await client.post(url, json=request_body, headers=headers)
            response.raise_for_status()
            logger.debug(f"[{model}] Chat Completions 响应: status={response.status_code}")
            return response.json()
        except httpx.HTTPStatusError as e:
            raise InferServiceError(f"Infer 服务请求失败: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise InferServiceError(f"Infer 服务请求异常: {str(e)}\n{traceback.format_exc()}")

    async def send_chat_completion_stream(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """发送 Chat Completions 请求（流式）

        直接转发 OpenAI 格式的 chat completions 流式请求到推理服务。

        Args:
            messages: OpenAI 格式的消息列表
            model: 模型名称
            **kwargs: 其他请求参数

        Yields:
            每个流式响应的 JSON 数据

        Raises:
            InferServiceError: 当请求失败时抛出
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 构建请求体
        request_body = {
            "model": model,
            "messages": messages,
            "stream": True
        }

        # 添加可选参数
        optional_params = [
            "max_tokens", "max_completion_tokens", "temperature", "top_p", "presence_penalty", "frequency_penalty",
            "tools", "tool_choice", "parallel_tool_calls", "stop", "n", "seed"
        ]
        for param in optional_params:
            if param in kwargs and kwargs[param] is not None:
                request_body[param] = kwargs[param]

        try:
            client = await self._get_client()
            async with client.stream("POST", url, json=request_body, headers=headers) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            yield json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
        except httpx.HTTPStatusError as e:
            raise InferServiceError(f"Infer 服务流式请求失败: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise InferServiceError(f"Infer 服务流式请求异常: {str(e)}\n{traceback.format_exc()}")

    def __repr__(self) -> str:
        return f"InferClient(base_url={self.base_url})"
