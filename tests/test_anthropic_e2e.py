#!/usr/bin/env python3
"""Anthropic /v1/messages 端到端测试

独立于真实 Postgres 与 Docker 部署，使用真实 HTTP 服务模拟 Anthropic 推理后端，
通过 FastAPI TestClient 走完整路由链路：

  TestClient → anthropic_routes → Processor.process_anthropic_request/stream
         → AnthropicDirectPipeline → InferClient → MockAnthropicInfer (HTTP)

依赖：pytest 或 httpx（项目已安装）。

运行方式：
    # 直接运行（输出摘要）
    PYTHONPATH=. python tests/test_anthropic_e2e.py

    # 通过 pytest
    PYTHONPATH=. pytest tests/test_anthropic_e2e.py -v

覆盖场景：
  - UC-01: 非流式普通文本请求（含 system + user）
  - UC-02: 非流式 tool_use 请求（assistant 返回工具调用）
  - UC-03: 非流式 400（缺少 max_tokens）
  - UC-04: 非流式 404（模型未注册）
  - UC-05: 流式文本请求（SSE 透传 + content 累积）
  - UC-06: 流式 tool_use 请求（input_json_delta 累积）
  - UC-07: 非流式路径式 session_id 提取（/s/{session_id}/v1/messages）
  - UC-08: Anthropic 格式错误响应结构校验
"""

import asyncio
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any, Dict, List, Optional, Tuple

# 测试环境：绕过项目真实 configs/config.yaml（YAML 语法可能有误，与 e2e 无关）。
# 在 import traj_proxy.* 之前，先写入一个最小合法的临时 config 并通过环境变量
# CONFIG_PATH（traj_proxy.utils.config 读取）让项目加载它。
_MIN_CONFIG = {
    "log_dir": "/tmp/trajproxy_e2e_logs",
    "log_level": "WARNING",
    "proxy_workers": {
        "count": 1, "base_port": 19000, "max_concurrent_requests": 1024,
        "semaphore_acquire_timeout": 1.0,
    },
    "database": {"host": "127.0.0.1", "port": 5432, "name": "e2e",
                 "user": "u", "password": "p"},
    "processor_manager": {"max_process_cache_size": 10},
    "infer_client": {
        "connect_timeout": 5.0, "read_timeout": 30.0,
        "max_connections": 50, "max_retries": 1,
    },
}
_tmp_cfg_dir = tempfile.mkdtemp(prefix="trajproxy_e2e_cfg_")
_tmp_cfg_path = os.path.join(_tmp_cfg_dir, "config.yaml")
with open(_tmp_cfg_path, "w", encoding="utf-8") as _fh:
    import yaml as _yaml
    _yaml.safe_dump(_MIN_CONFIG, _fh)
os.environ["TRAJ_PROXY_CONFIG"] = _tmp_cfg_path

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

# 被测模块
from traj_proxy.serve.anthropic_routes import anthropic_router
from traj_proxy.proxy_core.pipeline.anthropic_pipeline import AnthropicDirectPipeline
from traj_proxy.proxy_core.context import ProcessContext


# ============================================================
# Mock Anthropic 推理后端
# ============================================================

class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class MockAnthropicInfer:
    """启动真实 HTTP 服务，模拟 Anthropic 推理后端。

    端点：
      POST /v1/messages         正常响应（响应内容由 self.response_factory 决定）
      POST /mock/scenario/{id}  切换到预设场景（stream_text / stream_tool_use / error_500）
      GET  /mock/requests       查看所有收到的请求
      DELETE /mock/requests     清空请求历史
    """

    def __init__(self, port: int = 0):
        self.requests: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self.scenario: str = "stream_text"  # 流式场景切换
        self.port = port
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    # ---------- SSE 响应数据 ----------

    def _sse_text(self) -> List[Tuple[str, Dict[str, Any]]]:
        """简单的文本流式响应事件序列"""
        return [
            ("message_start", {
                "type": "message_start",
                "message": {
                    "id": "msg_01Test", "type": "message", "role": "assistant",
                    "content": [], "model": "claude-3-sonnet",
                    "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": 25, "output_tokens": 1},
                },
            }),
            ("content_block_start", {"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "text_delta", "text": "Hello"}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "text_delta", "text": " World!"}}),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {"type": "message_delta",
                               "delta": {"stop_reason": "end_turn"},
                               "usage": {"output_tokens": 3}}),
            ("message_stop", {"type": "message_stop"}),
        ]

    def _sse_tool_use(self) -> List[Tuple[str, Dict[str, Any]]]:
        """tool_use 流式响应事件序列"""
        return [
            ("message_start", {
                "type": "message_start",
                "message": {
                    "id": "msg_01Tool", "type": "message", "role": "assistant",
                    "content": [], "model": "claude-3-sonnet",
                    "usage": {"input_tokens": 30, "output_tokens": 1},
                },
            }),
            ("content_block_start", {"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "text_delta", "text": "Let me check."}}),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("content_block_start", {"type": "content_block_start", "index": 1,
                                     "content_block": {"type": "tool_use", "id": "toolu_01",
                                                       "name": "get_weather"}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 1,
                                     "delta": {"type": "input_json_delta",
                                               "partial_json": "{\"ci"}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 1,
                                     "delta": {"type": "input_json_delta",
                                               "partial_json": "ty\": \"SF\"}"}}),
            ("content_block_stop", {"type": "content_block_stop", "index": 1}),
            ("message_delta", {"type": "message_delta",
                               "delta": {"stop_reason": "tool_use"},
                               "usage": {"output_tokens": 15}}),
            ("message_stop", {"type": "message_stop"}),
        ]

    # ---------- HTTP Handler ----------

    def _make_handler(self):
        mock = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):  # noqa: A002
                pass  # 静默

            def _record(self, body: bytes):
                try:
                    body_json = json.loads(body) if body else {}
                except Exception:
                    body_json = {"_raw": body.decode("utf-8", "replace")}
                headers = {k: v for k, v in self.headers.items()}
                rec = {
                    "method": self.command,
                    "path": self.path,
                    "headers": headers,
                    "body": body_json,
                    "timestamp": time.time(),
                }
                with mock._lock:
                    mock.requests.append(rec)
                return body_json

            def _send_json(self, data: dict, status: int = 200):
                body = json.dumps(data, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_sse(self, events):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                for event_name, data in events:
                    self.wfile.write(f"event: {event_name}\n".encode("utf-8"))
                    payload = json.dumps(data, ensure_ascii=False)
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    time.sleep(0.005)  # 模拟网络延迟，便于观察流式行为

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length else b""
                body_json = self._record(body)

                if self.path == "/mock/requests":
                    self.send_error(405)
                    return

                if self.path == "/v1/messages":
                    if body_json.get("stream"):
                        scenario = mock.scenario
                        if scenario == "stream_tool_use":
                            self._send_sse(mock._sse_tool_use())
                        else:
                            self._send_sse(mock._sse_text())
                    else:
                        # 非流式：返回完整 Anthropic Message
                        self._send_json({
                            "id": "msg_01Direct",
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Hello from mock backend!"}],
                            "model": "claude-3-sonnet",
                            "stop_reason": "end_turn",
                            "stop_sequence": None,
                            "usage": {"input_tokens": 20, "output_tokens": 8},
                        })
                    return

                self.send_error(404, f"unknown path: {self.path}")

            def do_GET(self):
                if self.path == "/mock/requests":
                    with mock._lock:
                        data = list(mock.requests)
                    self._send_json({"count": len(data), "requests": data})
                    return
                self.send_error(404)

            def do_DELETE(self):
                if self.path == "/mock/requests":
                    with mock._lock:
                        mock.requests.clear()
                    self._send_json({"status": "ok"})
                    return
                self.send_error(404)

        return Handler

    def start(self) -> int:
        handler = self._make_handler()
        self.server = _ThreadingHTTPServer(("127.0.0.1", self.port), handler)
        if self.port == 0:
            self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self.port

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def last_request(self) -> Dict[str, Any]:
        with self._lock:
            return self.requests[-1] if self.requests else {}


# ============================================================
# TrajProxy App 装配（最小化：只挂 anthropic_router + 桩数据）
# ============================================================

class _StubProcessor:
    """封装 AnthropicDirectPipeline，提供 route 层需要的接口。"""

    def __init__(self, pipeline: AnthropicDirectPipeline):
        self._pipeline = pipeline
        self._anthropic_pipeline_cache = pipeline

    async def process_anthropic_request(
        self, body: dict, request_id: str, session_id=None,
        run_id=None, forward_headers=None,
    ) -> ProcessContext:
        ctx = self._pipeline._create_context(
            request_id=request_id, session_id=session_id, run_id=run_id,
            messages=body.get("messages", []), request_params=body,
            is_stream=False, forward_headers=forward_headers or {},
        )
        ctx.api_format = "anthropic"
        ctx.raw_request = dict(body)
        return await self._pipeline.process(body.get("messages", []), ctx)

    async def process_anthropic_stream(
        self, body: dict, request_id: str, session_id=None,
        run_id=None, context_holder=None, forward_headers=None,
    ):
        ctx = self._pipeline._create_context(
            request_id=request_id, session_id=session_id, run_id=run_id,
            messages=body.get("messages", []), request_params=body,
            is_stream=True, forward_headers=forward_headers or {},
        )
        ctx.api_format = "anthropic"
        ctx.raw_request = dict(body)
        try:
            async for raw_sse in self._pipeline.process_stream(
                body.get("messages", []), ctx,
            ):
                yield raw_sse.encode("utf-8") if isinstance(raw_sse, str) else raw_sse
        finally:
            if context_holder is not None:
                context_holder["context"] = ctx


class _StubProcessorManager:
    """可控制注册行为的 ProcessorManager 桩。"""

    def __init__(self, registered: Dict[str, _StubProcessor]):
        self._registered = registered

    async def get_processor_async(self, run_id: str, model: str):
        return self._registered.get(f"{run_id}|{model}")

    async def try_get_or_sync_from_db(self, run_id: str, model: str):
        return None


def _build_app(infer_base_url: str, context_captures: List[ProcessContext]) -> FastAPI:
    """构造最小化的 FastAPI 应用，含 anthropic 路由 + 桩中间件。

    context_captures：每次轨迹存储时追加 ProcessContext，供断言使用。
    """

    app = FastAPI()
    app.include_router(anthropic_router, prefix="/v1", tags=["Anthropic Messages"])
    app.include_router(anthropic_router, prefix="/s/{session_id}/v1",
                       tags=["Anthropic Messages (Path-based)"])

    # 构造一个真实的 AnthropicDirectPipeline（含真实 HTTP 客户端调用）
    # 使用真实 InferClient（指向 mock backend 的 base_url），让 pipeline 走真实 HTTP 链路
    from traj_proxy.proxy_core.infer_client import InferClient
    infer_client = InferClient(base_url=infer_base_url, api_key="test-infer-key")
    pipeline = AnthropicDirectPipeline(
        model="claude-3-test",
        infer_client=infer_client,
        request_repository=None,  # 不连真实 DB
    )

    # 劫持 _store_trajectory，把 context 捕获到 captures 里
    async def _capture_store(context: ProcessContext, tokenizer_path: str = "", run_id=None):
        context_captures.append(context)
        # 不真正写 DB（request_repository 为 None，原方法也是 no-op）
        return None

    pipeline._store_trajectory = _capture_store  # type: ignore[method-assign]

    stub_processor = _StubProcessor(pipeline)
    # run_id=DEFAULT 由 _extract_run_id（routes.py）在路径参数和 header 都无时从 None 规范化而来
    # 用 DEFAULT 作为 key，使 stub 能被查找到
    stub_manager = _StubProcessorManager({"DEFAULT|claude-3-test": stub_processor})

    # 关键：把 stub_manager 挂到 app.state（dependencies.get_processor_manager 从这里读）
    app.state.processor_manager = stub_manager
    # Semaphore 在测试环境置 None：asyncio.Semaphore 与 TestClient 的事件循环绑定时
    # 常出现 "coroutine 'Semaphore.acquire' was never awaited" 告警；
    # route handler 已做 `if semaphore:` 判空，设为 None 即可跳过限流逻辑。
    app.state.request_semaphore = None
    app.state.max_concurrent_requests = 100

    @app.middleware("http")
    async def inject_request_id(request: Request, call_next):
        request.state.request_id = str(uuid.uuid4())
        return await call_next(request)

    return app


# ============================================================
# 测试用例
# ============================================================

def test_uc01_nonstream_text(mock_infer: MockAnthropicInfer, captures: List[ProcessContext]):
    """UC-01: 非流式普通文本请求"""
    app = _build_app(mock_infer.base_url, captures)
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post("/v1/messages", json={
            "model": "claude-3-test",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream": False,
        })

    assert resp.status_code == 200
    data = resp.json()
    assert data["type"] == "message", f"期望 type=message, 实际: {data.get('type')}"
    assert data["role"] == "assistant"
    assert any(b["type"] == "text" for b in data.get("content", []))
    assert data["usage"]["input_tokens"] == 20
    assert data["usage"]["output_tokens"] == 8

    # 验证存储捕获
    assert len(captures) == 1
    ctx = captures[0]
    assert ctx.api_format == "anthropic"
    assert ctx.response_text == "Hello from mock backend!"
    assert ctx.prompt_tokens == 20
    assert ctx.completion_tokens == 8
    assert ctx.total_tokens == 28
    assert ctx.raw_request is not None and ctx.raw_response is not None
    raw_req: Dict[str, Any] = ctx.raw_request
    raw_resp: Dict[str, Any] = ctx.raw_response
    assert raw_req["max_tokens"] == 128
    assert raw_req["messages"][0]["role"] == "user"
    assert raw_resp["type"] == "message"

    # 验证后端收到的请求
    last = mock_infer.last_request()
    assert last["path"] == "/v1/messages"
    assert last["body"]["model"] == "claude-3-test"
    # 客户端未在请求中发送 x-api-key（本测试未注入），故转发路径不会带 x-api-key；
    # 实际部署时 InferClient._build_anthropic_headers 会用模型注册的 api_key 作为 x-api-key。
    # 这里校验：客户端若带 x-api-key 会被 HEADER_BLACKLIST 剔除，且当前测试场景下后端
    # 不应看到客户端 x-api-key。
    assert last["headers"].get("x-api-key") is None, (
        "测试场景未发送 x-api-key，转发路径也不应透传，"
        f"实际 headers={last['headers']}"
    )
    # 校验 Authorization 未注入（Anthropic 走 x-api-key）
    assert "authorization" not in {k.lower() for k in last["headers"]}, \
        "Anthropic 路径不应注入 Authorization header"
    print("    UC-01 非流式文本 OK")


def test_uc02_nonstream_tool_use(mock_infer: MockAnthropicInfer, captures: List[ProcessContext]):
    """UC-02: 非流式带 tools 定义"""
    # 让 mock 后端返回 tool_use 响应
    # 注意：当前简单 mock 只返回文本；这里主要验证 tools 参数被透传
    with TestClient(_build_app(mock_infer.base_url, captures),
                    raise_server_exceptions=False) as client:
        resp = client.post("/v1/messages", json={
            "model": "claude-3-test",
            "max_tokens": 1024,
            "tools": [{
                "name": "get_weather",
                "description": "Get weather for a city",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }],
            "messages": [{"role": "user", "content": "SF weather?"}],
            "stream": False,
        })

    assert resp.status_code == 200
    last = mock_infer.last_request()
    # tools 定义必须完整透传到后端
    assert "tools" in last["body"]
    assert last["body"]["tools"][0]["name"] == "get_weather"
    assert last["body"]["tools"][0]["input_schema"]["type"] == "object"
    print("    UC-02 tools 透传 OK")


def test_uc03_missing_max_tokens(mock_infer: MockAnthropicInfer):
    """UC-03: 缺少 max_tokens 应返回 400 Anthropic 错误格式"""
    captures: List[ProcessContext] = []
    with TestClient(_build_app(mock_infer.base_url, captures),
                    raise_server_exceptions=False) as client:
        resp = client.post("/v1/messages", json={
            "model": "claude-3-test",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        })

    assert resp.status_code == 400
    data = resp.json()
    assert data["type"] == "error"
    assert "error" in data
    assert data["error"]["type"] == "invalid_request_error"
    assert "max_tokens" in data["error"]["message"]
    print("    UC-03 max_tokens 必填校验 OK")


def test_uc04_model_not_registered(mock_infer: MockAnthropicInfer):
    """UC-04: 未注册模型应返回 404 Anthropic 格式"""
    captures: List[ProcessContext] = []
    with TestClient(_build_app(mock_infer.base_url, captures),
                    raise_server_exceptions=False) as client:
        resp = client.post("/v1/messages", json={
            "model": "unknown-model-xyz",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        })

    assert resp.status_code == 404
    data = resp.json()
    assert data["type"] == "error"
    assert data["error"]["type"] == "not_found_error"
    assert "unknown-model-xyz" in data["error"]["message"]
    print("    UC-04 模型未注册 404 OK")


def test_uc05_stream_text(mock_infer: MockAnthropicInfer, captures: List[ProcessContext]):
    """UC-05: 流式文本请求——SSE 透传 + content 累积"""
    mock_infer.scenario = "stream_text"
    app = _build_app(mock_infer.base_url, captures)

    with TestClient(app, raise_server_exceptions=False) as client:
        with client.stream("POST", "/v1/messages", json={
            "model": "claude-3-test",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "Hi!"}],
            "stream": True,
        }) as resp:
            assert resp.status_code == 200
            assert resp.headers.get("content-type", "").startswith("text/event-stream")
            raw_sse = "".join(resp.iter_text())

    # 客户端收到的 SSE 必须包含完整事件序列
    assert "event: message_start" in raw_sse
    assert "event: content_block_delta" in raw_sse
    assert "event: message_delta" in raw_sse
    assert "event: message_stop" in raw_sse
    assert "data: [DONE]" not in raw_sse, \
        "Anthropic 协议不应出现 data: [DONE]"

    # 累积的 context 验证
    assert len(captures) == 1
    ctx = captures[0]
    assert ctx.api_format == "anthropic"
    assert ctx.response_text == "Hello World!", f"累积文本不对: {ctx.response_text!r}"
    assert ctx.prompt_tokens == 25
    assert ctx.completion_tokens == 3
    assert ctx.total_tokens == 28
    assert ctx.raw_response is not None
    raw_resp: Dict[str, Any] = ctx.raw_response
    assert raw_resp["type"] == "message"
    assert raw_resp["stop_reason"] == "end_turn"
    assert any(b["type"] == "text" and b["text"] == "Hello World!" for b in
               raw_resp["content"])
    print("    UC-05 流式文本（透传 + 累积）OK")


def test_uc06_stream_tool_use(mock_infer: MockAnthropicInfer, captures: List[ProcessContext]):
    """UC-06: 流式 tool_use——input_json_delta 累积拼接"""
    mock_infer.scenario = "stream_tool_use"
    app = _build_app(mock_infer.base_url, captures)

    with TestClient(app, raise_server_exceptions=False) as client:
        with client.stream("POST", "/v1/messages", json={
            "model": "claude-3-test",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "SF?"}],
            "stream": True,
        }) as resp:
            assert resp.status_code == 200
            raw_sse = "".join(resp.iter_text())

    assert "event: message_stop" in raw_sse

    ctx = captures[0]
    assert ctx.stream_finish_reason == "tool_use"
    assert ctx.raw_response is not None
    raw_resp: Dict[str, Any] = ctx.raw_response
    # content 应包含 text + tool_use 两个 block
    content = raw_resp["content"]
    tool_block = next((b for b in content if b["type"] == "tool_use"), None)
    assert tool_block is not None, f"未找到 tool_use block: {content}"
    assert tool_block["id"] == "toolu_01"
    assert tool_block["name"] == "get_weather"
    # partial_json delta 累积后 json.loads 应为合法 dict
    assert tool_block["input"] == {"city": "SF"}, \
        f"input delta 累积错误: {tool_block['input']!r}"
    print("    UC-06 流式 tool_use (input_json_delta 累积) OK")


def test_uc07_path_session_id(mock_infer: MockAnthropicInfer, captures: List[ProcessContext]):
    """UC-07: /s/{session_id}/v1/messages 路径参数解析"""
    app = _build_app(mock_infer.base_url, captures)

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post(
            "/s/sess-from-path/v1/messages",
            json={
                "model": "claude-3-test",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
    # 该路径下 stub_manager 仅注册了 run_id="" + model="claude-3-test"，故应能命中
    assert resp.status_code == 200
    ctx = captures[0]
    assert ctx.session_id == "sess-from-path", \
        f"session_id 应为路径参数值，实际: {ctx.session_id!r}"
    print("    UC-07 路径式 session_id OK")


def test_uc08_error_format_contract(mock_infer: MockAnthropicInfer):
    """UC-08: 错误响应结构符合 Anthropic SDK 约定

    Anthropic SDK 期望:
      {"type": "error", "error": {"type": <str>, "message": <str>}}
    """
    captures: List[ProcessContext] = []
    with TestClient(_build_app(mock_infer.base_url, captures),
                    raise_server_exceptions=False) as client:
        # 触发 400
        r = client.post("/v1/messages", json={
            "model": "claude-3-test",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert r.status_code == 400

        # 触发 422（model 校验失败）
        r2 = client.post("/v1/messages", json={
            "model": "invalid,name,with,extra,commas",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "x"}],
        })

    for response in [r, r2]:
        data = response.json()
        assert "type" in data and data["type"] == "error"
        assert "error" in data and isinstance(data["error"], dict)
        assert isinstance(data["error"].get("type"), str)
        assert isinstance(data["error"].get("message"), str)
    print("    UC-08 错误响应结构契约 OK")


def test_uc09_api_key_blacklist(mock_infer: MockAnthropicInfer, captures: List[ProcessContext]):
    """UC-09: 客户端发送的 x-api-key 必须被 HEADER_BLACKLIST 剔除，不透传到后端

    保证两个语义：
      1) InferClient 注册的 api_key 作为 x-api-key 注入（独立于客户端 key）
      2) 客户端伪造的 x-api-key 不能穿透到后端
    """
    app = _build_app(mock_infer.base_url, captures)

    with TestClient(app, raise_server_exceptions=False) as client:
        # 客户端携带"伪造" x-api-key；按黑名单规则应被剔除
        client.post("/v1/messages", json={
            "model": "claude-3-test",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "check headers"}],
            "stream": False,
        }, headers={"x-api-key": "attacker-key"})

    last = mock_infer.last_request()
    # 后端的 x-api-key 由 InferClient 注入，使用注册的 api_key "test-infer-key"；
    # 客户端的 attacker-key 必须被黑名单剔除。
    # HTTP header 名大小写不敏感，需按小写键查找。
    lower_headers = {k.lower(): v for k, v in last["headers"].items()}
    assert lower_headers.get("x-api-key") == "test-infer-key", \
        f"后端期望收到注册的 api_key，实际 headers={last['headers']}"
    print("    UC-09 x-api-key 黑名单隔离 OK")


# ============================================================
# 测试夹具
# ============================================================

def _new_mock_infer():
    m = MockAnthropicInfer()
    m.start()
    return m


def _run_all():
    """以脚本方式执行全部用例并输出报告。"""
    mock_infer = _new_mock_infer()
    results = []

    cases = [
        ("UC-01 非流式文本", test_uc01_nonstream_text),
        ("UC-02 非流式 tools 透传", test_uc02_nonstream_tool_use),
        ("UC-03 max_tokens 必填", test_uc03_missing_max_tokens),
        ("UC-04 模型未注册", test_uc04_model_not_registered),
        ("UC-05 流式文本", test_uc05_stream_text),
        ("UC-06 流式 tool_use", test_uc06_stream_tool_use),
        ("UC-07 路径 session_id", test_uc07_path_session_id),
        ("UC-08 错误格式契约", test_uc08_error_format_contract),
        ("UC-09 x-api-key 黑名单", test_uc09_api_key_blacklist),
    ]

    try:
        for name, fn in cases:
            captures = []
            try:
                # 每次用例清空 mock 接收历史
                with mock_infer._lock:
                    mock_infer.requests.clear()
                # 根据函数签名决定传 captures 与否
                params = fn.__code__.co_varnames[:fn.__code__.co_argcount]
                if "captures" in params:
                    fn(mock_infer, captures)
                else:
                    fn(mock_infer)
                results.append((name, "PASS", ""))
            except Exception as e:
                results.append((name, "FAIL", f"{type(e).__name__}: {e}"))
    finally:
        mock_infer.stop()

    # 输出报告
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    print()
    print("=" * 64)
    print(f"Anthropic /v1/messages e2e 测试报告   PASSED: {passed}   FAILED: {failed}")
    print("=" * 64)
    for name, status, err in results:
        print(f"  [{status}] {name}")
        if err:
            for line in err.split("\n")[:5]:
                print(f"         {line}")
    print("=" * 64)
    return 0 if failed == 0 else 1


# ============================================================
# pytest 集成
# ============================================================

import pytest  # noqa: E402  放后面保证顶层脚本模式可用


@pytest.fixture(scope="module")
def mock_infer():
    m = _new_mock_infer()
    try:
        yield m
    finally:
        m.stop()


@pytest.fixture
def captures():
    return []


def test_e2e_uc01(mock_infer, captures): test_uc01_nonstream_text(mock_infer, captures)
def test_e2e_uc02(mock_infer, captures): test_uc02_nonstream_tool_use(mock_infer, captures)
def test_e2e_uc03(mock_infer): test_uc03_missing_max_tokens(mock_infer)
def test_e2e_uc04(mock_infer): test_uc04_model_not_registered(mock_infer)
def test_e2e_uc05(mock_infer, captures): test_uc05_stream_text(mock_infer, captures)
def test_e2e_uc06(mock_infer, captures): test_uc06_stream_tool_use(mock_infer, captures)
def test_e2e_uc07(mock_infer, captures): test_uc07_path_session_id(mock_infer, captures)
def test_e2e_uc08(mock_infer): test_uc08_error_format_contract(mock_infer)
def test_e2e_uc09(mock_infer, captures): test_uc09_api_key_blacklist(mock_infer, captures)


if __name__ == "__main__":
    sys.exit(_run_all())
