# TrajProxy 架构文档

## 概述

TrajProxy 是一个 LLM 请求代理服务，支持两种请求处理模式：

1. **Token-in-Token-out 模式**：完整的请求处理流程，包括 prompt 构建、token 编码/解码、前缀匹配缓存
2. **直接转发模式**：轻量级代理，直接转发 OpenAI 格式请求到推理服务

---

## 一、两种处理模式

### 1.1 Token-in-Token-out 模式

**适用场景：**
- 需要 token 级别的精确控制
- 需要前缀匹配缓存（减少重复计算）
- 需要 prompt_text 用于调试或分析
- 模型返回 token ids 而非文本

**必需参数：**
- `tokenizer_path`: Tokenizer 路径（本地路径或 HuggingFace 模型名称）

**处理流程：**
```
OpenAI Request
     │
     ▼
┌─────────────────┐
│  PromptBuilder  │  messages → prompt_text
│ (apply_chat_template)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  TokenBuilder   │  prompt_text → token_ids
│   (encode)      │  前缀匹配缓存
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   InferClient   │  /v1/completions
│   (token_ids)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  TokenBuilder   │  token_ids → response_text
│   (decode)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PromptBuilder  │  构建 OpenAI Response
│ (parse tool_calls, reasoning)
└────────┬────────┘
         │
         ▼
OpenAI Response
```

### 1.2 直接转发模式

**适用场景：**
- 推理服务已支持 `/v1/chat/completions` 接口
- 不需要 token 级别的控制
- 简单的请求代理和轨迹记录
- 快速集成，无需配置 tokenizer

**必需参数：**
- 无需 `tokenizer_path`

**处理流程：**
```
OpenAI Request
     │
     ▼
┌─────────────────┐
│   InferClient   │  直接转发
│ /v1/chat/completions
└────────┬────────┘
         │
         ▼
OpenAI Response (原格式返回)
```

---

## 二、关键组件

### 2.1 Processor（非流式处理）

**文件：** `traj_proxy/proxy_core/processor.py`

**职责：**
- 协调非流式请求的完整处理流程
- 根据模式选择处理路径

**核心方法：**
```python
async def process_request(
    self,
    messages: list,
    request_id: str,
    session_id: Optional[str] = None,
    **request_params
) -> ProcessContext:
    """处理非流式请求"""
    if not self.token_in_token_out:
        # 直接转发模式
        return await self._process_direct_forward(...)
    # Token-in-Token-out 模式
    # 完整处理流程...
```

### 2.2 StreamingProcessor（流式处理）

**文件：** `traj_proxy/proxy_core/streaming_processor.py`

**职责：**
- 协调流式请求的处理流程
- 管理流式状态

**核心方法：**
```python
async def process_stream(
    self,
    messages: list,
    request_id: str,
    session_id: Optional[str] = None,
    context_holder: Optional[dict] = None,
    **request_params
) -> AsyncIterator[Dict[str, Any]]:
    """流式处理请求"""
    if not self.token_in_token_out:
        # 直接转发模式
        async for chunk in self._process_stream_direct(...):
            yield chunk
        return
    # Token-in-Token-out 模式
    # 完整处理流程...
```

### 2.3 InferClient（推理服务客户端）

**文件：** `traj_proxy/proxy_core/infer_client.py`

**职责：**
- 与推理服务通信
- 支持两种 API 格式

**核心方法：**
```python
# Token-in-Token-out 模式使用
async def send_completion(prompt, model, **kwargs)
async def send_completion_stream(prompt, model, **kwargs)

# 直接转发模式使用
async def send_chat_completion(messages, model, **kwargs)
async def send_chat_completion_stream(messages, model, **kwargs)
```

### 2.4 ProcessorManager（处理器管理器）

**文件：** `traj_proxy/proxy_core/processor_manager.py`

**职责：**
- 管理多个 Processor 实例
- 动态注册/删除模型
- 从数据库同步模型配置

**注册模型示例：**
```python
# 直接转发模式（无需 tokenizer_path）
await processor_manager.register_dynamic_processor(
    model_name="my-model",
    url="http://infer-service:8000",
    api_key="xxx",
    token_in_token_out=False
)

# Token-in-Token-out 模式（需要 tokenizer_path）
await processor_manager.register_dynamic_processor(
    model_name="my-model",
    url="http://infer-service:8000",
    api_key="xxx",
    tokenizer_path="/path/to/tokenizer",
    token_in_token_out=True
)
```

---

## 三、数据流对比

### 3.1 请求转换

| 阶段 | Token-in-Token-out 模式 | 直接转发模式 |
|------|------------------------|--------------|
| 输入 | OpenAI messages | OpenAI messages |
| 转换 | messages → prompt_text → token_ids | 无转换 |
| 发送 | `/v1/completions` + token_ids | `/v1/chat/completions` + messages |

### 3.2 响应处理

| 阶段 | Token-in-Token-out 模式 | 直接转发模式 |
|------|------------------------|--------------|
| 接收 | token_ids 或 text | OpenAI Response |
| 转换 | token_ids → text → OpenAI Response | 无转换 |
| 解析 | tool_calls, reasoning | 无解析 |

### 3.3 数据库存储

| 字段 | Token-in-Token-out 模式 | 直接转发模式 |
|------|------------------------|--------------|
| tokenizer_path | 必填 | 空 |
| prompt_text | 填充 | 空 |
| token_ids | 填充 | 空 |
| response | OpenAI Response | OpenAI Response |
| messages | 原始消息 | 原始消息 |

---

## 四、API 接口

### 4.1 注册模型

**POST /model/register**

```json
// 直接转发模式
{
    "model_name": "my-model",
    "url": "http://infer-service:8000",
    "api_key": "xxx",
    "token_in_token_out": false
}

// Token-in-Token-out 模式
{
    "model_name": "my-model",
    "url": "http://infer-service:8000",
    "api_key": "xxx",
    "tokenizer_path": "/path/to/tokenizer",
    "token_in_token_out": true,
    "tool_parser": "deepseek_v3",
    "reasoning_parser": "deepseek_r1"
}
```

### 4.2 发送请求

**POST /v1/chat/completions**

```json
{
    "model": "my-model",
    "messages": [
        {"role": "user", "content": "Hello"}
    ],
    "stream": false
}
```

**Header:** `x-session-id: run_id,sample_id,task_id`

---

## 五、配置决策指南

### 5.1 选择处理模式

```
推理服务是否支持 /v1/chat/completions？
├── 是 → 是否需要前缀匹配缓存？
│         ├── 是 → 使用 Token-in-Token-out 模式
│         └── 否 → 使用直接转发模式
└── 否 → 使用 Token-in-Token-out 模式
         （需要推理服务支持 /v1/completions）
```

### 5.2 参数配置表

| 模式 | tokenizer_path | tool_parser | reasoning_parser |
|------|----------------|-------------|------------------|
| 直接转发 | 不需要 | 不需要 | 不需要 |
| Token-in-Token-out | 必需 | 可选 | 可选 |

---

## 六、关键文件路径

| 组件 | 路径 |
|------|------|
| 路由层 | `traj_proxy/proxy_core/routes.py` |
| 非流式处理器 | `traj_proxy/proxy_core/processor.py` |
| 流式处理器 | `traj_proxy/proxy_core/streaming_processor.py` |
| 推理客户端 | `traj_proxy/proxy_core/infer_client.py` |
| 处理器管理器 | `traj_proxy/proxy_core/processor_manager.py` |
| Prompt 构建器 | `traj_proxy/proxy_core/prompt_builder.py` |
| Token 构建器 | `traj_proxy/proxy_core/token_builder.py` |
| 上下文数据 | `traj_proxy/proxy_core/context.py` |
| 数据模型 | `traj_proxy/store/models.py` |
| 模型仓库 | `traj_proxy/store/model_repository.py` |
| 请求仓库 | `traj_proxy/store/request_repository.py` |
