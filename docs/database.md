# 数据库设计

TrajProxy 使用 PostgreSQL 存储模型配置和请求轨迹记录。

---

## 表结构概览

| 表名 | 说明 |
|------|------|
| `model_registry` | 模型配置注册表 |
| `request_records` | 请求轨迹记录 |

---

## model_registry 表

存储动态注册的模型配置。

### 表结构

```sql
CREATE TABLE model_registry (
    id SERIAL PRIMARY KEY,
    run_id TEXT NOT NULL DEFAULT '',       -- 运行ID，空字符串表示全局模型
    model_name TEXT NOT NULL,              -- 模型名称
    url TEXT NOT NULL,                     -- 推理服务 URL
    api_key TEXT NOT NULL,                 -- API 密钥
    tokenizer_path TEXT,                   -- Tokenizer 路径（可选，直接转发模式不需要）
    token_in_token_out BOOLEAN DEFAULT FALSE,  -- 是否启用 Token 模式
    tool_parser TEXT NOT NULL DEFAULT '',  -- 工具解析器名称
    reasoning_parser TEXT NOT NULL DEFAULT '', -- 推理解析器名称
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT unique_run_model UNIQUE (run_id, model_name)
);
```

### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id` | SERIAL | 是 | 主键 |
| `run_id` | TEXT | 是 | 运行ID，空字符串表示全局模型，用于多租户场景 |
| `model_name` | TEXT | 是 | 模型名称，请求时使用 |
| `url` | TEXT | 是 | 推理服务 URL |
| `api_key` | TEXT | 是 | API 密钥 |
| `tokenizer_path` | TEXT | 否 | Tokenizer 路径（本地路径或 HuggingFace 名称），直接转发模式不需要 |
| `token_in_token_out` | BOOLEAN | 否 | 是否启用 Token-in-Token-out 模式，默认 false |
| `tool_parser` | TEXT | 否 | 工具解析器名称（如 `deepseek_v3`） |
| `reasoning_parser` | TEXT | 否 | 推理解析器名称（如 `deepseek_r1`） |
| `updated_at` | TIMESTAMP | 否 | 最后更新时间 |

### 索引

| 索引名 | 字段 | 说明 |
|--------|------|------|
| `model_registry_run_id_idx` | `run_id` | 按运行ID查询 |
| `model_registry_model_name_idx` | `model_name` | 按模型名查询 |
| `model_registry_updated_at_idx` | `updated_at DESC` | 按更新时间查询 |

### 唯一约束

`(run_id, model_name)` 组合唯一，允许不同 run_id 下存在同名模型。

---

## ModelConfig 数据类

**文件：** `traj_proxy/store/models.py`

```python
@dataclass
class ModelConfig:
    """模型配置数据类"""
    url: str
    api_key: str
    tokenizer_path: Optional[str] = None  # 可选，直接转发模式下不需要
    run_id: str = ""                       # 运行ID，空字符串表示全局模型
    model_name: str = ""
    token_in_token_out: bool = False
    tool_parser: str = ""                  # 工具解析器名称
    reasoning_parser: str = ""             # 推理解析器名称
    updated_at: Optional[datetime] = None
```

---

## request_records 表

存储所有请求的完整轨迹记录。

### 表结构

```sql
CREATE TABLE request_records (
    id SERIAL PRIMARY KEY,
    unique_id TEXT NOT NULL UNIQUE,        -- 唯一标识
    request_id TEXT NOT NULL,              -- 请求 ID
    session_id TEXT NOT NULL,              -- 会话 ID
    model TEXT NOT NULL,                   -- 模型名称
    tokenizer_path TEXT,                   -- Tokenizer 路径（可选，直接转发模式为空）
    messages JSONB NOT NULL,               -- 原始消息列表

    -- 阶段1: OpenAI Chat 格式
    raw_request JSONB,                     -- 完整 OpenAI 请求
    raw_response JSONB,                    -- 完整 OpenAI 响应

    -- 阶段2: 文本推理格式（Token模式）
    text_request JSONB,                    -- 文本推理请求
    text_response JSONB,                   -- 文本推理响应

    -- 阶段3: Token 推理格式（Token模式）
    prompt_text TEXT,                      -- 文本 prompt
    token_ids INTEGER[],                   -- Token ID 列表
    token_request JSONB,                   -- Token 推理请求
    token_response JSONB,                  -- Token 推理响应

    -- 输出数据
    response_text TEXT,                    -- 响应文本
    response_ids INTEGER[],                -- 响应 Token ID

    -- 完整对话（用于前缀匹配缓存）
    full_conversation_text TEXT,           -- 完整对话文本（含历史）
    full_conversation_token_ids INTEGER[], -- 完整对话 Token ID

    -- 元数据
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    processing_duration_ms FLOAT,          -- 处理时长（毫秒）

    -- 统计信息
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    cache_hit_tokens INTEGER DEFAULT 0,    -- 缓存命中 Token 数

    -- 错误信息
    error TEXT,
    error_traceback TEXT,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### 字段分组说明

#### 标识字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `unique_id` | TEXT | 全局唯一标识，格式: `{session_id},{request_id}` |
| `request_id` | TEXT | 请求 ID |
| `session_id` | TEXT | 会话 ID，格式: `run_id,sample_id,task_id` |
| `model` | TEXT | 使用的模型名称 |

#### 请求阶段字段

| 阶段 | 字段 | 说明 |
|------|------|------|
| 阶段1 | `raw_request`, `raw_response` | OpenAI 格式原始请求/响应（所有模式都有） |
| 阶段2 | `text_request`, `text_response` | 文本推理格式（仅 Token 模式） |
| 阶段3 | `token_request`, `token_response` | Token 推理格式（仅 Token 模式） |

**模式差异：**
- **直接转发模式**：只填充 `raw_request` 和 `raw_response`
- **Token 模式**：填充所有阶段字段

#### Token 相关字段

| 字段 | 说明 |
|------|------|
| `prompt_text` | 文本格式的 prompt（仅 Token 模式） |
| `token_ids` | 输入 Token ID 数组（仅 Token 模式） |
| `response_text` | 响应文本 |
| `response_ids` | 输出 Token ID 数组（仅 Token 模式） |

#### 缓存相关字段

| 字段 | 说明 |
|------|------|
| `full_conversation_text` | 完整对话文本（用于前缀匹配缓存） |
| `full_conversation_token_ids` | 完整对话 Token ID（用于前缀匹配缓存） |
| `cache_hit_tokens` | 缓存命中的 Token 数 |

#### 统计字段

| 字段 | 说明 |
|------|------|
| `prompt_tokens` | 输入 Token 数 |
| `completion_tokens` | 输出 Token 数 |
| `total_tokens` | 总 Token 数 |
| `processing_duration_ms` | 请求处理时长（毫秒） |

### 索引

| 索引名 | 字段 | 说明 |
|--------|------|------|
| `request_records_session_id_idx` | `session_id` | 按会话查询（轨迹查询） |
| `request_records_request_id_idx` | `request_id` | 按请求ID查询 |
| `request_records_unique_id_idx` | `unique_id` | 唯一索引 |
| `request_records_start_time_idx` | `start_time DESC` | 按时间倒序查询 |

---

## RequestRecord 数据类

**文件：** `traj_proxy/store/models.py`

```python
@dataclass
class RequestRecord:
    """请求轨迹记录数据类"""
    unique_id: str
    request_id: str
    session_id: str
    model: str
    messages: List[Any]
    tokenizer_path: Optional[str] = None  # 可选，直接转发模式下不需要

    # 阶段1: OpenAI Chat 格式
    raw_request: Optional[Any] = None      # 完整 OpenAI 请求
    raw_response: Optional[Any] = None     # 完整 OpenAI 响应

    # 阶段2: 文本推理格式（Token模式）
    text_request: Optional[Any] = None     # 文本推理请求
    text_response: Optional[Any] = None    # 文本推理响应

    # 阶段3: Token 推理格式（Token模式）
    prompt_text: Optional[str] = None      # 文本 prompt
    token_ids: Optional[List[int]] = None  # Token ID 列表
    token_request: Optional[Any] = None    # Token 推理请求
    token_response: Optional[Any] = None   # Token 推理响应

    # 输出数据
    response_text: Optional[str] = None
    response_ids: Optional[List[int]] = None

    # 完整对话（用于前缀匹配缓存）
    full_conversation_text: Optional[str] = None
    full_conversation_token_ids: Optional[List[int]] = None

    # 元数据
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    processing_duration_ms: Optional[float] = None

    # 统计信息
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cache_hit_tokens: int = 0

    # 错误信息
    error: Optional[str] = None
    error_traceback: Optional[str] = None

    # 创建时间
    created_at: Optional[datetime] = None
```

---

## ProcessContext 数据类

**文件：** `traj_proxy/proxy_core/context.py`

ProcessContext 贯穿整个请求处理流程，包含输入数据、中间处理数据和输出结果。

```python
@dataclass
class ProcessContext:
    """处理上下文，贯穿整个请求处理流程"""
    
    # ========== 基础标识 ==========
    request_id: str
    model: str
    session_id: Optional[str] = None  # 格式: run_id,sample_id,task_id
    unique_id: Optional[str] = None   # 格式: {session_id},{request_id}

    # ========== 阶段1: OpenAI Chat 格式 ==========
    messages: List[Dict[str, Any]] = field(default_factory=list)
    request_params: Dict[str, Any] = field(default_factory=dict)
    raw_request: Optional[Dict[str, Any]] = None   # 完整 OpenAI 请求
    raw_response: Optional[Dict[str, Any]] = None  # 完整 OpenAI 响应

    # ========== 阶段2: 文本推理格式（Token模式）==========
    text_request: Optional[Dict[str, Any]] = None   # 文本推理请求
    text_response: Optional[Dict[str, Any]] = None  # 文本推理响应

    # ========== 阶段3: Token 推理格式（Token模式）==========
    prompt_text: Optional[str] = None               # 文本 prompt
    token_ids: Optional[List[int]] = None           # Token ID 列表
    cached_token_ids: Optional[List[int]] = None    # 缓存命中的 token
    uncached_text: Optional[str] = None             # 未缓存文本
    uncached_token_ids: Optional[List[int]] = None  # 未缓存 token ids
    token_request: Optional[Dict[str, Any]] = None  # Token 推理请求
    token_response: Optional[Dict[str, Any]] = None # Token 推理响应

    # ========== 输出数据 ==========
    response_text: Optional[str] = None             # 响应文本
    response_ids: Optional[List[int]] = None        # 响应 token ids

    # ========== 完整对话（用于前缀匹配）==========
    full_conversation_text: Optional[str] = None
    full_conversation_token_ids: Optional[List[int]] = None

    # ========== 元数据 ==========
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    processing_duration_ms: Optional[float] = None

    # ========== 统计信息 ==========
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cache_hit_tokens: Optional[int] = None      # 缓存命中的 token 数量

    # ========== 错误信息 ==========
    error: Optional[str] = None
    error_traceback: Optional[str] = None

    # ========== 流式处理状态 ==========
    is_stream: bool = False                    # 是否流式请求
    stream_buffer_text: str = ""               # 流式累积的响应文本
    stream_buffer_ids: List[int] = field(default_factory=list)  # 流式累积的 token ids
    stream_chunk_count: int = 0                # 已发送的 chunk 数量
    stream_finished: bool = False              # 流式是否结束

    # 流式累积字段（用于构建完整响应）
    stream_role: Optional[str] = None          # delta.role
    stream_reasoning: str = ""                 # delta.reasoning
    stream_tool_calls: Optional[List[Dict[str, Any]]] = None
    stream_function_call: Optional[Dict[str, Any]] = None
    stream_finish_reason: Optional[str] = None
    stream_logprobs: Optional[Dict[str, Any]] = None
    stream_stop_reason: Optional[Any] = None   # vLLM 扩展字段
    stream_token_ids: Optional[List[int]] = None
```

### 数据流向说明

**直接转发模式：**
```
raw_request → 推理服务 → raw_response
```

**Token 模式：**
```
raw_request → text_request → token_request → 推理服务 → token_response → text_response → raw_response
```

---

## 模型同步机制

TrajProxy 使用 **LISTEN/NOTIFY** 实现跨 Worker 的模型配置实时同步。

### 架构

```
┌─────────────────┐     NOTIFY      ┌─────────────────┐
│   Worker A      │ ──────────────► │   PostgreSQL    │
│ (注册模型)       │                 │  (NOTIFY 通道)  │
└─────────────────┘                 └────────┬────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │ LISTEN                 │ LISTEN                 │ LISTEN
                    ▼                        ▼                        ▼
            ┌───────────────┐        ┌───────────────┐        ┌───────────────┐
            │   Worker A    │        │   Worker B    │        │   Worker C    │
            │ (收到通知)     │        │ (收到通知)     │        │ (收到通知)     │
            └───────────────┘        └───────────────┘        └───────────────┘
```

### 实现细节

#### 1. NOTIFY 通道

通道名称: `model_registry_changes`

#### 2. Payload 格式

```json
{
    "action": "register",      // 或 "unregister"
    "run_id": "",
    "model_name": "gpt-4",
    "timestamp": 1712345678.9
}
```

#### 3. NotificationListener

位置: `traj_proxy/store/notification_listener.py`

- 维护一个专用的数据库连接（LISTEN 需要独占连接）
- 自动重连机制（指数退避）
- 连接失败时无限重试

#### 4. 兜底同步

为防止 NOTIFY 丢失，定期执行全量同步：

- 默认间隔: 300 秒（5 分钟）
- 配置项: `processor_manager.sync_fallback_interval`

### 同步流程

```
Worker A 注册模型
    │
    ├─► ModelRepository.register()
    │       │
    │       ├─► INSERT INTO model_registry
    │       │
    │       └─► NOTIFY model_registry_changes
    │
    └─► 本地 ProcessorManager 立即更新

PostgreSQL 广播 NOTIFY
    │
    ├─► Worker B NotificationListener 收到通知
    │       │
    │       └─► ProcessorManager 增量更新
    │
    └─► Worker C NotificationListener 收到通知
            │
            └─► ProcessorManager 增量更新
```

### 相关配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `sync_fallback_interval` | 300 | 兜底全量同步间隔（秒） |
| `sync_max_retries` | 3 | 同步失败最大重试次数 |
| `sync_retry_delay` | 5 | 重试初始延迟（秒） |
| `listen_reconnect_delay` | 5 | LISTEN 重连初始延迟（秒） |
| `listen_max_reconnect_delay` | 60 | LISTEN 重连最大延迟（秒） |

---

## 数据库初始化

使用 `scripts/init_db.py` 初始化数据库：

```bash
# 从环境变量读取 DATABASE_URL
python scripts/init_db.py

# 或指定连接 URL
python scripts/init_db.py --db-url postgresql://user:pass@host:port/dbname
```

脚本功能：
1. 创建数据库（如果不存在）
2. 创建 `model_registry` 表和索引
3. 创建 `request_records` 表和索引
4. 幂等执行，可重复运行

---

## 查询示例

### 查询特定会话的轨迹

```sql
SELECT unique_id, model, prompt_text, response_text, start_time
FROM request_records
WHERE session_id = 'app_001,sample_001,task_001'
ORDER BY start_time DESC
LIMIT 100;
```

### 查询最近的请求

```sql
SELECT session_id, model, prompt_tokens, completion_tokens, processing_duration_ms
FROM request_records
ORDER BY start_time DESC
LIMIT 20;
```

### 查询缓存命中率

```sql
SELECT
    model,
    COUNT(*) as total_requests,
    SUM(cache_hit_tokens) as total_cache_hits,
    SUM(prompt_tokens) as total_prompt_tokens,
    ROUND(SUM(cache_hit_tokens)::float / NULLIF(SUM(prompt_tokens), 0) * 100, 2) as cache_hit_rate
FROM request_records
GROUP BY model
ORDER BY total_requests DESC;
```

### 查询已注册模型

```sql
SELECT model_name, url, token_in_token_out, tool_parser, reasoning_parser, updated_at
FROM model_registry
ORDER BY model_name;
```
