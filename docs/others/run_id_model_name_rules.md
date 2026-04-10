# Run ID 和 Model Name 组合定义规则

> **导航**: [文档中心](../README.md) | [API 参考](../develop/api_reference.md)

本文档定义了 TrajProxy 中 `run_id`、`session_id`、`model` 的语义规范和组合行为。

---

## 概述

TrajProxy 使用 `(run_id, model_name)` 作为模型的唯一标识键，支持多租户场景下的模型隔离。

### 统一语义定义

| 字段 | 语义 | 格式 | 来源 | 用途 |
|------|------|------|------|------|
| **run_id** | 租户标识 | 任意非逗号字符串 | 路径 > model参数 > DEFAULT | 模型路由隔离 |
| **request_id** | 请求标识 | UUID | 服务端生成 | 单次请求追踪 |
| **session_id** | 会话标识 | **任意字符串** | 路径/Header | 轨迹分组查询 |
| **model** | 模型名称 | 任意字符串 | 请求体 | 模型类型 |

### 核心概念

- **run_id**: 运行ID，用于区分同一模型名称的不同实例（租户隔离）
- **model_name**: 模型名称，如 `gpt-4`、`qwen3.5-2b`
- **session_id**: 会话ID，用于轨迹分组查询，**不定义分割格式**
- **DEFAULT**: 当 run_id 为空时的默认值

---

## 路径格式定义

### 支持的路径格式

```
/s/{session_id}/v1/chat/completions          # 格式1：仅 session_id
/s/{run_id},{session_id}/v1/chat/completions # 格式2：run_id + session_id
```

**解析规则**：
- 路径段包含逗号 → 逗号前为 `run_id`，逗号后为 `session_id`
- 路径段无逗号 → 整体为 `session_id`，`run_id` 从 model 参数提取或默认 DEFAULT

---

## 注册模型接口

**端点**: `POST /models/register`

| 场景 | run_id | model_name | 行为 |
|------|--------|------------|------|
| 一 | 空 | 非空 | run_id 赋值为 `DEFAULT`，存储键为 `(DEFAULT, model_name)` |
| 二 | 非空 | 非空 | 直接存储，键为 `(run_id, model_name)` |

**示例**:

```bash
# 场景一：run_id 为空，自动使用 DEFAULT
curl -X POST http://localhost:12300/models/register \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "gpt-4",
    "url": "https://api.openai.com/v1",
    "api_key": "sk-xxxxx"
  }'
# 结果：存储为 (DEFAULT, gpt-4)

# 场景二：指定 run_id
curl -X POST http://localhost:12300/models/register \
  -H "Content-Type: application/json" \
  -d '{
    "run_id": "app_001",
    "model_name": "gpt-4",
    "url": "https://api.openai.com/v1",
    "api_key": "sk-xxxxx"
  }'
# 结果：存储为 (app_001, gpt-4)
```

---

## 删除模型接口

**端点**: `DELETE /models?model_name=xxx&run_id=xxx`

| 场景 | run_id | model_name | 行为 |
|------|--------|------------|------|
| 一 | 空 | 非空 | run_id 赋值为 `DEFAULT`，删除键 `(DEFAULT, model_name)` |
| 二 | 非空 | 非空 | 直接删除键 `(run_id, model_name)` |

**示例**:

```bash
# 场景一：删除 DEFAULT 模型
curl -X DELETE "http://localhost:12300/models?model_name=gpt-4"
# 结果：删除 (DEFAULT, gpt-4)

# 场景二：删除指定 run_id 的模型
curl -X DELETE "http://localhost:12300/models?model_name=gpt-4&run_id=app_001"
# 结果：删除 (app_001, gpt-4)
```

---

## 推理对话接口

**端点**: `POST /v1/chat/completions`

### run_id 提取优先级

```
优先级1（最高）: 路径参数逗号前
       /s/run_001,session-abc/... → run_id = "run_001"

优先级2: model 参数逗号后
       model: "gpt-4,run_002" → run_id = "run_002"

优先级3（最低）: DEFAULT
       无路径、无 model run_id → run_id = "DEFAULT"
```

### session_id 提取优先级

```
优先级1（最高）: 路径参数
       /s/session-abc/... → session_id = "session-abc"
       /s/run_001,session-abc/... → session_id = "session-abc"

优先级2: x-sandbox-traj-id Header

优先级3: x-session-id Header
```

### 场景矩阵

| 场景 | 路径 | model 格式 | run_id 来源 | session_id 来源 |
|------|------|-----------|-------------|-----------------|
| 1.1 | `/v1/chat/completions` | `model_name` | DEFAULT | Header |
| 1.2 | `/s/uuid/v1/chat/completions` | `model_name` | DEFAULT | 路径 |
| 1.3 | `/s/uuid/v1/chat/completions` | `model_name,run_id` | model | 路径 |
| 2.1 | `/s/run_id,session/v1/chat/completions` | `model_name` | 路径 | 路径 |
| 2.2 | `/s/run_id,session/v1/chat/completions` | `model_name,other_run` | **路径优先** | 路径 |

**重要**: 路径中的 run_id 优先级最高，会覆盖 model 参数中的 run_id。

### 示例

```bash
# 场景 1.1：无路径，使用 DEFAULT run_id
curl -X POST http://localhost:12300/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "你好"}]
  }'
# run_id = DEFAULT, session_id = None

# 场景 1.2：路径传 session_id，使用 DEFAULT run_id
curl -X POST http://localhost:12300/s/uuid-123/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "你好"}]
  }'
# run_id = DEFAULT, session_id = uuid-123

# 场景 1.3：路径传 session_id，model 参数传 run_id
curl -X POST http://localhost:12300/s/uuid-123/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4,app_001",
    "messages": [{"role": "user", "content": "你好"}]
  }'
# run_id = app_001, session_id = uuid-123

# 场景 2.1：路径同时传 run_id 和 session_id
curl -X POST http://localhost:12300/s/app_001,uuid-123/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "你好"}]
  }'
# run_id = app_001, session_id = uuid-123

# 场景 2.2：路径和 model 都有 run_id，路径优先
curl -X POST http://localhost:12300/s/app_001,uuid-123/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4,app_002",
    "messages": [{"role": "user", "content": "你好"}]
  }'
# run_id = app_001 (路径优先), session_id = uuid-123

# 兼容旧版三段格式
curl -X POST http://localhost:12300/s/app_001,sample_001,task_001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "你好"}]
  }'
# run_id = app_001, session_id = "sample_001,task_001"
```

---

## Session ID 说明

### session_id 语义

- **作用**: 轨迹分组标识，用于查询同一会话的多轮对话
- **格式**: 任意字符串，**不定义分割格式**
- **来源**: 路径参数 > x-sandbox-traj-id Header > x-session-id Header

### session_id 传递方式

1. **路径传递**: `/s/{session_id}/v1/chat/completions`
2. **路径传递（带 run_id）**: `/s/{run_id},{session_id}/v1/chat/completions`
3. **请求头传递**: `x-session-id` 或 `x-sandbox-traj-id` 请求头

### session_id 与 run_id 的关系

- **run_id**: 模型实例标识，用于模型路由（租户隔离）
- **session_id**: 会话标识，用于轨迹分组查询
- **两者解耦**: session_id 不强制包含 run_id，可独立变化

---

## 模型隔离说明

TrajProxy 采用**严格隔离**策略：

- 不同 `run_id` 的模型完全隔离
- 不存在回退逻辑（如找不到指定 run_id 的模型，不会回退到 DEFAULT 模型）
- 每个模型必须精确匹配 `(run_id, model_name)` 才能被访问

---

## 代码实现

### 核心常量

```python
# traj_proxy/utils/validators.py
DEFAULT_RUN_ID = "DEFAULT"
```

### 路径解析函数

```python
# traj_proxy/serve/routes.py
def _parse_path_session(path_segment: str) -> tuple:
    """
    解析路径中的 session 参数

    路径格式：
    - "uuid-string" → (None, "uuid-string")  # 无 run_id
    - "run_001,uuid-string" → ("run_001", "uuid-string")  # 有 run_id

    Returns:
        (run_id, session_id): run_id 可能为 None
    """
    if ',' in path_segment:
        parts = path_segment.split(',', 1)
        return parts[0].strip(), parts[1].strip()
    return None, path_segment
```

### 模型解析函数

```python
# traj_proxy/serve/routes.py
def _parse_model_and_run_id(model: str, run_id_from_path: str = None) -> tuple:
    """
    解析 model 参数和 run_id

    优先级：
    1. 路径参数中的 run_id（最高优先级）
    2. model 参数中的 run_id（格式：model_name,run_id）
    3. DEFAULT（默认值）

    Returns:
        (model_name, run_id)
    """
    # 优先级1: 路径中的 run_id
    if run_id_from_path:
        return model.strip(), run_id_from_path

    # 优先级2: model 参数中的 run_id
    if ',' in model:
        parts = model.split(',', 1)
        return parts[0].strip(), parts[1].strip()

    # 优先级3: DEFAULT
    return model.strip(), DEFAULT_RUN_ID
```

---

## 变更历史

| 日期 | 版本 | 变更说明 |
|------|------|----------|
| 2026-04-10 | v3.0 | 重新定义 run_id、session_id、model 语义，支持路径参数同时传递 run_id 和 session_id |
| 2026-04-02 | v2.0 | 重构 run_id 和 model_name 组合定义，引入 DEFAULT 常量，支持 model,run_id 格式 |
