# TrajProxy - LLM 代理服务

TrajProxy 是一个强大的 LLM 代理系统，提供统一的 OpenAI 兼容 API，支持 Token-in-Token-out 模式、多 Worker 并发处理、会话管理和监控等功能。

## 项目概述

TrajProxy采用基于Ray的分布式Worker架构，支持高并发的LLM推理请求处理和对话历史管理。

- **ProxyWorker**: 统一的代理 Worker，集成 LLM 推理请求处理和对话历史查询功能
  - 支持 Token-in-Token-out 模式和前缀匹配缓存
  - 提供轨迹记录查询服务

## 部署视图

```mermaid
graph TB
    subgraph "客户端"
        Client[客户端应用]
    end

    subgraph "Docker容器"
        direction TB
        subgraph "nginx容器"
            Nginx[Nginx<br/>统一入口<br/>端口:12345]
        end

        subgraph "litellm容器"
            LiteLLM[LiteLLM<br/>API网关<br/>端口:4000]
        end

        subgraph "traj_proxy容器"
            WM[WorkerManager<br/>主进程]
            subgraph "Ray Workers"
                PW0[ProxyWorker-0<br/>12300]
                PW1[ProxyWorker-1<br/>12301]
            end
        end

        subgraph "prometheus容器"
            Prom[Prometheus<br/>监控<br/>端口:9090]
        end

        subgraph "db容器"
            DB[(PostgreSQL<br/>端口:5432)]
        end
    end

    subgraph "外部服务"
        Infer[Infer服务<br/>LLM推理]
    end

    Client -->|推理请求| Nginx
    Client -->|模型管理/轨迹查询| Nginx

    Nginx -->|推理请求| LiteLLM
    Nginx -->|模型管理/轨迹查询| PW0

    LiteLLM -->|转发请求| PW0
    LiteLLM -->|转发请求| PW1

    PW0 -->|推理请求| Infer
    PW1 -->|推理请求| Infer
    PW0 -->|存储/查询轨迹| DB
    PW1 -->|存储/查询轨迹| DB

    PW0 -->|指标| Prom
    PW1 -->|指标| Prom

    WM -.->|管理| PW0
    WM -.->|管理| PW1

    style Client fill:#e1f5ff
    style Nginx fill:#fff9c4
    style LiteLLM fill:#c8e6c9
    style WM fill:#fff4e1
    style PW0 fill:#ffcc80
    style PW1 fill:#ffcc80
    style DB fill:#d1ecf1
    style Infer fill:#f8d7da
    style Prom fill:#e8f5e9
```

## 请求处理流程

```mermaid
flowchart LR
    Start[客户端请求] --> Nginx[Nginx<br/>端口:12345]

    Nginx --> Route{路由判断}

    Route -->|推理请求| LiteLLM[LiteLLM<br/>端口:4000]
    Route -->|模型管理| PW[ProxyWorker]
    Route -->|轨迹查询| PW
    Route -->|健康检查| PW

    LiteLLM -->|路由| PW

    subgraph "ProxyWorker处理流程"
        direction TB
        PW --> PRoute{内部路由}

        PRoute -->|聊天请求| PC[ProcessorManager<br/>LLM推理]
        PRoute -->|轨迹查询| TP[TranscriptProvider<br/>历史查询]

        subgraph "推理流程"
            PC --> PB[构建Prompt]
            PB --> MODE{处理模式}
            MODE -->|Text模式| PC_Text[发送文本到Infer]
            MODE -->|Token模式| PC_Token[前缀匹配编码<br/>发送Token到Infer]
            PC_Text --> Infer[Infer服务]
            PC_Token --> Infer
            Infer --> Response[接收响应]
            Response --> Build[构建OpenAI响应]
            Build --> Save[存储轨迹到DB]
        end

        TP --> Query[查询历史记录]
        Query --> DB[(PostgreSQL)]
        DB --> TP
    end

    Save --> Return[返回响应]
    TP --> Return

    Return --> End[客户端]

    style Start fill:#e1f5ff
    style Nginx fill:#fff9c4
    style LiteLLM fill:#c8e6c9
    style PW fill:#ffcc80
    style PC fill:#ffe0b2
    style TP fill:#ffe0b2
    style Infer fill:#f8d7da
    style DB fill:#d1ecf1
    style End fill:#e1f5ff
```

## 架构组件

| 组件 | 端口 | 说明 |
|------|------|------|
| Nginx | 12345 | 统一入口网关，路由推理请求和模型管理请求 |
| LiteLLM | 4000 | API 网关，提供统一的 OpenAI 兼容接口 |
| ProxyWorker | 12300-12320 | 统一的代理服务，集成 LLM 推理和轨迹查询功能 |
| PostgreSQL | 5432 | 数据库存储 |
| Prometheus | 9090 | 监控和指标收集 |

---

## 快速开始

### 前置要求

#### 通用要求
- LLM 推理服务（如 Ollama、vLLM 等）运行在 `http://localhost:1234`
- PostgreSQL 数据库（可本地运行或容器运行）

#### 本地开发模式
- Python 3.11+
- pip 依赖安装：`pip install -r requirements.txt`

#### Docker 容器模式
- Docker >= 20.10
- Docker Compose >= 2.0
- 至少 8GB 可用内存

### 1. 配置环境

编辑 `configs/config.yaml`，配置 LLM 推理服务地址和数据库连接：

```yaml
# ProxyWorker 配置
proxy_workers:
  count: 2
  base_port: 12300
  models:
    - model_name: qwen3.5-2b
      url: http://localhost:1234  # LLM 推理服务地址
      api_key: sk-1234
      tokenizer_path: Qwen/Qwen3.5-2B  # HuggingFace 模型名称或本地路径
      token_in_token_out: true  # 启用 Token-in-Token-out 模式，支持前缀匹配缓存

# Ray配置
ray:
  num_cpus: 4
  working_dir: "/app"  # 容器内路径，可通过环境变量 RAY_WORKING_DIR 覆盖
  pythonpath: "/app"   # 容器内 PYTHONPATH，可通过环境变量 RAY_PYTHONPATH 覆盖

# 数据库配置
database:
  url: "postgresql://llmproxy:dbpassword9090@localhost:5432/traj_proxy"
```

### 2. 启动服务

提供两种部署方式，根据需求选择：

#### 方式一：本地开发启动

适用于开发调试场景，直接在本地运行 Python 进程，连接外部数据库。

```bash
# 一键启动
./scripts/start_local.sh
```

**说明**：
- `start_local.sh` 会自动设置本地环境变量 `RAY_WORKING_DIR="."` 和 `RAY_PYTHONPATH="."`
- 依赖本地已安装的 PostgreSQL 数据库
- TrajProxy 运行在宿主机上，端口直接监听

#### 方式二：Docker 容器化部署

适用于生产环境，使用 Docker Compose 启动完整的容器组（包括 nginx、litellm、postgresdb、traj_proxy、prometheus）。

```bash
# 一键启动所有容器
./scripts/start_docker.sh
```

**说明**：
- `start_docker.sh` 会自动执行 `docker-compose up -d`
- 启动完整的容器组：nginx（统一入口）、litellm（网关）、db（数据库）、traj_proxy（代理）、prometheus（监控）
- 数据库 URL 配置为 `postgresql://llmproxy:dbpassword9090@db:5432/traj_proxy`（容器内网络）

### 3. 验证服务

```bash
# 检查 Nginx 健康状态（通过统一入口）
curl http://localhost:12345/health

# 检查 LiteLLM 健康状态
curl http://localhost:4000/health/liveliness

# 检查 TrajProxy 健康状态
curl http://localhost:12300/health

# 检查可用模型（通过 Nginx）
curl http://localhost:12345/models
```

---

## 部署教程

### 方式一：本地开发部署

适用于开发调试场景，直接在本地运行。

#### 配置说明

编辑 `configs/config.yaml`：

```yaml
proxy_workers:
  count: 2
  base_port: 12300
  models:
    - model_name: qwen3.5-2b
      url: http://localhost:1234  # 本地运行的推理服务
      api_key: sk-1234
      tokenizer_path: Qwen/Qwen3.5-2B  # HuggingFace 模型名称或本地路径
      token_in_token_out: true  # 启用前缀匹配缓存

ray:
  num_cpus: 4
  working_dir: "."      # 本地路径
  pythonpath: "."       # 本地 PYTHONPATH

# 数据库配置
database:
  url: "postgresql://llmproxy:dbpassword9090@localhost:5432/traj_proxy"
```

**关键点**：
- `url: http://localhost:1234` - 连接本地运行的 LLM 推理服务
- `database.url: ...localhost...` - 连接本地运行的 PostgreSQL
- `working_dir: "."` 和 `pythonpath: "."` - 使用当前目录

#### 启动步骤

```bash
# 确保本地数据库已运行
# docker run -d --name litellm_db -e POSTGRES_DB=litellm -e POSTGRES_USER=llmproxy -e POSTGRES_PASSWORD=dbpassword9090 -p 5432:5432 postgres:16

# 一键启动
./scripts/start_local.sh
```

#### 常用命令

```bash
# 停止服务
Ctrl+C  # 或者 kill 进程

# 查看日志
# 日志会直接输出到终端
```

---

### 方式二：Docker 容器化部署

适用于生产环境，使用 Docker Compose 启动完整的容器组。

#### 配置说明

编辑 `configs/config.yaml`：

```yaml
proxy_workers:
  count: 2
  base_port: 12300
  models:
    - model_name: qwen3.5-2b
      url: http://host.docker.internal:1234  # 宿主机推理服务
      api_key: sk-1234
      tokenizer_path: Qwen/Qwen3.5-2B  # HuggingFace 模型名称或本地路径
      token_in_token_out: true  # 启用前缀匹配缓存

ray:
  num_cpus: 4
  working_dir: "/app"  # 容器内路径
  pythonpath: "/app"   # 容器内 PYTHONPATH

# 数据库配置
database:
  url: "postgresql://llmproxy:dbpassword9090@db:5432/traj_proxy"
```

**关键点**：
- `url: http://host.docker.internal:1234` - 从容器访问宿主机服务
- `database.url: ...db:5432...` - 连接容器内的数据库服务
- `working_dir: "/app/traj_proxy"` - 容器内工作目录

#### 启动步骤

```bash
# 一键启动所有容器
./scripts/start_docker.sh

# 或直接使用 docker-compose
cd dockers && docker-compose up -d --build
```

#### 容器组说明

启动后包含以下容器：

| 容器名 | 端口 | 说明 |
|---------|-------|------|
| nginx | 12345 | 统一入口网关 |
| litellm | 4000 | API 网关 |
| db | 5432 | PostgreSQL 数据库 |
| traj_proxy | 12300-12320 | ProxyWorkers（统一服务） |
| prometheus | 9090 | 监控服务 |

#### 常用命令

```bash
# 查看服务状态
cd dockers && docker-compose ps

# 查看日志
cd dockers && docker-compose logs -f

# 停止服务
cd dockers && docker-compose down

# 重启服务
cd dockers && docker-compose restart

# 进入容器
cd dockers && docker-compose exec traj_proxy /bin/bash

# 进入数据库
cd dockers && docker-compose exec db psql -U llmproxy -d litellm
```

---

### 生产环境优化

#### 资源配置

修改 `docker-compose.yml` 添加资源限制：

```yaml
traj_proxy:
  deploy:
    resources:
      limits:
        cpus: '4'
        memory: 8G
      reservations:
        cpus: '2'
        memory: 4G
```

#### Worker 数量调优

根据服务器配置调整 `config.yaml`：

```yaml
proxy_workers:
  count: 4  # 根据 CPU 核心数调整

ray:
  num_cpus: 8  # 实际 CPU 核心数
```

### 配置说明

#### config.yaml

Worker 配置文件位于 `configs/config.yaml`，主要配置项：

```yaml
# ProxyWorker 配置
proxy_workers:
  count: 2                    # Worker数量
  base_port: 12300            # 起始端口（12300, 12301）
  models:                     # 预置模型配置
    - model_name: qwen3.5-2b
      url: http://host.docker.internal:8000  # LLM 推理服务地址
      api_key: sk-1234
      tokenizer_path: Qwen/Qwen3.5-2B  # HuggingFace 模型名称或本地路径
      token_in_token_out: true  # 启用 Token-in-Token-out 模式，支持前缀匹配缓存

# Ray 配置
ray:
  num_cpus: 4                 # CPU核心数
  working_dir: "/app"         # 默认容器内路径，可通过环境变量 RAY_WORKING_DIR 覆盖
  pythonpath: "/app"          # 默认容器内 PYTHONPATH，可通过环境变量 RAY_PYTHONPATH 覆盖

# 数据库配置
database:
  url: "postgresql://llmproxy:dbpassword9090@db:5432/traj_proxy"
  pool:
    min_size: 2               # 最小连接数
    max_size: 20              # 最大连接数
    timeout: 30               # 连接超时时间（秒）
```

**关键说明**：
- `host.docker.internal` 用于从 Docker 容器访问宿主机服务
- 本地部署时，使用 `url: http://localhost:1234` 和 `database.url: ...localhost...`
- 容器部署时，使用 `url: http://host.docker.internal:1234` 和 `database.url: ...db:5432...`
- `working_dir` 和 `pythonpath` 支持通过环境变量 `RAY_WORKING_DIR` 和 `RAY_PYTHONPATH` 覆盖

#### litellm.yaml

LiteLLM 配置文件，定义模型路由规则：

```yaml
model_list:
  # 通用匹配：所有模型请求都路由到 traj_proxy
  - model_name: "*"
    litellm_params:
      model: "openai/*"
      api_base: http://traj_proxy:12300/v1
      api_key: "sk-1234"
  - model_name: "*"
    litellm_params:
      model: "openai/*"
      api_base: http://traj_proxy:12301/v1
      api_key: "sk-1234"

litellm_settings:
  drop_params: true
  disable_request_checking: true
  use_chat_completions_url_for_anthropic_messages: true

general_settings:
  master_key: "sk-1234"
  forward_client_headers_to_llm_api: true
```

#### nginx.conf

Nginx 配置文件，定义请求路由规则：

```nginx
# 推理请求（/v1/chat/completions, /v1/messages）转发到 litellm
# 带 session_id 的路径格式: /s/{session_id}/v1/chat/completions
# 模型管理（/models）、轨迹查询（/trajectory）、健康检查（/health）转发到 traj_proxy
```

---

## 使用案例

### 案例 1：通过 Nginx 统一入口调用（推荐）

使用 OpenAI 兼容的聊天接口：

```bash
curl --location 'http://localhost:12345/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  -H "Authorization: Bearer sk-1234" \
  -H "x-session-id: app_001,sample_001,task_001" \
  --data '{
    "model": "qwen3.5-2b",
    "messages": [
      {
        "role": "user",
        "content": "你好，请介绍一下自己"
      }
    ]
  }'
```

### 案例 2：Anthropic Messages API

通过 Nginx 兼容 Anthropic 的消息接口：

```bash
curl --location 'http://localhost:12345/v1/messages' \
  --header 'x-api-key: sk-1234' \
  --header 'anthropic-version: 2023-06-01' \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "qwen3.5-2b",
    "max_tokens": 1024,
    "messages": [
      {
        "role": "user",
        "content": "what llm are you"
      }
    ]
  }'
```

### 案例 3：带 session_id 的路径格式

使用路径参数传递 session_id（格式：`{run_id},{sample_id},{task_id}`）：

```bash
curl --location 'http://localhost:12345/s/app_001,sample_001,task_001/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "qwen3.5-2b",
    "messages": [
      {
        "role": "user",
        "content": "what llm are you"
      }
    ]
  }'
```

### 案例 4：Python 客户端示例

```python
import openai

# 配置客户端（通过 Nginx 统一入口）
client = openai.OpenAI(
    base_url="http://localhost:12345/v1",
    api_key="sk-1234"
)

# 发送请求
response = client.chat.completions.create(
    model="qwen3.5-2b",
    messages=[
        {"role": "user", "content": "你好！"}
    ]
)

print(response.choices[0].message.content)
```

### 案例 5：查询对话历史

```bash
curl "http://localhost:12345/trajectory?session_id=app_001,sample_001,task_001"
```

---

## API端点

### Nginx 统一入口（端口 12345）
- `POST /v1/chat/completions` - OpenAI 兼容聊天补全
- `POST /v1/messages` - Anthropic 兼容消息接口
- `POST /s/{session_id}/v1/chat/completions` - 带 session_id 的聊天补全
- `GET /models` - 列出可用模型
- `GET /trajectory` - 查询对话轨迹记录
- `GET /health` - 健康检查

### LiteLLM 网关（端口 4000）
- `POST /v1/chat/completions` - OpenAI 兼容聊天补全
- `POST /v1/messages` - Anthropic 兼容消息接口
- `GET /v1/models` - 列出可用模型
- `GET /health/liveliness` - 健康检查

### TrajProxy Core（端口 12300-12320）
- `POST /v1/chat/completions` - 聊天补全
- `POST /s/{session_id}/v1/chat/completions` - 带session_id的聊天补全
- `GET /models` - 列出已注册模型
- `POST /models/register` - 注册新模型
- `DELETE /models/{model_name}` - 删除模型
- `GET /trajectory` - 查询对话轨迹记录
- `GET /health` - 健康检查

---

## 监控与日志

### Prometheus 监控（仅 Docker 容器模式）

访问 Prometheus UI：
```
http://localhost:9090
```

### 查看请求记录

连接到 PostgreSQL 数据库：

**本地开发模式：**
```bash
psql -h localhost -U llmproxy -d litellm

# 查询请求记录
SELECT * FROM request_records ORDER BY start_time DESC LIMIT 10;

# 查询特定会话的记录
SELECT * FROM request_records WHERE session_id = 'app_001,sample_001,task_001';
```

**Docker 容器模式：**
```bash
# 进入数据库
cd dockers && docker-compose exec db psql -U llmproxy -d litellm

# 查询请求记录
SELECT * FROM request_records ORDER BY start_time DESC LIMIT 10;

# 查询特定会话的记录
SELECT * FROM request_records WHERE session_id = 'app_001,sample_001,task_001';
```

### 查看日志

**本地开发模式：**
```bash
# 日志直接输出到终端，通过重定向保存
./scripts/start_local.sh 2>&1 | tee logs/traj_proxy.log
```

**Docker 容器模式：**
```bash
# 查看所有服务日志
cd dockers && docker-compose logs -f

# 查看特定服务日志
cd dockers && docker-compose logs -f traj_proxy
cd dockers && docker-compose logs -f litellm
cd dockers && docker-compose logs -f nginx
```

---

## 常用命令

### 本地开发模式

```bash
# 启动服务
./scripts/start_local.sh

# 停止服务
Ctrl+C
```

### Docker 容器模式

```bash
# 启动所有容器
./scripts/start_docker.sh

# 或直接使用 docker-compose
cd dockers && docker-compose up -d

# 停止所有容器
cd dockers && docker-compose down

# 停止所有容器并清理数据（谨慎使用）
cd dockers && docker-compose down -v

# 重启服务
cd dockers && docker-compose restart

# 查看服务状态
cd dockers && docker-compose ps

# 查看所有服务日志
cd dockers && docker-compose logs -f

# 查看特定服务日志
cd dockers && docker-compose logs -f traj_proxy
cd dockers && docker-compose logs -f litellm
cd dockers && docker-compose logs -f nginx
cd dockers && docker-compose logs -f db

# 进入容器
cd dockers && docker-compose exec traj_proxy /bin/bash

# 进入数据库
cd dockers && docker-compose exec db psql -U llmproxy -d litellm
```

---

## 故障排查

### 本地开发模式

#### 服务无法启动

```bash
# 检查 Python 环境
python --version  # 需要 Python 3.11+

# 检查端口占用
lsof -i :12300
lsof -i :12310

# 检查依赖安装
pip list | grep -E "fastapi|uvicorn|ray|psycopg"
```

#### 数据库连接失败

```bash
# 检查 PostgreSQL 是否运行
psql -h localhost -U llmproxy -d litellm -c "SELECT 1"

# 检查数据库连接配置
cat traj_proxy/config.yaml | grep database
```

### Docker 容器模式

#### 服务无法启动

```bash
# 检查端口占用
lsof -i :12345
lsof -i :4000
lsof -i :12300

# 检查 Docker 磁盘空间
docker system df

# 清理未使用的资源
docker system prune
```

#### 数据库连接失败

```bash
# 检查数据库状态
cd dockers && docker-compose ps db

# 查看数据库日志
cd dockers && docker-compose logs db
```

#### Worker 无响应

```bash
# 检查 Worker 进程
cd dockers && docker-compose exec traj_proxy ps aux | grep worker

# 检查 Worker 配置
cd dockers && docker-compose exec traj_proxy cat /app/configs/config.yaml
```

---

## 技术栈

- **Python 3.8+**
- **Ray** - 分布式计算框架
- **FastAPI** - Web框架
- **Uvicorn** - ASGI服务器
- **PostgreSQL** - 数据库
- **Psycopg3** - PostgreSQL驱动

## 项目结构

```
TrajProxy/
├── configs/                        # 配置文件目录
│   ├── config.yaml                 # TrajProxy 配置
│   ├── litellm.yaml                # LiteLLM 配置
│   ├── nginx.conf                  # Nginx 配置
│   └── prometheus.yml              # Prometheus 配置
├── dockers/                        # Docker 相关文件
│   ├── docker-compose.yml          # Docker Compose 编排
│   ├── Dockerfile                  # 镜像构建文件
│   └── images/                     # 导出的镜像文件
├── scripts/                        # 脚本目录
│   ├── start_local.sh              # 本地开发启动脚本
│   ├── start_docker.sh             # Docker Compose 启动脚本
│   └── download_tokenizer.py       # Tokenizer 下载脚本
├── tests/                          # 测试目录
├── traj_proxy/                     # 主代码目录
│   ├── app.py                      # 主入口
│   ├── proxy_core/                 # 推理核心模块
│   │   ├── processor.py           # 非流式请求处理器
│   │   ├── streaming_processor.py # 流式请求处理器
│   │   ├── processor_manager.py   # 处理器管理器
│   │   ├── prompt_builder.py      # 消息转换器
│   │   ├── token_builder.py       # Token处理器
│   │   ├── infer_client.py        # 推理客户端
│   │   ├── infer_response_parser.py # Infer响应解析器
│   │   ├── streaming.py           # 流式响应生成器
│   │   ├── context.py             # 上下文数据类
│   │   ├── routes.py              # API路由
│   │   └── parsers/               # 解析器模块
│   ├── transcript_provider/        # 对话历史模块
│   │   ├── provider.py            # 对话历史提供者
│   │   └── routes.py              # API路由
│   ├── store/                      # 存储模块
│   │   ├── database_manager.py    # 数据库管理器
│   │   ├── request_repository.py  # 请求记录仓库
│   │   └── model_repository.py    # 模型配置仓库
│   ├── workers/                    # Worker 模块
│   │   ├── worker.py              # 统一的 ProxyWorker 实现
│   │   ├── manager.py             # Worker 管理器
│   │   └── route_registrar.py     # 路由注册器
│   └── utils/                      # 工具模块
│       ├── config.py              # 配置管理
│       └── logger.py              # 日志系统
├── models/                         # Tokenizer 模型目录
├── logs/                           # 日志目录
├── requirements.txt                # Python 依赖
└── readme.md                       # 项目说明
```

---

## 端口映射

| 外部端口 | 内部端口 | 服务 | 说明 |
|----------|----------|------|------|
| 12345 | 12345 | Nginx | 统一入口网关 |
| 4000 | 4000 | LiteLLM API | API 网关入口 |
| 5432 | 5432 | PostgreSQL | 数据库访问 |
| 12300-12320 | 12300-12320 | ProxyWorkers | 统一代理服务（多实例） |
| 9090 | 9090 | Prometheus | 监控面板 |

**使用建议**：
- 客户端请求统一通过 `12345` 端口（Nginx）访问
- 内部服务端口（4000、12300+）用于直接调用和调试

---

## 许可证

MIT License
