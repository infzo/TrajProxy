# 开发指南

本文档介绍如何搭建本地开发环境、理解代码结构、运行测试。

---

## 开发环境搭建

### 1. 克隆项目

```bash
git clone <repository_url>
cd TrajProxy
```

### 2. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 准备数据库

```bash
# 启动 PostgreSQL（Docker 方式）
docker run -d --name traj_proxy_db \
    -e POSTGRES_DB=traj_proxy \
    -e POSTGRES_USER=llmproxy \
    -e POSTGRES_PASSWORD=dbpassword9090 \
    -p 5432:5432 postgres:16

# 初始化数据库
export DATABASE_URL="postgresql://llmproxy:dbpassword9090@localhost:5432/traj_proxy"
python scripts/init_db.py
```

### 5. 配置推理服务

确保 LLM 推理服务可用，例如：

```bash
# 使用 vLLM
vllm serve Qwen/Qwen3.5-2B --port 8000

# 或使用 Ollama
ollama serve
```

### 6. 启动开发服务

```bash
./scripts/start_local.sh
```

---

## 代码结构

```
TrajProxy/
├── configs/                        # 配置文件目录
│   ├── config.yaml                 # TrajProxy 主配置
│   ├── litellm.yaml                # LiteLLM 网关配置
│   ├── nginx.conf                  # Nginx 配置
│   └── prometheus.yml              # Prometheus 配置
│
├── traj_proxy/                     # 主代码目录
│   ├── app.py                      # 应用入口
│   ├── exceptions.py               # 自定义异常
│   │
│   ├── proxy_core/                 # 推理核心模块
│   │   ├── routes.py               # API 路由定义
│   │   ├── processor.py            # 非流式请求处理器
│   │   ├── streaming_processor.py  # 流式请求处理器
│   │   ├── processor_manager.py    # 处理器管理器
│   │   ├── prompt_builder.py       # 消息转换器
│   │   ├── token_builder.py        # Token 处理器
│   │   ├── infer_client.py         # 推理客户端
│   │   ├── infer_response_parser.py # Infer 响应解析器
│   │   ├── streaming.py            # 流式响应生成器
│   │   ├── context.py              # 处理上下文数据类
│   │   └── parsers/                # 解析器模块
│   │       ├── base.py             # 基础数据结构
│   │       ├── unified_parser.py   # 统一解析器
│   │       ├── parser_manager.py   # 解析器管理器
│   │       ├── tool_parsers/       # 工具解析器
│   │       └── reasoning_parsers/  # 推理解析器
│   │
│   ├── store/                      # 存储模块
│   │   ├── database_manager.py     # 数据库连接池管理
│   │   ├── model_repository.py     # 模型配置仓库
│   │   ├── request_repository.py   # 请求记录仓库
│   │   ├── notification_listener.py # LISTEN/NOTIFY 监听器
│   │   └── models.py               # 数据模型定义
│   │
│   ├── transcript_provider/        # 轨迹查询模块
│   │   ├── provider.py             # 轨迹提供者
│   │   └── routes.py               # API 路由
│   │
│   ├── workers/                    # Worker 模块
│   │   ├── worker.py               # ProxyWorker 实现
│   │   ├── manager.py              # Worker 管理器
│   │   └── route_registrar.py      # 路由注册器
│   │
│   └── utils/                      # 工具模块
│       ├── config.py               # 配置管理
│       └── logger.py               # 日志系统
│
├── tests/                          # 测试目录
│   └── e2e/                        # 端到端测试
│       ├── conftest.py             # 测试配置
│       ├── run_e2e.py              # 测试运行脚本
│       └── test_*.py               # 测试文件
│
├── scripts/                        # 脚本目录
│   ├── start_local.sh              # 本地启动脚本
│   ├── start_docker.sh             # Docker 启动脚本
│   ├── init_db.py                  # 数据库初始化
│   └── download_tokenizer.py       # Tokenizer 下载
│
├── dockers/                        # Docker 相关
│   ├── docker-compose.yml          # 容器编排
│   └── Dockerfile                  # 镜像构建
│
├── models/                         # Tokenizer 模型目录
├── docs/                           # 文档目录
├── requirements.txt                # Python 依赖
└── readme.md                       # 项目说明
```

---

## 核心模块说明

### proxy_core - 推理核心

核心请求处理流程：

```
请求 → routes.py → ProcessorManager.get_processor()
                     ↓
              Processor / StreamingProcessor
                     ↓
              PromptBuilder (messages → prompt_text)
                     ↓
              TokenBuilder (prompt_text → token_ids, 前缀匹配)
                     ↓
              InferClient (发送到推理服务)
                     ↓
              TokenBuilder (token_ids → text)
                     ↓
              Parser (解析 tool_calls, reasoning)
                     ↓
              PromptBuilder (构建 OpenAI Response)
```

### store - 存储层

- **DatabaseManager**: 管理连接池
- **ModelRepository**: 模型配置 CRUD
- **RequestRepository**: 请求轨迹存储
- **NotificationListener**: LISTEN/NOTIFY 实时同步

### workers - Worker 管理

- **WorkerManager**: 启动/管理多个 ProxyWorker
- **ProxyWorker**: FastAPI 应用，处理请求

---

## 运行测试

### 测试环境准备

```bash
# 1. 启动数据库
docker run -d --name test_db \
    -e POSTGRES_DB=traj_proxy \
    -e POSTGRES_USER=llmproxy \
    -e POSTGRES_PASSWORD=dbpassword9090 \
    -p 5432:5432 postgres:16

# 2. 初始化数据库
export DATABASE_URL="postgresql://llmproxy:dbpassword9090@localhost:5432/traj_proxy"
python scripts/init_db.py

# 3. 启动推理服务（或 mock 服务）
# 确保 http://localhost:8000 有可用的推理服务
```

### 运行 E2E 测试

```bash
cd tests/e2e

# 运行所有测试
python run_e2e.py

# 运行指定测试
python run_e2e.py test_health.py
python run_e2e.py test_model_management.py
python run_e2e.py test_token_mode.py
python run_e2e.py test_parsers.py
```

### 测试文件说明

| 测试文件 | 说明 |
|----------|------|
| `test_health.py` | 健康检查接口测试 |
| `test_model_management.py` | 模型注册/删除/列表测试 |
| `test_token_mode.py` | Token-in-Token-out 模式测试 |
| `test_parsers.py` | Parser 解析逻辑测试 |
| `test_http_request_formats.py` | HTTP 请求格式测试 |
| `test_session_id.py` | Session ID 传递测试 |
| `test_trajectory.py` | 轨迹查询测试 |

---

## 调试技巧

### 日志查看

日志输出到标准输出，可通过重定向保存：

```bash
./scripts/start_local.sh 2>&1 | tee logs/debug.log
```

### 修改日志级别

在代码中使用：

```python
from traj_proxy.utils.logger import get_logger
logger = get_logger(__name__)
logger.setLevel("DEBUG")
```

### 数据库调试

```bash
# 连接数据库
psql -h localhost -U llmproxy -d traj_proxy

# 查看最近的请求记录
SELECT * FROM request_records ORDER BY start_time DESC LIMIT 10;

# 查看已注册模型
SELECT * FROM model_registry;
```

### 单独测试组件

```python
# 测试 Tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
tokens = tokenizer.encode("Hello, world!")
print(tokens)

# 测试 Parser
from traj_proxy.proxy_core.parsers import ParserManager
parser_cls = ParserManager.get_tool_parser_cls("deepseek_v3")
print(parser_cls)
```

---

## 添加新的 Parser

### 1. 创建 Parser 类

在 `traj_proxy/proxy_core/parsers/tool_parsers/` 或 `reasoning_parsers/` 下创建文件：

```python
from traj_proxy.proxy_core.parsers.base import (
    ToolParser, ExtractedToolCallInfo, ToolCall, FunctionCall
)

class MyToolParser(ToolParser):
    """自定义工具解析器"""

    def extract_tool_calls(
        self,
        model_output: str,
        tools: Optional[List[dict]] = None,
        request: Optional[Any] = None
    ) -> ExtractedToolCallInfo:
        # 实现解析逻辑
        pass

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        tools: Optional[List[dict]] = None,
        request: Optional[Any] = None
    ) -> Optional[DeltaMessage]:
        # 实现流式解析逻辑
        pass
```

### 2. 注册 Parser

在 `tool_parser_manager.py` 或 `reasoning_parser_manager.py` 中注册：

```python
from .tool_parsers.my_parser import MyToolParser

ToolParserManager.register("my_parser", MyToolParser)
```

### 3. 使用 Parser

在模型配置中指定：

```yaml
models:
  - model_name: my-model
    tool_parser: "my_parser"
```

---

## 代码风格

- **缩进**: 4 个空格
- **命名**: snake_case（变量/函数），PascalCase（类）
- **注释**: 中文，详细说明
- **导入**: 绝对路径，项目根路径开头

示例：

```python
from traj_proxy.store.models import ModelConfig
from traj_proxy.utils.logger import get_logger

logger = get_logger(__name__)


def process_request(messages: list, model_name: str) -> dict:
    """处理请求

    Args:
        messages: 消息列表
        model_name: 模型名称

    Returns:
        处理结果字典
    """
    # 实现逻辑
    pass
```
