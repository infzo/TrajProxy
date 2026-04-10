# TrajProxy 变更日志

> **目录**: [文档中心](../README.md) | [经验总结](../experience/README.md)

> 本文档记录 TrajProxy 项目的所有重要变更。使用 `/record-change` 自动追加新条目。

---

## v0.5.0 (2026-03-31)

**类型**: 架构优化

### 新功能
- 流式处理器独立拆分（`streaming_processor.py`），从 processor 中解耦流式响应逻辑
- 推理响应解析器（`infer_response_parser.py`），统一解析推理服务返回结果
- 输入验证器（`validators.py`），集中请求参数校验
- E2E 测试框架大规模扩展：覆盖 parser、token 模式、session id、完整请求格式

### 改进
- Processor 职责拆分，降低单文件复杂度（350 行删除，逻辑分散到 streaming_processor 和 infer_response_parser）
- Parser 文档完善（`docs/parser.md`），详细说明各模型解析器行为
- 测试文件重新组织：合并重复用例，按功能模块拆分（`test_parsers.py`、`test_model_management.py`、`test_http_request_formats.py`、`test_session_id.py`）
- API 文档和 README 更新

### Bug 修复
- 修复 reasoning parser 格式解析错误
- 修复多个 tool parser 的边界情况

### 影响范围
- `traj_proxy/proxy_core/processor.py`
- `traj_proxy/proxy_core/streaming_processor.py`
- `traj_proxy/proxy_core/infer_response_parser.py`
- `traj_proxy/proxy_core/routes.py`
- `traj_proxy/proxy_core/parsers/`
- `traj_proxy/utils/validators.py`
- `docs/parser.md`
- `tests/e2e/`

---

## v0.4.0 (2026-03-30)

**类型**: 功能扩展 + 架构重构

### 新功能
- Tool Calling 解析器体系：支持 DeepSeek、Qwen、GLM、LLAMA 等多种模型格式
- Reasoning 解析器：提取模型思考过程内容
- Claude API 兼容：实现 Messages 接口，支持 Claude 协议转发
- Session ID 支持：通过路径头、请求头、模型名等多种方式传递会话标识

### 改进
- 目录结构调整：按功能模块重新组织代码
- 路由系统重构：统一接口格式，优化请求分发
- Processor Manager 增强：支持多种处理模式切换
- 代码整理，清理冗余逻辑

### 影响范围
- `traj_proxy/proxy_core/parsers/`（新增 tool_parsers/、reasoning_parsers/）
- `traj_proxy/proxy_core/routes.py`
- `traj_proxy/proxy_core/processor.py`
- `traj_proxy/proxy_core/processor_manager.py`
- `traj_proxy/proxy_core/prompt_builder.py`
- `traj_proxy/workers/`

---

## v0.3.0 (2026-03-28)

**类型**: 功能迭代

### 新功能
- Tokenizer 支持：集成 HuggingFace tokenizer，实现 token 编码/解码（Qwen2.5-7B、Qwen3-Coder 模型）
- Job ID 机制：请求级别唯一标识，支持全链路追踪
- Worker 合并：简化部署结构，统一工作进程管理

### 改进
- Processor Manager 增强 token 处理能力
- 数据库管理器扩展查询接口
- 模型仓库支持多 tokenizer 模型管理

### 影响范围
- `traj_proxy/proxy_core/processor_manager.py`
- `traj_proxy/proxy_core/routes.py`
- `traj_proxy/proxy_core/processor.py`
- `traj_proxy/store/database_manager.py`
- `traj_proxy/store/model_repository.py`
- `traj_proxy/store/models.py`
- `traj_proxy/workers/`
- `models/`（新增 tokenizer 模型文件）

---

## v0.2.0 (2026-03-24 ~ 2026-03-26)

**类型**: 功能迭代

### 新功能
- Provider 统一接口：对齐多个推理服务提供商的接口行为
- Processor Manager：独立管理请求处理器的创建和生命周期
- 模型注册表（`model_registry.py`）：动态模型管理，支持运行时增删模型
- LiteLLM 网关配置（`litellm.yaml`）：对接 LiteLLM 代理层
- API 文档（`api.md`）：完整的接口说明
- 部署脚本（`start_local.sh`、`start_docker.sh`）

### 改进
- 路由系统大幅扩展，支持多种请求格式
- 数据库管理器增强，支持更复杂的数据操作
- 文档全面更新（README 重写）
- Docker 镜像整理和部署流程优化

### 影响范围
- `traj_proxy/proxy_core/processor_manager.py`（新增）
- `traj_proxy/proxy_core/routes.py`
- `traj_proxy/proxy_core/worker.py`
- `traj_proxy/store/database_manager.py`
- `traj_proxy/store/model_registry.py`（新增）
- `traj_proxy/transcript_provider/`
- `configs/litellm.yaml`（新增）
- `scripts/`

---

## v0.1.0 (2026-03-20 ~ 2026-03-23)

**类型**: 初始版本

### 新功能
- 基于 Ray + FastAPI 的 LLM 代理服务基础架构
- Token-in-Token-out 模式：支持 token 级别的请求处理和缓存
- 直接转发模式：简单代理请求到推理服务
- 推理客户端（`infer_client.py`）：封装与推理服务的通信
- 数据持久层（`store/`）：PostgreSQL 数据库管理、模型和请求存储
- 轨迹记录（`transcript_provider/`）：请求-响应轨迹的存储和查询
- Worker 管理系统：基于 Ray Actor 的分布式工作进程管理
- 日志系统（`utils/logger.py`）
- Docker 部署支持：Dockerfile、docker-compose、Prometheus 监控

### Bug 修复
- 修复存储层写入失败问题

### 影响范围
- `traj_proxy/` 全部模块（初始创建）
- `configs/`
- `dockers/`
