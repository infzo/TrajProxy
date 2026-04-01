# TrajProxy 文档

## 文档导航

| 文档 | 说明 |
|------|------|
| [架构设计](architecture.md) | 系统架构概述、核心组件、处理流程 |
| [API 参考](api_reference.md) | 完整的 API 接口文档 |
| [配置详解](configuration.md) | 配置文件说明、环境变量 |
| [数据库设计](database.md) | 表结构、索引、同步机制 |
| [部署指南](deployment.md) | 本地开发、Docker 部署 |
| [开发指南](development.md) | 开发环境搭建、测试运行 |
| [Parser 文档](parser.md) | Tool/Reasoning Parser 行为逻辑 |
| [vLLM 对比](compare_vllm.md) | 与 vLLM OpenAI 接口对比 |

---

## 快速了解

### 项目概述

TrajProxy 是一个 LLM 请求代理服务，提供：

- **OpenAI 兼容 API** - 无缝对接现有客户端
- **Token-in-Token-out 模式** - 支持前缀匹配缓存
- **动态模型管理** - 运行时注册/删除模型
- **请求轨迹记录** - 完整的对话历史存储

### 核心架构

```
┌─────────────────────────────────────────────────────────────┐
│                      WorkerManager                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ ProxyWorker │  │ ProxyWorker │  │ ProxyWorker │  ...     │
│  │   :12300    │  │   :12301    │  │   :12302    │          │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │
└─────────┼────────────────┼────────────────┼─────────────────┘
          │                │                │
          ▼                ▼                ▼
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │PostgreSQL│     │ Infer服务 │     │Prometheus│
    │   (DB)   │     │  (LLM)   │     │ (监控)   │
    └──────────┘     └──────────┘     └──────────┘
```

### 两种处理模式

| 模式 | 特点 | 适用场景 |
|------|------|----------|
| Token-in-Token-out | prompt构建 + token编码 + 前缀缓存 | 需要缓存优化、token级别控制 |
| 直接转发 | 轻量代理，直接转发请求 | 快速集成、无需tokenizer |

---

## 推荐阅读顺序

1. **新用户**: [架构设计](architecture.md) → [部署指南](deployment.md) → [API 参考](api_reference.md)
2. **运维人员**: [部署指南](deployment.md) → [配置详解](configuration.md) → [数据库设计](database.md)
3. **开发者**: [开发指南](development.md) → [架构设计](architecture.md) → [Parser 文档](parser.md)
