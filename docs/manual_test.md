# 手动测试指南

本文档提供部署后的核心功能验证方法。

## 快速开始

```bash
# 执行完整测试流程（简洁模式）
bash docs/manual_test.sh

# 显示详细请求/响应信息（推荐）
bash docs/manual_test.sh --verbose

# 使用已有模型测试（跳过注册）
bash docs/manual_test.sh --verbose --skip-register --model qwen3.5-2b

# 指定服务地址
bash docs/manual_test.sh --verbose --proxy-url http://192.168.1.100:12300 --nginx-url http://192.168.1.100:12345
```

## 测试流程

脚本按照真实使用场景串联以下测试：

```
1. 健康检查
   └─ 验证 TrajProxy 和 Nginx 服务状态

2. 模型管理
   ├─ 查看已有模型
   ├─ 注册测试模型
   └─ 验证注册成功

3. OpenAI Chat 测试
   ├─ 非流式请求
   ├─ 流式请求
   └─ 路径传递 session_id

4. Claude Chat 测试
   ├─ 非流式请求（通过 LiteLLM）
   └─ 流式请求

5. Tool Call 测试
   ├─ OpenAI 格式 tool calling
   └─ Claude 格式 tool use

6. 轨迹查询
   └─ 按 session_id 查询请求记录

7. 清理
   └─ 删除测试模型
```

## 命令行选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--proxy-url` | TrajProxy 地址 | `http://localhost:12300` |
| `--nginx-url` | Nginx 网关地址 | `http://localhost:12345` |
| `--model` | 测试模型名称 | 自动检测或生成 |
| `--skip-register` | 跳过模型注册 | 否 |
| `--verbose, -v` | 显示详细请求/响应信息 | 否 |
| `--help` | 显示帮助 | - |

## 环境变量

也可通过环境变量配置：

```bash
export PROXY_URL="http://localhost:12300"
export NGINX_URL="http://localhost:12345"
export LITELLM_API_KEY="sk-1234"

bash docs/manual_test.sh
```

## 测试结果说明

脚本会输出彩色日志：

- 🟢 `[PASS]` - 测试通过
- 🔴 `[FAIL]` - 测试失败
- 🟡 `[SKIP]` - 测试跳过
- 🔵 `[INFO]` - 信息提示

测试结束后会显示摘要统计。

## 常见问题

### 1. 健康检查失败

确认服务已启动：

```bash
# 检查 TrajProxy
curl http://localhost:12300/health

# 检查 Nginx
curl http://localhost:12345/health
```

### 2. 模型注册失败

如果返回 400/409，可能是模型名称冲突，使用 `--skip-register` 跳过注册。

### 3. Chat 请求失败

- 检查模型是否已配置正确的推理服务地址
- 确认推理服务正常运行
- 查看 TrajProxy 日志排查问题

### 4. Claude 测试失败

确认 LiteLLM 网关已配置：

```bash
curl http://localhost:4000/health
```

## 手动验证命令

如需单独验证某个功能，可参考以下命令：

### 健康检查

```bash
curl http://localhost:12300/health
```

### 查看模型列表

```bash
curl http://localhost:12300/v1/models
```

### 注册模型

```bash
curl -X POST http://localhost:12300/models/register \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "test-model",
    "url": "http://localhost:1234",
    "api_key": "sk-test",
    "tokenizer_path": "Qwen/Qwen2.5-3B"
  }'
```

### OpenAI Chat

```bash
curl -X POST http://localhost:12300/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-session-id: test,001,001" \
  -d '{
    "model": "test-model",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 50
  }'
```

### Tool Calling

```bash
curl -X POST http://localhost:12300/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test-model",
    "messages": [{"role": "user", "content": "北京天气"}],
    "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
    "max_tokens": 100
  }'
```

### 查询轨迹

```bash
curl "http://localhost:12300/trajectory?session_id=test,001,001"
```

### 删除模型

```bash
curl -X DELETE "http://localhost:12300/models?model_name=test-model"
```
