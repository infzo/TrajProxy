# 多轮一致性 Prompt Template

## 定义

对于任意大模型的 Jinja 模板，如果：

- 第一轮 messages + response 拼接为字符串 a
- 第二轮 messages + response 拼接为字符串 b
- 第三轮 messages + response 拼接为字符串 c
- ...

则 **a 必须是 b 的子字符串，b 必须是 c 的子字符串**，以此类推。

## 为什么需要多轮一致性

在流式对话场景中，客户端通常需要：
1. 本地缓存已渲染的 prompt
2. 新轮次只发送增量部分

如果模板不满足多轮一致性，客户端无法复用缓存，必须每次完整渲染。

## Qwen3.5-2B 原模板问题

原模板使用 `last_query_index` 全局状态：

```jinja
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    ...
    {%- set ns.last_query_index = index %}
{%- endfor %}
...
{%- if loop.index0 > ns.last_query_index %}
    {#- 使用思考格式 #}
{%- else %}
    {#- 不使用思考格式 #}
{%- endif %}
```

**问题：** 当添加第二轮消息时，`last_query_index` 会变化，导致第一轮的 assistant 消息格式改变。

**示例：**

第一轮输出：
```
#@|im_start|#user
你好#@|im_end|#
#@|im_start|#assistant
#@think@#

#@/think@#

你好！#@|im_end|#
```

第二轮输出（历史消息格式变了！）：
```
#@|im_start|#user
你好#@|im_end|#
#@|im_start|#assistant
你好！#@|im_end|#   <-- 没有思考格式了
#@|im_start|#user
再见#@|im_end|#
#@|im_start|#assistant
#@think@#

#@/think@#

再见！#@|im_end|#
```

## 解决方案

删除 `last_query_index` 计算，**所有 assistant 消息统一使用思考格式**：

```jinja
{%- elif message.role == "assistant" %}
    ...
    {#- 多轮一致性：所有 assistant 都使用思考格式 #}
    {{- '#@|im_start|@#' + message.role + '\n#@think@#\n' + reasoning_content + '\n#@/think@#\n\n' + content }}
```

## 模板文件

| 文件 | 说明 |
|-----|------|
| `models/Qwen/Qwen3.5-2B/chat_template.jinja` | 原模板 |
| `models/Qwen/Qwen3.5-2B/chat_template_consistency.jinja` | 多轮一致性模板 |

## 测试用例

### 1. 纯对话

```python
messages_round1 = [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"}
]

messages_round2 = messages_round1 + [
    {"role": "user", "content": "再见"},
    {"role": "assistant", "content": "再见！"}
]

# 验证：round1_output 在 round2_output 中
```

### 2. tool 调用

```python
messages_round1 = [
    {"role": "user", "content": "北京天气"},
    {"role": "assistant", "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "北京"}}}]},
    {"role": "tool", "content": "晴天 25度"},
    {"role": "assistant", "content": "北京今天晴天，25度"}
]

messages_round2 = messages_round1 + [
    {"role": "user", "content": "上海呢"},
    {"role": "assistant", "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "上海"}}}]},
    {"role": "tool", "content": "多云 20度"},
    {"role": "assistant", "content": "上海多云，20度"}
]
```

### 3. reasoning_content

```python
messages_round1 = [
    {"role": "user", "content": "1+1=?"},
    {"role": "assistant", "reasoning_content": "这是一个简单的加法问题", "content": "答案是2"}
]

messages_round2 = messages_round1 + [
    {"role": "user", "content": "那2+2呢？"},
    {"role": "assistant", "reasoning_content": "继续加法运算", "content": "答案是4"}
]
```

### 4. 三轮传递性

```python
# 验证 round1 在 round2 中
# 验证 round2 在 round3 中
# 验证 round1 在 round3 中（传递性）
```

## 运行测试

```bash
pytest tests/test_template_consistency.py -v
```

## 扩展新模型

1. 找到类似 `last_query_index` 的全局状态依赖
2. 分析哪些消息属性决定格式（tool_calls、reasoning_content 等）
3. 修改条件判断，改为基于消息自身属性
4. 删除全局状态计算代码
5. 添加测试用例验证多轮一致性
