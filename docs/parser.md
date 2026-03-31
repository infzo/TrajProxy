# Parser 行为逻辑文档

## 概述

TrajProxy 的 Parser 模块参考 vLLM 0.16.0 接口设计，提供统一的工具调用和推理内容解析能力。支持非流式和流式两种模式。

---

## 一、核心数据结构

### 1.1 工具调用结构

```python
# 函数调用（符合 OpenAI 规范）
@dataclass
class FunctionCall:
    name: str = ""           # 函数名
    arguments: str = ""      # 参数（JSON 字符串）

# 工具调用（符合 OpenAI 规范）
@dataclass
class ToolCall:
    id: str                          # 唯一标识，格式: "call_xxx"
    type: str = "function"           # 类型固定为 "function"
    function: Optional[FunctionCall] = None  # 嵌套的函数对象

# 流式增量函数调用
@dataclass
class DeltaFunctionCall:
    name: Optional[str] = None       # 可选的函数名增量
    arguments: Optional[str] = None  # 可选的参数增量

# 流式增量工具调用
@dataclass
class DeltaToolCall:
    id: Optional[str] = None         # 工具调用 ID
    type: Optional[str] = None       # 类型
    index: int = 0                   # 索引（用于流式合并）
    function: Optional[DeltaFunctionCall] = None
```

### 1.2 提取结果结构

```python
# 非流式工具调用提取结果
@dataclass
class ExtractedToolCallInfo:
    tools_called: bool                  # 是否有工具调用
    tool_calls: List[ToolCall]          # 工具调用列表
    content: Optional[str] = None       # 工具调用前的文本内容

# 流式增量消息
@dataclass
class DeltaMessage:
    role: Optional[str] = None           # 角色
    content: Optional[str] = None        # 内容增量
    reasoning: Optional[str] = None      # 推理内容增量
    tool_calls: List[DeltaToolCall] = field(default_factory=list)
```

---

## 二、Tool Parser

### 2.1 已注册的 Tool Parsers

| 名称 | 类名 | 格式特点 |
|------|------|----------|
| `deepseek_v3` | DeepSeekV3ToolParser | Unicode 标记 `<｜tool▁calls▁begin｜>` |
| `deepseek_v31` | DeepSeekV31ToolParser | Unicode 标记变体 |
| `deepseek_v32` | DeepSeekV32ToolParser | DSML 格式 `<｜DSML｜>` |
| `qwen3_coder` | Qwen3CoderToolParser | `toral`/`Ranchi` 边界 + XML |
| `qwen_xml` | QwenXMLToolParser | `ournemouth`/`Ranchi` + XML |
| `glm45` | GLM45ToolParser | GLM 格式 |
| `glm47` | GLM47ToolParser | GLM 格式 |
| `llama3_json` | Llama3JsonParser | JSON 格式 |

### 2.2 非流式解析

**方法签名**:
```python
def extract_tool_calls(
    self,
    model_output: str,
    tools: Optional[List[dict]] = None,
    request: Optional[Any] = None
) -> ExtractedToolCallInfo
```

**返回结构**: `ExtractedToolCallInfo`

#### 场景1: 有工具调用

```python
# 输入 (DeepSeek V3 格式)
'<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"city": "北京"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>'

# 返回
ExtractedToolCallInfo(
    tools_called=True,
    tool_calls=[
        ToolCall(
            id="call_abc123...",
            type="function",
            function=FunctionCall(
                name="get_weather",
                arguments='{"city": "北京"}'
            )
        )
    ],
    content=None
)
```

#### 场景2: 有工具调用 + 前置文本

```python
# 输入
'好的，我来帮您查询。<｜tool▁calls▁begin｜>...<｜tool▁calls▁end｜>'

# 返回
ExtractedToolCallInfo(
    tools_called=True,
    tool_calls=[...],
    content='好的，我来帮您查询。'  # 工具调用前的内容
)
```

#### 场景3: 无工具调用

```python
# 输入
'这是普通的回复内容'

# 返回
ExtractedToolCallInfo(
    tools_called=False,
    tool_calls=[],
    content='这是普通的回复内容'  # 原内容
)
```

#### 场景4: 多个工具调用

```python
# 返回
ExtractedToolCallInfo(
    tools_called=True,
    tool_calls=[
        ToolCall(id="call_xxx1", function=FunctionCall(name="func1", arguments='{"a": 1}')),
        ToolCall(id="call_xxx2", function=FunctionCall(name="func2", arguments='{"b": 2}'))
    ],
    content=None
)
# 注意: 每个 tool_call 有唯一的 id
```

### 2.3 流式解析

**方法签名**:
```python
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
) -> Optional[DeltaMessage]
```

**返回结构**: `Optional[DeltaMessage]`

#### 流式增量示例

```python
# 1. 开始工具调用 - 发送工具头
DeltaMessage(
    tool_calls=[
        DeltaToolCall(
            index=0,
            id="call_abc123",
            type="function",
            function=DeltaFunctionCall(
                name="get_weather",
                arguments=""
            )
        )
    ]
)

# 2. 参数增量
DeltaMessage(
    tool_calls=[
        DeltaToolCall(
            index=0,
            function=DeltaFunctionCall(
                arguments='{"city": "北'
            )
        )
    ]
)

# 3. 参数结束
DeltaMessage(
    tool_calls=[
        DeltaToolCall(
            index=0,
            function=DeltaFunctionCall(
                arguments='"}'
            )
        )
    ]
)
```

### 2.4 不同模型格式示例

#### DeepSeek V3
```
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>func_name
```json
{"param": "value"}
```
<｜tool▁call▁end｜><｜tool▁calls▁end｜>
```

#### DeepSeek V3.2 DSML
```
<｜DSML｜function_calls>
<｜DSML｜invoke name="get_weather">
<｜DSML｜parameter name="location" string="true">杭州</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>
```

#### Qwen3 Coder XML
```
toral<function=get_weather>
<parameter=city>北京</parameter>
<parameter=unit>celsius</parameter>
</function> Ranchi
```

---

## 三、Reasoning Parser

### 3.1 已注册的 Reasoning Parsers

| 名称 | 类名 | 格式特点 |
|------|------|----------|
| `deepseek_r1` | DeepSeekR1ReasoningParser | `<thinky></thinke>` |
| `deepseek_v3` | DeepSeekV3ReasoningParser | `<thinky></thinke>` |
| `deepseek` | DeepSeekReasoningParser | `<｜begin▁of▁think｜>` Unicode |
| `qwen3` | Qwen3ReasoningParser | `<thinky></thinke>` |

### 3.2 非流式解析

**方法签名**:
```python
def extract_reasoning(
    self,
    model_output: str,
    request: Optional[Any] = None
) -> tuple[Optional[str], Optional[str]]
```

**返回结构**: `(reasoning, content)` 元组

#### 场景1: 有推理 + 有回复

```python
# 输入
'<thinky>\n用户询问天气情况。\n需要确认城市。\n</thinke>请问您想查询哪个城市的天气？'

# 返回
(
    '用户询问天气情况。\n需要确认城市。\n',  # reasoning（不含标记）
    '请问您想查询哪个城市的天气？'            # content
)
```

#### 场景2: 只有推理（无回复）

```python
# 输入
'<thinky>\n这是一个纯推理过程。\n</thinke>'

# 返回
(
    '这是一个纯推理过程。\n',  # reasoning
    None                        # content 为 None 或空
)
```

#### 场景3: 只有回复（无推理标记）

```python
# 输入
'这是普通的回复内容'

# 返回
(
    None,                       # reasoning 为 None
    '这是普通的回复内容'        # content 为原内容
)
```

#### 场景4: 只有 end_token（无 start_token）

```python
# 输入（vLLM MiniMaxM2 模式）
'这是推理内容</thinke>这是正常回复'

# 返回
(
    '这是推理内容',     # end_token 前的内容作为 reasoning
    '这是正常回复'      # end_token 后的内容作为 content
)
```

#### 场景5: 推理未结束（无 end_token）

```python
# 输入
'<thinky>\n推理内容没有结束\n持续推理中...'

# 返回
(
    '推理内容没有结束\n持续推理中...',  # 剩余内容作为 reasoning
    None                                  # content 为 None
)
```

### 3.3 流式解析

**方法签名**:
```python
def extract_reasoning_streaming(
    self,
    previous_text: str,
    current_text: str,
    delta_text: str,
    previous_token_ids: Sequence[int],
    current_token_ids: Sequence[int],
    delta_token_ids: Sequence[int]
) -> Optional[DeltaMessage]
```

**返回结构**: `Optional[DeltaMessage]`

#### 流式增量示例

```python
# 1. 在推理区域内
DeltaMessage(
    reasoning='推理内容增量'
)

# 2. 在 end_token 处（转换点）
DeltaMessage(
    reasoning='最后的推理',  # end_token 前
    content='正常回复开始'   # end_token 后
)

# 3. 推理结束后
DeltaMessage(
    content='正常回复内容增量'
)
```

### 3.4 不同模型格式示例

#### DeepSeek Reasoning (Unicode)
```
<｜begin▁of▁think｜>推理过程<｜end▁of▁think｜>正常回复
```

#### Qwen3 / DeepSeek R1 Reasoning
```
<thinky>推理过程</thinke>正常回复
```

---

## 四、组合场景（Reasoning + Tool Calls）

### 4.1 典型组合格式

```python
# 输入
'<thinky>\n用户询问北京天气。\n需要调用 API。\n</thinke><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"city": "北京"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
```

### 4.2 解析流程

```python
# 步骤1: 用 ReasoningParser 提取 reasoning
reasoning, remaining = reasoning_parser.extract_reasoning(output)
# reasoning = '用户询问北京天气。\n需要调用 API。\n'
# remaining = '<｜tool▁calls▁begin｜>...'

# 步骤2: 用 ToolParser 从 remaining 提取 tool_calls
tool_result = tool_parser.extract_tool_calls(remaining, tools=tools)
# tool_result.tools_called = True
# tool_result.tool_calls = [ToolCall(...)]
# tool_result.content = None
```

### 4.3 最终响应结构

```python
{
    "role": "assistant",
    "content": tool_result.content,        # 可能为 None
    "reasoning": reasoning,                 # 推理内容
    "tool_calls": [
        {
            "id": "call_xxx",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "北京"}'
            }
        }
    ]
}
```

---

## 五、响应格式决策表

### 5.1 非流式响应

| 场景 | reasoning | content | tool_calls | tools_called |
|------|-----------|---------|------------|--------------|
| 纯文本 | None | 原文本 | [] | False |
| 纯推理 | 推理内容 | None/空 | [] | False |
| 推理 + 文本 | 推理内容 | 文本内容 | [] | False |
| 工具调用 | None | 前置文本/None | [ToolCall...] | True |
| 推理 + 工具调用 | 推理内容 | None | [ToolCall...] | True |
| 推理 + 文本 + 工具调用 | 推理内容 | 文本内容 | [ToolCall...] | True |

### 5.2 流式响应

| 阶段 | DeltaMessage 字段 |
|------|-------------------|
| 推理中 | `reasoning` 有值 |
| 推理结束点 | `reasoning` + `content` 可能同时有值 |
| 正常回复 | `content` 有值 |
| 工具调用开始 | `tool_calls[0].id`, `tool_calls[0].function.name` |
| 工具参数增量 | `tool_calls[0].function.arguments` 增量 |
| 工具调用结束 | `tool_calls[0].function.arguments` 闭合 |

---

## 六、关键文件路径

| 类型 | 路径 |
|------|------|
| 基础结构 | `traj_proxy/proxy_core/parsers/base.py` |
| 统一 Parser | `traj_proxy/proxy_core/parsers/unified_parser.py` |
| Parser 管理器 | `traj_proxy/proxy_core/parsers/parser_manager.py` |
| Tool Parser 管理器 | `traj_proxy/proxy_core/parsers/tool_parser_manager.py` |
| Reasoning Parser 管理器 | `traj_proxy/proxy_core/parsers/reasoning_parser_manager.py` |
| Tool Parsers 目录 | `traj_proxy/proxy_core/parsers/tool_parsers/` |
| Reasoning Parsers 目录 | `traj_proxy/proxy_core/parsers/reasoning_parsers/` |
| 测试文件 | `tests/e2e/test_parsers.py`, `tests/e2e/test_parser_response_format.py` |

---

## 七、使用示例

### 7.1 获取 Parser

```python
from traj_proxy.proxy_core.parsers import ParserManager

# 获取 Tool Parser
tool_parser_cls = ParserManager.get_tool_parser_cls("deepseek_v3")
tool_parser = tool_parser_cls(tokenizer=tokenizer)

# 获取 Reasoning Parser
reasoning_parser_cls = ParserManager.get_reasoning_parser_cls("deepseek_r1")
reasoning_parser = reasoning_parser_cls(tokenizer=tokenizer)

# 或使用统一接口获取组合 Parser
parser_cls = ParserManager.get_parser(
    tool_parser_name="deepseek_v3",
    reasoning_parser_name="deepseek_r1"
)
parser = parser_cls(tokenizer=tokenizer)
```

### 7.2 非流式解析

```python
# 工具调用解析
result = tool_parser.extract_tool_calls(model_output, tools=tools)
if result.tools_called:
    for tc in result.tool_calls:
        print(f"Function: {tc.function.name}")
        print(f"Arguments: {tc.function.arguments}")

# 推理内容解析
reasoning, content = reasoning_parser.extract_reasoning(model_output)
if reasoning:
    print(f"Reasoning: {reasoning}")
if content:
    print(f"Content: {content}")
```

### 7.3 流式解析

```python
# 重置流式状态
tool_parser.reset_streaming_state()
reasoning_parser.reset_streaming_state()

# 流式处理
for chunk in stream:
    delta_text = chunk.choices[0].delta.content or ""
    
    # 工具调用流式解析
    tool_delta = tool_parser.extract_tool_calls_streaming(
        previous_text=previous_text,
        current_text=current_text,
        delta_text=delta_text,
        previous_token_ids=previous_token_ids,
        current_token_ids=current_token_ids,
        delta_token_ids=delta_token_ids,
        tools=tools
    )
    
    # 推理内容流式解析
    reasoning_delta = reasoning_parser.extract_reasoning_streaming(
        previous_text=previous_text,
        current_text=current_text,
        delta_text=delta_text,
        previous_token_ids=previous_token_ids,
        current_token_ids=current_token_ids,
        delta_token_ids=delta_token_ids
    )
    
    previous_text = current_text
    previous_token_ids = current_token_ids
```

### 7.4 列出已注册的 Parsers

```python
# 列出所有 Tool Parsers
tool_parsers = ParserManager.list_tool_parsers()
# ['deepseek_v3', 'deepseek_v31', 'deepseek_v32', 'qwen3_coder', 'qwen_xml', 'glm45', 'glm47', 'llama3_json']

# 列出所有 Reasoning Parsers
reasoning_parsers = ParserManager.list_reasoning_parsers()
# ['deepseek_r1', 'deepseek_v3', 'deepseek', 'qwen3']
```
