# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm/reasoning/abs_reasoning_parsers.py

"""
vLLM ReasoningParser 基类适配器

提供 ReasoningParser 基类和 ReasoningParserManager 注册器。
"""
import importlib
import os
from abc import abstractmethod
from collections.abc import Callable, Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.utils.collection_utils import is_list_of
from vllm.utils.import_utils import import_from_path

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.engine.protocol import (
        DeltaMessage,
    )
    from vllm.entrypoints.openai.responses.protocol import (
        ResponsesRequest,
    )
    from vllm.tokenizers import TokenizerLike
else:
    ChatCompletionRequest = Any
    DeltaMessage = Any
    ResponsesRequest = Any
    TokenizerLike = Any

logger = init_logger(__name__)


class ReasoningParser:
    """
    抽象推理解析器基类，不应直接使用。
    提供的方法应在子类中使用。

    用于从模型输出中提取推理内容。
    """

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        self.model_tokenizer = tokenizer

    @cached_property
    def vocab(self) -> dict[str, int]:
        # 注意：只有 PreTrainedTokenizerFast 保证有 .vocab
        # 但所有 tokenizer 都有 .get_vocab()
        return self.model_tokenizer.get_vocab()

    @abstractmethod
    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """
        检查推理内容是否在 input_ids 中结束。

        用于结构化引擎如 `xgrammar` 来检查推理内容是否在模型输出中结束。

        参数：
        input_ids: list[int]
            模型输出的 input_ids。

        返回：
        bool
            如果推理内容在 input_ids 中结束则为 True。
        """

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Sequence[int]
    ) -> bool:
        """
        检查推理内容是否在 decode 步骤的 input_ids 中结束。

        用于结构化引擎如 `xgrammar` 来检查推理内容是否在 decode 步骤期间在模型输出中结束。
        `input_ids` 是整个模型输出，`delta_ids` 是模型输出在当前 decode 步骤的最后几个计算出的 tokens。

        参数：
        input_ids: list[int]
            整个模型输出。
        delta_ids: list[int]
            当前 decode 步骤模型输出的最后几个计算出的 tokens。

        返回：
        bool
            如果推理内容在 decode 步骤的 `delta_ids` 中结束则为 True。
        """
        return self.is_reasoning_end(input_ids)

    @abstractmethod
    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        从 input_ids 中提取内容 token ids。
        参数：
        input_ids: list[int]
            模型输出的 input_ids。
        返回：
        list[int]
            从 input_ids 中提取的内容。
        """

    @abstractmethod
    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        """
        从完整的模型生成字符串中提取推理内容。

        用于非流式响应，我们在发送给客户端之前有完整的模型响应。

        参数：
        model_output: str
            要从中提取推理内容的模型生成字符串。

        request: ChatCompletionRequest
            用于生成 model_output 的请求对象。

        返回：
        tuple[Optional[str], Optional[str]]
            包含推理内容和内容的元组。
        """

    @abstractmethod
    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """
        实例方法，应该实现从不完整的响应中提取推理；
        用于处理推理调用和流式输出。
        必须是实例方法，因为它需要状态 -
        当前的 tokens/diffs，以及之前解析和提取的信息（见构造函数）
        """


class ReasoningParserManager:
    """
    ReasoningParser 实现的中心注册表。

    支持两种注册模式：
      - 通过 `register_module` 立即注册
      - 通过 `register_lazy_module` 延迟注册

    每个 reasoning parser 必须继承自 `ReasoningParser`。
    """

    reasoning_parsers: dict[str, type[ReasoningParser]] = {}
    lazy_parsers: dict[str, tuple[str, str]] = {}  # name -> (module_path, class_name)

    @classmethod
    def get_reasoning_parser(cls, name: str) -> type[ReasoningParser]:
        """
        获取已注册或延迟注册的 ReasoningParser 类。

        如果 parser 是延迟注册的，它将在首次访问时导入并缓存。

        抛出：
            KeyError: 如果在给定名称下未找到 parser。
        """
        if name in cls.reasoning_parsers:
            return cls.reasoning_parsers[name]

        if name in cls.lazy_parsers:
            return cls._load_lazy_parser(name)

        registered = ", ".join(cls.list_registered())
        raise KeyError(
            f"Reasoning parser '{name}' not found. Available parsers: {registered}"
        )

    @classmethod
    def list_registered(cls) -> list[str]:
        """返回所有已注册和延迟注册的 reasoning parser 名称"""
        return sorted(set(cls.reasoning_parsers.keys()) | set(cls.lazy_parsers.keys()))

    @classmethod
    def _load_lazy_parser(cls, name: str) -> type[ReasoningParser]:
        """导入并注册延迟加载的 reasoning parser"""
        module_path, class_name = cls.lazy_parsers[name]
        try:
            mod = importlib.import_module(module_path)
            parser_cls = getattr(mod, class_name)
            if not issubclass(parser_cls, ReasoningParser):
                raise TypeError(
                    f"{class_name} in {module_path} is not a ReasoningParser subclass."
                )

            cls.reasoning_parsers[name] = parser_cls  # 缓存
            return parser_cls
        except Exception as e:
            logger.exception(
                "Failed to import lazy reasoning parser '%s' from %s: %s",
                name,
                module_path,
                e,
            )
            raise

    @classmethod
    def _register_module(
        cls,
        module: type[ReasoningParser],
        module_name: str | list[str] | None = None,
        force: bool = True,
    ) -> None:
        """立即注册 ReasoningParser 类"""
        if not issubclass(module, ReasoningParser):
            raise TypeError(
                f"module must be subclass of ReasoningParser, but got {type(module)}"
            )

        if module_name is None:
            module_names = [module.__name__]
        elif isinstance(module_name, str):
            module_names = [module_name]
        elif is_list_of(module_name, str):
            module_names = module_name
        else:
            raise TypeError("module_name must be str, list[str], or None.")

        for name in module_names:
            if not force and name in cls.reasoning_parsers:
                existed = cls.reasoning_parsers[name]
                raise KeyError(f"{name} is already registered at {existed.__module__}")
            cls.reasoning_parsers[name] = module

    @classmethod
    def register_lazy_module(cls, name: str, module_path: str, class_name: str) -> None:
        """
        注册延迟模块映射以延迟导入。

        示例：
            ReasoningParserManager.register_lazy_module(
                name="qwen3",
                module_path="vllm.reasoning.parsers.qwen3_reasoning_parser",
                class_name="Qwen3ReasoningParser",
            )
        """
        cls.lazy_parsers[name] = (module_path, class_name)

    @classmethod
    def register_module(
        cls,
        name: str | list[str] | None = None,
        force: bool = True,
        module: type[ReasoningParser] | None = None,
    ) -> (
        type[ReasoningParser] | Callable[[type[ReasoningParser]], type[ReasoningParser]]
    ):
        """
        使用给定的名称或名称列表注册模块。
        可以用作装饰器（module 为 None）或普通函数（module 不为 None）。
        """
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, but got {type(force)}")

        # 立即注册（显式调用）
        if module is not None:
            cls._register_module(module=module, module_name=name, force=force)
            return module

        # 装饰器用法
        def _decorator(obj: type[ReasoningParser]) -> type[ReasoningParser]:
            module_path = obj.__module__
            class_name = obj.__name__

            if isinstance(name, str):
                names = [name]
            elif is_list_of(name, str):
                names = name
            else:
                names = [class_name]

            for n in names:
                cls.lazy_parsers[n] = (module_path, class_name)

            return obj

        return _decorator

    @classmethod
    def import_reasoning_parser(cls, plugin_path: str) -> None:
        """
        通过 reasoning parser 定义文件的路径导入用户定义的 reasoning parser。
        """
        module_name = os.path.splitext(os.path.basename(plugin_path))[0]

        try:
            import_from_path(module_name, plugin_path)
        except Exception:
            logger.exception(
                "Failed to load module '%s' from %s.", module_name, plugin_path
            )
            return


# 别名，方便使用
get_reasoning_parser = ReasoningParserManager.get_reasoning_parser
register_reasoning_parser = ReasoningParserManager.register_module
list_reasoning_parsers = ReasoningParserManager.list_registered

__all__ = [
    "ReasoningParser",
    "ReasoningParserManager",
    "get_reasoning_parser",
    "register_reasoning_parser",
    "list_reasoning_parsers",
]
