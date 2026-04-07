# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
vLLM 兼容层

提供 vllm 包的导入兼容，使得 `from vllm.xxx import yyy` 正常工作。
同时提供自动发现和注册机制。

使用方式：
1. 直接使用：
   from traj_proxy.proxy_core.parsers.vllm_compat import ensure_initialized
   ensure_initialized()  # 自动发现和注册所有 parser

2. 通过 ParserManager 使用（推荐）：
   from traj_proxy.proxy_core.parsers import ParserManager
   ParserManager.create_parsers(...)  # 内部会自动初始化
"""
import sys
import importlib
import logging
from pathlib import Path
from typing import List, Optional

# 兼容层根目录
_VLLM_COMPAT_DIR = Path(__file__).parent

# 将 vllm_compat 目录添加到 sys.path
# 这样 Python 就能找到 "vllm" 包了
_vllm_path = str(_VLLM_COMPAT_DIR)
if _vllm_path not in sys.path:
    # 插入到前面，确保优先级
    sys.path.insert(0, _vllm_path)

# 延迟初始化标记
_initialized = False
_logger: Optional[logging.Logger] = None


def _get_logger() -> logging.Logger:
    """获取或创建 logger"""
    global _logger
    if _logger is None:
        _logger = logging.getLogger(__name__)
        if not _logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            _logger.addHandler(handler)
            _logger.setLevel(logging.INFO)
    return _logger


def discover_and_register_parsers(
    tool_parsers_dir: str = "tool_parsers",
    reasoning_parsers_dir: str = "reasoning_parsers",
) -> List[str]:
    """自动发现并注册所有 parser

    扫描指定的 parser 目录，导入所有 .py 文件（除了 __init__.py），
    触发装饰器注册。

    Args:
        tool_parsers_dir: Tool Parser 目录名（相对于 vllm_compat 目录）
        reasoning_parsers_dir: Reasoning Parser 目录名（相对于 vllm_compat 目录）

    Returns:
        成功注册的 parser 名称列表
    """
    logger = _get_logger()
    registered = []

    # 发现 Tool Parsers
    tool_parsers_path = _VLLM_COMPAT_DIR / tool_parsers_dir
    if tool_parsers_path.exists():
        logger.debug(f"Scanning tool parsers directory: {tool_parsers_path}")
        for py_file in sorted(tool_parsers_path.glob("*.py")):
            if py_file.name.startswith("_"):
                continue

            module_name = py_file.stem
            try:
                # 构建模块路径
                full_module_path = f"traj_proxy.proxy_core.parsers.vllm_compat.{tool_parsers_dir}.{module_name}"
                importlib.import_module(full_module_path)
                registered.append(f"tool:{module_name}")
                logger.debug(f"Successfully loaded tool parser: {module_name}")
            except Exception as e:
                logger.warning(f"Failed to load tool parser {module_name}: {e}")

    # 发现 Reasoning Parsers
    reasoning_parsers_path = _VLLM_COMPAT_DIR / reasoning_parsers_dir
    if reasoning_parsers_path.exists():
        logger.debug(f"Scanning reasoning parsers directory: {reasoning_parsers_path}")
        for py_file in sorted(reasoning_parsers_path.glob("*.py")):
            if py_file.name.startswith("_"):
                continue

            module_name = py_file.stem
            try:
                full_module_path = f"traj_proxy.proxy_core.parsers.vllm_compat.{reasoning_parsers_dir}.{module_name}"
                importlib.import_module(full_module_path)
                registered.append(f"reasoning:{module_name}")
                logger.debug(f"Successfully loaded reasoning parser: {module_name}")
            except Exception as e:
                logger.warning(f"Failed to load reasoning parser {module_name}: {e}")

    if registered:
        logger.info(f"Discovered and registered parsers: {registered}")

    return registered


def ensure_initialized() -> bool:
    """确保兼容层已初始化

    首次调用时会执行自动发现和注册。
    后续调用会直接返回初始化状态。

    Returns:
        是否成功初始化（首次调用时）
    """
    global _initialized
    if _initialized:
        return True

    logger = _get_logger()
    logger.info("Initializing vLLM parser compatibility layer...")

    try:
        registered = discover_and_register_parsers()
        _initialized = True
        logger.info(f"vLLM parser compatibility layer initialized with {len(registered)} parsers")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize vLLM parser compatibility layer: {e}")
        return False


def is_initialized() -> bool:
    """检查兼容层是否已初始化"""
    return _initialized


def reset():
    """重置兼容层状态（主要用于测试）"""
    global _initialized
    _initialized = False


# 导出便捷函数
__all__ = [
    "discover_and_register_parsers",
    "ensure_initialized",
    "is_initialized",
    "reset",
]
