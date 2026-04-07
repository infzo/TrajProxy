# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm/logger/__init__.py

"""
vLLM logger 适配器

提供与 vllm.logger.init_logger 兼容的日志接口。
"""
import logging
from typing import Optional


def init_logger(name: str) -> logging.Logger:
    """初始化日志器

    Args:
        name: 日志器名称（通常是 __name__）

    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)

    # 如果没有 handler，添加一个基本的 handler
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


__all__ = ["init_logger"]
