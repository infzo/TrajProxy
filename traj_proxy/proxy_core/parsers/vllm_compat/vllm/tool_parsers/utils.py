# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm/tool_parsers/utils.py

"""
工具解析器工具函数
"""
import json
from typing import Any, Dict, List, Optional, Union


def get_json_schema_from_tools(
    tool_choice: Union[str, Dict[str, Any], None],
    tools: Optional[List[Dict[str, Any]]]
) -> Optional[Dict[str, Any]]:
    """从工具定义生成 JSON Schema

    Args:
        tool_choice: 工具选择（"auto", "none", "required", 或具体工具名）
        tools: 工具列表

    Returns:
        对应的 JSON Schema，如果无法生成则返回 None
    """
    if not tools:
        return None

    # 处理 "none" 情况
    if tool_choice == "none":
        return None

    # 处理 "auto" 情况 - 返回第一个工具的 schema
    if tool_choice == "auto" or tool_choice is None:
        # 返回一个通用的 schema
        return {
            "type": "object",
            "properties": {},
        }

    # 处理 "required" 情况
    if tool_choice == "required":
        # 返回一个强制调用工具的 schema
        return {
            "type": "object",
            "properties": {},
        }

    # 处理具体工具选择
    if isinstance(tool_choice, dict):
        function_name = tool_choice.get("function", {}).get("name")
        if function_name:
            for tool in tools:
                if tool.get("type") == "function":
                    func = tool.get("function", {})
                    if func.get("name") == function_name:
                        return {
                            "type": "object",
                            "properties": func.get("parameters", {}).get("properties", {}),
                            "required": func.get("parameters", {}).get("required", []),
                        }

    return None


__all__ = ["get_json_schema_from_tools"]
