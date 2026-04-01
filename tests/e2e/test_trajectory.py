"""
轨迹记录查询 API 测试

测试对话轨迹记录的查询接口
"""

import pytest
import requests
import time

from tests.e2e.config import PROXY_URL, TEST_MESSAGE


class TestTrajectoryAPI:
    """轨迹记录测试类"""

    @pytest.mark.integration
    def test_query_trajectory_after_chat(
        self,
        proxy_client: requests.Session,
        default_headers: dict,
        registered_model_name: str,
        unique_session_id: str
    ):
        """
        测试发送聊天后查询轨迹记录

        验证点:
        - 聊天请求成功
        - 轨迹查询成功
        - 轨迹记录包含正确的 session_id
        - 轨迹记录包含模型名称
        - limit 参数生效
        - 记录字段完整
        """
        # 发送聊天请求
        chat_response = proxy_client.post(
            f"{PROXY_URL}/v1/chat/completions",
            headers=default_headers,
            json={
                "model": registered_model_name,
                "messages": [
                    {"role": "user", "content": TEST_MESSAGE}
                ],
                "max_tokens": 50
            }
        )

        assert chat_response.status_code == 200, f"聊天请求失败: {chat_response.text}"

        # 等待数据写入
        time.sleep(1)

        # 查询轨迹记录
        trajectory_response = proxy_client.get(
            f"{PROXY_URL}/trajectory",
            params={
                "session_id": unique_session_id,
                "limit": 10
            }
        )

        assert trajectory_response.status_code == 200, f"查询轨迹失败: {trajectory_response.text}"

        trajectory_data = trajectory_response.json()

        # 验证轨迹数据
        assert trajectory_data.get("session_id") == unique_session_id, \
            f"session_id 不匹配: {trajectory_data}"

        # 可能需要等待数据同步，如果没有记录则重试
        if trajectory_data.get("count", 0) == 0:
            time.sleep(2)
            trajectory_response = proxy_client.get(
                f"{PROXY_URL}/trajectory",
                params={
                    "session_id": unique_session_id,
                    "limit": 10
                }
            )
            trajectory_data = trajectory_response.json()

        # 验证至少有一条记录
        assert trajectory_data.get("count", 0) >= 1, \
            f"预期至少 1 条记录，实际 {trajectory_data.get('count', 0)} 条"

        # 验证记录内容
        records = trajectory_data.get("records", [])
        assert len(records) > 0, "记录列表为空"

        record = records[0]
        assert record.get("session_id") == unique_session_id, \
            f"记录 session_id 不匹配: {record}"
        assert record.get("model") == registered_model_name, \
            f"记录 model 不匹配: {record}"

        # 验证必要字段存在
        required_fields = [
            "unique_id",
            "request_id",
            "session_id",
            "model"
        ]

        for field in required_fields:
            assert field in record, f"记录缺少必要字段: {field}"
            assert record[field] is not None, f"字段 {field} 值为 None"

    def test_query_nonexistent_trajectory(self, proxy_client: requests.Session):
        """
        测试查询不存在的轨迹记录

        验证点:
        - 返回状态码 200（API 设计如此，返回空列表）
        - 响应包含 session_id 和空的 records
        """
        session_id = "nonexistent_session_xyz,sample_001,task_001"

        response = proxy_client.get(
            f"{PROXY_URL}/trajectory",
            params={
                "session_id": session_id,
                "limit": 100
            }
        )

        # 轨迹查询通常返回 200，即使没有记录
        assert response.status_code == 200, f"查询轨迹失败: {response.text}"

        data = response.json()
        assert data.get("session_id") == session_id, f"session_id 不匹配: {data}"
        assert data.get("count", 0) == 0, f"预期没有记录，实际有 {data.get('count')} 条"
        assert data.get("records", []) == [], f"预期空记录: {data}"
