"""
Agent API 测试
"""

import pytest
from httpx import AsyncClient
from fastapi import status


@pytest.mark.asyncio
async def test_react_endpoint_stream(auth_client: AsyncClient):
    """测试 ReAct 流式接口"""
    response = await auth_client.post("/v1/agent/react", json={
        "session_id": "test-session-001",
        "user_input": "What is the weather in Beijing?",
        "max_iterations": 3
    })

    assert response.status_code == status.HTTP_200_OK
    assert response.headers["content-type"] == "text/event-stream"


@pytest.mark.asyncio
async def test_list_tools(auth_client: AsyncClient):
    """测试获取工具列表"""
    response = await auth_client.get("/v1/agent/tools")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["object"] == "list"
    # 应包含 internet_search, calculator, current_time 等工具
    tool_names = [t["name"] for t in data["data"]]
    assert "internet_search" in tool_names
