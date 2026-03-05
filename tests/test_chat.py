"""
聊天 API 测试
"""

import pytest
from httpx import AsyncClient
from fastapi import status


@pytest.mark.asyncio
async def test_chat_completions_non_stream(auth_client: AsyncClient):
    """测试非流式聊天补全"""
    response = await auth_client.post("/v1/chat/completions", json={
        "model": "nexus-agent",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "stream": False
    })

    # 由于需要 vLLM 服务，可能返回 503 服务不可用
    assert response.status_code in [status.HTTP_200_OK, status.HTTP_503_SERVICE_UNAVAILABLE]


@pytest.mark.asyncio
async def test_chat_completions_stream(auth_client: AsyncClient):
    """测试流式聊天补全"""
    response = await auth_client.post("/v1/chat/completions", json={
        "model": "nexus-agent",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "stream": True
    })

    # 由于需要 vLLM 服务，可能返回 503 服务不可用
    assert response.status_code in [status.HTTP_200_OK, status.HTTP_503_SERVICE_UNAVAILABLE]
    if response.status_code == 200:
        assert response.headers["content-type"] == "text/event-stream"


@pytest.mark.asyncio
async def test_list_models(auth_client: AsyncClient):
    """测试获取模型列表"""
    response = await auth_client.get("/v1/chat/models")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) > 0
