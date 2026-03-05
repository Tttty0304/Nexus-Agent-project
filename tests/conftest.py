"""
测试配置和Fixtures
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest_asyncio.fixture
async def client():
    """创建测试客户端"""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac


@pytest_asyncio.fixture
async def auth_client(client):
    """创建带认证的测试客户端"""
    # 先注册用户
    register_data = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpassword"
    }
    await client.post("/v1/auth/register", json=register_data)

    # 登录获取 token
    login_data = {
        "username": "testuser",
        "password": "testpassword"
    }
    response = await client.post("/v1/auth/login", data=login_data)
    token = response.json()["access_token"]

    # 设置认证头
    client.headers["Authorization"] = f"Bearer {token}"
    return client
