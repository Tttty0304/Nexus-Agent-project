"""
API V1 路由模块
"""

from fastapi import APIRouter

from app.api.v1 import auth, chat, agent, sessions, knowledge

# 创建主路由
api_router = APIRouter()

# 注册子路由
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(chat.router, prefix="/chat", tags=["Chat"])
api_router.include_router(agent.router, prefix="/agent", tags=["Agent"])
api_router.include_router(sessions.router, prefix="/sessions", tags=["Sessions"])
api_router.include_router(knowledge.router, prefix="/knowledge", tags=["Knowledge"])
