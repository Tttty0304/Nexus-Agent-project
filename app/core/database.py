"""
PostgreSQL + pgvector 数据库模块（满血版本）

特性：
- 异步连接池
- pgvector 向量扩展支持
- 自动重连机制
- 连接健康检查
"""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel
from sqlalchemy import text

from app.config import settings

# 创建异步引擎（满血版配置）
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_recycle=settings.DB_POOL_RECYCLE,
    pool_pre_ping=settings.DB_POOL_PRE_PING,
)

# 创建异步会话工厂
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# 导出异步会话 maker 供工具使用
async_session_maker = AsyncSessionLocal


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    获取数据库会话的依赖注入函数
    
    使用方式：
        @app.get("/items")
        async def read_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """
    初始化数据库表和扩展
    
    1. 启用 pgvector 扩展
    2. 创建所有表
    """
    from app.models.user import User
    from app.models.conversation import Conversation
    from app.models.message import Message
    from app.models.knowledge import Document, DocumentChunk
    
    async with engine.begin() as conn:
        # 启用 pgvector 扩展
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        
        # 创建所有表
        await conn.run_sync(SQLModel.metadata.create_all)
    
    print(f"✅ PostgreSQL 数据库已初始化: {settings.DATABASE_URL.split('@')[-1]}")


async def close_db():
    """关闭数据库连接池"""
    await engine.dispose()
    print("✅ 数据库连接池已关闭")


async def check_db_health() -> dict:
    """
    检查数据库健康状态
    
    Returns:
        {"status": "healthy" | "unhealthy", "latency_ms": float, "error": str}
    """
    import time
    
    start = time.time()
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(text("SELECT 1"))
            result.scalar()
            latency_ms = (time.time() - start) * 1000
            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 2),
                "error": None
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "latency_ms": None,
            "error": str(e)
        }
