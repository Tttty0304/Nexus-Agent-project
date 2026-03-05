"""
Nexus-Agent FastAPI 主应用入口（满血版本）

核心特性：
- 生命周期管理：启动时初始化连接池，关闭时优雅释放
- 全异步架构：Redis、PostgreSQL、HTTP 客户端全异步
- 健康检查：多维度服务健康探针
- 指标监控：Prometheus 指标暴露
- 认证授权：JWT Token 认证
- 限流保护：Redis 令牌桶限流
"""

import asyncio
import time
from contextlib import asynccontextmanager
from datetime import datetime

import httpx
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.core.cache import cache
from app.core.database import init_db, close_db, check_db_health
from app.services.llm_service import vllm_client
from app.api.v1 import chat, agent, sessions, knowledge, auth

# Prometheus metrics (conditional)
try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter, Histogram
    PROMETHEUS_AVAILABLE = True
    # 定义指标
    REQUEST_COUNT = Counter('nexus_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
    REQUEST_DURATION = Histogram('nexus_request_duration_seconds', 'Request duration')
except ImportError:
    PROMETHEUS_AVAILABLE = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    
    Startup: 初始化所有连接池
    Shutdown: 优雅释放资源
    """
    # ===== Startup =====
    print(f"🚀 [{settings.APP_NAME} v{settings.APP_VERSION}] Starting up...")
    
    # 1. 初始化 HTTP 客户端（vLLM 连接池）
    await vllm_client.connect()
    print("✅ VLLM client connected")
    
    # 2. 初始化 Redis 连接
    await cache.connect()
    print("✅ Redis cache connected")
    
    # 3. 初始化数据库
    try:
        await init_db()
        print("✅ Database initialized")
    except Exception as e:
        print(f"⚠️ Database initialization warning: {e}")
    
    # 4. 连接预热
    try:
        await vllm_client.health_check()
        print("✅ vLLM health check passed")
    except Exception as e:
        print(f"⚠️ vLLM health check failed: {e}")
    
    print(f"🟢 [{settings.APP_NAME}] Ready to serve!")
    
    yield  # 应用运行期间
    
    # ===== Shutdown =====
    print(f"🛑 [{settings.APP_NAME}] Shutting down...")
    
    # 1. 关闭 vLLM 客户端
    await vllm_client.close()
    print("✅ VLLM client closed")
    
    # 2. 关闭 Redis 连接
    await cache.disconnect()
    print("✅ Redis cache closed")
    
    # 3. 关闭数据库连接池
    await close_db()
    print("✅ Database connections closed")
    
    print(f"🔴 [{settings.APP_NAME}] Goodbye!")


# 创建 FastAPI 应用实例
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="高并发企业级 RAG 与 Agent 编排平台",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理"""
    import traceback
    import logging
    error_detail = traceback.format_exc()

    # 使用 logging 而不是 print
    logging.getLogger(__name__).error(f"Unhandled exception: {exc}", exc_info=True, extra={
        "path": request.url.path,
        "method": request.method,
        "client": request.client.host if request.client else None
    })

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "detail": error_detail if settings.DEBUG else None
        }
    )


# 请求耗时中间件
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """添加请求处理时间头"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# 健康检查端点
@app.get("/health")
async def health_check():
    """
    服务健康检查
    
    检查所有依赖服务的状态
    """
    checks = {
        "vllm": "unknown",
        "redis": "unknown",
        "postgres": "unknown"
    }
    
    # 检查 vLLM
    try:
        vllm_health = await vllm_client.health_check()
        checks["vllm"] = vllm_health.get("status", "degraded")
    except Exception as e:
        checks["vllm"] = f"unhealthy: {str(e)}"
    
    # 检查 Redis
    try:
        if await cache.ping():
            checks["redis"] = "healthy"
        else:
            checks["redis"] = "degraded"
    except Exception as e:
        checks["redis"] = f"unhealthy: {str(e)}"
    
    # 检查 PostgreSQL
    try:
        db_health = await check_db_health()
        checks["postgres"] = db_health["status"]
    except Exception as e:
        checks["postgres"] = f"unhealthy: {str(e)}"
    
    overall = "healthy" if all(
        c == "healthy" for c in checks.values()
    ) else "degraded"
    
    return {
        "status": overall,
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.APP_VERSION
    }


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


# Prometheus metrics endpoint (PDF架构要求)
@app.get("/metrics")
async def metrics():
    """
    Prometheus 指标暴露端点

    提供关键性能指标监控：
    - nexus_requests_total: 请求总数
    - nexus_request_duration_seconds: 请求处理时间
    - vllm_queue_length: vLLM 队列长度（待实现）
    """
    if not PROMETHEUS_AVAILABLE or not settings.METRICS_ENABLED:
        return JSONResponse(
            status_code=503,
            content={"error": "Metrics not available"}
        )

    from fastapi.responses import Response
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# 注册 API 路由
app.include_router(auth.router, prefix="/v1/auth", tags=["Authentication"])
app.include_router(chat.router, prefix="/v1", tags=["Chat"])
app.include_router(agent.router, prefix="/v1/agent", tags=["Agent"])
app.include_router(sessions.router, prefix="/v1/sessions", tags=["Sessions"])
app.include_router(knowledge.router, prefix="/v1/knowledge", tags=["Knowledge"])

# 挂载静态文件目录
import os
from fastapi.staticfiles import StaticFiles

# 使用绝对路径确保在 Docker 容器中也能找到 static 目录
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
# 如果目录不存在则创建（避免启动失败）
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# 限流中间件
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """请求限流中间件（Redis 令牌桶）"""
    # 如果限流禁用，直接放行
    if not settings.RATE_LIMIT_ENABLED:
        return await call_next(request)
    
    # 排除健康检查和文档端点
    if request.url.path in ["/health", "/", "/docs", "/redoc", "/openapi.json"]:
        return await call_next(request)
    
    # 获取用户标识
    user_id = request.headers.get("X-User-ID")
    client_ip = request.client.host
    key = f"ratelimit:{user_id or client_ip}"
    
    try:
        allowed = await cache.rate_limit_check(
            key,
            rate=settings.RATE_LIMIT_REQUESTS_PER_MINUTE / 60,
            capacity=settings.RATE_LIMIT_CAPACITY,
            burst=settings.RATE_LIMIT_BURST
        )
        
        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "limit": settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
                    "window": "1 minute"
                },
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(settings.RATE_LIMIT_REQUESTS_PER_MINUTE),
                    "X-RateLimit-Window": "60"
                }
            )
    except Exception as e:
        # 限流服务异常时记录日志但不阻塞请求（故障隔离原则）
        import logging
        logging.getLogger(__name__).warning(f"Rate limit check failed: {e}")
    
    response = await call_next(request)
    return response


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        loop="uvloop",
        http="httptools",
        reload=settings.DEBUG
    )
