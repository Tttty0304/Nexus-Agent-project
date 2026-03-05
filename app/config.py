"""
Nexus-Agent 配置管理（满血版本）
使用 Pydantic Settings 实现环境变量驱动的配置

支持：PostgreSQL + pgvector, Redis, Celery, 认证
"""

from functools import lru_cache
from typing import List, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置类"""

    # 应用基础配置
    APP_NAME: str = "Nexus-Agent"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8080
    WORKERS: int = 1
    
    # vLLM 服务配置
    VLLM_BASE_URL: str = "http://localhost:8001"
    VLLM_API_KEY: str = ""
    VLLM_MODEL: str = "/models/Qwen2-7B-Instruct-AWQ"  # vLLM 中实际的模型路径（Docker 容器内路径）
    VLLM_MAX_CONCURRENT: int = 20
    VLLM_TIMEOUT: float = 60.0
    
    # 数据库配置 (PostgreSQL + pgvector)
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/nexus"
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_RECYCLE: int = 3600
    DB_POOL_PRE_PING: bool = True
    
    # Redis 配置
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_POOL_SIZE: int = 50
    REDIS_POOL_MAX_CONNECTIONS: int = 100
    
    # Celery 配置
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    CELERY_WORKER_CONCURRENCY: int = 4
    CELERY_TASK_SOFT_TIME_LIMIT: int = 240
    CELERY_TASK_TIME_LIMIT: int = 300
    
    # 安全配置 (JWT)
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALGORITHM: str = "HS256"
    
    # 限流配置 (Redis 令牌桶)
    RATE_LIMIT_ENABLED: bool = False
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 60
    RATE_LIMIT_CAPACITY: int = 60
    RATE_LIMIT_BURST: int = 10
    
    # 会话配置
    MAX_SESSION_ROUNDS: int = 10
    SESSION_TTL_DAYS: int = 7
    SESSION_CLEANUP_INTERVAL: int = 3600  # 每小时清理过期会话
    
    # RAG 配置
    EMBEDDING_MODEL: str = "BAAI/bge-large-zh-v1.5"
    EMBEDDING_DIMENSION: int = 1024
    EMBEDDING_DEVICE: str = "cpu"  # 或 "cuda"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    VECTOR_TOP_K: int = 10
    VECTOR_THRESHOLD: float = 0.3  # 降低阈值，确保更多结果能被召回
    RAG_VECTOR_WEIGHT: float = 0.7
    RAG_KEYWORD_WEIGHT: float = 0.3
    RAG_RRF_K: int = 60  # RRF 融合常数
    
    # Query Rewrite 配置
    ENABLE_QUERY_REWRITE: bool = True
    QUERY_REWRITE_TEMPERATURE: float = 0.1
    QUERY_REWRITE_MAX_TOKENS: int = 256

    # Reranker 配置
    ENABLE_RERANKER: bool = True  # 开启重排序
    RERANKER_TYPE: str = "cross_encoder"  # 使用交叉编码器
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"  # 轻量级高精度重排序模型
    RERANKER_TOP_K: int = 20  # Rerank 的候选数量
    RERANKER_BATCH_SIZE: int = 8

    # Agent 配置
    AGENT_MAX_ITERATIONS: int = 5
    AGENT_TIMEOUT_SECONDS: float = 60.0  # 增加到60秒，避免长文本处理超时
    AGENT_MAX_TOKENS: int = 1024
    
    # 文档处理配置
    PDF_MAX_PAGES: int = 1000
    PDF_MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    OCR_ENABLED: bool = True
    OCR_LANGUAGES: List[str] = ["chi_sim", "eng"]
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json 或 text
    LOG_FILE: Optional[str] = None
    
    # 监控配置 (Prometheus)
    METRICS_ENABLED: bool = True
    METRICS_PORT: int = 9090
    
    # 追踪配置 (OpenTelemetry)
    TRACING_ENABLED: bool = False
    OTEL_EXPORTER_OTLP_ENDPOINT: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()


settings = get_settings()
