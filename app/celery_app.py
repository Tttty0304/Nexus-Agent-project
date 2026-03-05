"""
Celery 配置和任务定义

任务路由：
- pdf: PDF解析任务
- embedding: 向量化任务
- default: 默认队列
"""

from celery import Celery
from app.config import settings

# 创建 Celery 实例
celery_app = Celery(
    "nexus",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.tasks.document"]
)

# Celery 配置
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
    
    # 任务路由配置
    task_routes={
        "app.tasks.document.parse_pdf": {"queue": "pdf"},
        "app.tasks.document.generate_embeddings": {"queue": "embedding"},
    },
    
    # 任务确认策略
    task_acks_late=True,  # 任务完成才确认，保证至少执行一次
    worker_prefetch_multiplier=1,  # 避免单个 worker 积压
    task_track_started=True,  # 支持状态查询
    
    # 结果过期时间
    result_expires=3600,
    
    # 任务执行超时
    task_time_limit=300,  # 5分钟
    task_soft_time_limit=240,  # 4分钟软超时
)


# 定时任务配置（可选）
celery_app.conf.beat_schedule = {
    "cleanup-old-sessions": {
        "task": "app.tasks.document.cleanup_old_sessions",
        "schedule": 3600.0,  # 每小时执行
    },
}
