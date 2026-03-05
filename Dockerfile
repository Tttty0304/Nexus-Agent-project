# Nexus-Agent FastAPI 服务镜像
# 使用完整镜像路径，避免 registry-mirrors 失效问题

FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY pyproject.toml ./

# 安装所有 Python 依赖（使用清华镜像源）
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    fastapi>=0.110.0 \
    uvicorn[standard]>=0.27.0 \
    python-multipart>=0.0.6 \
    httpx[http2]>=0.26.0 \
    pydantic>=2.5.0 \
    pydantic-settings>=2.1.0 \
    sqlmodel>=0.0.14 \
    sqlalchemy>=2.0.0 \
    asyncpg>=0.29.0 \
    psycopg2-binary>=2.9.9 \
    pgvector>=0.2.4 \
    redis>=5.0.0 \
    celery>=5.3.0 \
    python-jose[cryptography]>=3.3.0 \
    passlib[bcrypt]>=1.7.4 \
    tenacity>=8.2.0 \
    tiktoken>=0.5.0 \
    sentence-transformers>=2.2.0 \
    torch>=2.0.0 \
    pdfplumber>=0.10.0 \
    pytesseract>=0.3.10 \
    pillow>=10.0.0 \
    aiohttp>=3.9.0 \
    beautifulsoup4>=4.12.0 \
    async-timeout>=4.0.0 \
    prometheus-client>=0.19.0 \
    structlog>=23.0.0 \
    numpy>=1.24.0 \
    && rm -rf /root/.cache/pip

# 复制应用代码
COPY app/ ./app/

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--loop", "uvloop", "--http", "httptools"]
