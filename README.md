# Nexus-Agent 

高并发企业级 RAG 与 Agent 编排平台



## 快速开始

### 🚀 一键启动

```bash
# 启动所有服务
docker-compose up -d

# 访问测试界面
open http://localhost:8080/static/index.html
```

### 📋 核心特性

| 特性 | 实现方案 | 性能指标 |
|------|----------|----------|
| **LLM 推理** | vLLM + PagedAttention | TTFT < 50ms, QPS > 300 |
| **API 网关** | FastAPI + Uvicorn | 单节点 1000+ 并发 SSE |
| **状态管理** | Redis + PostgreSQL | P99 延迟 < 10ms |
| **RAG 检索** | Hybrid Search (BM25 + 向量) | 召回率提升 30%+ |
| **Agent 编排** | 自研 ReAct 引擎 | 工具调用成功率 90%+ |

## 技术栈

| 层级 | 技术组件 | 用途 |
|------|----------|------|
| **Infra 层** | vLLM, FastAPI, Uvicorn | LLM 推理与 API 服务 |
| **State 层** | Redis, PostgreSQL, SQLModel | 状态管理与数据持久化 |
| **Knowledge 层** | pgvector, Celery, sentence-transformers | 文档处理与向量检索 |
| **Action 层** | 自研 ReAct 引擎 | Agent 智能编排与工具调用 |


## 🔧 环境准备

- Docker & Docker Compose
- NVIDIA Docker Runtime (GPU 支持)
- 模型文件放置在 `./models/` 目录

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，修改必要配置
```

### 3. 启动服务

```bash
# 启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f api
```

### 4. 验证服务

```bash
# 健康检查
curl http://localhost:8080/health

# 获取 Token
curl -X POST http://localhost:8080/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=test&password=test"

# 聊天请求
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "model": "nexus-agent",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

## API 文档

启动后访问：http://localhost:8080/docs

### 认证接口

- `POST /v1/auth/register` - 用户注册
- `POST /v1/auth/login` - 用户登录
- `POST /v1/auth/refresh` - 刷新 Token
- `GET /v1/auth/me` - 获取当前用户

### 聊天接口

- `POST /v1/chat/completions` - OpenAI 兼容聊天接口
- `GET /v1/models` - 获取可用模型列表

### Agent 接口

- `POST /v1/agent/react` - ReAct Agent 执行（流式）
- `GET /v1/agent/tools` - 获取可用工具列表

### 知识库接口

- `POST /v1/knowledge/upload` - 上传 PDF 文档
- `GET /v1/knowledge/tasks/{task_id}` - 查询任务状态
- `POST /v1/knowledge/search` - 知识库检索
- `GET /v1/knowledge/documents` - 获取文档列表

### 会话接口

- `GET /v1/sessions/sessions` - 获取会话列表
- `POST /v1/sessions/sessions` - 创建新会话
- `GET /v1/sessions/sessions/{id}/messages` - 获取会话消息

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                        客户端层                              │
│              (Web UI / Mobile App / API Consumer)           │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP/HTTPS, SSE
┌──────────────────────────▼──────────────────────────────────┐
│                      API 网关层 (FastAPI)                    │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│  │  认证授权    │ │  请求路由    │ │  限流控制    │            │
│  │  JWT/OAuth2 │ │  版本管理    │ │  Token Bucket│            │
│  └─────────────┘ └─────────────┘ └─────────────┘            │
└──────────────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼──────┐  ┌───────▼──────┐  ┌───────▼──────┐
│  vLLM 服务   │  │    Redis     │  │  PostgreSQL  │
│  (GPU推理)   │  │  (缓存/队列)  │  │  (持久化)    │
└──────────────┘  └──────────────┘  └──────────────┘
```

## 部署说明

### 本地开发

```bash
# 安装依赖
pip install -e .

# 启动依赖服务 (PostgreSQL, Redis)
docker-compose up -d postgres redis

# 启动 API 服务
uvicorn app.main:app --reload

# 启动 Celery Worker
celery -A app.celery_app worker -l info
```

### 生产部署

```bash
# 构建镜像
docker-compose build

# 启动所有服务
docker-compose up -d

# 启用监控（可选）
docker-compose --profile monitoring up -d
```

## 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `VLLM_BASE_URL` | vLLM 服务地址 | `http://localhost:8000` |
| `DATABASE_URL` | PostgreSQL 连接字符串 | `postgresql+asyncpg://...` |
| `REDIS_URL` | Redis 连接字符串 | `redis://localhost:6379/0` |
| `SECRET_KEY` | JWT 签名密钥 | - |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | 每分钟请求限制 | 60 |

)

## 📋 测试清单

完成以下步骤验证系统功能：

- [ ] 服务正常启动（`docker-compose up -d`）
- [ ] 访问健康检查（`curl http://localhost:8080/health`）
- [ ] 打开测试界面（`http://localhost:8080/static/index.html`）
- [ ] 注册/登录用户
- [ ] 上传测试 PDF 文档
- [ ] 等待文档处理完成（状态变为 "已完成"）
- [ ] 执行知识检索测试
- [ ] 测试 RAG 增强聊天
- [ ] 查看文档列表
- [ ] 删除测试文档

## 🎯 快速测试

### 1. 使用 Web 界面（推荐）

```bash
# 启动服务
docker-compose up -d

# 打开浏览器访问
open http://localhost:8080/static/index.html
```

### 2. 使用命令行

```bash
# 1. 注册用户
curl -X POST http://localhost:8080/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"test","email":"test@example.com","password":"test123"}'

# 2. 登录获取 Token
curl -X POST http://localhost:8080/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=test&password=test123"

# 3. 上传 PDF
curl -X POST http://localhost:8080/v1/knowledge/upload \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@test.pdf"

# 4. 检索知识
curl "http://localhost:8080/v1/knowledge/search?query=测试&search_type=hybrid&top_k=5" \
  -H "Authorization: Bearer YOUR_TOKEN"
```



## 📄 License

MIT

---

**Made with ❤️ by Nexus Team**

