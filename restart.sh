#!/bin/bash
# 快速重启修复脚本

echo "=== 重启 API 服务 ==="

cd /home/tengyue0304/projects/Nexus-Agent

echo "1. 停止 API 服务..."
sudo docker stop nexus-agent-api-1 nexus-agent-celery-worker-1 nexus-agent-celery-beat-1 2>/dev/null

echo "2. 删除 API 容器..."
sudo docker rm nexus-agent-api-1 nexus-agent-celery-worker-1 nexus-agent-celery-beat-1 2>/dev/null

echo "3. 重新创建服务..."
sudo docker-compose up -d api celery-worker celery-beat

echo "4. 等待 10 秒..."
sleep 10

echo "5. 检查状态..."
sudo docker ps --format "table {{.Names}}\t{{.Status}}" | grep nexus

echo ""
echo "6. 测试访问..."
curl -s http://localhost:8080/health | head -1

echo ""
echo "=== 完成 ==="
