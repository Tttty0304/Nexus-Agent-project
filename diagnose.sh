#!/bin/bash
# 诊断 8080 端口问题

echo "=== 诊断 API 服务问题 ==="

echo ""
echo "1. 检查 Docker 容器状态..."
sudo docker ps -a | grep -E "(nexus|api)" || echo "无相关容器"

echo ""
echo "2. 检查 API 容器日志..."
sudo docker-compose logs --tail=50 api 2>/dev/null || echo "无法获取日志"

echo ""
echo "3. 检查端口监听..."
sudo lsof -i :8080 || sudo ss -tlnp | grep 8080 || echo "8080 无监听"

echo ""
echo "4. 检查健康检查配置..."
sudo docker inspect nexus-agent-api-1 2>/dev/null | grep -A 10 "Health" || echo "无法获取 inspect 信息"

echo ""
echo "5. 尝试直接访问容器..."
sudo docker exec nexus-agent-api-1 curl -s http://localhost:8080/health 2>/dev/null || echo "容器内访问失败"

echo ""
echo "=== 诊断完成 ==="
