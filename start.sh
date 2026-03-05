#!/bin/bash
#
# Nexus-Agent 启动脚本
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_env() {
    print_info "检查环境..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker 未安装"
        exit 1
    fi
    
    if [ ! -f .env ]; then
        print_warn ".env 不存在，使用默认配置"
        cp .env.example .env 2>/dev/null || true
    fi
    
    print_info "环境检查通过"
}

show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  start       启动所有服务（默认，使用外部 vLLM）"
    echo "  gpu         启动所有服务（包含 Docker 内 vLLM，需要 nvidia-docker）"
    echo "  stop        停止所有服务"
    echo "  status      查看服务状态"
    echo "  logs        查看日志"
    echo ""
    echo "说明:"
    echo "  - 默认使用外部 vLLM（在主机上运行，推荐）"
    echo "  - vLLM 默认端口: 8001"
    echo "  - API 端口: 8080"
}

case "${1:-start}" in
    start)
        check_env
        print_info "启动服务（外部 vLLM 模式）..."
        print_info "请确保 vLLM 已在主机上运行: python -m vllm.entrypoints.openai.api_server --port 8001"
        sudo docker-compose up -d
        print_info "服务已启动"
        echo "  API: http://localhost:8080"
        ;;
    gpu)
        check_env
        print_info "启动服务（Docker GPU 模式）..."
        sudo docker-compose --profile gpu up -d
        print_info "服务已启动"
        ;;
    stop)
        print_info "停止服务..."
        sudo docker-compose down
        ;;
    status)
        sudo docker-compose ps
        ;;
    logs)
        sudo docker-compose logs -f
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "未知命令: $1"
        show_help
        exit 1
        ;;
esac
