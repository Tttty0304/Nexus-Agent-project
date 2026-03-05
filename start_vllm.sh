#!/bin/bash
# 启动 vLLM 服务加载本地 Qwen-7B-AWQ 模型

MODEL_PATH="./models/Qwen2-7B-Instruct-AWQ"
PORT=8001

echo "=========================================="
echo "启动 vLLM 服务 (Qwen-7B-AWQ)"
echo "=========================================="
echo "模型路径: $MODEL_PATH"
echo "服务端口: $PORT"
echo ""

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型目录不存在: $MODEL_PATH"
    exit 1
fi

# 检查是否有 GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU 状态:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    echo ""
    GPU_FLAG=""
else
    echo "警告: 未检测到 NVIDIA GPU，将使用 CPU 模式（性能较低）"
    GPU_FLAG="--device cpu"
fi

# 启动 vLLM
echo "正在启动 vLLM 服务..."
echo "命令: python -m vllm.entrypoints.openai.api_server \"
    --model $MODEL_PATH \"
    --quantization awq \"
    --dtype auto \"
    --max-model-len 8192 \"
    --gpu-memory-utilization 0.85 \"
    --max-num-seqs 256 \"
    --enable-prefix-caching \"
    --port $PORT \"
    $GPU_FLAG"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --quantization awq \
    --dtype auto \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 256 \
    --enable-prefix-caching \
    --port "$PORT" \
    $GPU_FLAG
