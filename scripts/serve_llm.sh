#!/bin/bash
# Serve the LLM Commander with vLLM (OpenAI-compatible API).
# Usage: bash serve_llm.sh [MODEL] [PORT]
# Run this on a GPU node (e.g. tmux session 37, RTX 6000 Ada 48GB).

MODEL=$1
PORT=$2

if [ -z "$MODEL" ]; then
    MODEL="openai/gpt-oss-20b"
fi
if [ -z "$PORT" ]; then
    PORT=8355
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

exec vllm serve "$MODEL" \
    --port "$PORT" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --disable-log-requests
