#!/bin/bash
# Serve the LLM Commander with vLLM (OpenAI-compatible API).
# Usage: bash serve_llm.sh [MODEL] [PORT]
# Run this inside an allocated GPU job with the vLLM environment active.
set -euo pipefail

MODEL=${1:-openai/gpt-oss-20b}
PORT=${2:-8355}
command -v vllm >/dev/null || { echo 'Activate the vLLM environment before serving.' >&2; exit 1; }

exec vllm serve "$MODEL" \
    --port "$PORT" \
    --max-model-len 8192 \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
