#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "usage: $0 PORT LOG TRACE TRAIN_ARGS..."
    exit 2
fi

PORT="$1"
LOG="$2"
TRACE="$3"
shift 3

VLLM_BIN=/home1/paul6598/miniconda3/envs/vllm/bin/vllm
PYTHON_BIN=/home1/paul6598/miniconda3/envs/permute/bin/python
MODEL="${MCA_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MAX_MODEL_LEN="${MCA_MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${MCA_GPU_MEMORY_UTILIZATION:-0.88}"
MODEL_DTYPE="${MCA_MODEL_DTYPE:-bfloat16}"
SERVER_LOG="${LOG%.log}_server.log"
VLLM_EXTRA_ARGS=()
if [ "${MCA_ENFORCE_EAGER:-0}" = "1" ]; then
    VLLM_EXTRA_ARGS+=(--enforce-eager)
fi
# Keep optional loading/quantization settings outside the training command so a
# paper-fidelity run can use a longer context without duplicating this runner.
if [ -n "${MCA_QUANTIZATION:-}" ]; then
    VLLM_EXTRA_ARGS+=(--quantization "$MCA_QUANTIZATION")
fi
if [ -n "${MCA_LOAD_FORMAT:-}" ]; then
    VLLM_EXTRA_ARGS+=(--load-format "$MCA_LOAD_FORMAT")
fi
export CUDA_HOME=/opt/ohpc/pub/cuda/13.1.1
export PATH="/home1/paul6598/miniconda3/envs/vllm/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

"$VLLM_BIN" serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --dtype "$MODEL_DTYPE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --enable-prefix-caching \
    --generation-config vllm \
    "${VLLM_EXTRA_ARGS[@]}" \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 600); do
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "vLLM server exited during startup; see $SERVER_LOG"
        exit 1
    fi
    sleep 1
done
curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null

TRAIN_ENV=()
if [ "${MCA_TRAIN_CPU_ONLY:-0}" = "1" ]; then
    # The already-running vLLM child keeps its GPU. Hide CUDA only from the policy
    # trainer so optimizer bookkeeping cannot allocate from a nearly-full device.
    TRAIN_ENV=(env CUDA_VISIBLE_DEVICES="")
fi

"${TRAIN_ENV[@]}" "$PYTHON_BIN" -u train.py \
    --algo llm_mca \
    --backbone rnn \
    --model "$MODEL" \
    --critic_backend vllm \
    --critic_api_base "http://127.0.0.1:${PORT}/v1" \
    --critic_trace "$TRACE" \
    "$@" 2>&1 | tee "$LOG"
