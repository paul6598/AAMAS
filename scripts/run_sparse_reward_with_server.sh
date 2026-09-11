#!/usr/bin/env bash
set -euo pipefail

MAP=${1:?usage: run_sparse_reward_with_server.sh MAP SEED}
SEED=${2:?usage: run_sparse_reward_with_server.sh MAP SEED}
PORT=${LLM_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 20000))}

cd /gpfs/home1/paul6598/AAMAS
server_pid=""
cleanup() {
  [[ -n "$server_pid" ]] && kill "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES=0 bash scripts/serve_llm.sh openai/gpt-oss-20b "$PORT" &
server_pid=$!
API="http://127.0.0.1:${PORT}/v1"
while ! curl --silent --fail "${API}/models" >/dev/null; do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "vLLM server exited before becoming ready" >&2
    exit 1
  fi
  sleep 10
done

set +e
CUDA_VISIBLE_DEVICES=0 bash scripts/run_sparse_reward_screen.sh \
  full "$MAP" "$SEED" "$API"
status=$?
set -e
exit "$status"
