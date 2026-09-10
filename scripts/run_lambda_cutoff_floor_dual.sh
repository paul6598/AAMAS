#!/usr/bin/env bash
set -euo pipefail

cd /gpfs/home1/paul6598/AAMAS

server0=""
server1=""
train0=""
train1=""
cleanup() {
  [[ -n "$train0" ]] && kill "$train0" 2>/dev/null || true
  [[ -n "$train1" ]] && kill "$train1" 2>/dev/null || true
  [[ -n "$server0" ]] && kill "$server0" 2>/dev/null || true
  [[ -n "$server1" ]] && kill "$server1" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES=0 bash scripts/serve_llm.sh openai/gpt-oss-20b 8356 &
server0=$!
CUDA_VISIBLE_DEVICES=1 bash scripts/serve_llm.sh openai/gpt-oss-20b 8357 &
server1=$!

for endpoint in http://127.0.0.1:8356/v1 http://127.0.0.1:8357/v1; do
  while ! curl --silent --fail "${endpoint}/models" >/dev/null; do
    if ! kill -0 "$server0" 2>/dev/null || ! kill -0 "$server1" 2>/dev/null; then
      echo "a vLLM server exited before both endpoints became ready" >&2
      exit 1
    fi
    sleep 10
  done
done

CUDA_VISIBLE_DEVICES=0 bash scripts/run_lambda_cutoff_screen.sh \
  fixed_floor 0 http://127.0.0.1:8356/v1 &
train0=$!
CUDA_VISIBLE_DEVICES=1 bash scripts/run_lambda_cutoff_screen.sh \
  fixed_floor 1 http://127.0.0.1:8357/v1 &
train1=$!

status=0
wait "$train0" || status=1
train0=""
wait "$train1" || status=1
train1=""
exit "$status"
