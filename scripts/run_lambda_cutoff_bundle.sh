#!/usr/bin/env bash
set -euo pipefail

BUNDLE=${1:?usage: run_lambda_cutoff_bundle.sh zero SEED | floor}
SEED=${2:-0}
PORT=8356
LEFT_DEVICE=${BUNDLE_LEFT_DEVICE:-1}
RIGHT_DEVICE=${BUNDLE_RIGHT_DEVICE:-2}

cd /gpfs/home1/paul6598/AAMAS

server_pid=""
left_pid=""
right_pid=""
cleanup() {
  [[ -n "$left_pid" ]] && kill "$left_pid" 2>/dev/null || true
  [[ -n "$right_pid" ]] && kill "$right_pid" 2>/dev/null || true
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

case "$BUNDLE" in
  zero)
    CUDA_VISIBLE_DEVICES="$LEFT_DEVICE" bash scripts/run_lambda_cutoff_screen.sh \
      fixed_zero "$SEED" "$API" &
    left_pid=$!
    CUDA_VISIBLE_DEVICES="$RIGHT_DEVICE" bash scripts/run_lambda_cutoff_screen.sh \
      rsvp_zero "$SEED" "$API" &
    right_pid=$!
    ;;
  floor)
    CUDA_VISIBLE_DEVICES="$LEFT_DEVICE" bash scripts/run_lambda_cutoff_screen.sh \
      fixed_floor 0 "$API" &
    left_pid=$!
    CUDA_VISIBLE_DEVICES="$RIGHT_DEVICE" bash scripts/run_lambda_cutoff_screen.sh \
      fixed_floor 1 "$API" &
    right_pid=$!
    ;;
  *) echo "unsupported bundle: $BUNDLE" >&2; exit 2 ;;
esac

left_status=0
right_status=0
wait "$left_pid" || left_status=$?
left_pid=""
wait "$right_pid" || right_status=$?
right_pid=""
if (( left_status != 0 || right_status != 0 )); then
  echo "bundle failed: left=${left_status}, right=${right_status}" >&2
  exit 1
fi
