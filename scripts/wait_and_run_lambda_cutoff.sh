#!/usr/bin/env bash
set -euo pipefail

SERVER_JOB=${1:?usage: wait_and_run_lambda_cutoff.sh SERVER_JOB ARM SEED}
ARM=${2:?usage: wait_and_run_lambda_cutoff.sh SERVER_JOB ARM SEED}
SEED=${3:?usage: wait_and_run_lambda_cutoff.sh SERVER_JOB ARM SEED}

while true; do
  SERVER_NODE=$(squeue -h -j "$SERVER_JOB" -o '%N' | head -n 1)
  if [[ -n "$SERVER_NODE" && "$SERVER_NODE" != "(null)" ]]; then
    break
  fi
  sleep 10
done

API="http://${SERVER_NODE}:8356/v1"
while ! curl --silent --fail "${API}/models" >/dev/null; do
  sleep 10
done

exec bash /gpfs/home1/paul6598/AAMAS/scripts/run_lambda_cutoff_screen.sh \
  "$ARM" "$SEED" "$API"
