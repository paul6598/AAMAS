#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=${1:?usage: run_lambda_cutoff_one.sh ENV ARM SEED}
ARM=${2:?usage: run_lambda_cutoff_one.sh ENV ARM SEED}
SEED=${3:?usage: run_lambda_cutoff_one.sh ENV ARM SEED}
PORT=8356

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

if [[ "$ENV_NAME" == "5m_vs_6m" ]]; then
  set +e
  CUDA_VISIBLE_DEVICES=0 bash scripts/run_lambda_cutoff_screen.sh \
    "$ARM" "$SEED" "$API"
  status=$?
  set -e
  exit "$status"
fi

if [[ "$ENV_NAME" != "pursuit" ]]; then
  echo "unsupported environment: $ENV_NAME" >&2
  exit 2
fi

case "$ARM" in
  fixed_floor) scheduler=fixed; f_update=50; lambda_zero_t=0; sched_trace=False ;;
  fixed_zero)  scheduler=fixed; f_update=50; lambda_zero_t=300000; sched_trace=False ;;
  rsvp_zero)   scheduler=vf;    f_update=200; lambda_zero_t=300000; sched_trace=True ;;
  *) echo "unsupported arm: $ARM" >&2; exit 2 ;;
esac

group="lambda_cutoff_20260910_pursuit_${ARM}"
export CUDA_VISIBLE_DEVICES=0
set +e
/home1/paul6598/miniconda3/envs/aamas/bin/python main.py \
  --config=rsvp --env-config=pursuit with \
  "seed=$SEED" t_max=1000000 \
  epsilon_start=1.0 epsilon_finish=0.05 epsilon_anneal_time=300000 \
  "scheduler=$scheduler" "f_update=$f_update" \
  sched_k=0.6 sched_h=3.0 sched_min_interval=10 sched_gamma=0.8 \
  sched_eps_frac=0.25 sched_warmup_episodes=20 "sched_trace=$sched_trace" \
  sched_target_early_per_ep=0.15 sched_h_adapt_episodes=20 \
  use_reward_shaping=True shaping_in_learner=True \
  use_action_masking=False use_masking_at_test=False \
  lambda_start=0.5 lambda_min=0.05 lambda_decay=0.9995 \
  lambda_floor_frac=0 "lambda_zero_t=$lambda_zero_t" \
  prompt_style=paper commander_include_training_stats=False dt_observable=True \
  llm_model=openai/gpt-oss-20b "llm_api_base=$API" \
  llm_temperature=0.2 llm_max_tokens=3072 llm_cache=False \
  test_interval=10000 test_nepisode=16 \
  save_model=True save_model_interval=100000 \
  use_wandb=True "wandb_group=$group" "wandb_run=${group}_seed${SEED}"
status=$?
set -e
exit "$status"
