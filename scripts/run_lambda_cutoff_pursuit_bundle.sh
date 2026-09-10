#!/usr/bin/env bash
set -euo pipefail

SEED=${1:?usage: run_lambda_cutoff_pursuit_bundle.sh SEED}
PORT=8356

cd /gpfs/home1/paul6598/AAMAS

server_pid=""
trainer_pids=()
cleanup() {
  for pid in "${trainer_pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
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

for arm in fixed_floor fixed_zero rsvp_zero; do
  case "$arm" in
    fixed_floor) scheduler=fixed; f_update=50; lambda_zero_t=0 ;;
    fixed_zero)  scheduler=fixed; f_update=50; lambda_zero_t=300000 ;;
    rsvp_zero)   scheduler=vf;    f_update=200; lambda_zero_t=300000 ;;
  esac
  group="lambda_cutoff_20260910_pursuit_${arm}"
  CUDA_VISIBLE_DEVICES=0 /home1/paul6598/miniconda3/envs/aamas/bin/python main.py \
    --config=rsvp --env-config=pursuit with \
    "seed=$SEED" t_max=1000000 \
    epsilon_start=1.0 epsilon_finish=0.05 epsilon_anneal_time=300000 \
    "scheduler=$scheduler" "f_update=$f_update" \
    sched_k=0.6 sched_h=3.0 sched_min_interval=10 sched_gamma=0.8 \
    sched_eps_frac=0.25 sched_warmup_episodes=20 \
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
    use_wandb=True "wandb_group=$group" "wandb_run=${group}_seed${SEED}" &
  trainer_pids+=("$!")
done

status=0
for pid in "${trainer_pids[@]}"; do
  wait "$pid" || status=1
done
trainer_pids=()
exit "$status"
