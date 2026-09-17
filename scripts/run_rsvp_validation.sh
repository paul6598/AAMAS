#!/usr/bin/env bash
# Use within an allocated compute job; API is an already available LLM server.
set -euo pipefail
MODE=${1:?audit|fixed|gate_timer|trial}
SEED=${2:?seed}
API=${3:?LLM_API_BASE}
HORIZON=${4:-1000000}
CHANNEL=${5:-shape}
cd "$(dirname "$0")/.."
export SC2PATH=${SC2PATH:-${HOME}/StarCraftII}
PY=${AAMAS_PYTHON:-python}
SCHED=vf
PERIOD=200
EVERY=${RSVP_VALIDATION_EVERY:-10}
TRIAL=False
MASK=False
BETA=0
case "$MODE" in
  audit) ;;
  fixed) SCHED=fixed; PERIOD=${FIXED_PERIOD:-200} ;;
  gate_timer) SCHED=gate_timer ;;
  trial) TRIAL=True; EVERY=1 ;;
  *) echo 'mode must be audit, fixed, gate_timer, or trial' >&2; exit 2 ;;
esac
case "$CHANNEL" in
  soft) MASK=True; BETA=0.1 ;;
  shape) ;;
  *) echo 'channel must be shape or soft' >&2; exit 2 ;;
esac
GROUP="${RSVP_EXPERIMENT_TAG:-rsvp_validation}_${MODE}_${CHANNEL}_F${PERIOD}_H${HORIZON}"
exec "$PY" main.py --config=rsvp --env-config=sc2 with \
  "seed=$SEED" "t_max=$HORIZON" env_args.map_name=2s3z \
  env_args.reward_sparse=False env_args.reward_scale=True epsilon_anneal_time=300000 \
  "scheduler=$SCHED" "f_update=$PERIOD" sched_trace=True \
  sched_gamma=0.8 sched_k=0.6 sched_h=3.0 sched_min_interval=10 sched_gate_interval=20 \
  sched_target_early_per_ep=0.0 \
  "rsvp_validation_every=$EVERY" "rsvp_refresh_trial=$TRIAL" rsvp_trial_probability=0.5 \
  rsvp_trial_window_episodes=5 rsvp_trial_control_probability=0.1 \
  rsvp_trial_max_per_kind=50 rsvp_trial_start_t=200000 \
  "use_action_masking=$MASK" soft_guidance_only=True use_masking_at_test=False "beta=$BETA" \
  shaping_in_learner=True lambda_start=0.5 lambda_min=0.05 \
  "lambda_floor_frac=$(awk -v h="$HORIZON" 'BEGIN {printf "%.12g", 400000/h}')" \
  lambda_zero_t=0 deduplicate_subgoals=True \
  prompt_style=paper commander_include_training_stats=False \
  llm_cache=False llm_temperature=0.2 "llm_api_base=$API" \
  test_interval=10000 test_nepisode=32 save_model=True save_model_interval=200000 \
  "use_wandb=${USE_WANDB:-False}" "wandb_group=$GROUP" "wandb_run=${GROUP}_seed${SEED}" "${@:6}"
