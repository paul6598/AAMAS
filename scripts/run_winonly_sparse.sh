#!/usr/bin/env bash
# 이미 준비된 API를 사용해 win-only 2s3z의 full/soft 채널을 재실행한다.
# Usage: bash scripts/run_winonly_sparse.sh qmix|lehca|rsvp|fixed50|fixed200 SEED [API] [HORIZON] [GROUP]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ARM=${1:?qmix|lehca|rsvp|fixed50|fixed200}
SEED=${2:?seed}
API=${3:-http://localhost:8355/v1}
HORIZON=${4:-300000}
GROUP=${5:-winonly_2s3z_${ARM}_H${HORIZON}}
PY=${AAMAS_PYTHON:-python}
export SC2PATH=${SC2PATH:-${HOME}/StarCraftII}

ARGS=(
  "seed=$SEED" env_args.map_name=2s3z "t_max=$HORIZON"
  env_args.reward_sparse=True env_args.reward_scale=False env_args.sparse_win_only=True
  epsilon_start=1.0 epsilon_finish=0.05 epsilon_anneal_time=300000
  test_interval=10000 test_nepisode=32 save_model=True save_model_interval=100000
  "use_wandb=${USE_WANDB:-False}" "wandb_group=$GROUP" "wandb_run=${GROUP}_seed${SEED}"
)
case "$ARM" in
  qmix)
    exec "$PY" main.py --config=qmix_paper --env-config=sc2 with "${ARGS[@]}"
    ;;
  lehca)
    exec "$PY" main.py --config=lehca --env-config=sc2 with "${ARGS[@]}" \
      commander=llm f_update=50 use_reward_shaping=True \
      use_action_masking=True use_masking_at_test=True shaping_in_learner=False \
      lambda_floor_frac=0 lambda_zero_t=0 beta=0.1 deduplicate_subgoals=True \
      prompt_style=paper commander_include_training_stats=False test_guidance_mode=fresh \
      dt_observable=True llm_model=openai/gpt-oss-20b "llm_api_base=$API" \
      llm_temperature=0.2 llm_max_tokens=3072 llm_cache=False
    ;;
  rsvp|fixed50|fixed200)
    MODE=audit
    if [[ "$ARM" != rsvp ]]; then
      MODE=fixed
      export FIXED_PERIOD=${ARM#fixed}
    fi
    exec bash scripts/run_rsvp_validation.sh "$MODE" "$SEED" "$API" "$HORIZON" soft "${ARGS[@]}"
    ;;
  *) echo "unsupported arm: $ARM" >&2; exit 2 ;;
esac
