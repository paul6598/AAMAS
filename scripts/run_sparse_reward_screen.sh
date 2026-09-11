#!/usr/bin/env bash
set -euo pipefail

ARM=${1:?usage: run_sparse_reward_screen.sh qmix|full MAP SEED [LLM_API_BASE]}
MAP=${2:?usage: run_sparse_reward_screen.sh qmix|full MAP SEED [LLM_API_BASE]}
SEED=${3:?usage: run_sparse_reward_screen.sh qmix|full MAP SEED [LLM_API_BASE]}
LLM_API_BASE=${4:-http://localhost:8356/v1}

case "$MAP" in
  5m_vs_6m|2s3z) T_MAX=300000 ;;
  *) echo "unsupported map: $MAP" >&2; exit 2 ;;
esac
case "$ARM" in
  qmix) CONFIG=qmix_paper ;;
  full) CONFIG=lehca ;;
  *) echo "unsupported arm: $ARM" >&2; exit 2 ;;
esac

cd /gpfs/home1/paul6598/AAMAS
export SC2PATH=/gpfs/home1/paul6598/StarCraftII

GROUP="sparse_reward_20260911_${MAP}_${ARM}_unscaled"
RUN="${GROUP}_seed${SEED}"
ARGS=(
  "seed=$SEED" "env_args.map_name=$MAP" "t_max=$T_MAX"
  env_args.reward_sparse=True env_args.reward_scale=False
  epsilon_start=1.0 epsilon_finish=0.05 epsilon_anneal_time=300000
  test_interval=10000 test_nepisode=32
  save_model=True save_model_interval=100000
  use_wandb=True "wandb_group=$GROUP" "wandb_run=$RUN"
)

if [[ "$ARM" == "full" ]]; then
  ARGS+=(
    commander=llm f_update=50
    use_reward_shaping=True use_action_masking=True use_masking_at_test=True
    prompt_style=paper commander_include_training_stats=False
    test_guidance_mode=fresh dt_observable=True
    llm_model=openai/gpt-oss-20b "llm_api_base=$LLM_API_BASE"
    llm_temperature=0.2 llm_max_tokens=3072 llm_cache=False
  )
fi

exec /home1/paul6598/miniconda3/envs/aamas/bin/python main.py \
  "--config=$CONFIG" --env-config=sc2 with "${ARGS[@]}"
