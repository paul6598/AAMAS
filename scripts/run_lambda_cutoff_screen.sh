#!/usr/bin/env bash
set -euo pipefail

ARM=${1:?usage: run_lambda_cutoff_screen.sh ARM SEED LLM_API_BASE}
SEED=${2:?usage: run_lambda_cutoff_screen.sh ARM SEED LLM_API_BASE}
LLM_API_BASE=${3:?usage: run_lambda_cutoff_screen.sh ARM SEED LLM_API_BASE}

case "$ARM" in
  fixed_floor)
    SCHEDULER=fixed
    F_UPDATE=50
    LAMBDA_ZERO_T=0
    ;;
  fixed_zero)
    SCHEDULER=fixed
    F_UPDATE=50
    LAMBDA_ZERO_T=300000
    ;;
  rsvp_zero)
    SCHEDULER=vf
    F_UPDATE=200
    LAMBDA_ZERO_T=300000
    ;;
  *) echo "unsupported arm: $ARM" >&2; exit 2 ;;
esac

cd /gpfs/home1/paul6598/AAMAS
export SC2PATH=/gpfs/home1/paul6598/StarCraftII

GROUP="lambda_cutoff_20260910_5m_vs_6m_${ARM}"
RUN="${GROUP}_seed${SEED}"

exec /home1/paul6598/miniconda3/envs/aamas/bin/python main.py \
  --config=rsvp --env-config=sc2 with \
  "seed=$SEED" env_args.map_name=5m_vs_6m t_max=1200000 \
  epsilon_start=1.0 epsilon_finish=0.05 epsilon_anneal_time=300000 \
  scheduler="$SCHEDULER" f_update="$F_UPDATE" \
  sched_k=0.6 sched_h=3.0 sched_min_interval=10 sched_gamma=0.8 \
  sched_eps_frac=0.25 sched_warmup_episodes=20 \
  sched_target_early_per_ep=0.15 sched_h_adapt_episodes=20 \
  use_reward_shaping=True shaping_in_learner=True \
  use_action_masking=False use_masking_at_test=False \
  lambda_start=0.5 lambda_min=0.05 lambda_decay=0.9995 \
  lambda_floor_frac=0 lambda_zero_t="$LAMBDA_ZERO_T" \
  prompt_style=paper commander_include_training_stats=False dt_observable=True \
  llm_model=openai/gpt-oss-20b "llm_api_base=$LLM_API_BASE" \
  llm_temperature=0.2 llm_max_tokens=3072 llm_cache=False \
  test_interval=10000 test_nepisode=32 \
  save_model=True save_model_interval=100000 \
  use_wandb=True "wandb_group=$GROUP" "wandb_run=$RUN"
