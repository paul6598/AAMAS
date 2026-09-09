#!/usr/bin/env bash
set -euo pipefail

SEED=${1:?usage: run_s0f_fixed_actual_f25.sh SEED LLM_API_BASE}
LLM_API_BASE=${2:?usage: run_s0f_fixed_actual_f25.sh SEED LLM_API_BASE}
T_MAX=${T_MAX:-300000}
F_UPDATE=${F_UPDATE:-25}
LAMBDA_FLOOR_FRAC=${LAMBDA_FLOOR_FRAC:-1.6}
SAVE_INTERVAL=${SAVE_INTERVAL:-100000}

cd /gpfs/home1/paul6598/AAMAS
export SC2PATH=/gpfs/home1/paul6598/StarCraftII

# Scheduler-only control for s0d: same RSVP shaping path and actual LLM, with
# adaptive vf/Fmax200 replaced by fixed F25.
exec /home1/paul6598/miniconda3/envs/aamas/bin/python main.py \
  --config=rsvp --env-config=sc2 with \
  "seed=$SEED" "env_args.map_name=5m_vs_6m" \
  "t_max=$T_MAX" "f_update=$F_UPDATE" \
  commander=llm scheduler=fixed \
  use_reward_shaping=True use_action_masking=False use_masking_at_test=False \
  shaping_in_learner=True \
  lambda_start=0.5 lambda_min=0.05 lambda_decay=0.9995 \
  "lambda_floor_frac=$LAMBDA_FLOOR_FRAC" shaping_clip=3.0 \
  prompt_style=default llm_model=openai/gpt-oss-20b \
  "llm_api_base=$LLM_API_BASE" llm_temperature=0.2 llm_cache=False \
  epsilon_start=1.0 epsilon_finish=0.05 epsilon_anneal_time=50000 \
  test_interval=10000 test_nepisode=32 \
  save_model=True "save_model_interval=$SAVE_INTERVAL" \
  use_wandb=true wandb_run=s0f_fixed_F25_rsvp_path_nc_t02_lam40
