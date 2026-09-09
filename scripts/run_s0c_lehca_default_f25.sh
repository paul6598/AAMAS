#!/usr/bin/env bash
set -euo pipefail

SEED=${1:?usage: run_s0c_lehca_default_f25.sh SEED LLM_API_BASE}
LLM_API_BASE=${2:?usage: run_s0c_lehca_default_f25.sh SEED LLM_API_BASE}
T_MAX=${T_MAX:-300000}
F_UPDATE=${F_UPDATE:-25}
LAMBDA_FLOOR_FRAC=${LAMBDA_FLOOR_FRAC:-1.6}
SAVE_INTERVAL=${SAVE_INTERVAL:-100000}

cd /gpfs/home1/paul6598/AAMAS
export SC2PATH=/gpfs/home1/paul6598/StarCraftII

# Matched to the completed LEHCA-original-style run (Sacred 244), except for
# F_update and the diagnostic horizon.  floor_frac=1.6 at 300k preserves the
# same absolute lambda schedule as floor_frac=0.4 at 1.2M (floor at 480k).
exec /home1/paul6598/miniconda3/envs/aamas/bin/python main.py \
  --config=lehca --env-config=sc2 with \
  "seed=$SEED" "env_args.map_name=5m_vs_6m" \
  "t_max=$T_MAX" "f_update=$F_UPDATE" \
  commander=llm \
  use_reward_shaping=True use_action_masking=True use_masking_at_test=True \
  beta=0.1 shaping_in_learner=False \
  lambda_start=0.5 lambda_min=0.05 lambda_decay=0.9995 \
  "lambda_floor_frac=$LAMBDA_FLOOR_FRAC" shaping_clip=3.0 \
  prompt_style=default llm_model=openai/gpt-oss-20b \
  "llm_api_base=$LLM_API_BASE" llm_temperature=0.2 llm_cache=False \
  epsilon_start=1.0 epsilon_finish=0.05 epsilon_anneal_time=50000 \
  test_interval=10000 test_nepisode=32 \
  save_model=True "save_model_interval=$SAVE_INTERVAL" \
  use_wandb=true wandb_run=s0c3_lehca_default_F25_b01_nc_t02_lam40
