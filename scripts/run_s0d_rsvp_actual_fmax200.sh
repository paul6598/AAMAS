#!/usr/bin/env bash
set -euo pipefail

SEED=${1:?usage: run_s0d_rsvp_actual_fmax200.sh SEED LLM_API_BASE}
LLM_API_BASE=${2:?usage: run_s0d_rsvp_actual_fmax200.sh SEED LLM_API_BASE}
T_MAX=${T_MAX:-300000}
F_MAX=${F_MAX:-200}
LAMBDA_FLOOR_FRAC=${LAMBDA_FLOOR_FRAC:-1.6}
SAVE_INTERVAL=${SAVE_INTERVAL:-100000}

cd /gpfs/home1/paul6598/AAMAS
export SC2PATH=/gpfs/home1/paul6598/StarCraftII

# Current RSVP (including Q-004 accounting fix), matched to the old actual-LLM
# Fmax200 arm except for code version and the 300k diagnostic horizon.
exec /home1/paul6598/miniconda3/envs/aamas/bin/python main.py \
  --config=rsvp --env-config=sc2 with \
  "seed=$SEED" "env_args.map_name=5m_vs_6m" \
  "t_max=$T_MAX" "f_update=$F_MAX" \
  commander=llm scheduler=vf \
  use_reward_shaping=True use_action_masking=False use_masking_at_test=False \
  shaping_in_learner=True \
  lambda_start=0.5 lambda_min=0.05 lambda_decay=0.9995 \
  "lambda_floor_frac=$LAMBDA_FLOOR_FRAC" shaping_clip=3.0 \
  sched_k=0.6 sched_h=3.0 sched_min_interval=10 sched_gamma=0.8 \
  sched_eps_frac=0.25 sched_warmup_episodes=20 \
  sched_trusted_weight_min=0.5 sched_target_early_per_ep=0.15 \
  sched_h_adapt_episodes=20 sched_fail_retry=5 sched_trace=False \
  critic_lr=0.001 critic_iters=50 critic_buffer=60000 \
  prompt_style=default llm_model=openai/gpt-oss-20b \
  "llm_api_base=$LLM_API_BASE" llm_temperature=0.2 llm_cache=False \
  epsilon_start=1.0 epsilon_finish=0.05 epsilon_anneal_time=50000 \
  test_interval=10000 test_nepisode=32 \
  save_model=True "save_model_interval=$SAVE_INTERVAL" \
  use_wandb=true wandb_run=s0d_rsvp_Fmax200_q004_nc_t02_lam40
