#!/usr/bin/env bash
set -euo pipefail

SEED=${1:?usage: run_s0b_shuffled_guidance.sh SEED POOL_JSONL}
POOL=${2:?usage: run_s0b_shuffled_guidance.sh SEED POOL_JSONL}
RUN_TAG=${RUN_TAG:-s0b3}
F_UPDATE=${F_UPDATE:-200}
T_MAX=${T_MAX:-1200000}
SAVE_INTERVAL=${SAVE_INTERVAL:-200000}
LAMBDA_FLOOR_FRAC=${LAMBDA_FLOOR_FRAC:-0.4}
cd /gpfs/home1/paul6598/AAMAS
export SC2PATH=/gpfs/home1/paul6598/StarCraftII

exec /home1/paul6598/miniconda3/envs/aamas/bin/python main.py \
  --config=rsvp --env-config=sc2 with \
  results/diagnostics/rsvp_fixed200_20260907/config.json \
  commander=shuffle "guidance_replay_path=$POOL" "seed=$SEED" \
  "f_update=$F_UPDATE" "t_max=$T_MAX" \
  "lambda_floor_frac=$LAMBDA_FLOOR_FRAC" \
  save_model=True "save_model_interval=$SAVE_INTERVAL" \
  "wandb_run=${RUN_TAG}_shuffle_F${F_UPDATE}_lam40_rsvp_path"
