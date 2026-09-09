#!/usr/bin/env bash
set -euo pipefail

SEED=${1:?usage: run_s0e_qmix_paper_5m.sh SEED}
T_MAX=${T_MAX:-5000000}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000000}

cd /gpfs/home1/paul6598/AAMAS
export SC2PATH=/gpfs/home1/paul6598/StarCraftII

# Full paper-budget QMIX audit for 5m_vs_6m.  qmix_paper.yaml is the frozen
# Table-2-compatible backbone used by LEHCA comparisons.
exec /home1/paul6598/miniconda3/envs/aamas/bin/python main.py \
  --config=qmix_paper --env-config=sc2 with \
  "seed=$SEED" "env_args.map_name=5m_vs_6m" "t_max=$T_MAX" \
  test_interval=10000 test_nepisode=32 \
  save_model=True "save_model_interval=$SAVE_INTERVAL" \
  use_wandb=true wandb_run=s0e_qmix_paper_5M_budget_audit
