#!/usr/bin/env bash
set -euo pipefail

MODE=${1:?usage: run_s0b_backbone_guidance.sh none|rule SEED}
SEED=${2:?usage: run_s0b_backbone_guidance.sh none|rule SEED}
RUN_TAG=${RUN_TAG:-s0b3}
F_UPDATE=${F_UPDATE:-200}
T_MAX=${T_MAX:-1200000}
SAVE_INTERVAL=${SAVE_INTERVAL:-200000}
LAMBDA_FLOOR_FRAC=${LAMBDA_FLOOR_FRAC:-0.4}
case "$MODE" in
  none|rule) ;;
  *) echo "MODE must be none or rule" >&2; exit 2 ;;
esac

cd /gpfs/home1/paul6598/AAMAS
export SC2PATH=/gpfs/home1/paul6598/StarCraftII

# Keep the RSVP/fixed-schedule learning path identical across both arms. The rule arm
# changes only Commander output; masking remains disabled, so action_rules do
# not act on the policy. Neither arm calls an LLM.
exec /home1/paul6598/miniconda3/envs/aamas/bin/python main.py \
  --config=rsvp --env-config=sc2 with \
  results/diagnostics/rsvp_fixed200_20260907/config.json \
  "commander=$MODE" "seed=$SEED" \
  "f_update=$F_UPDATE" "t_max=$T_MAX" \
  "lambda_floor_frac=$LAMBDA_FLOOR_FRAC" \
  save_model=True "save_model_interval=$SAVE_INTERVAL" \
  "wandb_run=${RUN_TAG}_${MODE}_F${F_UPDATE}_lam40_rsvp_path"
