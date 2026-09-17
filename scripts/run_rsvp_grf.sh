#!/bin/bash
# Usage: bash run_rsvp_grf.sh [SCHEDULER] [USE_WANDB] [GN] [LLM_API_BASE]
#   SCHEDULER    fixed | vf   (default vf)
#   USE_WANDB    true/false   (default true)
#   RUN          wandb run/group name: 알고리즘_(디테일)  (default <SCHEDULER>)
#   LLM_API_BASE OpenAI-compatible endpoint (default http://localhost:8356/v1)
# Requires an env with gfootball (conda grf: export LD_LIBRARY_PATH=$CONDA_PREFIX/lib).
# Extra overrides via EXTRA, e.g. EXTRA="sched_h=6 env_args.right_difficulty=0.6"
set -eo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
PY=${AAMAS_PYTHON:-python}

SCHEDULER=${1:-vf}
USE_WANDB=${2:-true}
RUN=${3:-rsvp_${SCHEDULER}}
LLM_API_BASE=${4:-http://localhost:8356/v1}

EXTRA_ARGS=()
if [ -n "$EXTRA" ]; then read -ra USER_EXTRA <<< "$EXTRA"; EXTRA_ARGS+=("${USER_EXTRA[@]}"); fi

for SEED in ${SEEDS:-0 1 2}
do
    "$PY" main.py --config=rsvp --env-config=gfootball with \
        "use_wandb=$USE_WANDB" "wandb_group=$RUN" "wandb_run=${RUN}_seed${SEED}" "seed=$SEED" \
        "t_max=500000" "test_interval=20000" "test_nepisode=4" "buffer_size=600" \
        "scheduler=$SCHEDULER" "llm_api_base=$LLM_API_BASE" \
        "${EXTRA_ARGS[@]}"
done
