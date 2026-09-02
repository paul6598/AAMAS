#!/bin/bash
# Usage: bash run_vigil_sc2.sh [MAP] [SCHEDULER] [USE_WANDB] [GN] [LLM_API_BASE]
#   MAP          SMAC map (default MMM2)
#   SCHEDULER    fixed | vf   (default vf)
#   USE_WANDB    true/false   (default true)
#   RUN          wandb run/group name: 알고리즘_(디테일)  (default <SCHEDULER>)
#   LLM_API_BASE OpenAI-compatible endpoint (default http://localhost:8356/v1)
# SEEDS="1" 처럼 시드 지정 가능 (기본 0 1 2). Extra pymarl overrides via EXTRA, e.g. EXTRA="f_update=200 sched_h=6" bash run_vigil_sc2.sh

cd ../
export SC2PATH=${SC2PATH:-/gpfs/home1/paul6598/StarCraftII}

MAP=${1:-MMM2}
SCHEDULER=${2:-vf}
USE_WANDB=${3:-true}
RUN=${4:-${SCHEDULER}}
LLM_API_BASE=${5:-http://localhost:8356/v1}

case "$MAP" in
    3s_vs_5z|5m_vs_6m|27m_vs_30m|MMM2) t_max=2000000 ;;
    *)                                 t_max=1000000 ;;
esac

EXTRA_ARGS=()
if [ -n "$EXTRA" ]; then read -ra USER_EXTRA <<< "$EXTRA"; EXTRA_ARGS+=("${USER_EXTRA[@]}"); fi

for SEED in ${SEEDS:-0 1 2}
do
    python main.py --config=vigil --env-config=sc2 with \
        "use_wandb=$USE_WANDB" "wandb_run=$RUN" "seed=$SEED" \
        "env_args.map_name=$MAP" "t_max=$t_max" \
        "test_interval=10000" "test_nepisode=32" \
        "scheduler=$SCHEDULER" "llm_api_base=$LLM_API_BASE" \
        "${EXTRA_ARGS[@]}"
done
