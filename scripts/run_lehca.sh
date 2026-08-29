#!/bin/bash
# Usage: bash run_lehca.sh [MAP] [USE_WANDB] [GN] [COMMANDER] [LLM_API_BASE] [LLM_MODEL]
#   MAP          SMAC map name (default 2s3z)
#   USE_WANDB    true/false   (default false)
#   GN           wandb group name (default LEHCA_<MAP>)
#   COMMANDER    llm | rule | none  (default llm)
#   LLM_API_BASE OpenAI-compatible endpoint (default http://localhost:8355/v1)
#   LLM_MODEL    model id as served (default openai/gpt-oss-20b)
# Extra pymarl overrides can be appended via the EXTRA env var, e.g.
#   EXTRA="f_update=400 beta=1.0" bash run_lehca.sh 2s3z

cd ../
export SC2PATH=${SC2PATH:-/gpfs/home1/paul6598/StarCraftII}

MAP=$1
USE_WANDB=$2
GN=$3
COMMANDER=$4
LLM_API_BASE=$5
LLM_MODEL=$6

if [ -z "$MAP" ]; then # if MAP is not provided, set default
    MAP="2s3z"
fi
if [ -z "$USE_WANDB" ]; then
    USE_WANDB=false
fi
if [ -z "$GN" ]; then
    GN="LEHCA_${MAP}"
fi
if [ -z "$COMMANDER" ]; then
    COMMANDER="llm"
fi
if [ -z "$LLM_API_BASE" ]; then
    LLM_API_BASE="http://localhost:8355/v1"
fi
if [ -z "$LLM_MODEL" ]; then
    LLM_MODEL="openai/gpt-oss-20b"
fi

# Training budget per map (paper Figs. 2-5)
case "$MAP" in
    2m_vs_1z)                 t_max=500000 ;;
    3s_vs_5z|5m_vs_6m|27m_vs_30m) t_max=5000000 ;;
    *)                        t_max=1000000 ;;   # 3m, 8m, MMM, 2s3z
esac

test_interval=10000
test_nepisode=32

EXTRA_ARGS=()
if [ "$COMMANDER" != "null" ]; then
    EXTRA_ARGS+=("commander=$COMMANDER")
fi
if [ "$LLM_API_BASE" != "null" ]; then
    EXTRA_ARGS+=("llm_api_base=$LLM_API_BASE")
fi
if [ "$LLM_MODEL" != "null" ]; then
    EXTRA_ARGS+=("llm_model=$LLM_MODEL")
fi
if [ -n "$EXTRA" ]; then
    read -ra USER_EXTRA <<< "$EXTRA"
    EXTRA_ARGS+=("${USER_EXTRA[@]}")
fi

for SEED in 0 1 2 3 4
do
    ARGS=(
        "use_wandb=$USE_WANDB"
        "wandb_group=$GN"
        "seed=$SEED"
        "env_args.map_name=$MAP"
        "t_max=$t_max"
        "test_interval=$test_interval"
        "test_nepisode=$test_nepisode"
        "${EXTRA_ARGS[@]}"
    )
    python main.py --config=lehca --env-config=sc2 with "${ARGS[@]}"
done
