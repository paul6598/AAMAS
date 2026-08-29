#!/bin/bash
# QMIX baseline with the paper's common hyperparameters (Table 2).
# Usage: bash run_qmix.sh [MAP] [USE_WANDB] [GN]
# Extra overrides: EXTRA="lr=0.0005" bash run_qmix.sh 2s3z

cd ../
export SC2PATH=${SC2PATH:-/gpfs/home1/paul6598/StarCraftII}

MAP=$1
USE_WANDB=$2
GN=$3

if [ -z "$MAP" ]; then
    MAP="2s3z"
fi
if [ -z "$USE_WANDB" ]; then
    USE_WANDB=false
fi
if [ -z "$GN" ]; then
    GN="QMIX_${MAP}"
fi

case "$MAP" in
    2m_vs_1z)                 t_max=500000 ;;
    3s_vs_5z|5m_vs_6m|27m_vs_30m) t_max=5000000 ;;
    *)                        t_max=1000000 ;;
esac

test_interval=10000
test_nepisode=32

EXTRA_ARGS=()
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
    python main.py --config=qmix_paper --env-config=sc2 with "${ARGS[@]}"
done
