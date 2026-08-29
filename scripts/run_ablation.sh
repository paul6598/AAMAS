#!/bin/bash
# LEHCA ablations (Table 5): toggle semantic reward shaping / action masking.
# Usage: bash run_ablation.sh [MAP] [REWARD] [MASK] [USE_WANDB] [GN] [LLM_API_BASE] [LLM_MODEL]
#   REWARD, MASK: true/false

cd ./
MAP=$1
REWARD=$2
MASK=$3
USE_WANDB=$4
GN=$5
LLM_API_BASE=$6
LLM_MODEL=$7

if [ -z "$MAP" ]; then
    MAP="2s3z"
fi
if [ -z "$REWARD" ]; then
    REWARD=true
fi
if [ -z "$MASK" ]; then
    MASK=true
fi
if [ -z "$USE_WANDB" ]; then
    USE_WANDB=false
fi
if [ -z "$GN" ]; then
    GN="LEHCA_${MAP}_r${REWARD}_m${MASK}"
fi

EXTRA="use_reward_shaping=$REWARD use_action_masking=$MASK ${EXTRA}" \
    bash run_lehca.sh "$MAP" "$USE_WANDB" "$GN" llm "$LLM_API_BASE" "$LLM_MODEL"
