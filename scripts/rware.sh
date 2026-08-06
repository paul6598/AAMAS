#!/bin/bash

cd "$(dirname "$0")/.."

# 2. 인자 받아오기
ALGO=$1        # 예: ddqn, mappo, rnn_iql, llm_mca
ENV=$2         # 예: rware-tiny-2ag-v2
USE_WANDB=$3   # true or false
GROUP_NAME=$4  # wandb 그룹명 (미지정 시 알고리즘명)

# 3. 기본값 설정
if [ -z "$ALGO" ]; then ALGO="rnn_iql"; fi
if [ -z "$ENV" ]; then ENV="rware-tiny-2ag-v2"; fi
if [ -z "$USE_WANDB" ]; then USE_WANDB="false"; fi
if [ -z "$GROUP_NAME" ]; then GROUP_NAME="$ALGO"; fi

# 4. 하이퍼파라미터 설정
# 4-1. 공통
GRAD_STEPS=40
BATCH_SIZE=64
GAMMA=0.99
EVAL_EPISODES=10
LOG_EVERY=10

# 4-2. LLM 계열 (에피소드당 LLM 호출 1회 -> 작게, 호라이즌 짧게)
LLM_ITERATIONS=200
LLM_EPISODES_PER_ITER=4
LLM_MAX_STEPS=25
MODEL=Qwen/Qwen2.5-7B-Instruct
MAX_NEW_TOKENS=1500

# 4-3. 비-LLM 계열 (LLM 호출 없어 크게; RWARE는 sparse라 호라이즌 여유 필요)
RL_ITERATIONS=1000
RL_EPISODES_PER_ITER=16
RL_MAX_STEPS=200

if [ "$ALGO" == "llm_mca" ] || [ "$ALGO" == "llm_taca" ] || [ "$ALGO" == "rnn_mca" ]; then
    ITERATIONS=$LLM_ITERATIONS
    EPISODES_PER_ITER=$LLM_EPISODES_PER_ITER
    MAX_STEPS=$LLM_MAX_STEPS
else
    ITERATIONS=$RL_ITERATIONS
    EPISODES_PER_ITER=$RL_EPISODES_PER_ITER
    MAX_STEPS=$RL_MAX_STEPS
fi

LOG_DIR=results/rware
mkdir -p "$LOG_DIR"

EXTRA_ARGS=""
if [ "$ALGO" == "llm_mca" ] || [ "$ALGO" == "llm_taca" ] || [ "$ALGO" == "rnn_mca" ]; then
    EXTRA_ARGS="--model $MODEL --max_new_tokens $MAX_NEW_TOKENS"
fi
if [ "$USE_WANDB" == "true" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --use_wandb --group $GROUP_NAME"
fi

# 5. 시드별 루프 실행
for i in 0 1 2
do
    echo "Starting Experiment: $ALGO on $ENV with SEED $i"

    python train.py \
        --algo $ALGO \
        --env $ENV \
        --seed $i \
        --iterations $ITERATIONS \
        --episodes_per_iter $EPISODES_PER_ITER \
        --grad_steps $GRAD_STEPS \
        --max_steps $MAX_STEPS \
        --batch_size $BATCH_SIZE \
        --gamma $GAMMA \
        --eval_episodes $EVAL_EPISODES \
        --log_every $LOG_EVERY \
        $EXTRA_ARGS \
        2>&1 | tee "$LOG_DIR/${ALGO}_${ENV}_seed${i}.log"
done
