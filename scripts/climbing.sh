#!/bin/bash

cd "$(dirname "$0")/.."

# 2. 인자 받아오기
ALGO=$1        # ddqn, mappo, llm_mca, llm_taca, rnn_iql, rnn_mca
USE_WANDB=$2   # true or false
GROUP_NAME=$3  # wandb 그룹명 (미지정 시 알고리즘명)

# 3. 기본값 설정
if [ -z "$ALGO" ]; then ALGO="llm_mca"; fi
if [ -z "$USE_WANDB" ]; then USE_WANDB="false"; fi
if [ -z "$GROUP_NAME" ]; then GROUP_NAME="$ALGO"; fi

ENV=Climbing

# 4. 하이퍼파라미터 (climbing: dense reward, 25-step matrix game)
MAX_STEPS=25
BATCH_SIZE=64
GAMMA=0.99
EVAL_EPISODES=20
LOG_EVERY=5

# 4-2. LLM 계열 (에피소드당 LLM 배치 콜)
LLM_ITERATIONS=60
LLM_EPISODES_PER_ITER=8
GRAD_STEPS=60
MODEL=Qwen/Qwen2.5-7B-Instruct
MAX_NEW_TOKENS=1500

# 4-3. 비-LLM 계열 (LLM 없어 크게)
RL_ITERATIONS=400
RL_EPISODES_PER_ITER=32

if [ "$ALGO" == "llm_mca" ] || [ "$ALGO" == "llm_taca" ] || [ "$ALGO" == "rnn_mca" ]; then
    ITERATIONS=$LLM_ITERATIONS
    EPISODES_PER_ITER=$LLM_EPISODES_PER_ITER
else
    ITERATIONS=$RL_ITERATIONS
    EPISODES_PER_ITER=$RL_EPISODES_PER_ITER
fi

LOG_DIR=results/climbing
mkdir -p "$LOG_DIR"

EXTRA_ARGS=""
if [ "$ALGO" == "llm_mca" ] || [ "$ALGO" == "llm_taca" ] || [ "$ALGO" == "rnn_mca" ]; then
    EXTRA_ARGS="--model $MODEL --max_new_tokens $MAX_NEW_TOKENS"
fi
if [ "$USE_WANDB" == "true" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --use_wandb --group $GROUP_NAME"
fi

# 5. 시드별 루프
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
        2>&1 | tee "$LOG_DIR/${ALGO}_seed${i}.log"
done
