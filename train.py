"""Entry point: pick an algorithm and an environment.

  python train.py --algo ddqn    --env Foraging-5x5-2p-1f-v3
  python train.py --algo llm_mca --env Foraging-8x8-2p-2f-coop-v3
"""
import argparse
import random

import numpy as np
import torch

from algorithms import get_algorithm
from envs import make_env


def seed_everything(seed):
    """Seed policy initialization, exploration, and replay sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="llm_mca",
                   help="ddqn | rnn_iql | oracle | llm_mca | llm_taca | mappo")
    p.add_argument("--env", default="Foraging-8x8-2p-2f-coop-v3")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--episodes_per_iter", type=int, default=8)
    p.add_argument("--grad_steps", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=25, help="cap episode length (LLM context)")
    p.add_argument("--max_new_tokens", type=int, default=700)
    p.add_argument("--critic_group", type=int, default=0,
                   help="episodes per LLM critic call (0=whole batch; 1=one episode per call)")
    p.add_argument("--critic_parallel", type=int, default=1,
                   help="concurrent vLLM critic calls (keeps group-size 1 semantics while batching)")
    p.add_argument("--critic_chunk_steps", type=int, default=0,
                   help="score long trajectories in windows of this many steps (0=whole episode)")
    p.add_argument("--critic_retries", type=int, default=1,
                   help="retry malformed LLM credit outputs this many times, one episode per call")
    p.add_argument("--critic_compact", action="store_true",
                   help="request only credit arrays (no explanation) to reduce truncation/parse failures")
    p.add_argument("--critic_single_prompt", action="store_true",
                   help="use the paper Figure-3 single-trajectory prompt when critic_group=1")
    p.add_argument("--critic_structured", action="store_true",
                   help="use vLLM JSON-schema decoding to enforce exact credit-array lengths")
    p.add_argument("--critic_grounded_filter", action="store_true",
                   help="zero LLM credits that contradict unambiguous LBF state transitions")
    p.add_argument("--critic_lenient_arrays", action="store_true",
                   help="accept finite free-form credit arrays with non-exact lengths (paper-style regex path)")
    p.add_argument("--critic_array_length_policy", choices=["right", "left"], default="right",
                   help="how to align non-exact free-form credit arrays; left preserves a t=0 prefix")
    p.add_argument("--critic_paper_faithful", action="store_true",
                   help="omit local progress hints from the LBF prompt; use only paper-style observations/actions/rewards")
    p.add_argument("--critic_fallback", choices=["env", "zero"], default="env",
                   help="credit used after all format retries fail (env=equal-split global reward)")
    p.add_argument("--critic_trace", default=None,
                   help="optional JSONL file recording raw LLM critic outputs and parse validity")
    p.add_argument("--backbone", choices=["rnn", "mlp"], default="rnn",
                   help="decentralized policy backbone module for LLM-MCA/TACA (default: rnn)")
    p.add_argument("--train_device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="policy learner device; use cpu when a large vLLM critic occupies the GPU")
    p.add_argument("--critic_backend", choices=["hf", "vllm"], default="hf",
                   help="LLM critic backend: local transformers (hf) or a vLLM OpenAI server (vllm)")
    p.add_argument("--critic_api_base", default="http://localhost:8000/v1",
                   help="vLLM OpenAI-compatible endpoint (used when --critic_backend vllm)")
    p.add_argument("--credit_scale", type=float, default=1.0,
                   help="multiply the LLM credit before training (match paper's ~integer scale)")
    p.add_argument("--task_wean_frac", type=float, default=0.8,
                   help="llm_taca: fraction of training over which task input is weaned to zero")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--value_norm", action="store_true",
                   help="normalize RIQL value targets (off by default; changes sparse LBF dynamics)")
    p.add_argument("--rnn_lr", type=float, default=5e-4,
                   help="RIQL learning rate")
    p.add_argument("--rnn_hidden", type=int, default=64,
                   help="RIQL GRU hidden width")
    p.add_argument("--rnn_batch_size", type=int, default=16,
                   help="episodes sampled per RIQL gradient update")
    p.add_argument("--rnn_target_sync", type=int, default=200,
                   help="RIQL target-network synchronization interval in gradient updates")
    p.add_argument("--rnn_buffer_cap", type=int, default=5000,
                   help="maximum episodes retained by the RIQL replay buffer")
    p.add_argument("--rnn_separate", action="store_true",
                   help="use one recurrent Q network per agent instead of parameter sharing")
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eps_decay_frac", type=float, default=0.6,
                   help="fraction of training over which epsilon decays to its floor")
    # logging
    p.add_argument("--use_wandb", action="store_true", help="mirror metrics to Weights & Biases")
    p.add_argument("--wandb_project", default=None, help="wandb project (default: AAMAS_{env})")
    p.add_argument("--wandb_entity", default="joonhuk6598-university-of-seoul",
                   help="wandb team entity to log under")
    p.add_argument("--exp_name", default=None, help="wandb run name (default: {algo}_seed{seed})")
    p.add_argument("--group", default=None, help="wandb group (default: {algo})")
    # MAPPO-only hyperparameters (ignored by other algorithms)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--minibatch_size", type=int, default=256)
    return p


def main():
    args = build_argparser().parse_args()
    seed_everything(args.seed)
    if args.wandb_project is None:
        args.wandb_project = f"AAMAS_{args.env}"
    if args.exp_name is None:
        args.exp_name = f"{args.algo}_seed{args.seed}"
    if args.group is None:
        args.group = args.algo
    env = make_env(args.env, seed=args.seed, max_episode_steps=args.max_steps)
    eval_env = make_env(args.env, seed=args.seed + 10_000, max_episode_steps=args.max_steps)
    train = get_algorithm(args.algo)
    train(env, eval_env, args)


if __name__ == "__main__":
    main()
