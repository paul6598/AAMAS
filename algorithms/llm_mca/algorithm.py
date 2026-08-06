"""LLM-MCA (the paper's method): an LLM centralized critic decomposes each episode's global
reward into per-agent credit, and decentralized policies train on that credit.

The decentralized policy backbone is a swappable module: `--backbone rnn` (default, recurrent
IQL) or `--backbone mlp` (feed-forward Double DQN). Same LLM-MCA method either way.
"""
from .critic import LLMCritic


def train(env, eval_env, args):
    if getattr(args, "backbone", "rnn") == "mlp":
        from ..common.trainer import train_with_critic
        critic = LLMCritic(model_name=args.model, max_new_tokens=args.max_new_tokens,
                       group_size=getattr(args, "critic_group", 0),
                       parallel_requests=getattr(args, "critic_parallel", 1),
                       chunk_steps=getattr(args, "critic_chunk_steps", 0),
                           max_retries=getattr(args, "critic_retries", 1),
                           compact_output=getattr(args, "critic_compact", False),
                           single_prompt=getattr(args, "critic_single_prompt", False),
                       structured_output=getattr(args, "critic_structured", False),
                       grounded_filter=getattr(args, "critic_grounded_filter", False),
                       lenient_arrays=getattr(args, "critic_lenient_arrays", False),
                       array_length_policy=getattr(args, "critic_array_length_policy", "right"),
                       paper_faithful=getattr(args, "critic_paper_faithful", False),
                       fallback=getattr(args, "critic_fallback", "env"),
                           backend=getattr(args, "critic_backend", "hf"),
                           api_base=getattr(args, "critic_api_base", "http://localhost:8000/v1"),
                           trace_path=getattr(args, "critic_trace", None))
        return train_with_critic(env, eval_env, critic, args)
    from ..rnn_iql.algorithm import train_mca      # rnn backbone (default)
    return train_mca(env, eval_env, args)
