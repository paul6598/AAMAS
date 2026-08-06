"""Recurrent IQL training loop. Same critic interface as the MLP path, so it runs with the
equal-split critic (true reward -- the backbone gate test) or the LLM critic (rnn_mca)."""
import numpy as np
import torch

from ..common.rnn import RIQL, SeparateRIQL
from ..common.trainer import epsilon_at, evaluate
from ..ddqn.algorithm import EqualSplitCritic
from utils.logger import Logger


def train_rnn(env, eval_env, critic, args):
    requested_device = getattr(args, "train_device", "auto")
    device = ("cuda" if torch.cuda.is_available() else "cpu"
              if requested_device == "auto" else requested_device)
    if requested_device != "auto":
        device = requested_device
    controller_cls = SeparateRIQL if getattr(args, "rnn_separate", False) else RIQL
    ctrl = controller_cls(
        env.obs_dim,
        env.n_agents,
        env.n_actions,
        device,
        lr=getattr(args, "rnn_lr", 5e-4),
        gamma=args.gamma,
        hidden=getattr(args, "rnn_hidden", 64),
        target_sync=getattr(args, "rnn_target_sync", 200),
        buffer_cap=getattr(args, "rnn_buffer_cap", 5000),
        value_norm=getattr(args, "value_norm", False),
    )
    actors = ctrl.actors()
    # Shared RIQL actors append a global agent-id one-hot; separate learners append their
    # singleton id.  Reuse the actor's exact feature here so replay and rollout inputs agree.
    agent_features = [np.asarray(actor.id_onehot, dtype=np.float32) for actor in actors]
    logger = Logger(args.use_wandb, args.wandb_project, args.exp_name, args.group,
                    args.wandb_entity, vars(args))

    for it in range(args.iterations):
        eps = epsilon_at(it, args.iterations, decay_frac=args.eps_decay_frac)
        batch = env.collect_batch(actors, eps, args.episodes_per_iter)
        all_credits, _ = critic.assign_credits_batch(batch, env)
        scale = getattr(args, "credit_scale", 1.0)
        oracle_corrs = []
        if critic.__class__.__name__ == "LLMCritic":
            # Diagnostic only: never used as a training target.  It reveals when syntactically
            # valid LLM arrays cease to track concrete directional progress as the policy shifts.
            from ..common.oracle import OracleDenseCritic
            oracle_credits, _ = OracleDenseCritic(gamma=args.gamma).assign_credits_batch(batch, env)
            for llm_credit, oracle_credit in zip(all_credits, oracle_credits):
                for i in range(env.n_agents):
                    x = np.asarray(llm_credit[i], dtype=np.float32)
                    y = np.asarray(oracle_credit[i], dtype=np.float32)
                    if x.std() > 1e-8 and y.std() > 1e-8:
                        oracle_corrs.append(float(np.corrcoef(x, y)[0, 1]))

        ep_returns, credit_totals = [], []
        for traj, credits in zip(batch, all_credits):
            credits = credits * scale       # match the paper's integer-scale credit (~1-5/step)
            ep_returns.append(sum(traj.global_reward))
            credit_totals.append(float(np.abs(credits).sum()))
            T = len(traj)
            obs_seq = np.zeros((env.n_agents, T, ctrl.input_dim), np.float32)
            act_seq = np.zeros((env.n_agents, T), np.int64)
            for i in range(env.n_agents):
                for k in range(T):
                    obs_seq[i, k] = np.concatenate([traj.obs[k][i], agent_features[i]])
                    act_seq[i, k] = int(traj.actions[k][i])
            ctrl.push_episode(obs_seq, act_seq, credits, np.array(traj.done, np.float32))

        rnn_batch_size = getattr(args, "rnn_batch_size", 16)
        losses = [ctrl.update(batch_size=rnn_batch_size) for _ in range(args.grad_steps)]
        losses = [x for x in losses if x is not None]

        metrics = {
            "train_return": float(np.mean(ep_returns)),
            "epsilon": float(eps),
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "credit_total": float(np.mean(credit_totals)),
            "buffer_episodes": len(ctrl.buffer),
        }
        metrics.update(getattr(critic, "last_stats", {}))
        if oracle_corrs:
            metrics["critic_oracle_corr"] = float(np.mean(oracle_corrs))
        if it % args.log_every == 0 or it == args.iterations - 1:
            metrics["eval_return"] = evaluate(eval_env, actors, args.eval_episodes)
        logger.log(metrics, step=it)

    logger.finish()
    print("TRAIN_DONE", flush=True)
    return ctrl


def train(env, eval_env, args):
    """rnn_iql: recurrent IQL on the true reward (equal split) -- the backbone gate test."""
    return train_rnn(env, eval_env, EqualSplitCritic(), args)


def train_oracle(env, eval_env, args):
    """rnn_oracle: recurrent IQL trained on the oracle DENSE credit (diagnostic upper bound for
    densification -- see algorithms/common/oracle.py)."""
    from ..common.oracle import OracleDenseCritic
    return train_rnn(env, eval_env, OracleDenseCritic(gamma=args.gamma), args)


def train_mca(env, eval_env, args):
    """rnn_mca: recurrent IQL trained on the LLM critic's per-agent credit (LLM-MCA on the RNN
    backbone)."""
    from ..llm_mca.critic import LLMCritic
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
    return train_rnn(env, eval_env, critic, args)


def train_taca(env, eval_env, args):
    """rnn_taca: LLM-TACA (forward guidance + task-input weaning) on the recurrent IQL backbone --
    the RNN counterpart of llm_taca, so TACA is judged on a backbone that isn't frozen (like MLP)."""
    from ..common.rnn import RIQL
    from ..llm_taca.critic import LLMTACACritic
    from ..llm_taca.algorithm import build_task_vec, task_keep_prob

    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(args.seed)
    task_dim = 1 + env.n_actions
    ctrl = RIQL(env.obs_dim, env.n_agents, env.n_actions, device, gamma=args.gamma,
                task_dim=task_dim, value_norm=getattr(args, "value_norm", False))
    actors = ctrl.actors()
    eye = np.eye(env.n_agents, dtype=np.float32)
    critic = LLMTACACritic(model_name=args.model, max_new_tokens=args.max_new_tokens,
                           group_size=getattr(args, "critic_group", 0),
                           backend=getattr(args, "critic_backend", "hf"),
                           api_base=getattr(args, "critic_api_base", "http://localhost:8000/v1"))
    logger = Logger(args.use_wandb, args.wandb_project, args.exp_name, args.group,
                    args.wandb_entity, vars(args))

    current_task = [None] * env.n_agents
    for it in range(args.iterations):
        eps = epsilon_at(it, args.iterations, decay_frac=args.eps_decay_frac)
        keep = task_keep_prob(it, args.iterations, args.task_wean_frac)
        ctrl.task_dropout = min(0.9, 1.0 - keep)

        batch, ep_tvecs = [], []
        for _ in range(args.episodes_per_iter):
            tvecs = []
            for i in range(env.n_agents):
                use = current_task[i] is not None and np.random.random() < keep
                tv = build_task_vec(current_task[i] if use else None, env.n_actions)
                actors[i].task = tv
                tvecs.append(tv)
            batch.append(env.rollout(actors, eps))
            ep_tvecs.append(tvecs)

        all_credits, rec_actions, _ = critic.assign_batch(batch, env)
        current_task = [rec_actions[i] if rec_actions[i] is not None else current_task[i]
                        for i in range(env.n_agents)]
        n_targets = sum(a is not None for a in current_task)

        ep_returns, credit_totals = [], []
        for traj, credits, tvecs in zip(batch, all_credits, ep_tvecs):
            ep_returns.append(sum(traj.global_reward))
            credit_totals.append(float(np.abs(credits).sum()))
            T = len(traj)
            obs_seq = np.zeros((env.n_agents, T, ctrl.input_dim), np.float32)
            act_seq = np.zeros((env.n_agents, T), np.int64)
            for i in range(env.n_agents):
                for k in range(T):
                    obs_seq[i, k] = np.concatenate([traj.obs[k][i], eye[i], tvecs[i]])
                    act_seq[i, k] = int(traj.actions[k][i])
            ctrl.push_episode(obs_seq, act_seq, credits, np.array(traj.done, np.float32))

        losses = [x for x in (ctrl.update() for _ in range(args.grad_steps)) if x is not None]
        metrics = {
            "train_return": float(np.mean(ep_returns)), "epsilon": float(eps),
            "task_keep_prob": float(keep), "n_recommended": n_targets,
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "credit_total": float(np.mean(credit_totals)), "buffer_episodes": len(ctrl.buffer),
        }
        if it % args.log_every == 0 or it == args.iterations - 1:
            for a in actors:                       # execution: no task input
                a.task = np.zeros(task_dim, np.float32)
            metrics["eval_return"] = evaluate(eval_env, actors, args.eval_episodes)
        logger.log(metrics, step=it)

    logger.finish()
    print("TRAIN_DONE", flush=True)
    return ctrl
