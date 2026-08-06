"""Shared CTDE training loop: collect trajectories, let a critic decompose the global reward
into per-agent credit, and train one Double DQN per agent on its credit.

Algorithms differ only in the critic they inject (equal-split baseline, LLM critic, ...).
"""
import numpy as np
import torch

from .ddqn import DDQNAgent
from utils.logger import Logger


def epsilon_at(it, n_iters, eps_start=1.0, eps_end=0.05, decay_frac=0.6):
    frac = min(1.0, it / max(1, int(n_iters * decay_frac)))
    return eps_start + frac * (eps_end - eps_start)


def linear_decay(start, end, it, n_iters, decay_frac=1.0):
    """Linearly interpolate start -> end over the first `decay_frac` of training, then hold at
    end. Reusable schedule (lr, entropy coefficient, ...)."""
    frac = min(1.0, it / max(1, int(n_iters * decay_frac)))
    return start + frac * (end - start)


@torch.no_grad()
def evaluate(env, agents, n_episodes=5):
    # Evaluate on the same held-out layouts at every checkpoint. Besides reducing curve noise,
    # restoring the episode counter prevents logging frequency from changing the training run.
    start_ep = getattr(env, "_ep", None)
    returns = []
    for _ in range(n_episodes):
        traj = env.rollout(agents, epsilon=0.0)
        returns.append(sum(traj.global_reward))
    if start_ep is not None:
        env._ep = start_ep
    return float(np.mean(returns))


def train_with_critic(env, eval_env, critic, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agents = [DDQNAgent(env.obs_dim, env.n_actions, device, gamma=args.gamma,
                        batch_size=args.batch_size) for _ in range(env.n_agents)]
    logger = Logger(args.use_wandb, args.wandb_project, args.exp_name, args.group,
                    args.wandb_entity, vars(args))

    for it in range(args.iterations):
        eps = epsilon_at(it, args.iterations, decay_frac=args.eps_decay_frac)
        batch = env.collect_batch(agents, eps, args.episodes_per_iter)

        # one critic call for the whole batch, so an LLM critic can compare episodes
        all_credits, _ = critic.assign_credits_batch(batch, env)
        ep_returns, credit_totals = [], []
        for traj, credits in zip(batch, all_credits):
            ep_returns.append(sum(traj.global_reward))
            credit_totals.append(float(np.abs(credits).sum()))
            for i in range(env.n_agents):
                for k in range(len(traj)):
                    agents[i].buffer.push(
                        traj.obs[k][i], int(traj.actions[k][i]), float(credits[i, k]),
                        traj.next_obs[k][i], traj.done[k],
                    )

        losses = []
        for _ in range(args.grad_steps):
            for ag in agents:
                loss = ag.update()
                if loss is not None:
                    losses.append(loss)

        metrics = {
            "train_return": float(np.mean(ep_returns)),
            "epsilon": float(eps),
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "credit_total": float(np.mean(credit_totals)),
            "buffer_size": len(agents[0].buffer),
        }
        metrics.update(getattr(critic, "last_stats", {}))
        if it % args.log_every == 0 or it == args.iterations - 1:
            metrics["eval_return"] = evaluate(eval_env, agents, args.eval_episodes)
        logger.log(metrics, step=it)

    logger.finish()
    print("TRAIN_DONE", flush=True)
    return agents
