"""LLM-TACA (paper Sec. 3.3): LLM-MCA plus explicit task assignment.

The critic returns, per agent, both per-timestep credit AND a recommended action. Each agent's
Double DQN policy is given an augmented input [observation, task_vec], where task_vec is a
[present_flag, one-hot(recommended action)] encoding. Following the paper, we wean the policy off
the task input over training by progressively overwriting it with zeros (task_keep_prob 1 -> 0),
so the decentralized policy (which sees task_vec = 0 at execution) does not depend on it.

Task usage is forward guidance: during collection each agent acts with task_vec set to the
critic's last recommended action (kept with probability task_keep_prob, which weans 1 -> 0 over
training), so the recommendation actively steers exploration. At evaluation the policy sees
task_vec = 0.
"""
import numpy as np
import torch

from ..common.ddqn import DDQNAgent
from ..common.trainer import epsilon_at, evaluate
from .critic import LLMTACACritic
from utils.logger import Logger


class TaskPolicy:
    """Wraps a DDQNAgent so env.rollout can call act(obs); appends the current task_vec.
    For forward guidance, `task` is set per episode during collection (the critic's recommendation)
    and reset to zero at evaluation."""

    def __init__(self, agent, task_dim):
        self.agent = agent
        self.task = np.zeros(task_dim, dtype=np.float32)

    def act(self, obs, epsilon):
        return self.agent.act(np.concatenate([obs, self.task]), epsilon)


def build_task_vec(action, n_actions):
    """[present_flag, one-hot(recommended action)]; zeros if no recommendation."""
    vec = np.zeros(1 + n_actions, dtype=np.float32)
    if action is not None:
        vec[0] = 1.0
        vec[1 + action] = 1.0
    return vec


def task_keep_prob(it, n_iters, wean_frac):
    """Fraction of task inputs kept (not zeroed) at iteration `it`. Linearly 1 -> 0 over the
    first `wean_frac` of training, then 0."""
    horizon = max(1, int(n_iters * wean_frac))
    return max(0.0, 1.0 - it / horizon)


def train(env, eval_env, args):
    """LLM-TACA with a swappable policy backbone: --backbone rnn (default) or mlp."""
    if getattr(args, "backbone", "rnn") == "mlp":
        return _train_mlp(env, eval_env, args)
    from ..rnn_iql.algorithm import train_taca         # rnn backbone (default)
    return train_taca(env, eval_env, args)


def _train_mlp(env, eval_env, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(args.seed)
    task_dim = 1 + env.n_actions
    agents = [DDQNAgent(env.obs_dim + task_dim, env.n_actions, device, gamma=args.gamma,
                        batch_size=args.batch_size, task_dim=task_dim) for _ in range(env.n_agents)]
    policies = [TaskPolicy(a, task_dim) for a in agents]
    critic = LLMTACACritic(model_name=args.model, max_new_tokens=args.max_new_tokens)
    logger = Logger(args.use_wandb, args.wandb_project, args.exp_name, args.group,
                    args.wandb_entity, vars(args))

    current_task = [None] * env.n_agents  # forward guidance: last LLM action recommendation

    for it in range(args.iterations):
        eps = epsilon_at(it, args.iterations)
        keep = task_keep_prob(it, args.iterations, args.task_wean_frac)
        # paper's controlled dropout on the task-input neurons, increasing 0 -> 0.9 as we wean off
        dropout = min(0.9, 1.0 - keep)
        for ag in agents:
            ag.task_dropout = dropout

        # collect one episode at a time so the (weaned) task recommendation actively guides the
        # policy's actions during collection -- forward guidance, not hindsight
        batch, ep_tvecs = [], []
        for _ in range(args.episodes_per_iter):
            tvecs = []
            for i in range(env.n_agents):
                use = current_task[i] is not None and np.random.random() < keep
                tv = build_task_vec(current_task[i] if use else None, env.n_actions)
                policies[i].task = tv
                tvecs.append(tv)
            batch.append(env.rollout(policies, eps))
            ep_tvecs.append(tvecs)

        all_credits, rec_actions, _ = critic.assign_batch(batch, env)
        current_task = [rec_actions[i] if rec_actions[i] is not None else current_task[i]
                        for i in range(env.n_agents)]  # update guidance
        n_targets = sum(a is not None for a in current_task)

        ep_returns, credit_totals = [], []
        for traj, credits, tvecs in zip(batch, all_credits, ep_tvecs):
            ep_returns.append(sum(traj.global_reward))
            credit_totals.append(float(np.abs(credits).sum()))
            for i in range(env.n_agents):
                for k in range(len(traj)):
                    agents[i].buffer.push(
                        np.concatenate([traj.obs[k][i], tvecs[i]]), int(traj.actions[k][i]),
                        float(credits[i, k]), np.concatenate([traj.next_obs[k][i], tvecs[i]]),
                        traj.done[k],
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
            "task_keep_prob": float(keep),
            "n_recommended": n_targets,
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "credit_total": float(np.mean(credit_totals)),
            "buffer_size": len(agents[0].buffer),
        }
        if it % args.log_every == 0 or it == args.iterations - 1:
            for p in policies:                      # execution: no task input
                p.task = np.zeros(task_dim, np.float32)
            metrics["eval_return"] = evaluate(eval_env, policies, args.eval_episodes)
        logger.log(metrics, step=it)

    logger.finish()
    print("TRAIN_DONE", flush=True)
    return agents
