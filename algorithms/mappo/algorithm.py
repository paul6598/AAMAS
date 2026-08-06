"""MAPPO (Multi-Agent PPO): a CTDE baseline, not a credit-assignment method.

Per-agent actors act on their local observation; a centralized critic estimates V(state) where
state is the concatenation of all agents' observations. Trained on the TRUE global reward with
GAE + the clipped PPO objective. This is on-policy, so it has its own trainer rather than the
shared train_with_critic loop.
"""
import numpy as np
import torch
import torch.nn as nn

from ..common.trainer import evaluate, linear_decay
from .networks import Actor, Critic, RecurrentActor, RecurrentCritic, ValueNormalizer
from utils.logger import Logger


def _concat_state(obs_tuple):
    return np.concatenate([np.asarray(o, dtype=np.float32) for o in obs_tuple])


def _gae(rewards, values, dones, last_value, gamma, lam):
    """Generalized Advantage Estimation for one episode. values has length T; last_value
    bootstraps the step after the episode (0 on a true terminal). Returns (adv[T], ret[T])."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for k in reversed(range(T)):
        next_value = last_value if k == T - 1 else values[k + 1]
        mask = 1.0 - float(dones[k])
        delta = rewards[k] + gamma * next_value * mask - values[k]
        gae = delta + gamma * lam * mask * gae
        adv[k] = gae
    return adv, adv + values


def train(env, eval_env, args):
    """MAPPO with a swappable policy/critic backbone: --backbone rnn (default, recurrent) or mlp."""
    if getattr(args, "backbone", "rnn") == "mlp":
        return _train_mlp(env, eval_env, args)
    return _train_rnn(env, eval_env, args)


def train_with_reward_critic(env, eval_env, args, reward_critic):
    """MAPPO whose on-policy advantages use centralized dense credit during training."""
    if getattr(args, "backbone", "rnn") == "mlp":
        return _train_mlp(env, eval_env, args, reward_critic=reward_critic)
    return _train_rnn(env, eval_env, args, reward_critic=reward_critic)


def _train_mlp(env, eval_env, args, reward_critic=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_agents, obs_dim, n_act = env.n_agents, env.obs_dim, env.n_actions
    state_dim = obs_dim * n_agents

    actors = [Actor(obs_dim, n_act, device).to(device) for _ in range(n_agents)]
    critic = Critic(state_dim).to(device)
    params = list(critic.parameters()) + [p for a in actors for p in a.parameters()]
    opt = torch.optim.Adam(params, lr=args.lr)
    logger = Logger(args.use_wandb, args.wandb_project, args.exp_name, args.group,
                    args.wandb_entity, vars(args))

    for it in range(args.iterations):
        # linear decay schedules (both logged, so the decay curves show up in wandb)
        lr = linear_decay(args.lr, 0.0, it, args.iterations)
        ent_coef = linear_decay(args.entropy_coef, 0.0, it, args.iterations)
        for g in opt.param_groups:
            g["lr"] = lr

        batch = env.collect_batch(actors, epsilon=1.0, n_episodes=args.episodes_per_iter)
        if reward_critic is None:
            reward_seqs = [np.asarray(t.global_reward, np.float32) for t in batch]
        else:
            decomposed, _ = reward_critic.assign_credits_batch(batch, env)
            reward_seqs = [np.asarray(c, np.float32).sum(axis=0) for c in decomposed]

        # Flatten the batch into per-timestep training tensors (advantages shared across agents).
        states, advs, rets = [], [], []
        obs_by_agent = [[] for _ in range(n_agents)]
        act_by_agent = [[] for _ in range(n_agents)]
        logp_by_agent = [[] for _ in range(n_agents)]
        ep_returns, shaped_returns = [], []

        for traj, rewards in zip(batch, reward_seqs):
            T = len(traj)
            ep_returns.append(sum(traj.global_reward))
            shaped_returns.append(float(rewards.sum()))
            ep_states = np.stack([_concat_state(traj.obs[k]) for k in range(T)])
            with torch.no_grad():
                st = torch.as_tensor(ep_states, dtype=torch.float32, device=device)
                values = critic(st).cpu().numpy()
                if traj.done[-1]:
                    last_value = 0.0
                else:
                    last_st = torch.as_tensor(_concat_state(traj.next_obs[-1]),
                                              dtype=torch.float32, device=device).unsqueeze(0)
                    last_value = float(critic(last_st).item())
            adv, ret = _gae(rewards, values, traj.done, last_value,
                            args.gamma, args.gae_lambda)
            states.append(ep_states)
            advs.append(adv)
            rets.append(ret)
            for i in range(n_agents):
                obs_i = np.stack([traj.obs[k][i] for k in range(T)])
                act_i = np.array([traj.actions[k][i] for k in range(T)], dtype=np.int64)
                with torch.no_grad():
                    lp, _ = actors[i].evaluate_actions(
                        torch.as_tensor(obs_i, dtype=torch.float32, device=device),
                        torch.as_tensor(act_i, dtype=torch.int64, device=device))
                obs_by_agent[i].append(obs_i)
                act_by_agent[i].append(act_i)
                logp_by_agent[i].append(lp.cpu().numpy())

        # concatenate across episodes
        states = torch.as_tensor(np.concatenate(states), dtype=torch.float32, device=device)
        advs = torch.as_tensor(np.concatenate(advs), dtype=torch.float32, device=device)
        rets = torch.as_tensor(np.concatenate(rets), dtype=torch.float32, device=device)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        obs_t = [torch.as_tensor(np.concatenate(obs_by_agent[i]), dtype=torch.float32, device=device)
                 for i in range(n_agents)]
        act_t = [torch.as_tensor(np.concatenate(act_by_agent[i]), dtype=torch.int64, device=device)
                 for i in range(n_agents)]
        logp_t = [torch.as_tensor(np.concatenate(logp_by_agent[i]), dtype=torch.float32, device=device)
                  for i in range(n_agents)]

        N = states.shape[0]
        p_losses, v_losses, entropies = [], [], []
        for _ in range(args.ppo_epochs):
            perm = torch.randperm(N, device=device)
            for start in range(0, N, args.minibatch_size):
                idx = perm[start:start + args.minibatch_size]
                policy_loss, entropy_term = 0.0, 0.0
                for i in range(n_agents):
                    new_lp, entropy = actors[i].evaluate_actions(obs_t[i][idx], act_t[i][idx])
                    ratio = torch.exp(new_lp - logp_t[i][idx])
                    surr1 = ratio * advs[idx]
                    surr2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * advs[idx]
                    policy_loss = policy_loss - torch.min(surr1, surr2).mean()
                    entropy_term = entropy_term + entropy.mean()
                value_loss = nn.functional.mse_loss(critic(states[idx]), rets[idx])
                loss = policy_loss - ent_coef * entropy_term + args.value_coef * value_loss
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 0.5)
                opt.step()
                p_losses.append(float(policy_loss)); v_losses.append(float(value_loss))
                entropies.append(float(entropy_term) / n_agents)

        metrics = {
            "train_return": float(np.mean(ep_returns)),
            "shaped_return": float(np.mean(shaped_returns)),
            "lr": float(lr),
            "entropy_coef": float(ent_coef),
            "policy_loss": float(np.mean(p_losses)),
            "value_loss": float(np.mean(v_losses)),
            "entropy": float(np.mean(entropies)),
            "steps": N,
        }
        if reward_critic is not None:
            metrics.update(getattr(reward_critic, "last_stats", {}))
        if it % args.log_every == 0 or it == args.iterations - 1:
            metrics["eval_return"] = evaluate(eval_env, actors, args.eval_episodes)
        logger.log(metrics, step=it)

    logger.finish()
    print("TRAIN_DONE", flush=True)
    return actors


def _pad(seqs, device, dtype):
    """seqs: list of np arrays [T_i, ...] (or [T_i]). Returns padded tensor [B, maxT, ...] and
    mask [B, maxT]."""
    B = len(seqs)
    maxT = max(s.shape[0] for s in seqs)
    tail = seqs[0].shape[1:] if seqs[0].ndim > 1 else ()
    out = np.zeros((B, maxT, *tail), dtype=seqs[0].dtype)
    mask = np.zeros((B, maxT), np.float32)
    for b, s in enumerate(seqs):
        out[b, : s.shape[0]] = s
        mask[b, : s.shape[0]] = 1.0
    return (torch.as_tensor(out, dtype=dtype, device=device),
            torch.as_tensor(mask, dtype=torch.float32, device=device))


def _train_rnn(env, eval_env, args, reward_critic=None):
    """Recurrent (GRU) actors + recurrent centralized critic + value normalization -- the
    paper-grade MAPPO. Fixes the MLP MAPPO's low ceiling (no memory) and high variance (no value
    normalization)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_agents, obs_dim, n_act = env.n_agents, env.obs_dim, env.n_actions
    state_dim = obs_dim * n_agents

    actors = [RecurrentActor(obs_dim, n_act, device).to(device) for _ in range(n_agents)]
    critic = RecurrentCritic(state_dim, device).to(device)
    vnorm = ValueNormalizer(device)
    params = list(critic.parameters()) + [p for a in actors for p in a.parameters()]
    opt = torch.optim.Adam(params, lr=args.lr)
    logger = Logger(args.use_wandb, args.wandb_project, args.exp_name, args.group,
                    args.wandb_entity, vars(args))

    for it in range(args.iterations):
        lr = linear_decay(args.lr, 0.0, it, args.iterations)
        ent_coef = linear_decay(args.entropy_coef, 0.0, it, args.iterations)
        for g in opt.param_groups:
            g["lr"] = lr

        batch = env.collect_batch(actors, epsilon=1.0, n_episodes=args.episodes_per_iter)
        if reward_critic is None:
            reward_seqs = [np.asarray(t.global_reward, np.float32) for t in batch]
        else:
            decomposed, _ = reward_critic.assign_credits_batch(batch, env)
            # Structural credits sum back to the task reward; potential terms add dense progress.
            reward_seqs = [np.asarray(c, np.float32).sum(axis=0) for c in decomposed]

        episodes, ep_returns, shaped_returns = [], [], []
        for traj, rewards in zip(batch, reward_seqs):
            T = len(traj)
            ep_returns.append(sum(traj.global_reward))
            shaped_returns.append(float(rewards.sum()))
            states = np.stack([_concat_state(traj.obs[k]) for k in range(T)])
            ext = np.concatenate([states, _concat_state(traj.next_obs[-1])[None]], 0)  # [T+1]
            with torch.no_grad():
                v_ext = vnorm.denormalize(
                    critic.forward_seq(torch.as_tensor(ext[None], dtype=torch.float32, device=device))
                )[0].cpu().numpy()
            values = v_ext[:T]
            last_value = 0.0 if traj.done[-1] else float(v_ext[T])
            adv, ret = _gae(rewards, values, traj.done, last_value,
                            args.gamma, args.gae_lambda)
            obs_i = [np.stack([traj.obs[k][i] for k in range(T)]) for i in range(n_agents)]
            act_i = [np.array([traj.actions[k][i] for k in range(T)], np.int64) for i in range(n_agents)]
            with torch.no_grad():
                old_lp = [actors[i].seq_logp_entropy(
                    torch.as_tensor(obs_i[i][None], dtype=torch.float32, device=device),
                    torch.as_tensor(act_i[i][None], dtype=torch.int64, device=device))[0][0].cpu().numpy()
                    for i in range(n_agents)]
            episodes.append({"states": states, "obs": obs_i, "act": act_i, "old_lp": old_lp,
                             "adv": adv, "ret": ret})

        all_adv = np.concatenate([e["adv"] for e in episodes])
        adv_mean, adv_std = all_adv.mean(), all_adv.std() + 1e-8
        vnorm.update(torch.as_tensor(np.concatenate([e["ret"] for e in episodes]),
                                     dtype=torch.float32, device=device))

        p_losses, v_losses, entropies = [], [], []
        E = len(episodes)
        ep_per_mb = max(1, args.minibatch_size // max(1, int(np.mean([len(e["adv"]) for e in episodes]))))
        for _ in range(args.ppo_epochs):
            order = np.random.permutation(E)
            for s in range(0, E, ep_per_mb):
                mb = [episodes[j] for j in order[s:s + ep_per_mb]]
                st, mask = _pad([e["states"] for e in mb], device, torch.float32)
                adv, _ = _pad([(e["adv"] - adv_mean) / adv_std for e in mb], device, torch.float32)
                ret, _ = _pad([e["ret"] for e in mb], device, torch.float32)
                m = mask
                policy_loss, entropy_term = 0.0, 0.0
                for i in range(n_agents):
                    obs_s, _ = _pad([e["obs"][i] for e in mb], device, torch.float32)
                    act_s, _ = _pad([e["act"][i] for e in mb], device, torch.int64)
                    olp_s, _ = _pad([e["old_lp"][i] for e in mb], device, torch.float32)
                    new_lp, ent = actors[i].seq_logp_entropy(obs_s, act_s)
                    ratio = torch.exp(new_lp - olp_s)
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv
                    policy_loss = policy_loss - (torch.min(surr1, surr2) * m).sum() / m.sum()
                    entropy_term = entropy_term + (ent * m).sum() / m.sum()
                pred = critic.forward_seq(st)
                v_target = vnorm.normalize(ret)
                value_loss = (((pred - v_target) ** 2) * m).sum() / m.sum()
                loss = policy_loss - ent_coef * entropy_term + args.value_coef * value_loss
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 0.5)
                opt.step()
                p_losses.append(float(policy_loss)); v_losses.append(float(value_loss))
                entropies.append(float(entropy_term) / n_agents)

        metrics = {
            "train_return": float(np.mean(ep_returns)), "lr": float(lr),
            "shaped_return": float(np.mean(shaped_returns)),
            "entropy_coef": float(ent_coef), "policy_loss": float(np.mean(p_losses)),
            "value_loss": float(np.mean(v_losses)), "entropy": float(np.mean(entropies)),
            "steps": int(sum(len(e["adv"]) for e in episodes)),
        }
        if reward_critic is not None:
            metrics.update(getattr(reward_critic, "last_stats", {}))
        if it % args.log_every == 0 or it == args.iterations - 1:
            metrics["eval_return"] = evaluate(eval_env, actors, args.eval_episodes)
        logger.log(metrics, step=it)

    logger.finish()
    print("TRAIN_DONE", flush=True)
    return actors
