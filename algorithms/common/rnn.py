"""Recurrent Independent Q-Learning backbone (EPyMARL-style): a shared GRU Q-network with an
agent-id input and per-agent hidden state, trained on whole episodes with BPTT. The recurrence
gives each agent memory of the unfolding episode, which is what lets independent learners
implicitly coordinate on Level-Based Foraging (who goes to which apple) -- something a memoryless
MLP cannot do.

Trains on per-agent per-timestep credit, so it plugs into the same critic interface (equal-split
= true reward, or the LLM critic) as the MLP DDQN.
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNQNet(nn.Module):
    def __init__(self, input_dim, n_actions, hidden=64, aux_classes=0):
        super().__init__()
        self.hidden = hidden
        self.aux_classes = aux_classes
        self.fc1 = nn.Linear(input_dim, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.aux_head = nn.Linear(hidden, aux_classes) if aux_classes else None
        # When enabled, Q explicitly consumes its own differentiable subgoal prediction.
        self.fc2 = nn.Linear(hidden + aux_classes, n_actions)

    def forward(self, x, h, return_aux=False):
        z = torch.relu(self.fc1(x))
        h = self.gru(z, h)
        aux = self.aux_head(h) if self.aux_head is not None else None
        features = torch.cat([h, torch.softmax(aux, dim=-1)], dim=-1) if aux is not None else h
        q = self.fc2(features)
        if return_aux:
            return q, h, aux
        return q, h

    def init_hidden(self, batch, device):
        return torch.zeros(batch, self.hidden, device=device)


class RecurrentActor:
    """Per-agent rollout wrapper: holds this agent's hidden state and id, acts via the shared net.
    env.rollout calls reset() at each episode start (duck-typed)."""

    def __init__(self, controller, agent_idx, n_agents, task_dim=0):
        self.ctrl = controller
        self.id_onehot = np.eye(n_agents, dtype=np.float32)[agent_idx]
        self.task = np.zeros(task_dim, dtype=np.float32)  # LLM-TACA: per-episode task input (0 = none)
        self.h = None

    def reset(self):
        self.h = None

    def act(self, obs, epsilon):
        x = np.concatenate([np.asarray(obs, dtype=np.float32), self.id_onehot, self.task])
        q, self.h = self.ctrl.act_step(x, self.h)
        # Do not consume the training RNG during greedy evaluation.
        if epsilon > 0.0 and random.random() < epsilon:
            return random.randrange(self.ctrl.n_actions)
        return int(np.argmax(q))


class RIQL:
    """Shared recurrent Q backbone + episode buffer + BPTT update."""

    def __init__(self, obs_dim, n_agents, n_actions, device, lr=5e-4, gamma=0.99,
                 hidden=64, target_sync=200, buffer_cap=5000, task_dim=0,
                 aux_classes=0, aux_coef=0.2, value_norm=False):
        self.device = device
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gamma = gamma
        # LLM-TACA: last `task_dim` input dims are the task assignment; `task_dropout` zeros them
        # during updates with rising probability as the policy is weaned off the task.
        self.task_dim = task_dim
        self.task_dropout = 0.0
        self.aux_classes = aux_classes
        self.aux_coef = aux_coef
        self.use_value_norm = value_norm
        self.last_aux_loss = float("nan")
        self.last_aux_accuracy = float("nan")
        self.input_dim = obs_dim + n_agents + task_dim
        self.online = RNNQNet(self.input_dim, n_actions, hidden, aux_classes).to(device)
        self.target = RNNQNet(self.input_dim, n_actions, hidden, aux_classes).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)
        if value_norm:
            from .normalizer import ValueNormalizer
            self.vnorm = ValueNormalizer(device)
        else:
            self.vnorm = None
        self.buffer = deque(maxlen=buffer_cap)
        self.target_sync = target_sync
        self._updates = 0

    def actors(self):
        return [RecurrentActor(self, i, self.n_agents, self.task_dim) for i in range(self.n_agents)]

    @torch.no_grad()
    def act_step(self, x, h):
        xt = torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        if h is None:
            h = self.online.init_hidden(1, self.device)
        q, h2 = self.online(xt, h)
        return q.squeeze(0).cpu().numpy(), h2

    def push_episode(self, obs_seq, act_seq, rew_seq, done_seq, aux_seq=None):
        """obs_seq: [n_agents, T, input_dim]; act_seq/rew_seq: [n_agents, T]; done_seq: [T]."""
        if aux_seq is None:
            aux_seq = np.full(act_seq.shape, -1, dtype=np.int64)
        self.buffer.append((obs_seq.astype(np.float32), act_seq.astype(np.int64),
                            rew_seq.astype(np.float32), done_seq.astype(np.float32),
                            np.asarray(aux_seq, dtype=np.int64)))

    def _run(self, net, obs, return_aux=False):
        """obs [N,T,D] -> Q sequence and, optionally, subgoal logits."""
        N, T, _ = obs.shape
        h = net.init_hidden(N, self.device)
        qs, auxes = [], []
        for t in range(T):
            if return_aux:
                q, h, aux = net(obs[:, t], h, return_aux=True)
                auxes.append(aux)
            else:
                q, h = net(obs[:, t], h)
            qs.append(q)
        q_seq = torch.stack(qs, dim=1)
        if return_aux:
            return q_seq, torch.stack(auxes, dim=1)
        return q_seq

    def update(self, batch_size=16):
        if len(self.buffer) < batch_size:
            return None
        eps = random.sample(self.buffer, batch_size)
        maxT = max(e[0].shape[1] for e in eps)
        N = batch_size * self.n_agents
        obs = np.zeros((N, maxT, self.input_dim), np.float32)
        act = np.zeros((N, maxT), np.int64)
        rew = np.zeros((N, maxT), np.float32)
        done = np.zeros((N, maxT), np.float32)
        mask = np.zeros((N, maxT), np.float32)
        aux_label = np.full((N, maxT), -1, np.int64)
        for b, (o, a, r, d, aux) in enumerate(eps):
            T = o.shape[1]
            for i in range(self.n_agents):
                row = b * self.n_agents + i
                obs[row, :T] = o[i]
                act[row, :T] = a[i]
                rew[row, :T] = r[i]
                done[row, :T] = d           # done is per-timestep (shared)
                mask[row, :T] = 1.0
                aux_label[row, :T] = aux[i]
        if self.task_dim and self.task_dropout > 0.0:      # controlled dropout on task-input neurons
            drop = np.random.random(obs.shape[:2]) < self.task_dropout
            obs[drop, -self.task_dim:] = 0.0
        obs = torch.as_tensor(obs, device=self.device)
        act = torch.as_tensor(act, device=self.device)
        rew = torch.as_tensor(rew, device=self.device)
        done = torch.as_tensor(done, device=self.device)
        mask = torch.as_tensor(mask, device=self.device)
        aux_label = torch.as_tensor(aux_label, device=self.device)

        if self.aux_classes:
            online_q, aux_logits = self._run(self.online, obs, return_aux=True)
        else:
            online_q = self._run(self.online, obs)
            aux_logits = None
        with torch.no_grad():
            target_q = self._run(self.target, obs)                 # [N, T, A]
        q_taken = online_q.gather(2, act.unsqueeze(2)).squeeze(2)  # [N, T]

        with torch.no_grad():
            next_act = online_q.detach()[:, 1:].argmax(dim=2)      # double DQN: online selects
            next_q = target_q[:, 1:].gather(2, next_act.unsqueeze(2)).squeeze(2)
            next_q = torch.cat([next_q, torch.zeros(next_q.shape[0], 1, device=self.device)], dim=1)
            if self.vnorm is not None:
                # Optional for unusually large/noisy LLM credits; off by default because it
                # changes the sparse-reward RNN behavior that reaches the published LBF ceiling.
                tgt_real = rew + self.gamma * (1.0 - done) * self.vnorm.denormalize(next_q)
                self.vnorm.update(tgt_real[mask > 0])
                tgt = self.vnorm.normalize(tgt_real)
            else:
                tgt = rew + self.gamma * (1.0 - done) * next_q

        td = (q_taken - tgt) * mask
        td_loss = (td.pow(2) * 0.5).sum() / mask.sum().clamp(min=1)
        loss = td_loss
        if aux_logits is not None:
            valid_aux = (aux_label >= 0) & (mask > 0)
            if valid_aux.any():
                aux_loss = F.cross_entropy(aux_logits[valid_aux], aux_label[valid_aux])
                loss = loss + self.aux_coef * aux_loss
                with torch.no_grad():
                    accuracy = (aux_logits[valid_aux].argmax(-1) == aux_label[valid_aux]).float().mean()
                self.last_aux_loss = float(aux_loss.item())
                self.last_aux_accuracy = float(accuracy.item())
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()

        self._updates += 1
        if self._updates % self.target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())


class SeparateRIQL:
    """One recurrent Q learner per environment agent.

    The shared :class:`RIQL` is sample-efficient, but parameter sharing is an additional
    assumption that the LLM-MCA paper does not state: it describes a set of decentralized policy
    networks.  Separate learners provide the literal architecture and allow the two LBF agents to
    specialize without gradients from the other role.  The wrapper preserves RIQL's trainer
    interface so shared and separate policies differ only in this one choice.
    """

    def __init__(self, obs_dim, n_agents, n_actions, device, **kwargs):
        self.n_agents = n_agents
        self.controllers = [
            RIQL(obs_dim, 1, n_actions, device, **kwargs) for _ in range(n_agents)
        ]
        self.input_dim = self.controllers[0].input_dim

    @property
    def buffer(self):
        # Every controller receives exactly one copy of every collected episode.
        return self.controllers[0].buffer

    def actors(self):
        return [ctrl.actors()[0] for ctrl in self.controllers]

    def push_episode(self, obs_seq, act_seq, rew_seq, done_seq, aux_seq=None):
        for i, ctrl in enumerate(self.controllers):
            ctrl_aux = None if aux_seq is None else aux_seq[i:i + 1]
            ctrl.push_episode(
                obs_seq[i:i + 1],
                act_seq[i:i + 1],
                rew_seq[i:i + 1],
                done_seq,
                ctrl_aux,
            )

    def update(self, batch_size=16):
        losses = [ctrl.update(batch_size=batch_size) for ctrl in self.controllers]
        losses = [loss for loss in losses if loss is not None]
        return float(np.mean(losses)) if losses else None
