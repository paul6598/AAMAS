"""Double DQN agent (one per agent), as used for the decentralized policies in LLM-MCA.

Each agent trains on the *credit* the LLM critic assigns to it, not the env reward.
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buf.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        obs, act, rew, nobs, done = zip(*batch)
        return (
            np.stack(obs), np.array(act), np.array(rew, dtype=np.float32),
            np.stack(nobs), np.array(done, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


class DDQNAgent:
    def __init__(self, obs_dim, n_actions, device, lr=5e-4, gamma=0.99,
                 batch_size=64, target_sync=200, buffer_cap=100_000, task_dim=0):
        self.device = device
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync = target_sync
        self.online = QNet(obs_dim, n_actions).to(device)
        self.target = QNet(obs_dim, n_actions).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_cap)
        self._steps = 0
        # LLM-TACA: the last `task_dim` input dims are the task assignment; `task_dropout` is the
        # probability of zeroing them during a training update (the paper's controlled dropout on
        # the task-input neurons), increased over training so the policy learns not to rely on it.
        self.task_dim = task_dim
        self.task_dropout = 0.0

    @torch.no_grad()
    def act(self, obs, epsilon):
        # Do not consume the training RNG during greedy evaluation.
        if epsilon > 0.0 and random.random() < epsilon:
            return random.randrange(self.n_actions)
        q = self.online(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
        return int(q.argmax(dim=1).item())

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None
        obs, act, rew, nobs, done = self.buffer.sample(self.batch_size)
        if self.task_dim and self.task_dropout > 0.0:
            drop = np.random.random(len(obs)) < self.task_dropout   # per-transition task dropout
            obs[drop, -self.task_dim:] = 0.0
            nobs[drop, -self.task_dim:] = 0.0
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act = torch.as_tensor(act, dtype=torch.int64, device=self.device).unsqueeze(1)
        rew = torch.as_tensor(rew, dtype=torch.float32, device=self.device)
        nobs = torch.as_tensor(nobs, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device)

        q = self.online(obs).gather(1, act).squeeze(1)
        with torch.no_grad():
            next_act = self.online(nobs).argmax(dim=1, keepdim=True)      # Double DQN: online selects
            next_q = self.target(nobs).gather(1, next_act).squeeze(1)     # target evaluates
            tgt = rew + self.gamma * (1.0 - done) * next_q
        loss = nn.functional.smooth_l1_loss(q, tgt)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()

        self._steps += 1
        if self._steps % self.target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())
