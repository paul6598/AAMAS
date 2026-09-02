"""Multi-head shaping-value critic V_j(s) with online training.

Targets are discounted suffix sums of per-predicate values over each finished
episode (Monte Carlo). A recency ring buffer keeps the critic tracking the
current policy. Heads whose targets have (so far) near-zero variance are
reported as untrusted so the scheduler can treat those sub-goals as
undecidable.
"""
import numpy as np
import torch
import torch.nn as nn


class ValueCritic:
    def __init__(self, in_dim, n_heads, gamma=0.8, lr=1e-3, hidden=128,
                 buffer_steps=60000, device="cpu"):
        self.gamma = gamma
        self.n_heads = n_heads
        self.device = device
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_heads)).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.X, self.Y = [], []
        self.buffer_steps = buffer_steps
        self.n_steps = 0
        self.mu = torch.zeros(n_heads)
        self.sd = torch.ones(n_heads)
        self.head_var = np.zeros(n_heads)
        self.loss_ema = None

    def add_episode(self, X, F):
        """X: list of feature vectors; F: list of per-head predicate values."""
        F = np.asarray(F, dtype=np.float32)
        Y = np.zeros_like(F)
        run = np.zeros(F.shape[1], dtype=np.float32)
        for t in range(len(F) - 1, -1, -1):
            run = F[t] + self.gamma * run
            Y[t] = run
        self.X.append(np.asarray(X, dtype=np.float32))
        self.Y.append(Y)
        self.n_steps += len(Y)
        while self.n_steps > self.buffer_steps and len(self.X) > 1:
            self.n_steps -= len(self.X.pop(0))
            self.Y.pop(0)
        allY = np.concatenate(self.Y)
        self.mu = torch.tensor(allY.mean(0))
        self.sd = torch.tensor(allY.std(0)).clamp(min=1e-3)
        self.head_var = allY.var(0)

    def train(self, iters=100, batch=1024):
        if not self.X:
            return None
        X = torch.tensor(np.concatenate(self.X), device=self.device)
        Y = torch.tensor(np.concatenate(self.Y), device=self.device)
        tgt = (Y - self.mu.to(self.device)) / self.sd.to(self.device)
        for _ in range(iters):
            idx = torch.randint(0, len(X), (min(batch, len(X)),), device=self.device)
            loss = ((self.net(X[idx]) - tgt[idx]) ** 2).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        v = float(loss.item())
        self.loss_ema = v if self.loss_ema is None else 0.9 * self.loss_ema + 0.1 * v
        return v

    def predict(self, x):
        with torch.no_grad():
            out = self.net(torch.tensor(x, dtype=torch.float32,
                                        device=self.device).unsqueeze(0))[0].cpu()
        return (out * self.sd + self.mu).numpy()

    def trusted(self, head, min_var=1e-4):
        return self.head_var[head] > min_var
