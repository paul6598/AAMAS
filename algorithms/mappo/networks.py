"""Actor and centralized-critic networks for MAPPO."""
import torch
import torch.nn as nn
from torch.distributions import Categorical


class Actor(nn.Module):
    """Per-agent stochastic policy over discrete actions. Duck-types with env.rollout via act()."""

    def __init__(self, obs_dim, n_actions, device, hidden=128):
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs):
        return self.net(obs)  # logits

    @torch.no_grad()
    def act(self, obs, epsilon):
        """Sample an action during collection (epsilon > 0); take the greedy action for
        evaluation (epsilon == 0, the convention env/evaluate uses)."""
        logits = self.forward(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
        if epsilon == 0.0:
            return int(logits.argmax(dim=1).item())
        return int(Categorical(logits=logits).sample().item())

    def evaluate_actions(self, obs, actions):
        """log-probs and entropy for given (obs, actions) under the current policy."""
        dist = Categorical(logits=self.forward(obs))
        return dist.log_prob(actions), dist.entropy()


class Critic(nn.Module):
    """Centralized value function V(state), where state = concatenation of all agents' obs."""

    def __init__(self, state_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


# ---- Recurrent MAPPO (GRU) + value normalization (paper-grade MAPPO) -------------------------

from ..common.normalizer import ValueNormalizer  # noqa: E402,F401  (shared with the RIQL backbone)


class RecurrentActor(nn.Module):
    """Per-agent recurrent policy (fc -> GRUCell -> head). Tracks its own hidden state across an
    episode during rollout (reset() each episode, duck-typed by env.rollout)."""

    def __init__(self, obs_dim, n_actions, device, hidden=128):
        super().__init__()
        self.device = device
        self.hidden = hidden
        self.n_actions = n_actions
        self.fc = nn.Linear(obs_dim, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.head = nn.Linear(hidden, n_actions)
        self._h = None

    def init_hidden(self, batch):
        return torch.zeros(batch, self.hidden, device=self.device)

    def _step(self, obs, h):
        h = self.gru(torch.tanh(self.fc(obs)), h)
        return self.head(h), h

    def reset(self):
        self._h = None

    @torch.no_grad()
    def act(self, obs, epsilon):
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if self._h is None:
            self._h = self.init_hidden(1)
        logits, self._h = self._step(x, self._h)
        if epsilon == 0.0:
            return int(logits.argmax(dim=1).item())
        return int(Categorical(logits=logits).sample().item())

    def forward_seq(self, obs_seq):
        """obs_seq [B, T, obs_dim] -> logits [B, T, n_actions] (BPTT from h=0)."""
        B, T, _ = obs_seq.shape
        h = self.init_hidden(B)
        outs = []
        for t in range(T):
            logit, h = self._step(obs_seq[:, t], h)
            outs.append(logit)
        return torch.stack(outs, dim=1)

    def seq_logp_entropy(self, obs_seq, act_seq):
        dist = Categorical(logits=self.forward_seq(obs_seq))
        return dist.log_prob(act_seq), dist.entropy()


class RecurrentCritic(nn.Module):
    """Centralized recurrent value function V(state_seq), state = concat of all agents' obs."""

    def __init__(self, state_dim, device, hidden=128):
        super().__init__()
        self.device = device
        self.hidden = hidden
        self.fc = nn.Linear(state_dim, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.head = nn.Linear(hidden, 1)

    def init_hidden(self, batch):
        return torch.zeros(batch, self.hidden, device=self.device)

    def forward_seq(self, state_seq):
        """state_seq [B, T, state_dim] -> values [B, T]."""
        B, T, _ = state_seq.shape
        h = self.init_hidden(B)
        outs = []
        for t in range(T):
            h = self.gru(torch.tanh(self.fc(state_seq[:, t])), h)
            outs.append(self.head(h).squeeze(-1))
        return torch.stack(outs, dim=1)
