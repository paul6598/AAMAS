"""QMIX learner with the paper's optimizer settings (Adam, Table 2) and the
progressive decay of the shaping weight lambda after each training update
(Algorithm 1, line 22)."""
from torch.optim import Adam

from .state import get_state
from algorithm.src.learners.q_learner import QLearner


class LehcaQLearner(QLearner):
    def __init__(self, mac, scheme, logger, args):
        super(LehcaQLearner, self).__init__(mac, scheme, logger, args)
        if getattr(args, "optimizer", "rmsprop") == "adam":
            self.optimiser = Adam(params=self.params, lr=args.lr)
        self.state = get_state()
        self.shaping_in_learner = getattr(args, "shaping_in_learner", False)
        # > 0: lambda follows a horizon-proportional exponential that reaches
        # lambda_min at (lambda_floor_frac * t_max) env steps; 0 = legacy
        # per-update multiplicative decay
        self.lambda_floor_frac = getattr(args, "lambda_floor_frac", 0.0)
        # > 0 takes precedence over floor_frac and reaches exactly zero at
        # this environment step. This enables an exact shaping-only cutoff.
        self.lambda_zero_t = getattr(args, "lambda_zero_t", 0)

    def train(self, batch, t_env, episode_num):
        if self.lambda_zero_t > 0:
            self.state.set_lambda_cosine_zero(t_env, self.lambda_zero_t)
        elif self.lambda_floor_frac > 0:
            self.state.set_lambda_progress(t_env, self.lambda_floor_frac * self.args.t_max)
        if self.shaping_in_learner:
            # Compose r + lambda_now * F_t on the sampled copy so every
            # replayed transition uses the CURRENT lambda, not the one at
            # collection time.
            batch.data.transition_data["reward"] = (
                batch["reward"] + self.state.lambda_val * batch["shaping_f"])
        super(LehcaQLearner, self).train(batch, t_env, episode_num)
        if self.lambda_zero_t <= 0 and self.lambda_floor_frac <= 0:
            self.state.decay_lambda()
