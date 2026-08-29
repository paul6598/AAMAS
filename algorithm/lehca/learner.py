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

    def train(self, batch, t_env, episode_num):
        if self.shaping_in_learner:
            # Compose r + lambda_now * F_t on the sampled copy so every
            # replayed transition uses the CURRENT lambda, not the one at
            # collection time.
            batch.data.transition_data["reward"] = (
                batch["reward"] + self.state.lambda_val * batch["shaping_f"])
        super(LehcaQLearner, self).train(batch, t_env, episode_num)
        self.state.decay_lambda()
