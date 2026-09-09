"""Shared LEHCA state: current guidance and the shaping weight lambda.

The runner reads lambda when composing r_total = r_env + lambda * F_t at
collection time; the learner decays lambda after each training update
(Algorithm 1, line 22). A module-level singleton keeps pymarl's
runner/learner registries untouched.
"""


class LehcaState:
    def __init__(self):
        self.guidance = None
        self.lambda_val = 0.0
        self.lambda_min = 0.0
        self.lambda_decay = 1.0

    def configure(self, args):
        self.lambda_start = getattr(args, "lambda_start", 0.5)
        self.lambda_val = self.lambda_start
        self.lambda_min = getattr(args, "lambda_min", 0.05)
        self.lambda_decay = getattr(args, "lambda_decay", 0.9995)

    def decay_lambda(self):
        self.lambda_val = max(self.lambda_min, self.lambda_val * self.lambda_decay)

    def set_lambda_progress(self, t_env, t_floor):
        # horizon-proportional exponential: lambda_start at t=0, lambda_min
        # exactly at t_floor, independent of t_max and of update cadence
        p = min(1.0, t_env / max(1.0, t_floor))
        ratio = max(self.lambda_min, 1e-8) / self.lambda_start
        self.lambda_val = max(self.lambda_min, self.lambda_start * ratio ** p)


_STATE = LehcaState()


def get_state():
    return _STATE
