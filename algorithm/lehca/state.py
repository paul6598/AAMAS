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

    # 학습 업데이트마다 λ를 곱셈 감쇠시키되 하한을 유지한다.
    def decay_lambda(self):
        self.lambda_val = max(self.lambda_min, self.lambda_val * self.lambda_decay)

    # 환경 스텝 기준 감쇠로 지정 시점에 λ의 하한에 도달하도록 한다.
    def set_lambda_progress(self, t_env, t_floor):
        # horizon-proportional exponential: lambda_start at t=0, lambda_min
        # exactly at t_floor, independent of t_max and of update cadence
        p = min(1.0, t_env / max(1.0, t_floor))
        ratio = max(self.lambda_min, 1e-8) / self.lambda_start
        self.lambda_val = max(self.lambda_min, self.lambda_start * ratio ** p)

    # 지정 종료 시점까지 λ를 코사인 곡선으로 줄여 정확히 0으로 만든다.
    def set_lambda_cosine_zero(self, t_env, t_off):
        """Cosine anneal from lambda_start to exactly zero at t_off."""
        import math
        p = min(1.0, max(0.0, t_env / max(1.0, t_off)))
        self.lambda_val = self.lambda_start * 0.5 * (1.0 + math.cos(math.pi * p))
        if p >= 1.0:
            self.lambda_val = 0.0


_STATE = LehcaState()


def get_state():
    return _STATE
