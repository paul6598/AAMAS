from functools import partial
try:
    from smac.env import StarCraft2Env
except ImportError:  # e.g. GRF-only environments without SMAC installed
    StarCraft2Env = None
from .multiagentenv import MultiAgentEnv
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


class _WinOnlyRewardEnv:
    """SMAC with the sparse defeat penalty removed (win:+1, otherwise 0).

    SMAC's native sparse reward is +1 win / -1 defeat / 0 timeout. This opt-in
    variant clips negative rewards to zero. It removes the defeat-vs-timeout
    reward difference, while sparse exploration can still fail. Existing
    results retain their actual reward flags in Sacred config.json.
    Enabled with env_args.sparse_win_only=True; requires reward_sparse=True.
    """

    def __init__(self, env, sparse_win_only=False, **kwargs):
        self._env = env(**kwargs)
        self._win_only = sparse_win_only
        if sparse_win_only and not kwargs.get("reward_sparse", False):
            raise ValueError("sparse_win_only requires reward_sparse=True")

    def step(self, actions):
        reward, terminated, info = self._env.step(actions)
        if self._win_only and reward < 0:
            reward = 0.0
        return reward, terminated, info

    def __getattr__(self, name):
        return getattr(self._env, name)


def sc2_env_fn(**kwargs) -> MultiAgentEnv:
    if kwargs.pop("sparse_win_only", False):
        return _WinOnlyRewardEnv(StarCraft2Env, sparse_win_only=True, **kwargs)
    return StarCraft2Env(**kwargs)


REGISTRY = {}
if StarCraft2Env is not None:
    REGISTRY["sc2"] = sc2_env_fn

# Google Research Football (gfootball imported lazily inside the wrapper)
from .gfootball import GFootballEnv  # noqa: E402
REGISTRY["gfootball"] = partial(env_fn, env=GFootballEnv)

# PettingZoo SISL Pursuit (pettingzoo imported lazily inside the wrapper)
from .pettingzoo_pursuit import PursuitEnv  # noqa: E402
REGISTRY["pursuit"] = partial(env_fn, env=PursuitEnv)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", os.path.join(os.path.expanduser("~"), "StarCraftII"))
