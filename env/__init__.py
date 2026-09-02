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

REGISTRY = {}
if StarCraft2Env is not None:
    REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

# Google Research Football (gfootball imported lazily inside the wrapper)
from .gfootball import GFootballEnv  # noqa: E402
REGISTRY["gfootball"] = partial(env_fn, env=GFootballEnv)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "/gpfs/home1/paul6598/StarCraftII")
