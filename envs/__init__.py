"""Environment registry. New environments: add a module and an entry in make_env."""
from .base import Trajectory


def make_env(env_id, **kwargs):
    """Create an environment by id. Dispatches on the id's naming convention."""
    if env_id.startswith("Foraging-"):
        from .lbf import LBFEnv
        return LBFEnv(env_id, **kwargs)
    if env_id == "Climbing":
        from .climbing import ClimbingEnv
        return ClimbingEnv(env_id, **kwargs)
    if env_id.startswith("rware"):
        from .rware import RWAREEnv
        return RWAREEnv(env_id, **kwargs)
    raise ValueError(f"Unknown environment id: {env_id}")
