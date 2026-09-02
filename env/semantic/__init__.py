from .sc2 import SC2SemanticInterface
from .grf import GRFSemanticInterface

# Keyed by the pymarl env name (args.env). Add new environments here.
IFACE_REGISTRY = {
    "sc2": SC2SemanticInterface,
    "gfootball": GRFSemanticInterface,
}


def make_interface(env_name, env, args):
    if env_name not in IFACE_REGISTRY:
        raise ValueError(
            "No LEHCA semantic interface registered for env '%s'" % env_name)
    return IFACE_REGISTRY[env_name](env, args)
