from .llm_commander import LLMCommander
from .rule_commander import RuleCommander
from .shuffled_commander import ShuffledCommander
from .aligned_commander import AlignedCommander

REGISTRY = {
    "llm": LLMCommander,
    "rule": RuleCommander,
    "shuffle": ShuffledCommander,
    "aligned": AlignedCommander,
}


def make_commander(args, iface, logger=None):
    from .base import set_vocab_mode
    set_vocab_mode(getattr(args, "mask_vocab", "full"))
    kind = getattr(args, "commander", "none")
    if kind in (None, "none", "null", False):
        return None
    if kind == "llm":
        return LLMCommander(args, iface, logger)
    if kind == "rule":
        return RuleCommander()
    if kind == "shuffle":
        return ShuffledCommander(args)
    if kind == "aligned":
        return AlignedCommander()
    raise ValueError("Unknown commander type: %s" % kind)
