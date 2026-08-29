from .llm_commander import LLMCommander
from .rule_commander import RuleCommander

REGISTRY = {
    "llm": LLMCommander,
    "rule": RuleCommander,
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
    raise ValueError("Unknown commander type: %s" % kind)
