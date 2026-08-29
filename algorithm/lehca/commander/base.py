"""Commander interface and guidance schema.

A Commander maps the structured summary d_t to 'guidance':
    {
      "strategy":     str,                       # one-line rationale
      "subgoals":     [{"predicate": str, "weight": float, "unit_type": str?}],
      "action_rules": [{"applies_to": str, "forbid": [str], "prefer": [str],
                        "prefer_weight": float}],
    }
"""
from ..shaping.predicates import ALL_PREDICATES, TYPED_PREDICATES

MAX_SUBGOALS = 6
MAX_RULES = 6

# "strategic" vocabulary mode (set from args.mask_vocab): forbids may only
# name stop/attack_type/attack_all; per-step micro tokens are dropped.
VOCAB_MODE = "full"
STRATEGIC_FORBID_OK = ("attack_type:",)  # never allow forbidding all attacks
STRATEGIC_DROP = ("attack_lowest_health", "attack_nearest")


def set_vocab_mode(mode):
    global VOCAB_MODE
    VOCAB_MODE = mode

EMPTY_GUIDANCE = {"strategy": "", "subgoals": [], "action_rules": []}

VALID_TOKEN_PREFIXES = (
    "noop", "stop", "move_north", "move_south", "move_east", "move_west",
    "move_all", "attack_all", "attack_type:", "attack_lowest_health",
    "attack_nearest",
)


def _valid_token(tok):
    return isinstance(tok, str) and any(
        tok == p or (p.endswith(":") and tok.startswith(p))
        for p in VALID_TOKEN_PREFIXES)


def sanitize_guidance(g):
    """Validate/clean raw parsed guidance; returns None if unusable."""
    if not isinstance(g, dict):
        return None
    out = {"strategy": str(g.get("strategy", ""))[:300],
           "subgoals": [], "action_rules": []}
    for sg in (g.get("subgoals") or [])[:MAX_SUBGOALS]:
        if not isinstance(sg, dict):
            continue
        pred = sg.get("predicate")
        if pred not in ALL_PREDICATES:
            continue
        item = {"predicate": pred,
                "weight": max(0.0, min(1.0, float(sg.get("weight", 0.5))))}
        if pred in TYPED_PREDICATES:
            ut = sg.get("unit_type")
            if not isinstance(ut, str) or not ut:
                continue
            item["unit_type"] = ut
        out["subgoals"].append(item)
    for r in (g.get("action_rules") or [])[:MAX_RULES]:
        if not isinstance(r, dict):
            continue
        forbid = [t for t in (r.get("forbid") or []) if _valid_token(t)]
        prefer = [t for t in (r.get("prefer") or []) if _valid_token(t)]
        if VOCAB_MODE == "strategic":
            forbid = [t for t in forbid if any(t == p or (p.endswith(":") and t.startswith(p))
                                               for p in STRATEGIC_FORBID_OK)]
            prefer = [t for t in prefer if t not in STRATEGIC_DROP]
        rule = {"applies_to": r.get("applies_to", "all")
                if isinstance(r.get("applies_to", "all"), str) else "all",
                "forbid": forbid, "prefer": prefer}
        try:
            rule["prefer_weight"] = float(r.get("prefer_weight", 2.0))
        except (TypeError, ValueError):
            rule["prefer_weight"] = 2.0
        if rule["forbid"] or rule["prefer"]:
            out["action_rules"].append(rule)
    if not out["subgoals"] and not out["action_rules"]:
        return None
    return out


class Commander:
    def __call__(self, summary, cache_key, iface):
        """Return guidance dict or None on failure."""
        raise NotImplementedError

    def stats(self):
        return {}
