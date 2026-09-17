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

# Broad hard constraints are unsafe in SMAC: the environment availability
# mask already removes infeasible actions, while these tokens can erase an
# entire tactical action class.  The paper describes hard constraints as
# exceptional safety constraints, not as blanket tactical commands.
UNSAFE_FORBID_TOKENS = {"noop", "stop", "move_all", "attack_all"}


def _valid_token(tok):
    return isinstance(tok, str) and any(
        tok == p or (p.endswith(":") and tok.startswith(p))
        for p in VALID_TOKEN_PREFIXES)


# 출력 스키마와 허용 어휘를 검증하고 설정에 따라 중복 서브골을 제거한다.
def sanitize_guidance(g, deduplicate_subgoals=True):
    """Validate/clean raw parsed guidance; returns None if unusable."""
    if not isinstance(g, dict):
        return None
    out = {"strategy": str(g.get("strategy", ""))[:300],
           "subgoals": [], "action_rules": []}
    # A predicate is a semantic feature, not a separate reward term each time
    # it appears in the LLM response.  Exact repeats would otherwise be summed
    # by compute_shaping and silently multiply that feature's reward.  Dedup
    # before applying MAX_SUBGOALS so repeats also cannot crowd out later,
    # distinct features.  For repeats, keep the strongest requested weight.
    subgoal_index = {}
    for sg in (g.get("subgoals") or []):
        if not isinstance(sg, dict):
            continue
        pred = sg.get("predicate")
        if pred not in ALL_PREDICATES:
            continue
        try:
            weight = max(0.0, min(1.0, float(sg.get("weight", 0.5))))
        except (TypeError, ValueError):
            weight = 0.5
        item = {"predicate": pred, "weight": weight}
        unit_key = None
        if pred in TYPED_PREDICATES:
            ut = sg.get("unit_type")
            if not isinstance(ut, str) or not ut:
                continue
            ut = ut.strip()
            if not ut:
                continue
            item["unit_type"] = ut
            unit_key = ut.casefold()
        key = (pred, unit_key)
        if deduplicate_subgoals and key in subgoal_index:
            old = out["subgoals"][subgoal_index[key]]
            old["weight"] = max(old["weight"], item["weight"])
            continue
        if len(out["subgoals"]) >= MAX_SUBGOALS:
            continue
        if deduplicate_subgoals:
            subgoal_index[key] = len(out["subgoals"])
        out["subgoals"].append(item)
    for r in (g.get("action_rules") or [])[:MAX_RULES]:
        if not isinstance(r, dict):
            continue
        forbid = [t for t in (r.get("forbid") or [])
                  if _valid_token(t) and t not in UNSAFE_FORBID_TOKENS]
        prefer = [t for t in (r.get("prefer") or []) if _valid_token(t)]
        # A soft preference and a hard prohibition for the same symbolic
        # action are semantically contradictory.  Preserve the preference and
        # discard the prohibition rather than silently letting -inf win.
        preferred = set(prefer)
        forbid = [t for t in forbid if t not in preferred]
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
