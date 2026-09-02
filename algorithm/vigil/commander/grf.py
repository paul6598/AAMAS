"""LLM Commander for GRF: GRF vocabulary (tokens, selectors, predicates) and a
football-specific system prompt. Reuses LEHCA's LLMCommander transport
(endpoint, retries, cache, stats); only prompt construction and guidance
sanitisation differ, so `__call__` is re-implemented here rather than
editing algorithm/lehca (baseline stays untouched).
"""
import time

import requests

from algorithm.lehca.commander.llm_commander import LLMCommander, extract_json
from env.semantic.grf import VALID_TOKENS, SELECTORS

PREDICATES = ("ball_progress", "keep_possession", "shot_in_box", "pass_forward",
              "regain_possession", "defensive_shape", "press_carrier",
              "compactness", "no_slide_foul")
MAX_SUBGOALS, MAX_RULES = 4, 4
MOVE_AXES = (("move_forward", "move_back"), ("move_up", "move_down"))

SYSTEM_PROMPT = """You are the LLM Commander of a football team in a 5-a-side match (Google Research Football). You operate at a coarse tactical timescale; decentralized reinforcement-learning agents control the players at high frequency and learn the fine motor skills themselves. Your job is team-level tactical cognition, NOT step-by-step control.

{env_context}

Reason in three stages before answering:
1. Phase assessment: are we ATTACKING (we have the ball), DEFENDING (they have it), in TRANSITION (loose ball / just changed hands), or at a SET PIECE? Where is the ball, how is our shape?
2. Tactical plan for THIS phase (it will be replaced when the phase changes).
3. Decompose into sub-goals (reward shaping) and action rules (soft/hard constraints).

Output STRICT JSON only (no markdown):
{{
  "strategy": "<one concise sentence naming the phase and the plan>",
  "subgoals": [ {{"predicate": "<predicate>", "weight": <0.0-1.0>}} ],
  "action_rules": [ {{"applies_to": "<selector>", "forbid": [<token>...], "prefer": [<token>...], "prefer_weight": <1.1-3.0>}} ]
}}

Sub-goal predicates (at most 4; weight = priority):
- attacking: "ball_progress", "keep_possession", "pass_forward", "shot_in_box"
- defending: "regain_possession", "defensive_shape", "press_carrier", "no_slide_foul"
- either:    "compactness"

Selectors for applies_to: "all", "carrier" (our ball carrier), "off_ball", "nearest_to_ball", "deepest" (our deepest field player), "role:<GK|CB|LB|RB|DM|CM|LM|RM|AM|CF>".

Action tokens: "move_forward", "move_back", "move_up", "move_down", "move_toward_ball", "move_toward_goal", "move_toward_own_goal", "sprint", "release_sprint", "short_pass", "long_pass", "high_pass", "shot", "dribble", "release_dribble", "slide", "idle".

Rules:
- "forbid" is a HARD constraint: use it only for clearly wrong actions in this phase (e.g. "slide" for off-ball defenders is a foul risk; "move_back" for the carrier with open space ahead). Never forbid both directions of one axis. Most rules should have an empty forbid list.
- "prefer" is a SOFT bias (prefer_weight 1.5-2.5) expressing the phase plan, e.g. defending: off_ball prefer "move_toward_ball"/"move_back"; attacking: carrier prefer "move_forward"/"short_pass", off_ball prefer "move_forward".
- At most 4 rules. Keep guidance minimal and coherent with the phase."""


def sanitize_grf(g):
    if not isinstance(g, dict):
        return None
    out = {"strategy": str(g.get("strategy", ""))[:300], "subgoals": [], "action_rules": []}
    for sg in (g.get("subgoals") or [])[:MAX_SUBGOALS]:
        if isinstance(sg, dict) and sg.get("predicate") in PREDICATES:
            try:
                w = float(sg.get("weight", 0.5))
            except (TypeError, ValueError):
                w = 0.5
            out["subgoals"].append({"predicate": sg["predicate"], "weight": max(0.0, min(1.0, w))})
    for r in (g.get("action_rules") or [])[:MAX_RULES]:
        if not isinstance(r, dict):
            continue
        sel = r.get("applies_to", "all")
        if not isinstance(sel, str) or sel not in SELECTORS:
            sel = "all"
        forbid = [t for t in (r.get("forbid") or []) if t in VALID_TOKENS]
        prefer = [t for t in (r.get("prefer") or []) if t in VALID_TOKENS]
        for a, b in MOVE_AXES:           # never forbid both directions of an axis
            if a in forbid and b in forbid:
                forbid = [t for t in forbid if t not in (a, b)]
        try:
            pw = float(r.get("prefer_weight", 2.0))
        except (TypeError, ValueError):
            pw = 2.0
        if forbid or prefer:
            out["action_rules"].append({"applies_to": sel, "forbid": forbid,
                                        "prefer": prefer, "prefer_weight": pw})
    if not out["subgoals"] and not out["action_rules"]:
        return None
    return out


class GRFLLMCommander(LLMCommander):

    def __call__(self, summary, cache_key, iface):
        if self.system_prompt is None:
            self.system_prompt = SYSTEM_PROMPT.format(env_context=iface.prompt_context())
        if self.use_cache and cache_key in self._cache:
            self.n_cache_hits += 1
            return self._cache[cache_key]
        payload = {"model": self.model,
                   "messages": [{"role": "system", "content": self.system_prompt},
                                {"role": "user", "content": summary +
                                 "\n\nProduce your tactical guidance now as strict JSON."}],
                   "temperature": self.temperature, "max_tokens": self.max_tokens}
        if self._reasoning_effort_ok:
            payload["reasoning_effort"] = self.reasoning_effort
        guidance = None
        for _ in range(2):
            t0 = time.time()
            try:
                r = self._session.post(self.api_base + "/chat/completions",
                                       json=payload, timeout=self.timeout)
                if r.status_code == 400 and "reasoning_effort" in payload:
                    self._reasoning_effort_ok = False
                    payload.pop("reasoning_effort")
                    continue
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"]
                self.n_calls += 1
                self.total_latency += time.time() - t0
                self.last_plan_text = content
                guidance = sanitize_grf(extract_json(content))
                if guidance is not None:
                    break
            except (requests.RequestException, KeyError, IndexError, ValueError):
                self.n_failures += 1
                time.sleep(0.5)
        if guidance is not None and self.use_cache:
            self._cache[cache_key] = guidance
            if len(self._cache) > 5000:
                self._cache.pop(next(iter(self._cache)))
        return guidance
