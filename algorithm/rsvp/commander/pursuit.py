"""LLM Commander for PettingZoo Pursuit. Reuses LEHCA's LLMCommander transport;
pursuit guidance is shaping-only (no action rules). The LLM names sub-goals as
(predicate, sector); sanitisation maps "sector" onto the shared "unit_type"
field so library/head plumbing works unchanged.
"""
import time

import requests

from algorithm.lehca.commander.llm_commander import LLMCommander, extract_json
from algorithm.rsvp.shaping.pursuit import (SECTORS, SECTOR_PREDICATES,
                                             GLOBAL_PREDICATES)

MAX_SUBGOALS = 4

SYSTEM_PROMPT = """You are the commander of a pursuit team. Decentralized reinforcement-learning agents control the 8 pursuers at every step and learn the low-level movement themselves; your job is coarse team strategy that stays valid for a while, NOT step-by-step control.

{env_context}

Reason briefly:
1. Where are the evaders concentrated, and where are your pursuers?
2. Pick a plan: which cluster to collapse on, which sector to block, whether to spread and search.
3. Decompose into at most 4 sub-goals (reward shaping priorities).

Output STRICT JSON only (no markdown):
{{
  "strategy": "<one concise sentence>",
  "subgoals": [ {{"predicate": "<predicate>", "sector": "<NW|NE|SW|SE|null>", "weight": <0.0-1.0>}} ]
}}

Sub-goal predicates:
- "approach"     (sector required): close distance to the evaders in that sector
- "encircle"     (sector required): surround the evaders in that sector from multiple sides
- "blockade"     (sector required): keep at least one pursuer stationed in that sector
- "catch"        (sector null): reward each capture
- "tag_pressure" (sector null): keep pursuers adjacent to evaders

Weights express priority. Focus on 1-2 sectors; avoid spreading weight over everything."""


def sanitize_pursuit(g):
    if not isinstance(g, dict):
        return None
    out = {"strategy": str(g.get("strategy", ""))[:300], "subgoals": [],
           "action_rules": []}
    for sg in (g.get("subgoals") or [])[:MAX_SUBGOALS]:
        if not isinstance(sg, dict):
            continue
        pred = sg.get("predicate")
        sector = sg.get("sector") if sg.get("sector") in SECTORS else None
        if pred in SECTOR_PREDICATES and sector is None:
            continue
        if pred in GLOBAL_PREDICATES:
            sector = None
        elif pred not in SECTOR_PREDICATES:
            continue
        try:
            w = float(sg.get("weight", 0.5))
        except (TypeError, ValueError):
            w = 0.5
        out["subgoals"].append({"predicate": pred, "unit_type": sector,
                                "weight": max(0.0, min(1.0, w))})
    if not out["subgoals"]:
        return None
    return out


class PursuitLLMCommander(LLMCommander):

    def __call__(self, summary, cache_key, iface):
        if self.system_prompt is None:
            self.system_prompt = SYSTEM_PROMPT.format(env_context=iface.prompt_context())
        if self.use_cache and cache_key in self._cache:
            self.n_cache_hits += 1
            return self._cache[cache_key]
        payload = {"model": self.model,
                   "messages": [{"role": "system", "content": self.system_prompt},
                                {"role": "user", "content": summary +
                                 "\n\nProduce your guidance now as strict JSON."}],
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
                guidance = sanitize_pursuit(extract_json(content))
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
