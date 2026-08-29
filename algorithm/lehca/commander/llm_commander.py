"""LLM Commander: queries an OpenAI-compatible endpoint (e.g. vLLM serving
gpt-oss-20b) with the structured summary d_t and parses the JSON guidance.

Robustness (paper: agents fall back to last valid sub-goals when the
Commander is unavailable): failures return None and the caller keeps the
previous guidance. Responses are cached on a coarse state key to amortize
LLM cost across similar situations.
"""
import json
import logging
import re
import time

import requests

logging.getLogger("urllib3").setLevel(logging.WARNING)

from .base import Commander, sanitize_guidance
from ..shaping.predicates import SIMPLE_PREDICATES, TYPED_PREDICATES

SYSTEM_PROMPT = """You are the strategic Commander of an allied combat team in a StarCraft II micromanagement battle (SMAC benchmark). You do NOT control units directly. Low-level reinforcement-learning agents execute actions at high frequency; your job is coarse-timescale strategic guidance: assess the situation, choose sub-goals that shape their reward, and constrain/bias their action selection.

{env_context}

Respond with STRICT JSON only (no markdown, no commentary) using this schema:
{{
  "strategy": "<one concise sentence describing your macro strategy>",
  "subgoals": [
    {{"predicate": "<predicate>", "weight": <0.0-1.0>, "unit_type": "<TypeName, only for typed predicates>"}}
  ],
  "action_rules": [
    {{"applies_to": "all" | "type:<AllyTypeName>", "forbid": ["<token>", ...], "prefer": ["<token>", ...], "prefer_weight": <1.1-5.0>}}
  ]
}}

Available reward predicates (pick at most 6, weight = priority):
- "enemy_kill": reward for each enemy eliminated
- "enemy_damage": reward proportional to damage dealt to enemies
- "ally_survive": penalty when an allied unit dies
- "focus_fire": reward when attacking agents concentrate on the same target
- "retreat_low_health": reward when critically wounded allies move away
- "kill_type": reward for eliminating enemies of a given unit_type
- "damage_type": reward for damaging enemies of a given unit_type
- "protect_type": penalty when allies of a given unit_type take damage or die

Available action tokens for forbid/prefer (at most 6 rules):
"stop", "move_north", "move_south", "move_east", "move_west", "move_all",
"attack_all", "attack_type:<EnemyTypeName>", "attack_lowest_health", "attack_nearest"

Guidelines: the environment already densely rewards damage, kills and winning, so prefer coordination-level sub-goals the environment does NOT reward (focus_fire, protect_type, kill_type prioritization, retreat_low_health) over generic enemy_damage/enemy_kill. Prioritize eliminating the enemy's highest-threat damage dealers first, protect fragile key allies (e.g. Medivac), encourage focus fire, and never forbid all attack actions. Keep guidance minimal and strategically coherent."""


def extract_json(text):
    if text is None:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


class LLMCommander(Commander):
    def __init__(self, args, iface, logger=None):
        self.args = args
        self.logger = logger
        self.api_base = args.llm_api_base.rstrip("/")
        self.model = args.llm_model
        self.temperature = args.llm_temperature
        self.max_tokens = args.llm_max_tokens
        self.timeout = args.llm_timeout
        self.use_cache = getattr(args, "llm_cache", True)
        # Built lazily: unit info only exists after the env has been reset.
        self.system_prompt = None
        self._cache = {}
        self._session = requests.Session()
        self._reasoning_effort_ok = True
        self.n_calls = 0
        self.n_cache_hits = 0
        self.n_failures = 0
        self.total_latency = 0.0

    def __call__(self, summary, cache_key, iface):
        if self.system_prompt is None:
            self.system_prompt = SYSTEM_PROMPT.format(
                env_context=iface.prompt_context())
        if self.use_cache and cache_key in self._cache:
            self.n_cache_hits += 1
            return self._cache[cache_key]

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content":
                    summary + "\n\nProduce your strategic guidance now as strict JSON."},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self._reasoning_effort_ok:
            payload["reasoning_effort"] = "low"

        guidance = None
        for attempt in range(2):
            t0 = time.time()
            try:
                r = self._session.post(self.api_base + "/chat/completions",
                                       json=payload, timeout=self.timeout)
                if r.status_code == 400 and "reasoning_effort" in payload:
                    # Server/model without reasoning support: drop and retry.
                    self._reasoning_effort_ok = False
                    payload.pop("reasoning_effort")
                    continue
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"]
                self.n_calls += 1
                self.total_latency += time.time() - t0
                guidance = sanitize_guidance(extract_json(content))
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

    def stats(self):
        return {
            "llm_calls": self.n_calls,
            "llm_cache_hits": self.n_cache_hits,
            "llm_failures": self.n_failures,
            "llm_mean_latency": self.total_latency / max(1, self.n_calls),
        }
