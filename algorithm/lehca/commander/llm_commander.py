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


PAPER_SYSTEM_PROMPT = """You are the LLM Commander of an allied team in a StarCraft II micromanagement battle (SMAC). You operate at a coarse strategic timescale; decentralized reinforcement-learning agents execute actions at high frequency and learn fine-grained control on their own. Your job is expert strategic cognition, NOT low-level control.

{env_context}

Reason in three explicit stages before answering:
1. Situation assessment: threats, opportunities, force balance, engagement phase.
2. Strategic planning: a coherent macro-strategy for this phase.
3. Task decomposition: concrete, evaluable sub-goals with priorities.

Then output STRICT JSON only (no markdown) with this schema:
{{
  "strategy": "<one concise sentence>",
  "subgoals": [ {{"predicate": "<predicate>", "weight": <0.0-1.0>, "unit_type": "<TypeName, typed predicates only>"}} ],
  "action_rules": [ {{"applies_to": "all" | "type:<AllyTypeName>", "forbid": [<token>...], "prefer": [<token>...], "prefer_weight": <1.1-3.0>}} ]
}}

Reward predicates (at most 6; weight = priority): "enemy_kill", "enemy_damage", "ally_survive", "focus_fire", "retreat_low_health", "kill_type", "damage_type", "protect_type" (last three need unit_type).

Action tokens: "stop", "move_north", "move_south", "move_east", "move_west", "move_all", "attack_all", "attack_type:<EnemyTypeName>", "attack_lowest_health", "attack_nearest".

Rules for action_rules:
- "forbid" is a HARD constraint. Use it ONLY to rule out clearly infeasible or risky decisions (e.g. never forbid all attacks; never forbid retreat or stop, which agents need for kiting). Most of the time "forbid" should be an empty list.
- "prefer" is a SOFT, moderate preference that biases but does not dictate action choice. Express strategic priorities (e.g. focus fire on the enemy's damage dealers via "attack_type:<X>"), not step-by-step micro. Keep prefer_weight moderate (1.5-2.5).
- The environment already rewards damage, kills and winning; prefer coordination-level sub-goals (focus_fire, protect_type, kill_type prioritization, retreat_low_health)."""


# --- two-stage (paper-style) pipeline: free-form plan -> grounding module ---
PLAN_SYSTEM_PROMPT = """You are the LLM Commander of an allied team in a StarCraft II micromanagement battle (SMAC). Decentralized reinforcement-learning agents execute actions at high frequency and learn fine-grained control themselves; you provide expert strategic cognition at a coarse timescale.

{env_context}

Write your reasoning in prose, in three stages:
1. Strategic objectives and battlefield assessment (threats, opportunities, force balance, phase).
2. Macro-strategy for this phase.
3. Sub-goal decomposition: 2-5 concrete, evaluable sub-goals. For EACH sub-goal give a Rationale and an Incentive sentence describing what team behavior or state transition should be rewarded and how strongly (high/medium/low priority).
Finally, list any Action constraints: hard prohibitions ONLY for clearly infeasible or risky actions (usually none), and soft preferences (which targets or unit types to prioritize). Do not micro-manage individual steps."""

GROUNDING_SYSTEM_PROMPT = """You are the semantic grounding module of a hierarchical multi-agent RL system. You receive a Commander's free-form strategic plan and must translate it into the executable schema below so that automated modules can compute reward shaping and action masks. Map each sub-goal/incentive to the closest available reward predicate (weight = its priority: high 0.8-1.0, medium 0.5-0.7, low 0.2-0.4) and each action constraint to action tokens. Use ONLY the listed predicates and tokens; drop anything that cannot be expressed. "forbid" must contain only actions the Commander explicitly prohibited as infeasible/risky (usually empty). "prefer" carries the Commander's soft priorities with moderate prefer_weight (1.5-2.5).

{env_context}

Output STRICT JSON only:
{{
  "strategy": "<one sentence summary of the plan>",
  "subgoals": [ {{"predicate": "<predicate>", "weight": <0.0-1.0>, "unit_type": "<TypeName, typed predicates only>"}} ],
  "action_rules": [ {{"applies_to": "all" | "type:<AllyTypeName>", "forbid": [<token>...], "prefer": [<token>...], "prefer_weight": <1.1-3.0>}} ]
}}
Predicates (max 6): "enemy_kill", "enemy_damage", "ally_survive", "focus_fire", "retreat_low_health", "kill_type", "damage_type", "protect_type" (last three need unit_type).
Action tokens: "stop", "move_north", "move_south", "move_east", "move_west", "move_all", "attack_all", "attack_type:<EnemyTypeName>", "attack_lowest_health", "attack_nearest"."""


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
        self.reasoning_effort = getattr(args, "llm_reasoning_effort", "low")
        self.prompt_style = getattr(args, "prompt_style", "default")
        self._ground_prompt = None
        self.last_plan_text = None
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
            ctx = iface.prompt_context()
            if self.prompt_style == "paper":
                self.system_prompt = PAPER_SYSTEM_PROMPT.format(env_context=ctx)
            elif self.prompt_style == "twostage":
                self.system_prompt = PLAN_SYSTEM_PROMPT.format(env_context=ctx)
                self._ground_prompt = GROUNDING_SYSTEM_PROMPT.format(env_context=ctx)
            else:
                self.system_prompt = SYSTEM_PROMPT.format(env_context=ctx)
        if self.use_cache and cache_key in self._cache:
            self.n_cache_hits += 1
            return self._cache[cache_key]
        if self.prompt_style == "twostage":
            return self._call_twostage(summary, cache_key)

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
            payload["reasoning_effort"] = self.reasoning_effort

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

    def _chat(self, system, user, max_tokens):
        payload = {"model": self.model,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}],
                   "temperature": self.temperature, "max_tokens": max_tokens}
        if self._reasoning_effort_ok:
            payload["reasoning_effort"] = self.reasoning_effort
        r = self._session.post(self.api_base + "/chat/completions",
                               json=payload, timeout=self.timeout)
        if r.status_code == 400 and "reasoning_effort" in payload:
            self._reasoning_effort_ok = False
            payload.pop("reasoning_effort")
            r = self._session.post(self.api_base + "/chat/completions",
                                   json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def _call_twostage(self, summary, cache_key):
        """Paper-style: free-form plan (stage 1) -> grounding module (stage 2)."""
        guidance = None
        for attempt in range(2):
            t0 = time.time()
            try:
                plan = self._chat(self.system_prompt,
                                  summary + "\n\nWrite your strategic plan now.",
                                  self.max_tokens)
                self.last_plan_text = plan
                grounded = self._chat(
                    self._ground_prompt,
                    "Commander plan:\n" + (plan or "") +
                    "\n\nTranslate this plan into the JSON schema now.",
                    1024)
                self.n_calls += 1
                self.total_latency += time.time() - t0
                guidance = sanitize_guidance(extract_json(grounded))
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
