# 실제 SMAC → LLM → JSON → 마스크 스모크 테스트

생성 시각(UTC): 2026-09-12T12:14:03.361160+00:00

별도의 실제 2s3z 에피소드에서 수집한 관측입니다. 과거 로그의 재현이나 학습 정책 평가가 아닙니다. 아군은 공격 가능한 적이 있으면 첫 대상을 공격하고, 없으면 동쪽으로 이동합니다. 한 에피소드 최대 120스텝에서 상황별 첫 관측과 마지막 비종료 관측을 선택합니다.

같은 관측을 paper와 paper_v2에 각 1회 입력합니다(실패 시 Commander의 기존 재시도 적용). 모델 openai/gpt-oss-20b, temperature=0.2, reasoning_effort=low, max_tokens=3072, 응답 캐시 off, f_update=200. 반복 통계나 성능 비교가 아닙니다.

전체 원자료: [JSON](llm-pipeline-smoke-20260912.json). 여기에 per-agent get_obs, availability, 요청·응답 전체, 파싱 및 마스크를 보존했습니다.

주의: snapshot에는 training-time 전체 상태도 포함됩니다. 이것을 LLM에 직접 보내지 않습니다. 실제 전송 내용은 아래 system/user message이며, 적 정보는 기존 summary의 visibility 필터를 통과합니다. JSON 파싱과 sanitizer는 조건 검증기가 아니므로 정제 성공이 의미적 타당성을 보장하지 않습니다.

## 적 미관측 — env step 0

### 1. 실제 환경에서 추출한 숫자 snapshot

```json
{
  "allies": [
    {
      "type": "Stalker",
      "hp": 160.0,
      "hp_max": 160.0,
      "x": 9.0,
      "y": 14.8115234375,
      "alive": true
    },
    {
      "type": "Stalker",
      "hp": 160.0,
      "hp_max": 160.0,
      "x": 9.0,
      "y": 16.0,
      "alive": true
    },
    {
      "type": "Zealot",
      "hp": 150.0,
      "hp_max": 150.0,
      "x": 8.248046875,
      "y": 16.751708984375,
      "alive": true
    },
    {
      "type": "Zealot",
      "hp": 150.0,
      "hp_max": 150.0,
      "x": 9.751708984375,
      "y": 16.751708984375,
      "alive": true
    },
    {
      "type": "Zealot",
      "hp": 150.0,
      "hp_max": 150.0,
      "x": 9.94140625,
      "y": 15.50537109375,
      "alive": true
    }
  ],
  "enemies": [
    {
      "type": "Zealot",
      "hp": 150.0,
      "hp_max": 150.0,
      "x": 23.751708984375,
      "y": 16.751708984375,
      "alive": true,
      "visible": false
    },
    {
      "type": "Zealot",
      "hp": 150.0,
      "hp_max": 150.0,
      "x": 22.248046875,
      "y": 16.751708984375,
      "alive": true,
      "visible": false
    },
    {
      "type": "Zealot",
      "hp": 150.0,
      "hp_max": 150.0,
      "x": 23.94140625,
      "y": 15.50537109375,
      "alive": true,
      "visible": false
    },
    {
      "type": "Stalker",
      "hp": 160.0,
      "hp_max": 160.0,
      "x": 23.0,
      "y": 14.8115234375,
      "alive": true,
      "visible": false
    },
    {
      "type": "Stalker",
      "hp": 160.0,
      "hp_max": 160.0,
      "x": 23.0,
      "y": 16.0,
      "alive": true,
      "visible": false
    }
  ],
  "n_actions": 11
}
```

원래 per-agent 관측 벡터와 availability는 첨부 JSON의 obs/avail_actions에 있습니다.

### 2. 기존 summary 함수가 만든 자연어

```text
Scenario: SMAC map '2s3z' (5 allied vs 5 enemy units).
Allied force: 5/5 alive.
  - Stalker: 2/2 alive, avg health 100% [ranged mechanical: mobile, shields regenerate]
  - Zealot: 3/3 alive, avg health 100% [melee: high damage up close, must reach targets]
Enemy force: no enemy units currently in sight (5 on the map at start).
Spatial: enemy position unknown (none in sight).
```

### 3–6. paper

#### 요청 1: 실제 messages

```json
[
  {
    "role": "system",
    "content": "You are the LLM Commander of an allied team in a StarCraft II micromanagement battle (SMAC). You operate at a coarse strategic timescale; decentralized reinforcement-learning agents execute actions at high frequency and learn fine-grained control on their own. Your job is expert strategic cognition, NOT low-level control.\n\nEach allied agent has this discrete action space: noop (only when dead), stop, move_north, move_south, move_east, move_west, and attack_enemy_i for each enemy unit i (Medivacs heal allies instead of attacking).\nAllied unit types: Stalker, Zealot. Enemy unit types: Stalker, Zealot.\nThe task objective is to eliminate all enemy units while keeping the allied team effective. Do not assume or infer the simulator's true reward function; reason only from the observable situation and this task objective.\n\nReason in three explicit stages before answering:\n1. Situation assessment: threats, opportunities, force balance, engagement phase.\n2. Strategic planning: a coherent macro-strategy for this phase.\n3. Task decomposition: concrete, evaluable sub-goals with priorities.\n\nThen output STRICT JSON only (no markdown) with this schema:\n{\n  \"strategy\": \"<one concise sentence>\",\n  \"subgoals\": [ {\"predicate\": \"<predicate>\", \"weight\": <0.0-1.0>, \"unit_type\": \"<TypeName, typed predicates only>\"} ],\n  \"action_rules\": [ {\"applies_to\": \"all\" | \"type:<AllyTypeName>\", \"forbid\": [<token>...], \"prefer\": [<token>...], \"prefer_weight\": <1.1-3.0>} ]\n}\n\nReward predicates (at most 6; weight = priority): \"enemy_kill\", \"enemy_damage\", \"ally_survive\", \"focus_fire\", \"retreat_low_health\", \"kill_type\", \"damage_type\", \"protect_type\" (last three need unit_type).\n\nAction tokens: \"stop\", \"move_north\", \"move_south\", \"move_east\", \"move_west\", \"move_all\", \"attack_all\", \"attack_type:<EnemyTypeName>\", \"attack_lowest_health\", \"attack_nearest\".\n\nRules for action_rules:\n- \"forbid\" is a HARD constraint. Use it ONLY to rule out clearly infeasible or risky decisions (e.g. never forbid all attacks; never forbid retreat or stop, which agents need for kiting). Most of the time \"forbid\" should be an empty list.\n- \"prefer\" is a SOFT, moderate preference that biases but does not dictate action choice. Express strategic priorities (e.g. focus fire on the enemy's damage dealers via \"attack_type:<X>\"), not step-by-step micro. Keep prefer_weight moderate (1.5-2.5).\n- Do not assume, request, or refer to the simulator's true reward function. Choose only phase-relevant sub-goals and avoid generic/typed duplicates on homogeneous maps.\n- \"forbid\" MUST NOT contain \"noop\", \"stop\", \"move_all\", or \"attack_all\". Never put the same token in both \"forbid\" and \"prefer\"."
  },
  {
    "role": "user",
    "content": "Scenario: SMAC map '2s3z' (5 allied vs 5 enemy units).\nAllied force: 5/5 alive.\n  - Stalker: 2/2 alive, avg health 100% [ranged mechanical: mobile, shields regenerate]\n  - Zealot: 3/3 alive, avg health 100% [melee: high damage up close, must reach targets]\nEnemy force: no enemy units currently in sight (5 on the map at start).\nSpatial: enemy position unknown (none in sight).\n\nProduce your strategic guidance now as strict JSON."
  }
]
```

#### 실제 응답 content 원문

```text
{"strategy":"Search for enemies, then engage with ranged stalkers while keeping Zealots close for melee support.","subgoals":[{"predicate":"enemy_kill","weight":1.0,"unit_type":"Stalker"},{"predicate":"enemy_damage","weight":0.8,"unit_type":"Stalker"},{"predicate":"ally_survive","weight":0.9,"unit_type":"Zealot"},{"predicate":"focus_fire","weight":0.7,"unit_type":"Stalker"},{"predicate":"retreat_low_health","weight":0.5,"unit_type":"Zealot"}],"action_rules":[{"applies_to":"all","forbid":[],"prefer":["move_north","move_south","move_east","move_west"],"prefer_weight":1.5},{"applies_to":"type:Stalker","forbid":[],"prefer":["attack_nearest"],"prefer_weight":2.0},{"applies_to":"type:Zealot","forbid":[],"prefer":["attack_nearest"],"prefer_weight":1.8}]}
```

응답 메타데이터:

```json
{
  "status": 200,
  "finish_reason": "stop",
  "usage": {
    "prompt_tokens": 850,
    "total_tokens": 1242,
    "completion_tokens": 392,
    "prompt_tokens_details": null
  },
  "error": null
}
```

#### JSON 추출 결과（sanitizer 이전）

```json
{
  "strategy": "Search for enemies, then engage with ranged stalkers while keeping Zealots close for melee support.",
  "subgoals": [
    {
      "predicate": "enemy_kill",
      "weight": 1.0,
      "unit_type": "Stalker"
    },
    {
      "predicate": "enemy_damage",
      "weight": 0.8,
      "unit_type": "Stalker"
    },
    {
      "predicate": "ally_survive",
      "weight": 0.9,
      "unit_type": "Zealot"
    },
    {
      "predicate": "focus_fire",
      "weight": 0.7,
      "unit_type": "Stalker"
    },
    {
      "predicate": "retreat_low_health",
      "weight": 0.5,
      "unit_type": "Zealot"
    }
  ],
  "action_rules": [
    {
      "applies_to": "all",
      "forbid": [],
      "prefer": [
        "move_north",
        "move_south",
        "move_east",
        "move_west"
      ],
      "prefer_weight": 1.5
    },
    {
      "applies_to": "type:Stalker",
      "forbid": [],
      "prefer": [
        "attack_nearest"
      ],
      "prefer_weight": 2.0
    },
    {
      "applies_to": "type:Zealot",
      "forbid": [],
      "prefer": [
        "attack_nearest"
      ],
      "prefer_weight": 1.8
    }
  ]
}
```

#### sanitizer 이후 실제 반환 지침

```json
{
  "strategy": "Search for enemies, then engage with ranged stalkers while keeping Zealots close for melee support.",
  "subgoals": [
    {
      "predicate": "enemy_kill",
      "weight": 1.0
    },
    {
      "predicate": "enemy_damage",
      "weight": 0.8
    },
    {
      "predicate": "ally_survive",
      "weight": 0.9
    },
    {
      "predicate": "focus_fire",
      "weight": 0.7
    },
    {
      "predicate": "retreat_low_health",
      "weight": 0.5
    }
  ],
  "action_rules": [
    {
      "applies_to": "all",
      "forbid": [],
      "prefer": [
        "move_north",
        "move_south",
        "move_east",
        "move_west"
      ],
      "prefer_weight": 1.5
    },
    {
      "applies_to": "type:Stalker",
      "forbid": [],
      "prefer": [
        "attack_nearest"
      ],
      "prefer_weight": 2.0
    },
    {
      "applies_to": "type:Zealot",
      "forbid": [],
      "prefer": [
        "attack_nearest"
      ],
      "prefer_weight": 1.8
    }
  ]
}
```

#### 컴파일된 마스크

```json
{
  "hard": [
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ]
  ],
  "soft": [
    [
      1.0,
      1.0,
      1.5,
      1.5,
      1.5,
      1.5,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.5,
      1.5,
      1.5,
      1.5,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.5,
      1.5,
      1.5,
      1.5,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.5,
      1.5,
      1.5,
      1.5,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.5,
      1.5,
      1.5,
      1.5,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ]
  ],
  "avail_actions": [
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ]
  ]
}
```

행: ally index, 열: [noop, stop, north, south, east, west, enemy0…enemy4]. soft>1은 선호이며 확률이 아닙니다. availability가 최종 실행 가능성을 제한합니다. RSVP soft-only 실행에서는 아래 hard를 모두 1로 덮어씁니다.

### 3–6. paper_v2

#### 요청 1: 실제 messages

```json
[
  {
    "role": "system",
    "content": "You are the expert strategic Commander of an allied team in a StarCraft II micromanagement battle (SMAC). Decentralized reinforcement-learning agents choose individual actions and learn fine-grained control. You provide coarse-timescale strategic guidance, not a script of individual moves.\n\nTASK AND INFORMATION BOUNDARY\nEach allied agent has this discrete action space: noop (only when dead), stop, move_north, move_south, move_east, move_west, and attack_enemy_i for each enemy unit i (Medivacs heal allies instead of attacking).\nAllied unit types: Stalker, Zealot. Enemy unit types: Stalker, Zealot.\nThe task objective is to eliminate all enemy units while keeping the allied team effective. Do not assume or infer the simulator's true reward function; reason only from the observable situation and this task objective.\nUse only the supplied observable situation and task objective. Do not assume, request, infer, or refer to the simulator's true reward function, hidden enemy states, future states, or training statistics. Enemies not currently seen are not necessarily dead. A team-level engagement description is not proof that every ally can attack. Do not invent health, cooldowns, positions, or terrain absent from the input.\n\nGUIDANCE LIFETIME\nGuidance is reused between refreshes, with a configured interval or ceiling of 200 environment steps; it is not a one-step instruction. Prefer objectives that remain meaningful as agents move. A failed refresh can leave the previous guidance in use longer. Do not assume a new instruction will arrive immediately after a move.\n\nDECISION PROCEDURE\n1. Situation assessment: identify the observed threats, opportunities, force balance, phase, and important uncertainty.\n2. Strategic planning: choose a coherent phase objective that contributes to winning while leaving low-level execution to the agents.\n3. Task decomposition and grounding: choose a small set of evaluable sub-goals and priorities for this guidance interval. Check that each predicate and action rule actually implements the intended behavior.\nReturn only the JSON decision, not a reasoning transcript. In strategy give a concise observable reason and phase objective (at most 250 characters). This short rationale must agree with the executable fields.\n\nEXECUTABLE SUB-GOAL INTERFACE\nThe following are our auxiliary shaping predicates, NOT the environment reward. Weights in [0, 1] are nonnegative priorities multiplying signed predicate values, not probabilities. Equal weights do not imply equal numerical effects. Choose at most 6 unique, relevant predicates; an empty list is allowed. Do not fill the list just to reach a quota.\n- enemy_kill: +1 per newly eliminated enemy.\n- enemy_damage: +10 times enemy HP decrease divided by total enemy maximum HP.\n- ally_survive: -1 per newly dead ally; it does not give a positive bonus merely for staying alive.\n- kill_type: +1.5 per newly eliminated enemy of unit_type.\n- damage_type: +5 times HP decrease among enemies of unit_type divided by their total maximum HP.\n- protect_type: -1.5 per newly dead ally of unit_type, minus 3 times their HP decrease divided by their total maximum HP; this penalizes damage/death, not rewards healing or idling.\n- focus_fire: zero unless at least two living non-Medivac allies choose attack actions. Otherwise +0.3 times the largest same-target share, only if that share exceeds one half. It measures selected attack-target concentration, not confirmed damage. Omit when fewer than two combat allies remain.\n- retreat_low_health: +0.1 per living ally below 30% HP choosing a movement action that increases distance from the pre-step living-enemy centroid. It is not a generic bonus for moving, safe kiting, or reaching cover. The summary's 'critically low' label alone does not certify this exact threshold.\nThe last three typed names are kill_type, damage_type, protect_type: these require unit_type (an enemy type for kill/damage, an allied type for protect). All other predicates omit unit_type. Use only types supplied by the interface. Do not duplicate a predicate/type pair. Avoid overlapping generic and typed goals unless there is a distinct strategic reason; never pair them for the only type on a homogeneous side. Do not select type-targeted goals for an enemy type without current observational support. Do not assume a shaping predicate's name implements extra conditions or a longer temporal plan.\n\nEXECUTABLE ACTION INTERFACE\nUse at most 6 rules. Each applies_to selector is either all or type:<AllyTypeName>. Rules apply to EVERY living ally matching that selector until refresh; there is NO health, range, cooldown, time, or if/then condition in this schema. Text in strategy does not create executable conditions. If a conditional tactic cannot be expressed faithfully, omit that action rule; use an appropriate conditional predicate if one exists.\nAvailable tokens: stop, move_north, move_south, move_east, move_west, move_all, attack_all, attack_type:<EnemyTypeName>, attack_lowest_health, attack_nearest.\nTarget tokens are resolved anew at each environment step, with environment availability enforced separately. attack_nearest and attack_lowest_health resolve per agent and need not coordinate a shared target. Cardinal movement tokens retain their absolute direction, NOT a relative direction away from the enemy. Use a direction only if its persistence is justified by the observed situation; never encode 'retreat when hurt' as unconditional movement for a whole type. Without visible targets, attack preferences alone do not implement scouting.\nforbid is a hard prohibition; prefer is a soft bias, not a forced action. Usually leave forbid empty. Never forbid noop, stop, move_all, attack_all, all movement directions, or all enemy types. Do not forbid an action also preferred by any overlapping rule. Prefer weights should be moderate (1.5–2.5; allowed 1.1–3.0). Repeated or overlapping preferences combine by maximum weight, not addition. An empty action_rules list is valid when no durable, executable strategic bias is justified.\n\nOUTPUT CONTRACT\nOutput STRICT JSON only, with these three keys and no markdown. Replace placeholders with valid JSON values; omit unit_type for untyped predicates:\n{\n  \"strategy\": \"<brief observable reason and phase objective>\",\n  \"subgoals\": [{\"predicate\": \"<available predicate>\", \"weight\": 0.5, \"unit_type\": \"<typed predicates only>\"}],\n  \"action_rules\": [{\"applies_to\": \"all\", \"forbid\": [], \"prefer\": [\"<available token>\"], \"prefer_weight\": 1.5}]\n}\nBefore emitting JSON, verify type validity, predicate activation conditions, non-duplication, consistency with strategy, and safety over the guidance lifetime. Do not add unsupported conditional fields or promise behavior the executable interface cannot express.\n"
  },
  {
    "role": "user",
    "content": "Scenario: SMAC map '2s3z' (5 allied vs 5 enemy units).\nAllied force: 5/5 alive.\n  - Stalker: 2/2 alive, avg health 100% [ranged mechanical: mobile, shields regenerate]\n  - Zealot: 3/3 alive, avg health 100% [melee: high damage up close, must reach targets]\nEnemy force: no enemy units currently in sight (5 on the map at start).\nSpatial: enemy position unknown (none in sight).\n\nProduce your strategic guidance now as strict JSON."
  }
]
```

#### 실제 응답 content 원문

```text
{
  "strategy": "No enemies in sight; direct all units to move north to scout for enemy positions.",
  "subgoals": [],
  "action_rules": [
    {
      "applies_to": "all",
      "forbid": [],
      "prefer": ["move_north"],
      "prefer_weight": 1.5
    }
  ]
}
```

응답 메타데이터:

```json
{
  "status": 200,
  "finish_reason": "stop",
  "usage": {
    "prompt_tokens": 1597,
    "total_tokens": 1833,
    "completion_tokens": 236,
    "prompt_tokens_details": null
  },
  "error": null
}
```

#### JSON 추출 결과（sanitizer 이전）

```json
{
  "strategy": "No enemies in sight; direct all units to move north to scout for enemy positions.",
  "subgoals": [],
  "action_rules": [
    {
      "applies_to": "all",
      "forbid": [],
      "prefer": [
        "move_north"
      ],
      "prefer_weight": 1.5
    }
  ]
}
```

#### sanitizer 이후 실제 반환 지침

```json
{
  "strategy": "No enemies in sight; direct all units to move north to scout for enemy positions.",
  "subgoals": [],
  "action_rules": [
    {
      "applies_to": "all",
      "forbid": [],
      "prefer": [
        "move_north"
      ],
      "prefer_weight": 1.5
    }
  ]
}
```

#### 컴파일된 마스크

```json
{
  "hard": [
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ]
  ],
  "soft": [
    [
      1.0,
      1.0,
      1.5,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.5,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.5,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.5,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.5,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ]
  ],
  "avail_actions": [
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ]
  ]
}
```

행: ally index, 열: [noop, stop, north, south, east, west, enemy0…enemy4]. soft>1은 선호이며 확률이 아닙니다. availability가 최종 실행 가능성을 제한합니다. RSVP soft-only 실행에서는 아래 hard를 모두 1로 덮어씁니다.

## 다수 아군 교전 — env step 2

### 1. 실제 환경에서 추출한 숫자 snapshot

```json
{
  "allies": [
    {
      "type": "Stalker",
      "hp": 160.0,
      "hp_max": 160.0,
      "x": 11.77099609375,
      "y": 14.521484375,
      "alive": true
    },
    {
      "type": "Stalker",
      "hp": 160.0,
      "hp_max": 160.0,
      "x": 11.262451171875,
      "y": 16.01611328125,
      "alive": true
    },
    {
      "type": "Zealot",
      "hp": 150.0,
      "hp_max": 150.0,
      "x": 10.4970703125,
      "y": 16.75439453125,
      "alive": true
    },
    {
      "type": "Zealot",
      "hp": 150.0,
      "hp_max": 150.0,
      "x": 11.998291015625,
      "y": 16.79248046875,
      "alive": true
    },
    {
      "type": "Zealot",
      "hp": 150.0,
      "hp_max": 150.0,
      "x": 12.190673828125,
      "y": 15.497802734375,
      "alive": true
    }
  ],
  "enemies": [
    {
      "type": "Zealot",
      "hp": 150.0,
      "hp_max": 150.0,
      "x": 21.501708984375,
      "y": 16.751708984375,
      "alive": true,
      "visible": false
    },
    {
      "type": "Zealot",
      "hp": 150.0,
      "hp_max": 150.0,
      "x": 19.998046875,
      "y": 16.8427734375,
      "alive": true,
      "visible": true
    },
    {
      "type": "Zealot",
      "hp": 150.0,
      "hp_max": 150.0,
      "x": 21.69140625,
      "y": 15.50537109375,
      "alive": true,
      "visible": false
    },
    {
      "type": "Stalker",
      "hp": 160.0,
      "hp_max": 160.0,
      "x": 20.046875,
      "y": 14.741455078125,
      "alive": true,
      "visible": true
    },
    {
      "type": "Stalker",
      "hp": 160.0,
      "hp_max": 160.0,
      "x": 20.420654296875,
      "y": 15.8671875,
      "alive": true,
      "visible": true
    }
  ],
  "n_actions": 11
}
```

원래 per-agent 관측 벡터와 availability는 첨부 JSON의 obs/avail_actions에 있습니다.

### 2. 기존 summary 함수가 만든 자연어

```text
Scenario: SMAC map '2s3z' (5 allied vs 5 enemy units).
Allied force: 5/5 alive.
  - Stalker: 2/2 alive, avg health 100% [ranged mechanical: mobile, shields regenerate]
  - Zealot: 3/3 alive, avg health 100% [melee: high damage up close, must reach targets]
Enemy force: 3 unit(s) currently in sight (5 on the map at start).
  - Stalker: 2/2 alive, avg health 100% [ranged mechanical: mobile, shields regenerate]
  - Zealot: 1/1 alive, avg health 100% [melee: high damage up close, must reach targets]
Spatial: army separation 8.6 map units; phase: approaching (out of attack range).
Enemy centroid lies to the east of the allied force.
```

### 3–6. paper

#### 요청 1: 실제 messages

```json
[
  {
    "role": "system",
    "content": "You are the LLM Commander of an allied team in a StarCraft II micromanagement battle (SMAC). You operate at a coarse strategic timescale; decentralized reinforcement-learning agents execute actions at high frequency and learn fine-grained control on their own. Your job is expert strategic cognition, NOT low-level control.\n\nEach allied agent has this discrete action space: noop (only when dead), stop, move_north, move_south, move_east, move_west, and attack_enemy_i for each enemy unit i (Medivacs heal allies instead of attacking).\nAllied unit types: Stalker, Zealot. Enemy unit types: Stalker, Zealot.\nThe task objective is to eliminate all enemy units while keeping the allied team effective. Do not assume or infer the simulator's true reward function; reason only from the observable situation and this task objective.\n\nReason in three explicit stages before answering:\n1. Situation assessment: threats, opportunities, force balance, engagement phase.\n2. Strategic planning: a coherent macro-strategy for this phase.\n3. Task decomposition: concrete, evaluable sub-goals with priorities.\n\nThen output STRICT JSON only (no markdown) with this schema:\n{\n  \"strategy\": \"<one concise sentence>\",\n  \"subgoals\": [ {\"predicate\": \"<predicate>\", \"weight\": <0.0-1.0>, \"unit_type\": \"<TypeName, typed predicates only>\"} ],\n  \"action_rules\": [ {\"applies_to\": \"all\" | \"type:<AllyTypeName>\", \"forbid\": [<token>...], \"prefer\": [<token>...], \"prefer_weight\": <1.1-3.0>} ]\n}\n\nReward predicates (at most 6; weight = priority): \"enemy_kill\", \"enemy_damage\", \"ally_survive\", \"focus_fire\", \"retreat_low_health\", \"kill_type\", \"damage_type\", \"protect_type\" (last three need unit_type).\n\nAction tokens: \"stop\", \"move_north\", \"move_south\", \"move_east\", \"move_west\", \"move_all\", \"attack_all\", \"attack_type:<EnemyTypeName>\", \"attack_lowest_health\", \"attack_nearest\".\n\nRules for action_rules:\n- \"forbid\" is a HARD constraint. Use it ONLY to rule out clearly infeasible or risky decisions (e.g. never forbid all attacks; never forbid retreat or stop, which agents need for kiting). Most of the time \"forbid\" should be an empty list.\n- \"prefer\" is a SOFT, moderate preference that biases but does not dictate action choice. Express strategic priorities (e.g. focus fire on the enemy's damage dealers via \"attack_type:<X>\"), not step-by-step micro. Keep prefer_weight moderate (1.5-2.5).\n- Do not assume, request, or refer to the simulator's true reward function. Choose only phase-relevant sub-goals and avoid generic/typed duplicates on homogeneous maps.\n- \"forbid\" MUST NOT contain \"noop\", \"stop\", \"move_all\", or \"attack_all\". Never put the same token in both \"forbid\" and \"prefer\"."
  },
  {
    "role": "user",
    "content": "Scenario: SMAC map '2s3z' (5 allied vs 5 enemy units).\nAllied force: 5/5 alive.\n  - Stalker: 2/2 alive, avg health 100% [ranged mechanical: mobile, shields regenerate]\n  - Zealot: 3/3 alive, avg health 100% [melee: high damage up close, must reach targets]\nEnemy force: 3 unit(s) currently in sight (5 on the map at start).\n  - Stalker: 2/2 alive, avg health 100% [ranged mechanical: mobile, shields regenerate]\n  - Zealot: 1/1 alive, avg health 100% [melee: high damage up close, must reach targets]\nSpatial: army separation 8.6 map units; phase: approaching (out of attack range).\nEnemy centroid lies to the east of the allied force.\n\nProduce your strategic guidance now as strict JSON."
  }
]
```

#### 실제 응답 content 원문

```text
{"strategy":"Engage the enemy from the east, using allied stalkers to keep distance and Zealots to close in, focusing fire on enemy stalkers first.","subgoals":[{"predicate":"enemy_kill","weight":1.0,"unit_type":"all"},{"predicate":"enemy_damage","weight":0.8,"unit_type":"all"},{"predicate":"ally_survive","weight":0.9,"unit_type":"all"},{"predicate":"focus_fire","weight":0.7,"unit_type":"all"},{"predicate":"kill_type","weight":0.6,"unit_type":"Stalker"},{"predicate":"damage_type","weight":0.5,"unit_type":"Stalker"}],"action_rules":[{"applies_to":"all","forbid":[],"prefer":["attack_type:Stalker"],"prefer_weight":2.0},{"applies_to":"type:Zealot","forbid":[],"prefer":["attack_nearest"],"prefer_weight":1.5}]}
```

응답 메타데이터:

```json
{
  "status": 200,
  "finish_reason": "stop",
  "usage": {
    "prompt_tokens": 929,
    "total_tokens": 1373,
    "completion_tokens": 444,
    "prompt_tokens_details": null
  },
  "error": null
}
```

#### JSON 추출 결과（sanitizer 이전）

```json
{
  "strategy": "Engage the enemy from the east, using allied stalkers to keep distance and Zealots to close in, focusing fire on enemy stalkers first.",
  "subgoals": [
    {
      "predicate": "enemy_kill",
      "weight": 1.0,
      "unit_type": "all"
    },
    {
      "predicate": "enemy_damage",
      "weight": 0.8,
      "unit_type": "all"
    },
    {
      "predicate": "ally_survive",
      "weight": 0.9,
      "unit_type": "all"
    },
    {
      "predicate": "focus_fire",
      "weight": 0.7,
      "unit_type": "all"
    },
    {
      "predicate": "kill_type",
      "weight": 0.6,
      "unit_type": "Stalker"
    },
    {
      "predicate": "damage_type",
      "weight": 0.5,
      "unit_type": "Stalker"
    }
  ],
  "action_rules": [
    {
      "applies_to": "all",
      "forbid": [],
      "prefer": [
        "attack_type:Stalker"
      ],
      "prefer_weight": 2.0
    },
    {
      "applies_to": "type:Zealot",
      "forbid": [],
      "prefer": [
        "attack_nearest"
      ],
      "prefer_weight": 1.5
    }
  ]
}
```

#### sanitizer 이후 실제 반환 지침

```json
{
  "strategy": "Engage the enemy from the east, using allied stalkers to keep distance and Zealots to close in, focusing fire on enemy stalkers first.",
  "subgoals": [
    {
      "predicate": "enemy_kill",
      "weight": 1.0
    },
    {
      "predicate": "enemy_damage",
      "weight": 0.8
    },
    {
      "predicate": "ally_survive",
      "weight": 0.9
    },
    {
      "predicate": "focus_fire",
      "weight": 0.7
    },
    {
      "predicate": "kill_type",
      "weight": 0.6,
      "unit_type": "Stalker"
    },
    {
      "predicate": "damage_type",
      "weight": 0.5,
      "unit_type": "Stalker"
    }
  ],
  "action_rules": [
    {
      "applies_to": "all",
      "forbid": [],
      "prefer": [
        "attack_type:Stalker"
      ],
      "prefer_weight": 2.0
    },
    {
      "applies_to": "type:Zealot",
      "forbid": [],
      "prefer": [
        "attack_nearest"
      ],
      "prefer_weight": 1.5
    }
  ]
}
```

#### 컴파일된 마스크

```json
{
  "hard": [
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ]
  ],
  "soft": [
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      2.0,
      2.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      2.0,
      2.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.5,
      1.0,
      2.0,
      2.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.5,
      1.0,
      2.0,
      2.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      2.0,
      2.0
    ]
  ],
  "avail_actions": [
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ]
  ]
}
```

행: ally index, 열: [noop, stop, north, south, east, west, enemy0…enemy4]. soft>1은 선호이며 확률이 아닙니다. availability가 최종 실행 가능성을 제한합니다. RSVP soft-only 실행에서는 아래 hard를 모두 1로 덮어씁니다.

### 3–6. paper_v2

#### 요청 1: 실제 messages

```json
[
  {
    "role": "system",
    "content": "You are the expert strategic Commander of an allied team in a StarCraft II micromanagement battle (SMAC). Decentralized reinforcement-learning agents choose individual actions and learn fine-grained control. You provide coarse-timescale strategic guidance, not a script of individual moves.\n\nTASK AND INFORMATION BOUNDARY\nEach allied agent has this discrete action space: noop (only when dead), stop, move_north, move_south, move_east, move_west, and attack_enemy_i for each enemy unit i (Medivacs heal allies instead of attacking).\nAllied unit types: Stalker, Zealot. Enemy unit types: Stalker, Zealot.\nThe task objective is to eliminate all enemy units while keeping the allied team effective. Do not assume or infer the simulator's true reward function; reason only from the observable situation and this task objective.\nUse only the supplied observable situation and task objective. Do not assume, request, infer, or refer to the simulator's true reward function, hidden enemy states, future states, or training statistics. Enemies not currently seen are not necessarily dead. A team-level engagement description is not proof that every ally can attack. Do not invent health, cooldowns, positions, or terrain absent from the input.\n\nGUIDANCE LIFETIME\nGuidance is reused between refreshes, with a configured interval or ceiling of 200 environment steps; it is not a one-step instruction. Prefer objectives that remain meaningful as agents move. A failed refresh can leave the previous guidance in use longer. Do not assume a new instruction will arrive immediately after a move.\n\nDECISION PROCEDURE\n1. Situation assessment: identify the observed threats, opportunities, force balance, phase, and important uncertainty.\n2. Strategic planning: choose a coherent phase objective that contributes to winning while leaving low-level execution to the agents.\n3. Task decomposition and grounding: choose a small set of evaluable sub-goals and priorities for this guidance interval. Check that each predicate and action rule actually implements the intended behavior.\nReturn only the JSON decision, not a reasoning transcript. In strategy give a concise observable reason and phase objective (at most 250 characters). This short rationale must agree with the executable fields.\n\nEXECUTABLE SUB-GOAL INTERFACE\nThe following are our auxiliary shaping predicates, NOT the environment reward. Weights in [0, 1] are nonnegative priorities multiplying signed predicate values, not probabilities. Equal weights do not imply equal numerical effects. Choose at most 6 unique, relevant predicates; an empty list is allowed. Do not fill the list just to reach a quota.\n- enemy_kill: +1 per newly eliminated enemy.\n- enemy_damage: +10 times enemy HP decrease divided by total enemy maximum HP.\n- ally_survive: -1 per newly dead ally; it does not give a positive bonus merely for staying alive.\n- kill_type: +1.5 per newly eliminated enemy of unit_type.\n- damage_type: +5 times HP decrease among enemies of unit_type divided by their total maximum HP.\n- protect_type: -1.5 per newly dead ally of unit_type, minus 3 times their HP decrease divided by their total maximum HP; this penalizes damage/death, not rewards healing or idling.\n- focus_fire: zero unless at least two living non-Medivac allies choose attack actions. Otherwise +0.3 times the largest same-target share, only if that share exceeds one half. It measures selected attack-target concentration, not confirmed damage. Omit when fewer than two combat allies remain.\n- retreat_low_health: +0.1 per living ally below 30% HP choosing a movement action that increases distance from the pre-step living-enemy centroid. It is not a generic bonus for moving, safe kiting, or reaching cover. The summary's 'critically low' label alone does not certify this exact threshold.\nThe last three typed names are kill_type, damage_type, protect_type: these require unit_type (an enemy type for kill/damage, an allied type for protect). All other predicates omit unit_type. Use only types supplied by the interface. Do not duplicate a predicate/type pair. Avoid overlapping generic and typed goals unless there is a distinct strategic reason; never pair them for the only type on a homogeneous side. Do not select type-targeted goals for an enemy type without current observational support. Do not assume a shaping predicate's name implements extra conditions or a longer temporal plan.\n\nEXECUTABLE ACTION INTERFACE\nUse at most 6 rules. Each applies_to selector is either all or type:<AllyTypeName>. Rules apply to EVERY living ally matching that selector until refresh; there is NO health, range, cooldown, time, or if/then condition in this schema. Text in strategy does not create executable conditions. If a conditional tactic cannot be expressed faithfully, omit that action rule; use an appropriate conditional predicate if one exists.\nAvailable tokens: stop, move_north, move_south, move_east, move_west, move_all, attack_all, attack_type:<EnemyTypeName>, attack_lowest_health, attack_nearest.\nTarget tokens are resolved anew at each environment step, with environment availability enforced separately. attack_nearest and attack_lowest_health resolve per agent and need not coordinate a shared target. Cardinal movement tokens retain their absolute direction, NOT a relative direction away from the enemy. Use a direction only if its persistence is justified by the observed situation; never encode 'retreat when hurt' as unconditional movement for a whole type. Without visible targets, attack preferences alone do not implement scouting.\nforbid is a hard prohibition; prefer is a soft bias, not a forced action. Usually leave forbid empty. Never forbid noop, stop, move_all, attack_all, all movement directions, or all enemy types. Do not forbid an action also preferred by any overlapping rule. Prefer weights should be moderate (1.5–2.5; allowed 1.1–3.0). Repeated or overlapping preferences combine by maximum weight, not addition. An empty action_rules list is valid when no durable, executable strategic bias is justified.\n\nOUTPUT CONTRACT\nOutput STRICT JSON only, with these three keys and no markdown. Replace placeholders with valid JSON values; omit unit_type for untyped predicates:\n{\n  \"strategy\": \"<brief observable reason and phase objective>\",\n  \"subgoals\": [{\"predicate\": \"<available predicate>\", \"weight\": 0.5, \"unit_type\": \"<typed predicates only>\"}],\n  \"action_rules\": [{\"applies_to\": \"all\", \"forbid\": [], \"prefer\": [\"<available token>\"], \"prefer_weight\": 1.5}]\n}\nBefore emitting JSON, verify type validity, predicate activation conditions, non-duplication, consistency with strategy, and safety over the guidance lifetime. Do not add unsupported conditional fields or promise behavior the executable interface cannot express.\n"
  },
  {
    "role": "user",
    "content": "Scenario: SMAC map '2s3z' (5 allied vs 5 enemy units).\nAllied force: 5/5 alive.\n  - Stalker: 2/2 alive, avg health 100% [ranged mechanical: mobile, shields regenerate]\n  - Zealot: 3/3 alive, avg health 100% [melee: high damage up close, must reach targets]\nEnemy force: 3 unit(s) currently in sight (5 on the map at start).\n  - Stalker: 2/2 alive, avg health 100% [ranged mechanical: mobile, shields regenerate]\n  - Zealot: 1/1 alive, avg health 100% [melee: high damage up close, must reach targets]\nSpatial: army separation 8.6 map units; phase: approaching (out of attack range).\nEnemy centroid lies to the east of the allied force.\n\nProduce your strategic guidance now as strict JSON."
  }
]
```

#### 실제 응답 content 원문

```text
{"strategy":"Engage the visible enemies, focus fire on the two stalkers first, then the lone zealot, while keeping all units alive.","subgoals":[{"predicate":"enemy_kill","weight":1},{"predicate":"enemy_damage","weight":0.5},{"predicate":"kill_type","weight":1.5,"unit_type":"Stalker"},{"predicate":"damage_type","weight":1,"unit_type":"Stalker"},{"predicate":"protect_type","weight":1.5,"unit_type":"Zealot"}],"action_rules":[{"applies_to":"all","forbid":[],"prefer":["attack_nearest"],"prefer_weight":1.5}]}
```

응답 메타데이터:

```json
{
  "status": 200,
  "finish_reason": "stop",
  "usage": {
    "prompt_tokens": 1676,
    "total_tokens": 1992,
    "completion_tokens": 316,
    "prompt_tokens_details": null
  },
  "error": null
}
```

#### JSON 추출 결과（sanitizer 이전）

```json
{
  "strategy": "Engage the visible enemies, focus fire on the two stalkers first, then the lone zealot, while keeping all units alive.",
  "subgoals": [
    {
      "predicate": "enemy_kill",
      "weight": 1
    },
    {
      "predicate": "enemy_damage",
      "weight": 0.5
    },
    {
      "predicate": "kill_type",
      "weight": 1.5,
      "unit_type": "Stalker"
    },
    {
      "predicate": "damage_type",
      "weight": 1,
      "unit_type": "Stalker"
    },
    {
      "predicate": "protect_type",
      "weight": 1.5,
      "unit_type": "Zealot"
    }
  ],
  "action_rules": [
    {
      "applies_to": "all",
      "forbid": [],
      "prefer": [
        "attack_nearest"
      ],
      "prefer_weight": 1.5
    }
  ]
}
```

#### sanitizer 이후 실제 반환 지침

```json
{
  "strategy": "Engage the visible enemies, focus fire on the two stalkers first, then the lone zealot, while keeping all units alive.",
  "subgoals": [
    {
      "predicate": "enemy_kill",
      "weight": 1.0
    },
    {
      "predicate": "enemy_damage",
      "weight": 0.5
    },
    {
      "predicate": "kill_type",
      "weight": 1.0,
      "unit_type": "Stalker"
    },
    {
      "predicate": "damage_type",
      "weight": 1.0,
      "unit_type": "Stalker"
    },
    {
      "predicate": "protect_type",
      "weight": 1.0,
      "unit_type": "Zealot"
    }
  ],
  "action_rules": [
    {
      "applies_to": "all",
      "forbid": [],
      "prefer": [
        "attack_nearest"
      ],
      "prefer_weight": 1.5
    }
  ]
}
```

#### 컴파일된 마스크

```json
{
  "hard": [
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ]
  ],
  "soft": [
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.5,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.5,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.5,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.5,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.5,
      1.0
    ]
  ],
  "avail_actions": [
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ]
  ]
}
```

행: ally index, 열: [noop, stop, north, south, east, west, enemy0…enemy4]. soft>1은 선호이며 확률이 아닙니다. availability가 최종 실행 가능성을 제한합니다. RSVP soft-only 실행에서는 아래 hard를 모두 1로 덮어씁니다.

## 소수 아군 생존 — env step 30

### 1. 실제 환경에서 추출한 숫자 snapshot

```json
{
  "allies": [
    {
      "type": "Stalker",
      "hp": 160.0,
      "hp_max": 160.0,
      "x": 13.22802734375,
      "y": 14.431640625,
      "alive": true
    },
    {
      "type": "Stalker",
      "hp": 160.0,
      "hp_max": 160.0,
      "x": 13.436767578125,
      "y": 16.11962890625,
      "alive": true
    },
    {
      "type": "Zealot",
      "hp": 0.0,
      "hp_max": 150.0,
      "x": 18.031982421875,
      "y": 14.76806640625,
      "alive": false
    },
    {
      "type": "Zealot",
      "hp": 0.0,
      "hp_max": 150.0,
      "x": 16.638427734375,
      "y": 17.208984375,
      "alive": false
    },
    {
      "type": "Zealot",
      "hp": 0.0,
      "hp_max": 150.0,
      "x": 18.21533203125,
      "y": 16.7216796875,
      "alive": false
    }
  ],
  "enemies": [
    {
      "type": "Zealot",
      "hp": 0.0,
      "hp_max": 150.0,
      "x": 17.346435546875,
      "y": 16.56787109375,
      "alive": false,
      "visible": false
    },
    {
      "type": "Zealot",
      "hp": 0.0,
      "hp_max": 150.0,
      "x": 17.670654296875,
      "y": 15.75537109375,
      "alive": false,
      "visible": false
    },
    {
      "type": "Zealot",
      "hp": 150.0,
      "hp_max": 150.0,
      "x": 17.01611328125,
      "y": 15.06396484375,
      "alive": true,
      "visible": true
    },
    {
      "type": "Stalker",
      "hp": 147.875,
      "hp_max": 160.0,
      "x": 19.1240234375,
      "y": 14.728759765625,
      "alive": true,
      "visible": true
    },
    {
      "type": "Stalker",
      "hp": 160.0,
      "hp_max": 160.0,
      "x": 19.730712890625,
      "y": 15.837890625,
      "alive": true,
      "visible": true
    }
  ],
  "n_actions": 11
}
```

원래 per-agent 관측 벡터와 availability는 첨부 JSON의 obs/avail_actions에 있습니다.

### 2. 기존 summary 함수가 만든 자연어

```text
Scenario: SMAC map '2s3z' (5 allied vs 5 enemy units).
Allied force: 2/5 alive.
  - Stalker: 2/2 alive, avg health 100% [ranged mechanical: mobile, shields regenerate]
  - Zealot: 0/3 alive (eliminated) [melee: high damage up close, must reach targets]
Enemy force: 3 unit(s) currently in sight (5 on the map at start).
  - Stalker: 2/2 alive, avg health 96% [ranged mechanical: mobile, shields regenerate]
  - Zealot: 1/1 alive, avg health 100% [melee: high damage up close, must reach targets]
Spatial: army separation 5.3 map units; phase: engaged (within typical attack range).
Enemy centroid lies to the east of the allied force.
```

### 3–6. paper

#### 요청 1: 실제 messages

```json
[
  {
    "role": "system",
    "content": "You are the LLM Commander of an allied team in a StarCraft II micromanagement battle (SMAC). You operate at a coarse strategic timescale; decentralized reinforcement-learning agents execute actions at high frequency and learn fine-grained control on their own. Your job is expert strategic cognition, NOT low-level control.\n\nEach allied agent has this discrete action space: noop (only when dead), stop, move_north, move_south, move_east, move_west, and attack_enemy_i for each enemy unit i (Medivacs heal allies instead of attacking).\nAllied unit types: Stalker, Zealot. Enemy unit types: Stalker, Zealot.\nThe task objective is to eliminate all enemy units while keeping the allied team effective. Do not assume or infer the simulator's true reward function; reason only from the observable situation and this task objective.\n\nReason in three explicit stages before answering:\n1. Situation assessment: threats, opportunities, force balance, engagement phase.\n2. Strategic planning: a coherent macro-strategy for this phase.\n3. Task decomposition: concrete, evaluable sub-goals with priorities.\n\nThen output STRICT JSON only (no markdown) with this schema:\n{\n  \"strategy\": \"<one concise sentence>\",\n  \"subgoals\": [ {\"predicate\": \"<predicate>\", \"weight\": <0.0-1.0>, \"unit_type\": \"<TypeName, typed predicates only>\"} ],\n  \"action_rules\": [ {\"applies_to\": \"all\" | \"type:<AllyTypeName>\", \"forbid\": [<token>...], \"prefer\": [<token>...], \"prefer_weight\": <1.1-3.0>} ]\n}\n\nReward predicates (at most 6; weight = priority): \"enemy_kill\", \"enemy_damage\", \"ally_survive\", \"focus_fire\", \"retreat_low_health\", \"kill_type\", \"damage_type\", \"protect_type\" (last three need unit_type).\n\nAction tokens: \"stop\", \"move_north\", \"move_south\", \"move_east\", \"move_west\", \"move_all\", \"attack_all\", \"attack_type:<EnemyTypeName>\", \"attack_lowest_health\", \"attack_nearest\".\n\nRules for action_rules:\n- \"forbid\" is a HARD constraint. Use it ONLY to rule out clearly infeasible or risky decisions (e.g. never forbid all attacks; never forbid retreat or stop, which agents need for kiting). Most of the time \"forbid\" should be an empty list.\n- \"prefer\" is a SOFT, moderate preference that biases but does not dictate action choice. Express strategic priorities (e.g. focus fire on the enemy's damage dealers via \"attack_type:<X>\"), not step-by-step micro. Keep prefer_weight moderate (1.5-2.5).\n- Do not assume, request, or refer to the simulator's true reward function. Choose only phase-relevant sub-goals and avoid generic/typed duplicates on homogeneous maps.\n- \"forbid\" MUST NOT contain \"noop\", \"stop\", \"move_all\", or \"attack_all\". Never put the same token in both \"forbid\" and \"prefer\"."
  },
  {
    "role": "user",
    "content": "Scenario: SMAC map '2s3z' (5 allied vs 5 enemy units).\nAllied force: 2/5 alive.\n  - Stalker: 2/2 alive, avg health 100% [ranged mechanical: mobile, shields regenerate]\n  - Zealot: 0/3 alive (eliminated) [melee: high damage up close, must reach targets]\nEnemy force: 3 unit(s) currently in sight (5 on the map at start).\n  - Stalker: 2/2 alive, avg health 96% [ranged mechanical: mobile, shields regenerate]\n  - Zealot: 1/1 alive, avg health 100% [melee: high damage up close, must reach targets]\nSpatial: army separation 5.3 map units; phase: engaged (within typical attack range).\nEnemy centroid lies to the east of the allied force.\n\nProduce your strategic guidance now as strict JSON."
  }
]
```

#### 실제 응답 content 원문

```text
{"strategy":"Use the two allied stalkers to kite the enemy stalkers while protecting the lone Zealot from melee contact, focusing fire on the enemy stalkers and retreating if the Zealot is threatened.","subgoals":[{"predicate":"enemy_kill","weight":0.9,"unit_type":"Stalker"},{"predicate":"enemy_kill","weight":0.8,"unit_type":"Zealot"},{"predicate":"focus_fire","weight":0.85,"unit_type":"Stalker"},{"predicate":"protect_type","weight":0.8,"unit_type":"Zealot"},{"predicate":"retreat_low_health","weight":0.7,"unit_type":"Zealot"}],"action_rules":[{"applies_to":"type:Stalker","forbid":[],"prefer":["attack_type:Stalker","attack_lowest_health"],"prefer_weight":2.0},{"applies_to":"type:Zealot","forbid":[],"prefer":["move_west","move_north"],"prefer_weight":1.5},{"applies_to":"all","forbid":[],"prefer":["attack_type:Stalker"],"prefer_weight":1.8}]}
```

응답 메타데이터:

```json
{
  "status": 200,
  "finish_reason": "stop",
  "usage": {
    "prompt_tokens": 928,
    "total_tokens": 1191,
    "completion_tokens": 263,
    "prompt_tokens_details": null
  },
  "error": null
}
```

#### JSON 추출 결과（sanitizer 이전）

```json
{
  "strategy": "Use the two allied stalkers to kite the enemy stalkers while protecting the lone Zealot from melee contact, focusing fire on the enemy stalkers and retreating if the Zealot is threatened.",
  "subgoals": [
    {
      "predicate": "enemy_kill",
      "weight": 0.9,
      "unit_type": "Stalker"
    },
    {
      "predicate": "enemy_kill",
      "weight": 0.8,
      "unit_type": "Zealot"
    },
    {
      "predicate": "focus_fire",
      "weight": 0.85,
      "unit_type": "Stalker"
    },
    {
      "predicate": "protect_type",
      "weight": 0.8,
      "unit_type": "Zealot"
    },
    {
      "predicate": "retreat_low_health",
      "weight": 0.7,
      "unit_type": "Zealot"
    }
  ],
  "action_rules": [
    {
      "applies_to": "type:Stalker",
      "forbid": [],
      "prefer": [
        "attack_type:Stalker",
        "attack_lowest_health"
      ],
      "prefer_weight": 2.0
    },
    {
      "applies_to": "type:Zealot",
      "forbid": [],
      "prefer": [
        "move_west",
        "move_north"
      ],
      "prefer_weight": 1.5
    },
    {
      "applies_to": "all",
      "forbid": [],
      "prefer": [
        "attack_type:Stalker"
      ],
      "prefer_weight": 1.8
    }
  ]
}
```

#### sanitizer 이후 실제 반환 지침

```json
{
  "strategy": "Use the two allied stalkers to kite the enemy stalkers while protecting the lone Zealot from melee contact, focusing fire on the enemy stalkers and retreating if the Zealot is threatened.",
  "subgoals": [
    {
      "predicate": "enemy_kill",
      "weight": 0.9
    },
    {
      "predicate": "focus_fire",
      "weight": 0.85
    },
    {
      "predicate": "protect_type",
      "weight": 0.8,
      "unit_type": "Zealot"
    },
    {
      "predicate": "retreat_low_health",
      "weight": 0.7
    }
  ],
  "action_rules": [
    {
      "applies_to": "type:Stalker",
      "forbid": [],
      "prefer": [
        "attack_type:Stalker",
        "attack_lowest_health"
      ],
      "prefer_weight": 2.0
    },
    {
      "applies_to": "type:Zealot",
      "forbid": [],
      "prefer": [
        "move_west",
        "move_north"
      ],
      "prefer_weight": 1.5
    },
    {
      "applies_to": "all",
      "forbid": [],
      "prefer": [
        "attack_type:Stalker"
      ],
      "prefer_weight": 1.8
    }
  ]
}
```

#### 컴파일된 마스크

```json
{
  "hard": [
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ]
  ],
  "soft": [
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      2.0,
      2.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      2.0,
      2.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ]
  ],
  "avail_actions": [
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      1,
      1,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      1,
      1,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ]
  ]
}
```

행: ally index, 열: [noop, stop, north, south, east, west, enemy0…enemy4]. soft>1은 선호이며 확률이 아닙니다. availability가 최종 실행 가능성을 제한합니다. RSVP soft-only 실행에서는 아래 hard를 모두 1로 덮어씁니다.

### 3–6. paper_v2

#### 요청 1: 실제 messages

```json
[
  {
    "role": "system",
    "content": "You are the expert strategic Commander of an allied team in a StarCraft II micromanagement battle (SMAC). Decentralized reinforcement-learning agents choose individual actions and learn fine-grained control. You provide coarse-timescale strategic guidance, not a script of individual moves.\n\nTASK AND INFORMATION BOUNDARY\nEach allied agent has this discrete action space: noop (only when dead), stop, move_north, move_south, move_east, move_west, and attack_enemy_i for each enemy unit i (Medivacs heal allies instead of attacking).\nAllied unit types: Stalker, Zealot. Enemy unit types: Stalker, Zealot.\nThe task objective is to eliminate all enemy units while keeping the allied team effective. Do not assume or infer the simulator's true reward function; reason only from the observable situation and this task objective.\nUse only the supplied observable situation and task objective. Do not assume, request, infer, or refer to the simulator's true reward function, hidden enemy states, future states, or training statistics. Enemies not currently seen are not necessarily dead. A team-level engagement description is not proof that every ally can attack. Do not invent health, cooldowns, positions, or terrain absent from the input.\n\nGUIDANCE LIFETIME\nGuidance is reused between refreshes, with a configured interval or ceiling of 200 environment steps; it is not a one-step instruction. Prefer objectives that remain meaningful as agents move. A failed refresh can leave the previous guidance in use longer. Do not assume a new instruction will arrive immediately after a move.\n\nDECISION PROCEDURE\n1. Situation assessment: identify the observed threats, opportunities, force balance, phase, and important uncertainty.\n2. Strategic planning: choose a coherent phase objective that contributes to winning while leaving low-level execution to the agents.\n3. Task decomposition and grounding: choose a small set of evaluable sub-goals and priorities for this guidance interval. Check that each predicate and action rule actually implements the intended behavior.\nReturn only the JSON decision, not a reasoning transcript. In strategy give a concise observable reason and phase objective (at most 250 characters). This short rationale must agree with the executable fields.\n\nEXECUTABLE SUB-GOAL INTERFACE\nThe following are our auxiliary shaping predicates, NOT the environment reward. Weights in [0, 1] are nonnegative priorities multiplying signed predicate values, not probabilities. Equal weights do not imply equal numerical effects. Choose at most 6 unique, relevant predicates; an empty list is allowed. Do not fill the list just to reach a quota.\n- enemy_kill: +1 per newly eliminated enemy.\n- enemy_damage: +10 times enemy HP decrease divided by total enemy maximum HP.\n- ally_survive: -1 per newly dead ally; it does not give a positive bonus merely for staying alive.\n- kill_type: +1.5 per newly eliminated enemy of unit_type.\n- damage_type: +5 times HP decrease among enemies of unit_type divided by their total maximum HP.\n- protect_type: -1.5 per newly dead ally of unit_type, minus 3 times their HP decrease divided by their total maximum HP; this penalizes damage/death, not rewards healing or idling.\n- focus_fire: zero unless at least two living non-Medivac allies choose attack actions. Otherwise +0.3 times the largest same-target share, only if that share exceeds one half. It measures selected attack-target concentration, not confirmed damage. Omit when fewer than two combat allies remain.\n- retreat_low_health: +0.1 per living ally below 30% HP choosing a movement action that increases distance from the pre-step living-enemy centroid. It is not a generic bonus for moving, safe kiting, or reaching cover. The summary's 'critically low' label alone does not certify this exact threshold.\nThe last three typed names are kill_type, damage_type, protect_type: these require unit_type (an enemy type for kill/damage, an allied type for protect). All other predicates omit unit_type. Use only types supplied by the interface. Do not duplicate a predicate/type pair. Avoid overlapping generic and typed goals unless there is a distinct strategic reason; never pair them for the only type on a homogeneous side. Do not select type-targeted goals for an enemy type without current observational support. Do not assume a shaping predicate's name implements extra conditions or a longer temporal plan.\n\nEXECUTABLE ACTION INTERFACE\nUse at most 6 rules. Each applies_to selector is either all or type:<AllyTypeName>. Rules apply to EVERY living ally matching that selector until refresh; there is NO health, range, cooldown, time, or if/then condition in this schema. Text in strategy does not create executable conditions. If a conditional tactic cannot be expressed faithfully, omit that action rule; use an appropriate conditional predicate if one exists.\nAvailable tokens: stop, move_north, move_south, move_east, move_west, move_all, attack_all, attack_type:<EnemyTypeName>, attack_lowest_health, attack_nearest.\nTarget tokens are resolved anew at each environment step, with environment availability enforced separately. attack_nearest and attack_lowest_health resolve per agent and need not coordinate a shared target. Cardinal movement tokens retain their absolute direction, NOT a relative direction away from the enemy. Use a direction only if its persistence is justified by the observed situation; never encode 'retreat when hurt' as unconditional movement for a whole type. Without visible targets, attack preferences alone do not implement scouting.\nforbid is a hard prohibition; prefer is a soft bias, not a forced action. Usually leave forbid empty. Never forbid noop, stop, move_all, attack_all, all movement directions, or all enemy types. Do not forbid an action also preferred by any overlapping rule. Prefer weights should be moderate (1.5–2.5; allowed 1.1–3.0). Repeated or overlapping preferences combine by maximum weight, not addition. An empty action_rules list is valid when no durable, executable strategic bias is justified.\n\nOUTPUT CONTRACT\nOutput STRICT JSON only, with these three keys and no markdown. Replace placeholders with valid JSON values; omit unit_type for untyped predicates:\n{\n  \"strategy\": \"<brief observable reason and phase objective>\",\n  \"subgoals\": [{\"predicate\": \"<available predicate>\", \"weight\": 0.5, \"unit_type\": \"<typed predicates only>\"}],\n  \"action_rules\": [{\"applies_to\": \"all\", \"forbid\": [], \"prefer\": [\"<available token>\"], \"prefer_weight\": 1.5}]\n}\nBefore emitting JSON, verify type validity, predicate activation conditions, non-duplication, consistency with strategy, and safety over the guidance lifetime. Do not add unsupported conditional fields or promise behavior the executable interface cannot express.\n"
  },
  {
    "role": "user",
    "content": "Scenario: SMAC map '2s3z' (5 allied vs 5 enemy units).\nAllied force: 2/5 alive.\n  - Stalker: 2/2 alive, avg health 100% [ranged mechanical: mobile, shields regenerate]\n  - Zealot: 0/3 alive (eliminated) [melee: high damage up close, must reach targets]\nEnemy force: 3 unit(s) currently in sight (5 on the map at start).\n  - Stalker: 2/2 alive, avg health 96% [ranged mechanical: mobile, shields regenerate]\n  - Zealot: 1/1 alive, avg health 100% [melee: high damage up close, must reach targets]\nSpatial: army separation 5.3 map units; phase: engaged (within typical attack range).\nEnemy centroid lies to the east of the allied force.\n\nProduce your strategic guidance now as strict JSON."
  }
]
```

#### 실제 응답 content 원문

```text
{
  "strategy": "Engage enemy stalkers first to reduce ranged threat, then target the lone Zealot.",
  "subgoals": [
    {"predicate": "enemy_kill", "weight": 1.0},
    {"predicate": "enemy_damage", "weight": 0.5},
    {"predicate": "kill_type", "weight": 1.5, "unit_type": "Stalker"},
    {"predicate": "damage_type", "weight": 2.5, "unit_type": "Stalker"}
  ],
  "action_rules": [
    {
      "applies_to": "all",
      "forbid": [],
      "prefer": ["attack_type:Stalker"],
      "prefer_weight": 2.0
    }
  ]
}
```

응답 메타데이터:

```json
{
  "status": 200,
  "finish_reason": "stop",
  "usage": {
    "prompt_tokens": 1675,
    "total_tokens": 2143,
    "completion_tokens": 468,
    "prompt_tokens_details": null
  },
  "error": null
}
```

#### JSON 추출 결과（sanitizer 이전）

```json
{
  "strategy": "Engage enemy stalkers first to reduce ranged threat, then target the lone Zealot.",
  "subgoals": [
    {
      "predicate": "enemy_kill",
      "weight": 1.0
    },
    {
      "predicate": "enemy_damage",
      "weight": 0.5
    },
    {
      "predicate": "kill_type",
      "weight": 1.5,
      "unit_type": "Stalker"
    },
    {
      "predicate": "damage_type",
      "weight": 2.5,
      "unit_type": "Stalker"
    }
  ],
  "action_rules": [
    {
      "applies_to": "all",
      "forbid": [],
      "prefer": [
        "attack_type:Stalker"
      ],
      "prefer_weight": 2.0
    }
  ]
}
```

#### sanitizer 이후 실제 반환 지침

```json
{
  "strategy": "Engage enemy stalkers first to reduce ranged threat, then target the lone Zealot.",
  "subgoals": [
    {
      "predicate": "enemy_kill",
      "weight": 1.0
    },
    {
      "predicate": "enemy_damage",
      "weight": 0.5
    },
    {
      "predicate": "kill_type",
      "weight": 1.0,
      "unit_type": "Stalker"
    },
    {
      "predicate": "damage_type",
      "weight": 1.0,
      "unit_type": "Stalker"
    }
  ],
  "action_rules": [
    {
      "applies_to": "all",
      "forbid": [],
      "prefer": [
        "attack_type:Stalker"
      ],
      "prefer_weight": 2.0
    }
  ]
}
```

#### 컴파일된 마스크

```json
{
  "hard": [
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ]
  ],
  "soft": [
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      2.0,
      2.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      2.0,
      2.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ]
  ],
  "avail_actions": [
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      1,
      1,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      1,
      1,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ]
  ]
}
```

행: ally index, 열: [noop, stop, north, south, east, west, enemy0…enemy4]. soft>1은 선호이며 확률이 아닙니다. availability가 최종 실행 가능성을 제한합니다. RSVP soft-only 실행에서는 아래 hard를 모두 1로 덮어씁니다.

## 마지막 비종료 관측 — env step 51

### 1. 실제 환경에서 추출한 숫자 snapshot

```json
{
  "allies": [
    {
      "type": "Stalker",
      "hp": 3.0,
      "hp_max": 160.0,
      "x": 13.22802734375,
      "y": 14.431640625,
      "alive": true
    },
    {
      "type": "Stalker",
      "hp": 0.0,
      "hp_max": 160.0,
      "x": 13.436767578125,
      "y": 16.11962890625,
      "alive": false
    },
    {
      "type": "Zealot",
      "hp": 0.0,
      "hp_max": 150.0,
      "x": 18.031982421875,
      "y": 14.76806640625,
      "alive": false
    },
    {
      "type": "Zealot",
      "hp": 0.0,
      "hp_max": 150.0,
      "x": 16.638427734375,
      "y": 17.208984375,
      "alive": false
    },
    {
      "type": "Zealot",
      "hp": 0.0,
      "hp_max": 150.0,
      "x": 18.21533203125,
      "y": 16.7216796875,
      "alive": false
    }
  ],
  "enemies": [
    {
      "type": "Zealot",
      "hp": 0.0,
      "hp_max": 150.0,
      "x": 17.346435546875,
      "y": 16.56787109375,
      "alive": false,
      "visible": false
    },
    {
      "type": "Zealot",
      "hp": 0.0,
      "hp_max": 150.0,
      "x": 17.670654296875,
      "y": 15.75537109375,
      "alive": false,
      "visible": false
    },
    {
      "type": "Zealot",
      "hp": 39.0,
      "hp_max": 150.0,
      "x": 14.091064453125,
      "y": 15.283203125,
      "alive": true,
      "visible": true
    },
    {
      "type": "Stalker",
      "hp": 160.0,
      "hp_max": 160.0,
      "x": 19.1240234375,
      "y": 14.728759765625,
      "alive": true,
      "visible": true
    },
    {
      "type": "Stalker",
      "hp": 160.0,
      "hp_max": 160.0,
      "x": 19.730712890625,
      "y": 15.837890625,
      "alive": true,
      "visible": true
    }
  ],
  "n_actions": 11
}
```

원래 per-agent 관측 벡터와 availability는 첨부 JSON의 obs/avail_actions에 있습니다.

### 2. 기존 summary 함수가 만든 자연어

```text
Scenario: SMAC map '2s3z' (5 allied vs 5 enemy units).
Allied force: 1/5 alive.
  - Stalker: 1/2 alive, avg health 2%, 1 critically low [ranged mechanical: mobile, shields regenerate]
  - Zealot: 0/3 alive (eliminated) [melee: high damage up close, must reach targets]
Enemy force: 3 unit(s) currently in sight (5 on the map at start).
  - Stalker: 2/2 alive, avg health 100% [ranged mechanical: mobile, shields regenerate]
  - Zealot: 1/1 alive, avg health 26%, 1 critically low [melee: high damage up close, must reach targets]
Spatial: army separation 4.5 map units; phase: engaged (within typical attack range).
Enemy centroid lies to the east of the allied force.
```

### 3–6. paper

#### 요청 1: 실제 messages

```json
[
  {
    "role": "system",
    "content": "You are the LLM Commander of an allied team in a StarCraft II micromanagement battle (SMAC). You operate at a coarse strategic timescale; decentralized reinforcement-learning agents execute actions at high frequency and learn fine-grained control on their own. Your job is expert strategic cognition, NOT low-level control.\n\nEach allied agent has this discrete action space: noop (only when dead), stop, move_north, move_south, move_east, move_west, and attack_enemy_i for each enemy unit i (Medivacs heal allies instead of attacking).\nAllied unit types: Stalker, Zealot. Enemy unit types: Stalker, Zealot.\nThe task objective is to eliminate all enemy units while keeping the allied team effective. Do not assume or infer the simulator's true reward function; reason only from the observable situation and this task objective.\n\nReason in three explicit stages before answering:\n1. Situation assessment: threats, opportunities, force balance, engagement phase.\n2. Strategic planning: a coherent macro-strategy for this phase.\n3. Task decomposition: concrete, evaluable sub-goals with priorities.\n\nThen output STRICT JSON only (no markdown) with this schema:\n{\n  \"strategy\": \"<one concise sentence>\",\n  \"subgoals\": [ {\"predicate\": \"<predicate>\", \"weight\": <0.0-1.0>, \"unit_type\": \"<TypeName, typed predicates only>\"} ],\n  \"action_rules\": [ {\"applies_to\": \"all\" | \"type:<AllyTypeName>\", \"forbid\": [<token>...], \"prefer\": [<token>...], \"prefer_weight\": <1.1-3.0>} ]\n}\n\nReward predicates (at most 6; weight = priority): \"enemy_kill\", \"enemy_damage\", \"ally_survive\", \"focus_fire\", \"retreat_low_health\", \"kill_type\", \"damage_type\", \"protect_type\" (last three need unit_type).\n\nAction tokens: \"stop\", \"move_north\", \"move_south\", \"move_east\", \"move_west\", \"move_all\", \"attack_all\", \"attack_type:<EnemyTypeName>\", \"attack_lowest_health\", \"attack_nearest\".\n\nRules for action_rules:\n- \"forbid\" is a HARD constraint. Use it ONLY to rule out clearly infeasible or risky decisions (e.g. never forbid all attacks; never forbid retreat or stop, which agents need for kiting). Most of the time \"forbid\" should be an empty list.\n- \"prefer\" is a SOFT, moderate preference that biases but does not dictate action choice. Express strategic priorities (e.g. focus fire on the enemy's damage dealers via \"attack_type:<X>\"), not step-by-step micro. Keep prefer_weight moderate (1.5-2.5).\n- Do not assume, request, or refer to the simulator's true reward function. Choose only phase-relevant sub-goals and avoid generic/typed duplicates on homogeneous maps.\n- \"forbid\" MUST NOT contain \"noop\", \"stop\", \"move_all\", or \"attack_all\". Never put the same token in both \"forbid\" and \"prefer\"."
  },
  {
    "role": "user",
    "content": "Scenario: SMAC map '2s3z' (5 allied vs 5 enemy units).\nAllied force: 1/5 alive.\n  - Stalker: 1/2 alive, avg health 2%, 1 critically low [ranged mechanical: mobile, shields regenerate]\n  - Zealot: 0/3 alive (eliminated) [melee: high damage up close, must reach targets]\nEnemy force: 3 unit(s) currently in sight (5 on the map at start).\n  - Stalker: 2/2 alive, avg health 100% [ranged mechanical: mobile, shields regenerate]\n  - Zealot: 1/1 alive, avg health 26%, 1 critically low [melee: high damage up close, must reach targets]\nSpatial: army separation 4.5 map units; phase: engaged (within typical attack range).\nEnemy centroid lies to the east of the allied force.\n\nProduce your strategic guidance now as strict JSON."
  }
]
```

#### 실제 응답 content 원문

```text
{"strategy":"Use the lone stalker to kite the enemy, focusing fire on the enemy stalker while retreating when health is critically low.","subgoals":[{"predicate":"enemy_kill","weight":1.0,"unit_type":"Stalker"},{"predicate":"enemy_damage","weight":0.8,"unit_type":"Stalker"},{"predicate":"ally_survive","weight":0.9,"unit_type":"Stalker"},{"predicate":"focus_fire","weight":0.7,"unit_type":"Stalker"},{"predicate":"retreat_low_health","weight":0.6,"unit_type":"Stalker"}],"action_rules":[{"applies_to":"type:Stalker","forbid":[],"prefer":["attack_type:Stalker","move_west"],"prefer_weight":2.0}]}
```

응답 메타데이터:

```json
{
  "status": 200,
  "finish_reason": "stop",
  "usage": {
    "prompt_tokens": 936,
    "total_tokens": 1199,
    "completion_tokens": 263,
    "prompt_tokens_details": null
  },
  "error": null
}
```

#### JSON 추출 결과（sanitizer 이전）

```json
{
  "strategy": "Use the lone stalker to kite the enemy, focusing fire on the enemy stalker while retreating when health is critically low.",
  "subgoals": [
    {
      "predicate": "enemy_kill",
      "weight": 1.0,
      "unit_type": "Stalker"
    },
    {
      "predicate": "enemy_damage",
      "weight": 0.8,
      "unit_type": "Stalker"
    },
    {
      "predicate": "ally_survive",
      "weight": 0.9,
      "unit_type": "Stalker"
    },
    {
      "predicate": "focus_fire",
      "weight": 0.7,
      "unit_type": "Stalker"
    },
    {
      "predicate": "retreat_low_health",
      "weight": 0.6,
      "unit_type": "Stalker"
    }
  ],
  "action_rules": [
    {
      "applies_to": "type:Stalker",
      "forbid": [],
      "prefer": [
        "attack_type:Stalker",
        "move_west"
      ],
      "prefer_weight": 2.0
    }
  ]
}
```

#### sanitizer 이후 실제 반환 지침

```json
{
  "strategy": "Use the lone stalker to kite the enemy, focusing fire on the enemy stalker while retreating when health is critically low.",
  "subgoals": [
    {
      "predicate": "enemy_kill",
      "weight": 1.0
    },
    {
      "predicate": "enemy_damage",
      "weight": 0.8
    },
    {
      "predicate": "ally_survive",
      "weight": 0.9
    },
    {
      "predicate": "focus_fire",
      "weight": 0.7
    },
    {
      "predicate": "retreat_low_health",
      "weight": 0.6
    }
  ],
  "action_rules": [
    {
      "applies_to": "type:Stalker",
      "forbid": [],
      "prefer": [
        "attack_type:Stalker",
        "move_west"
      ],
      "prefer_weight": 2.0
    }
  ]
}
```

#### 컴파일된 마스크

```json
{
  "hard": [
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ]
  ],
  "soft": [
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      2.0,
      1.0,
      1.0,
      1.0,
      2.0,
      2.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ]
  ],
  "avail_actions": [
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      1,
      1,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ]
  ]
}
```

행: ally index, 열: [noop, stop, north, south, east, west, enemy0…enemy4]. soft>1은 선호이며 확률이 아닙니다. availability가 최종 실행 가능성을 제한합니다. RSVP soft-only 실행에서는 아래 hard를 모두 1로 덮어씁니다.

### 3–6. paper_v2

#### 요청 1: 실제 messages

```json
[
  {
    "role": "system",
    "content": "You are the expert strategic Commander of an allied team in a StarCraft II micromanagement battle (SMAC). Decentralized reinforcement-learning agents choose individual actions and learn fine-grained control. You provide coarse-timescale strategic guidance, not a script of individual moves.\n\nTASK AND INFORMATION BOUNDARY\nEach allied agent has this discrete action space: noop (only when dead), stop, move_north, move_south, move_east, move_west, and attack_enemy_i for each enemy unit i (Medivacs heal allies instead of attacking).\nAllied unit types: Stalker, Zealot. Enemy unit types: Stalker, Zealot.\nThe task objective is to eliminate all enemy units while keeping the allied team effective. Do not assume or infer the simulator's true reward function; reason only from the observable situation and this task objective.\nUse only the supplied observable situation and task objective. Do not assume, request, infer, or refer to the simulator's true reward function, hidden enemy states, future states, or training statistics. Enemies not currently seen are not necessarily dead. A team-level engagement description is not proof that every ally can attack. Do not invent health, cooldowns, positions, or terrain absent from the input.\n\nGUIDANCE LIFETIME\nGuidance is reused between refreshes, with a configured interval or ceiling of 200 environment steps; it is not a one-step instruction. Prefer objectives that remain meaningful as agents move. A failed refresh can leave the previous guidance in use longer. Do not assume a new instruction will arrive immediately after a move.\n\nDECISION PROCEDURE\n1. Situation assessment: identify the observed threats, opportunities, force balance, phase, and important uncertainty.\n2. Strategic planning: choose a coherent phase objective that contributes to winning while leaving low-level execution to the agents.\n3. Task decomposition and grounding: choose a small set of evaluable sub-goals and priorities for this guidance interval. Check that each predicate and action rule actually implements the intended behavior.\nReturn only the JSON decision, not a reasoning transcript. In strategy give a concise observable reason and phase objective (at most 250 characters). This short rationale must agree with the executable fields.\n\nEXECUTABLE SUB-GOAL INTERFACE\nThe following are our auxiliary shaping predicates, NOT the environment reward. Weights in [0, 1] are nonnegative priorities multiplying signed predicate values, not probabilities. Equal weights do not imply equal numerical effects. Choose at most 6 unique, relevant predicates; an empty list is allowed. Do not fill the list just to reach a quota.\n- enemy_kill: +1 per newly eliminated enemy.\n- enemy_damage: +10 times enemy HP decrease divided by total enemy maximum HP.\n- ally_survive: -1 per newly dead ally; it does not give a positive bonus merely for staying alive.\n- kill_type: +1.5 per newly eliminated enemy of unit_type.\n- damage_type: +5 times HP decrease among enemies of unit_type divided by their total maximum HP.\n- protect_type: -1.5 per newly dead ally of unit_type, minus 3 times their HP decrease divided by their total maximum HP; this penalizes damage/death, not rewards healing or idling.\n- focus_fire: zero unless at least two living non-Medivac allies choose attack actions. Otherwise +0.3 times the largest same-target share, only if that share exceeds one half. It measures selected attack-target concentration, not confirmed damage. Omit when fewer than two combat allies remain.\n- retreat_low_health: +0.1 per living ally below 30% HP choosing a movement action that increases distance from the pre-step living-enemy centroid. It is not a generic bonus for moving, safe kiting, or reaching cover. The summary's 'critically low' label alone does not certify this exact threshold.\nThe last three typed names are kill_type, damage_type, protect_type: these require unit_type (an enemy type for kill/damage, an allied type for protect). All other predicates omit unit_type. Use only types supplied by the interface. Do not duplicate a predicate/type pair. Avoid overlapping generic and typed goals unless there is a distinct strategic reason; never pair them for the only type on a homogeneous side. Do not select type-targeted goals for an enemy type without current observational support. Do not assume a shaping predicate's name implements extra conditions or a longer temporal plan.\n\nEXECUTABLE ACTION INTERFACE\nUse at most 6 rules. Each applies_to selector is either all or type:<AllyTypeName>. Rules apply to EVERY living ally matching that selector until refresh; there is NO health, range, cooldown, time, or if/then condition in this schema. Text in strategy does not create executable conditions. If a conditional tactic cannot be expressed faithfully, omit that action rule; use an appropriate conditional predicate if one exists.\nAvailable tokens: stop, move_north, move_south, move_east, move_west, move_all, attack_all, attack_type:<EnemyTypeName>, attack_lowest_health, attack_nearest.\nTarget tokens are resolved anew at each environment step, with environment availability enforced separately. attack_nearest and attack_lowest_health resolve per agent and need not coordinate a shared target. Cardinal movement tokens retain their absolute direction, NOT a relative direction away from the enemy. Use a direction only if its persistence is justified by the observed situation; never encode 'retreat when hurt' as unconditional movement for a whole type. Without visible targets, attack preferences alone do not implement scouting.\nforbid is a hard prohibition; prefer is a soft bias, not a forced action. Usually leave forbid empty. Never forbid noop, stop, move_all, attack_all, all movement directions, or all enemy types. Do not forbid an action also preferred by any overlapping rule. Prefer weights should be moderate (1.5–2.5; allowed 1.1–3.0). Repeated or overlapping preferences combine by maximum weight, not addition. An empty action_rules list is valid when no durable, executable strategic bias is justified.\n\nOUTPUT CONTRACT\nOutput STRICT JSON only, with these three keys and no markdown. Replace placeholders with valid JSON values; omit unit_type for untyped predicates:\n{\n  \"strategy\": \"<brief observable reason and phase objective>\",\n  \"subgoals\": [{\"predicate\": \"<available predicate>\", \"weight\": 0.5, \"unit_type\": \"<typed predicates only>\"}],\n  \"action_rules\": [{\"applies_to\": \"all\", \"forbid\": [], \"prefer\": [\"<available token>\"], \"prefer_weight\": 1.5}]\n}\nBefore emitting JSON, verify type validity, predicate activation conditions, non-duplication, consistency with strategy, and safety over the guidance lifetime. Do not add unsupported conditional fields or promise behavior the executable interface cannot express.\n"
  },
  {
    "role": "user",
    "content": "Scenario: SMAC map '2s3z' (5 allied vs 5 enemy units).\nAllied force: 1/5 alive.\n  - Stalker: 1/2 alive, avg health 2%, 1 critically low [ranged mechanical: mobile, shields regenerate]\n  - Zealot: 0/3 alive (eliminated) [melee: high damage up close, must reach targets]\nEnemy force: 3 unit(s) currently in sight (5 on the map at start).\n  - Stalker: 2/2 alive, avg health 100% [ranged mechanical: mobile, shields regenerate]\n  - Zealot: 1/1 alive, avg health 26%, 1 critically low [melee: high damage up close, must reach targets]\nSpatial: army separation 4.5 map units; phase: engaged (within typical attack range).\nEnemy centroid lies to the east of the allied force.\n\nProduce your strategic guidance now as strict JSON."
  }
]
```

#### 실제 응답 content 원문

```text
{"strategy":"Retreat west to avoid being killed while still attempting to damage enemies.","subgoals":[{"predicate":"protect_type","weight":1.5,"unit_type":"Stalker"},{"predicate":"enemy_damage","weight":0.5}],"action_rules":[{"applies_to":"all","forbid":[],"prefer":["move_west","attack_nearest"],"prefer_weight":1.5}]}
```

응답 메타데이터:

```json
{
  "status": 200,
  "finish_reason": "stop",
  "usage": {
    "prompt_tokens": 1683,
    "total_tokens": 2170,
    "completion_tokens": 487,
    "prompt_tokens_details": null
  },
  "error": null
}
```

#### JSON 추출 결과（sanitizer 이전）

```json
{
  "strategy": "Retreat west to avoid being killed while still attempting to damage enemies.",
  "subgoals": [
    {
      "predicate": "protect_type",
      "weight": 1.5,
      "unit_type": "Stalker"
    },
    {
      "predicate": "enemy_damage",
      "weight": 0.5
    }
  ],
  "action_rules": [
    {
      "applies_to": "all",
      "forbid": [],
      "prefer": [
        "move_west",
        "attack_nearest"
      ],
      "prefer_weight": 1.5
    }
  ]
}
```

#### sanitizer 이후 실제 반환 지침

```json
{
  "strategy": "Retreat west to avoid being killed while still attempting to damage enemies.",
  "subgoals": [
    {
      "predicate": "protect_type",
      "weight": 1.0,
      "unit_type": "Stalker"
    },
    {
      "predicate": "enemy_damage",
      "weight": 0.5
    }
  ],
  "action_rules": [
    {
      "applies_to": "all",
      "forbid": [],
      "prefer": [
        "move_west",
        "attack_nearest"
      ],
      "prefer_weight": 1.5
    }
  ]
}
```

#### 컴파일된 마스크

```json
{
  "hard": [
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ]
  ],
  "soft": [
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.5,
      1.0,
      1.0,
      1.5,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ]
  ],
  "avail_actions": [
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      1,
      1,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ]
  ]
}
```

행: ally index, 열: [noop, stop, north, south, east, west, enemy0…enemy4]. soft>1은 선호이며 확률이 아닙니다. availability가 최종 실행 가능성을 제한합니다. RSVP soft-only 실행에서는 아래 hard를 모두 1로 덮어씁니다.
