# 실제 LLM 지침 검토 사례 — 2026-09-12

## 먼저 알아둘 기록 한계

이 문서는 최근 dense 2s3z 실험 4개에서 추출한 **실제 저장 지침 16개**입니다. 모두 기존 `paper` 프롬프트를 사용하는 실행이며, `paper_v2`의 결과가 아닙니다.

**요청하신 입력·출력 원문 전체는 기존 로그만으로 제공할 수 없습니다.** 당시 관측 요약 문장과 API 응답 원문을 저장하지 않았기 때문입니다. 없는 내용을 만들어 채우지 않고 아래처럼 구분합니다.

| 항목 | 보존 여부 / 이 문서의 처리 |
|---|---|
| 매 호출의 관측 입력 문장 `summary` | 미저장. 복원 불가 |
| 상태 키 `cache_key` | 저장됨. 사례마다 원문 첨부. 전체 관측 문장을 대체하지 않음 |
| 공통 system prompt | 호출 당시 원문 미저장. 현재 코드의 `PAPER_SYSTEM_PROMPT` 템플릿을 부록에 첨부 |
| API 응답 원문 | 미저장. 파싱·sanitizer 처리 이후 `guidance`만 첨부 |
| 내부 추론 / reasoning 내용 | 미저장. `strategy`는 짧은 출력 요약이지 내부 추론 원문이 아님 |
| 학습 성능 | 판단 편향을 줄이기 위해 이 문서에서 제외 |

`cache_key`는 압축·구간화된 상태입니다. 같은 키라고 같은 LLM 입력이었다고 볼 수 없습니다. 또한 이름에 cache가 들어가지만 이것만으로 응답 캐싱이 활성화됐다는 뜻은 아닙니다.

## 선정 방법과 읽는 순서

각 실행에서 (1) 적 미관측, (2) 다수 아군 생존·적 관측, (3) 아군 한 명 생존의 **첫 번째 해당 기록**, 그리고 (4) 추출 시점의 마지막 유효 기록을 골랐습니다. 같은 실행 안의 중복 행은 제외했습니다. 이는 상황별 검토 표본이며 오류율이나 전체 출력 분포를 추정하는 무작위 표본이 아닙니다.

먼저 저장된 상태 키와 출력만 읽고 판단해 주세요. 전체 관측이 필요한 판단은 '판단 불가'로 남기는 것이 맞습니다.

- 전략 자체가 관측으로 뒷받침되는가? → 입력 원문이 없으므로 상당 부분 판단 불가
- 전략의 의도와 선택한 predicate/action rule이 일치하는가?
- 조건부 전략을 무조건 이동 선호로 잘못 옮기지는 않았는가?
- 아군 한 명에게 focus_fire처럼 활성화 불가능한 지침이 있는가?
- 다음 갱신까지 유지될 때 위험할 수 있는 방향 지침이 있는가?

실행별 설정은 서로 다릅니다. RSVP와 LEHCA 출력을 여기서 나란히 보여준다고 해서 통제된 알고리즘 비교가 되는 것은 아닙니다.

## 마스킹 RSVP seed 0

원자료: JSONL: `results/guidance/rsvp__2026-09-12_18-52-21__p645636_dense_softtrain_20260912_2s3z_soft_rsvp_b01_1m_s0_p645636.jsonl` (서버 보존 자료). 추출 시점 읽은 기록 1344개. 실행 중인 파일이므로 이후 행이 추가될 수 있습니다.

### 사례 01 — 적 미관측

원자료 행: **1**. `t_env=0`, `t_global=0`, `cache_hit=false`, `early=false`.

입력 — 당시 관측 문장: **미저장**. 저장된 상태 키 원문:

```text
2s3z|A:Stalker:2:4|A:Zealot:3:4|nosight
```

출력 — 파싱·정제 후 실제 저장 JSON (들여쓰기만 변경):

```json
{
  "strategy": "Scout for enemy units, then engage with ranged Stalkers first while Zealots close in for melee damage, maintaining optimal distance and avoiding overcommitment.",
  "subgoals": [
    {
      "predicate": "enemy_kill",
      "weight": 0.9
    },
    {
      "predicate": "enemy_damage",
      "weight": 0.7
    },
    {
      "predicate": "ally_survive",
      "weight": 0.8
    },
    {
      "predicate": "focus_fire",
      "weight": 0.7
    }
  ],
  "action_rules": [
    {
      "applies_to": "all",
      "forbid": [],
      "prefer": [
        "attack_nearest"
      ],
      "prefer_weight": 1.8
    },
    {
      "applies_to": "type:Stalker",
      "forbid": [],
      "prefer": [
        "attack_type:Stalker"
      ],
      "prefer_weight": 2
    },
    {
      "applies_to": "type:Zealot",
      "forbid": [],
      "prefer": [
        "attack_type:Zealot"
      ],
      "prefer_weight": 1.5
    }
  ]
}
```

검토 메모: 전략–실행 일치 여부 / 유지 시 위험 / 추가로 필요한 관측 정보를 기록해 주세요.

### 사례 02 — 다수 아군 생존·적 관측

원자료 행: **3**. `t_env=372`, `t_global=400`, `cache_hit=false`, `early=false`.

입력 — 당시 관측 문장: **미저장**. 저장된 상태 키 원문:

```text
2s3z|A:Stalker:2:3|A:Zealot:1:4|E:Stalker:2:4|E:Zealot:3:2|d1
```

출력 — 파싱·정제 후 실제 저장 JSON (들여쓰기만 변경):

```json
{
  "strategy": "Use the two Stalkers to kite the enemy Stalkers while the Zealot closes in on the nearest Zealot, focusing fire on the enemy Stalkers to reduce ranged threat.",
  "subgoals": [
    {
      "predicate": "enemy_kill",
      "weight": 0.9
    },
    {
      "predicate": "focus_fire",
      "weight": 0.7
    },
    {
      "predicate": "enemy_damage",
      "weight": 0.6
    },
    {
      "predicate": "ally_survive",
      "weight": 0.5
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
      "prefer_weight": 2
    },
    {
      "applies_to": "type:Zealot",
      "forbid": [],
      "prefer": [
        "attack_lowest_health",
        "move_north"
      ],
      "prefer_weight": 1.8
    },
    {
      "applies_to": "all",
      "forbid": [],
      "prefer": [
        "attack_type:Zealot"
      ],
      "prefer_weight": 1.5
    }
  ]
}
```

검토 메모: 전략–실행 일치 여부 / 유지 시 위험 / 추가로 필요한 관측 정보를 기록해 주세요.

### 사례 03 — 아군 1명 생존

원자료 행: **2**. `t_env=160`, `t_global=200`, `cache_hit=false`, `early=false`.

입력 — 당시 관측 문장: **미저장**. 저장된 상태 키 원문:

```text
2s3z|A:Stalker:1:4|A:Zealot:0:-1|E:Stalker:1:4|E:Zealot:3:2|d1
```

출력 — 파싱·정제 후 실제 저장 JSON (들여쓰기만 변경):

```json
{
  "strategy": "Use the lone stalker to kite the enemy, focus fire on the enemy stalker, and retreat when low on health while preserving the stalker's mobility.",
  "subgoals": [
    {
      "predicate": "enemy_kill",
      "weight": 0.9
    },
    {
      "predicate": "enemy_damage",
      "weight": 0.8
    },
    {
      "predicate": "ally_survive",
      "weight": 0.7
    },
    {
      "predicate": "focus_fire",
      "weight": 0.6
    },
    {
      "predicate": "retreat_low_health",
      "weight": 0.5
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
      "prefer_weight": 1.8
    },
    {
      "applies_to": "type:Stalker",
      "forbid": [],
      "prefer": [
        "move_north",
        "move_west"
      ],
      "prefer_weight": 1.5
    }
  ]
}
```

검토 메모: 전략–실행 일치 여부 / 유지 시 위험 / 추가로 필요한 관측 정보를 기록해 주세요.

### 사례 04 — 로그 후반 사례

원자료 행: **1344**. `t_env=193367`, `t_global=193387`, `cache_hit=false`, `early=true`.

입력 — 당시 관측 문장: **미저장**. 저장된 상태 키 원문:

```text
2s3z|A:Stalker:2:3|A:Zealot:3:1|E:Stalker:2:2|E:Zealot:3:3|d0
```

출력 — 파싱·정제 후 실제 저장 JSON (들여쓰기만 변경):

```json
{
  "strategy": "Use Zealots to absorb Zealot damage while Stalkers focus fire on enemy Stalkers, then kite the enemy to the north and finish the Zealots.",
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
        "attack_type:Stalker"
      ],
      "prefer_weight": 2
    },
    {
      "applies_to": "type:Zealot",
      "forbid": [],
      "prefer": [
        "attack_lowest_health",
        "attack_nearest"
      ],
      "prefer_weight": 1.5
    },
    {
      "applies_to": "all",
      "forbid": [],
      "prefer": [
        "move_north"
      ],
      "prefer_weight": 1.3
    }
  ]
}
```

검토 메모: 전략–실행 일치 여부 / 유지 시 위험 / 추가로 필요한 관측 정보를 기록해 주세요.

## 마스킹 RSVP seed 1

원자료: JSONL: `results/guidance/rsvp__2026-09-12_18-52-22__p3249831_dense_softtrain_20260912_2s3z_soft_rsvp_b01_1m_s1_p3249831.jsonl` (서버 보존 자료). 추출 시점 읽은 기록 1312개. 실행 중인 파일이므로 이후 행이 추가될 수 있습니다.

### 사례 05 — 적 미관측

원자료 행: **1**. `t_env=0`, `t_global=0`, `cache_hit=false`, `early=false`.

입력 — 당시 관측 문장: **미저장**. 저장된 상태 키 원문:

```text
2s3z|A:Stalker:2:4|A:Zealot:3:4|nosight
```

출력 — 파싱·정제 후 실제 저장 JSON (들여쓰기만 변경):

```json
{
  "strategy": "Search the map for hidden enemies while maintaining a defensive formation and be ready to engage immediately upon detection.",
  "subgoals": [
    {
      "predicate": "enemy_kill",
      "weight": 0.9
    },
    {
      "predicate": "enemy_damage",
      "weight": 0.8
    },
    {
      "predicate": "ally_survive",
      "weight": 0.7
    },
    {
      "predicate": "focus_fire",
      "weight": 0.6
    }
  ],
  "action_rules": [
    {
      "applies_to": "all",
      "forbid": [],
      "prefer": [
        "move_all"
      ],
      "prefer_weight": 1.8
    },
    {
      "applies_to": "type:Stalker",
      "forbid": [],
      "prefer": [
        "attack_nearest"
      ],
      "prefer_weight": 2
    },
    {
      "applies_to": "type:Zealot",
      "forbid": [],
      "prefer": [
        "attack_nearest"
      ],
      "prefer_weight": 2
    }
  ]
}
```

검토 메모: 전략–실행 일치 여부 / 유지 시 위험 / 추가로 필요한 관측 정보를 기록해 주세요.

### 사례 06 — 다수 아군 생존·적 관측

원자료 행: **12**. `t_env=1993`, `t_global=2022`, `cache_hit=false`, `early=false`.

입력 — 당시 관측 문장: **미저장**. 저장된 상태 키 원문:

```text
2s3z|A:Stalker:2:2|A:Zealot:1:1|E:Stalker:2:3|E:Zealot:3:2|d0
```

출력 — 파싱·정제 후 실제 저장 JSON (들여쓰기만 변경):

```json
{
  "strategy": "Use the Stalkers to kite the enemy Stalkers while the Zealot engages the nearest Zealot, prioritizing focus fire on enemy Stalkers and protecting the Zealot from melee damage.",
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
      "predicate": "ally_survive",
      "weight": 0.7
    },
    {
      "predicate": "enemy_damage",
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
      "prefer_weight": 2
    },
    {
      "applies_to": "type:Zealot",
      "forbid": [],
      "prefer": [
        "attack_nearest",
        "move_north"
      ],
      "prefer_weight": 1.8
    },
    {
      "applies_to": "all",
      "forbid": [],
      "prefer": [
        "attack_lowest_health"
      ],
      "prefer_weight": 1.5
    }
  ]
}
```

검토 메모: 전략–실행 일치 여부 / 유지 시 위험 / 추가로 필요한 관측 정보를 기록해 주세요.

### 사례 07 — 아군 1명 생존

원자료 행: **3**. `t_env=350`, `t_global=400`, `cache_hit=false`, `early=false`.

입력 — 당시 관측 문장: **미저장**. 저장된 상태 키 원문:

```text
2s3z|A:Stalker:0:-1|A:Zealot:1:0|E:Stalker:2:3|E:Zealot:3:2|d0
```

출력 — 파싱·정제 후 실제 저장 JSON (들여쓰기만 변경):

```json
{
  "strategy": "Retreat the lone Zealot while focusing fire on the enemy stalkers to reduce the enemy threat before regrouping.",
  "subgoals": [
    {
      "predicate": "enemy_kill",
      "weight": 0.9
    },
    {
      "predicate": "enemy_damage",
      "weight": 0.7
    },
    {
      "predicate": "ally_survive",
      "weight": 0.8
    },
    {
      "predicate": "retreat_low_health",
      "weight": 0.6
    },
    {
      "predicate": "focus_fire",
      "weight": 0.8
    }
  ],
  "action_rules": [
    {
      "applies_to": "type:Zealot",
      "forbid": [],
      "prefer": [
        "move_west",
        "attack_type:Stalker"
      ],
      "prefer_weight": 2
    },
    {
      "applies_to": "all",
      "forbid": [],
      "prefer": [
        "attack_type:Stalker"
      ],
      "prefer_weight": 1.5
    }
  ]
}
```

검토 메모: 전략–실행 일치 여부 / 유지 시 위험 / 추가로 필요한 관측 정보를 기록해 주세요.

### 사례 08 — 로그 후반 사례

원자료 행: **1312**. `t_env=190657`, `t_global=190697`, `cache_hit=false`, `early=false`.

입력 — 당시 관측 문장: **미저장**. 저장된 상태 키 원문:

```text
2s3z|A:Stalker:1:1|A:Zealot:1:0|E:Stalker:2:2|E:Zealot:3:2|d0
```

출력 — 파싱·정제 후 실제 저장 JSON (들여쓰기만 변경):

```json
{
  "strategy": "Prioritize kiting the enemy Stalkers while funneling Zealots into a melee kill zone, preserving the lone Zealot and the remaining Stalker.",
  "subgoals": [
    {
      "predicate": "enemy_kill",
      "weight": 0.9
    },
    {
      "predicate": "ally_survive",
      "weight": 0.7
    },
    {
      "predicate": "focus_fire",
      "weight": 0.85
    },
    {
      "predicate": "protect_type",
      "weight": 0.75,
      "unit_type": "Zealot"
    }
  ],
  "action_rules": [
    {
      "applies_to": "all",
      "forbid": [],
      "prefer": [
        "attack_type:Stalker",
        "move_west"
      ],
      "prefer_weight": 1.8
    },
    {
      "applies_to": "type:Zealot",
      "forbid": [],
      "prefer": [
        "attack_lowest_health",
        "move_north"
      ],
      "prefer_weight": 1.6
    }
  ]
}
```

검토 메모: 전략–실행 일치 여부 / 유지 시 위험 / 추가로 필요한 관측 정보를 기록해 주세요.

## corrected LEHCA seed 2

원자료: JSONL: `results/guidance/lehca__2026-09-12_16-15-21__p563296_dense_dedup300000_20260912_2s3z_lehca_b01_eps300k_s2_p563296.jsonl` (서버 보존 자료). 추출 시점 읽은 기록 3627개. 실행 중인 파일이므로 이후 행이 추가될 수 있습니다.

### 사례 09 — 적 미관측

원자료 행: **1**. `t_env=0`, `t_global=0`, `cache_hit=false`.

입력 — 당시 관측 문장: **미저장**. 저장된 상태 키 원문:

```text
2s3z|A:Stalker:2:4|A:Zealot:3:4|nosight
```

출력 — 파싱·정제 후 실제 저장 JSON (들여쓰기만 변경):

```json
{
  "strategy": "Search for enemies, then engage with ranged units first and melee units to finish off.",
  "subgoals": [
    {
      "predicate": "enemy_kill",
      "weight": 0.9
    },
    {
      "predicate": "enemy_damage",
      "weight": 0.7
    },
    {
      "predicate": "ally_survive",
      "weight": 0.5
    }
  ],
  "action_rules": [
    {
      "applies_to": "all",
      "forbid": [],
      "prefer": [
        "attack_type:Stalker",
        "attack_lowest_health"
      ],
      "prefer_weight": 1.8
    },
    {
      "applies_to": "type:Zealot",
      "forbid": [],
      "prefer": [
        "attack_type:Stalker",
        "attack_lowest_health"
      ],
      "prefer_weight": 1.6
    }
  ]
}
```

검토 메모: 전략–실행 일치 여부 / 유지 시 위험 / 추가로 필요한 관측 정보를 기록해 주세요.

### 사례 10 — 다수 아군 생존·적 관측

원자료 행: **11**. `t_env=472`, `t_global=500`, `cache_hit=false`.

입력 — 당시 관측 문장: **미저장**. 저장된 상태 키 원문:

```text
2s3z|A:Stalker:2:4|A:Zealot:1:1|E:Stalker:2:4|E:Zealot:3:3|d1
```

출력 — 파싱·정제 후 실제 저장 JSON (들여쓰기만 변경):

```json
{
  "strategy": "Use the two Stalkers to kite the enemy Stalkers while the Zealot closes in on the nearest Zealot, focusing fire on the enemy Stalkers first and protecting the Zealot from melee damage.",
  "subgoals": [
    {
      "predicate": "enemy_kill",
      "weight": 0.9
    },
    {
      "predicate": "protect_type",
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
      "prefer_weight": 2
    },
    {
      "applies_to": "type:Zealot",
      "forbid": [],
      "prefer": [
        "attack_nearest",
        "attack_type:Zealot"
      ],
      "prefer_weight": 1.8
    },
    {
      "applies_to": "all",
      "forbid": [],
      "prefer": [
        "move_north",
        "move_west"
      ],
      "prefer_weight": 1.5
    }
  ]
}
```

검토 메모: 전략–실행 일치 여부 / 유지 시 위험 / 추가로 필요한 관측 정보를 기록해 주세요.

### 사례 11 — 아군 1명 생존

원자료 행: **2**. `t_env=0`, `t_global=50`, `cache_hit=false`.

입력 — 당시 관측 문장: **미저장**. 저장된 상태 키 원문:

```text
2s3z|A:Stalker:0:-1|A:Zealot:1:0|E:Stalker:2:3|E:Zealot:3:2|d0
```

출력 — 파싱·정제 후 실제 저장 JSON (들여쓰기만 변경):

```json
{
  "strategy": "Retreat the lone Zealot while focusing fire on the nearest enemy Stalker to reduce damage intake.",
  "subgoals": [
    {
      "predicate": "retreat_low_health",
      "weight": 0.9
    },
    {
      "predicate": "focus_fire",
      "weight": 0.8
    },
    {
      "predicate": "enemy_kill",
      "weight": 0.5
    }
  ],
  "action_rules": [
    {
      "applies_to": "type:Zealot",
      "forbid": [],
      "prefer": [
        "move_west",
        "move_south"
      ],
      "prefer_weight": 2
    },
    {
      "applies_to": "type:Zealot",
      "forbid": [],
      "prefer": [
        "attack_type:Stalker",
        "attack_lowest_health"
      ],
      "prefer_weight": 1.8
    }
  ]
}
```

검토 메모: 전략–실행 일치 여부 / 유지 시 위험 / 추가로 필요한 관측 정보를 기록해 주세요.

### 사례 12 — 로그 후반 사례

원자료 행: **3627**. `t_env=181258`, `t_global=181300`, `cache_hit=false`.

입력 — 당시 관측 문장: **미저장**. 저장된 상태 키 원문:

```text
2s3z|A:Stalker:1:0|A:Zealot:0:-1|E:Stalker:2:2|E:Zealot:3:1|d1
```

출력 — 파싱·정제 후 실제 저장 JSON (들여쓰기만 변경):

```json
{
  "strategy": "Retreat while kiting the enemy, focusing fire on the lowest‑health units to preserve the lone stalker.",
  "subgoals": [
    {
      "predicate": "enemy_kill",
      "weight": 0.9
    },
    {
      "predicate": "enemy_damage",
      "weight": 0.8
    },
    {
      "predicate": "ally_survive",
      "weight": 1
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
        "attack_lowest_health",
        "attack_type:Stalker"
      ],
      "prefer_weight": 2
    },
    {
      "applies_to": "all",
      "forbid": [],
      "prefer": [
        "move_west"
      ],
      "prefer_weight": 1.5
    }
  ]
}
```

검토 메모: 전략–실행 일치 여부 / 유지 시 위험 / 추가로 필요한 관측 정보를 기록해 주세요.

## corrected LEHCA seed 3

원자료: JSONL: `results/guidance/lehca__2026-09-12_16-22-11__p566000_dense_dedup300000_20260912_2s3z_lehca_b01_eps300k_s3_p566000.jsonl` (서버 보존 자료). 추출 시점 읽은 기록 3611개. 실행 중인 파일이므로 이후 행이 추가될 수 있습니다.

### 사례 13 — 적 미관측

원자료 행: **1**. `t_env=0`, `t_global=0`, `cache_hit=false`.

입력 — 당시 관측 문장: **미저장**. 저장된 상태 키 원문:

```text
2s3z|A:Stalker:2:4|A:Zealot:3:4|nosight
```

출력 — 파싱·정제 후 실제 저장 JSON (들여쓰기만 변경):

```json
{
  "strategy": "Search the map for hidden enemies while maintaining a defensive formation and be ready to engage immediately upon detection.",
  "subgoals": [
    {
      "predicate": "enemy_kill",
      "weight": 0.9
    },
    {
      "predicate": "enemy_damage",
      "weight": 0.8
    },
    {
      "predicate": "ally_survive",
      "weight": 0.7
    },
    {
      "predicate": "focus_fire",
      "weight": 0.6
    }
  ],
  "action_rules": [
    {
      "applies_to": "all",
      "forbid": [],
      "prefer": [
        "move_all"
      ],
      "prefer_weight": 1.5
    },
    {
      "applies_to": "type:Stalker",
      "forbid": [],
      "prefer": [
        "attack_nearest"
      ],
      "prefer_weight": 2
    },
    {
      "applies_to": "type:Zealot",
      "forbid": [],
      "prefer": [
        "attack_nearest"
      ],
      "prefer_weight": 2
    }
  ]
}
```

검토 메모: 전략–실행 일치 여부 / 유지 시 위험 / 추가로 필요한 관측 정보를 기록해 주세요.

### 사례 14 — 다수 아군 생존·적 관측

원자료 행: **19**. `t_env=874`, `t_global=900`, `cache_hit=false`.

입력 — 당시 관측 문장: **미저장**. 저장된 상태 키 원문:

```text
2s3z|A:Stalker:2:3|A:Zealot:1:4|E:Stalker:2:4|E:Zealot:3:3|d1
```

출력 — 파싱·정제 후 실제 저장 JSON (들여쓰기만 변경):

```json
{
  "strategy": "Use the two Stalkers to kite the enemy Stalkers while the Zealot closes in on the Zealot cluster, focusing fire on the enemy Stalkers first to reduce ranged threat.",
  "subgoals": [
    {
      "predicate": "enemy_kill",
      "weight": 0.9
    },
    {
      "predicate": "focus_fire",
      "weight": 0.7
    },
    {
      "predicate": "ally_survive",
      "weight": 0.6
    },
    {
      "predicate": "retreat_low_health",
      "weight": 0.5
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
      "prefer_weight": 1.8
    },
    {
      "applies_to": "type:Zealot",
      "forbid": [],
      "prefer": [
        "attack_type:Zealot",
        "move_east"
      ],
      "prefer_weight": 1.6
    },
    {
      "applies_to": "all",
      "forbid": [],
      "prefer": [
        "attack_lowest_health"
      ],
      "prefer_weight": 1.5
    }
  ]
}
```

검토 메모: 전략–실행 일치 여부 / 유지 시 위험 / 추가로 필요한 관측 정보를 기록해 주세요.

### 사례 15 — 아군 1명 생존

원자료 행: **2**. `t_env=0`, `t_global=50`, `cache_hit=false`.

입력 — 당시 관측 문장: **미저장**. 저장된 상태 키 원문:

```text
2s3z|A:Stalker:1:2|A:Zealot:0:-1|E:Zealot:3:3|d0
```

출력 — 파싱·정제 후 실제 저장 JSON (들여쓰기만 변경):

```json
{
  "strategy": "Use the lone stalker to kite and focus fire on the enemy Zealots while preserving its health, retreating if its health drops below 30%.",
  "subgoals": [
    {
      "predicate": "enemy_kill",
      "weight": 0.9
    },
    {
      "predicate": "ally_survive",
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
        "attack_type:Zealot",
        "attack_lowest_health"
      ],
      "prefer_weight": 1.8
    },
    {
      "applies_to": "all",
      "forbid": [],
      "prefer": [
        "move_north",
        "move_west"
      ],
      "prefer_weight": 1.5
    }
  ]
}
```

검토 메모: 전략–실행 일치 여부 / 유지 시 위험 / 추가로 필요한 관측 정보를 기록해 주세요.

### 사례 16 — 로그 후반 사례

원자료 행: **3611**. `t_env=180469`, `t_global=180500`, `cache_hit=false`.

입력 — 당시 관측 문장: **미저장**. 저장된 상태 키 원문:

```text
2s3z|A:Stalker:1:1|A:Zealot:2:1|E:Stalker:2:2|E:Zealot:2:3|d0
```

출력 — 파싱·정제 후 실제 저장 JSON (들여쓰기만 변경):

```json
{
  "strategy": "Prioritize kiting the enemy stalkers while funneling enemy zealots into a kill zone, preserving the low‑health allied units.",
  "subgoals": [
    {
      "predicate": "enemy_kill",
      "weight": 0.9
    },
    {
      "predicate": "focus_fire",
      "weight": 0.7
    },
    {
      "predicate": "protect_type",
      "weight": 0.6,
      "unit_type": "Zealot"
    },
    {
      "predicate": "retreat_low_health",
      "weight": 0.5
    },
    {
      "predicate": "enemy_damage",
      "weight": 0.4
    }
  ],
  "action_rules": [
    {
      "applies_to": "all",
      "forbid": [],
      "prefer": [
        "attack_type:Stalker",
        "attack_lowest_health"
      ],
      "prefer_weight": 1.8
    },
    {
      "applies_to": "type:Zealot",
      "forbid": [],
      "prefer": [
        "attack_type:Zealot",
        "attack_nearest"
      ],
      "prefer_weight": 2
    }
  ]
}
```

검토 메모: 전략–실행 일치 여부 / 유지 시 위험 / 추가로 필요한 관측 정보를 기록해 주세요.

## 부록 A. 공통 입력 템플릿 — 코드 기준, 호출 원문 아님

출처: [llm_commander.py](../../../algorithm/lehca/commander/llm_commander.py).
실제 요청은 아래 system template의 `{env_context}`를 환경 인터페이스의 설명으로 치환합니다. 아래는 **미치환 코드 템플릿**이므로 JSON 예시의 이중 중괄호도 그대로 두었습니다.

```text
You are the LLM Commander of an allied team in a StarCraft II micromanagement battle (SMAC). You operate at a coarse strategic timescale; decentralized reinforcement-learning agents execute actions at high frequency and learn fine-grained control on their own. Your job is expert strategic cognition, NOT low-level control.

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
- Do not assume, request, or refer to the simulator's true reward function. Choose only phase-relevant sub-goals and avoid generic/typed duplicates on homogeneous maps.
- "forbid" MUST NOT contain "noop", "stop", "move_all", or "attack_all". Never put the same token in both "forbid" and "prefer".
```

user message는 아래 구조입니다. 대괄호 부분은 실제 로그에 없는 입력을 뜻하며, 실제 입력값이 아닙니다.

```text
[당시 iface.summary(snap, stats)가 만든 관측 문장 — 미저장]

Produce your strategic guidance now as strict JSON.
```

환경 문맥은 [SC2SemanticInterface.prompt_context](../../../env/semantic/sc2.py)가 제공하는 행동 공간, 아군·적 타입 목록, 과제 목표입니다. 이 2s3z 실행의 타입은 Stalker와 Zealot입니다. 호출 시점의 전체 system message가 보존되어 있지 않으므로 이를 '저장된 실제 입력'으로 표시하지 않았습니다.

## 부록 B. 실행 의미를 확인할 코드

- [predicate 계산](../../../algorithm/lehca/shaping/predicates.py): 부호·가중치·활성화 조건·clipping.
- [출력 정제](../../../algorithm/lehca/commander/base.py): 원 응답에서 제거·정제되는 필드.
- [행동 마스크 컴파일](../../../algorithm/lehca/masking/compiler.py): 타입별 적용, hard/soft 조합.
- [대상 토큰 해석](../../../env/semantic/sc2.py): 매 스텝 토큰을 실제 대상으로 변환.
- [신 프롬프트 설계](prompt-reconstruction-20260912.md): 기존 프롬프트에서 보완하려는 문제.

향후 입력–출력 전체를 감사하려면 요청 messages, 응답 content, 정제 후 guidance를 같은 호출 ID로 별도 저장해야 합니다. 이 문서 작성에서는 로깅 코드·실험 설정을 변경하지 않았습니다.
