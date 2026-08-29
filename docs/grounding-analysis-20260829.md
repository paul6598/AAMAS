# 논문의 텍스트→수치 신호 변환(그라운딩) 프로세스 정밀 검토 (2026-08-29)

근거: 논문 pp.4–9 (Overall architecture, Environment interaction and semantic
representation, Prompt engineering, High-level strategy generation, Dynamic semantic
reward shaping, Dynamic action masking, Eq. 5–7, Algorithm 1, Fig. 1, Fig. 8).
비교 대상: 우리 구현 `algorithm/lehca/{commander,shaping,masking}`, `env/semantic/sc2.py`.

## 1. 논문이 기술한 파이프라인 (5단계)

```
obs {o_i}  ─①─▶ d_t (텍스트)  ─②─▶ LLM JSON {전략, 서브골 G(+w_j), NL 보상규칙 R, 액션제약}
                                        │                              │
                              ③ semantic grounding module      ④ mask parsing module
                                        ▼                              ▼
                              f_j(s,a,s') predicates             M_hard, W_soft (per-agent)
                                        └──────── ⑤ Eq.1-2 / Eq.5-7 / Alg.1 ────────┘
```

### ① Semantic_Transform (입력 측) — 명시됨, 우리와 일치
"Data filtering and aggregation / Key entity identification / Relation inference /
Template-based semantic generation" — 템플릿 + 규칙 기반. 우리 `summary()`(타입별
집계, 중심 거리·방향, phase, 템플릿)와 같은 종류. **차이**: 논문은 "observable
information ... not privileged"를 강조하는데 우리는 `env.enemies` 전체 상태(시야 밖
적 HP·위치 포함)를 사용 → 재현 편차(리뷰 2-B). 정직하게 보고 필요.

### ② Commander 출력 — "structured JSON"이지만 내용물은 자연어 규칙
- 프롬프트: CoT식 "situation analysis → strategic planning → task decomposition",
  출력은 "structured JSON format, facilitating automated parsing".
- 서브골 g_j에 대해 LLM은 **자연어 보상 규칙 서술**을 낸다. 논문 예: 서브골
  "prioritize eliminating enemy damage dealers" → 규칙 *"Provide a positive team reward
  for each eliminated enemy ranged unit, and grant an additional guiding reward if the
  unit is focus-fired within a short time window."* + 우선순위 w_j ∈ [0,1].
- 액션 제약도 텍스트: *"prioritize focus fire on enemy damage dealers"*, *"move south"*.
- Fig. 8의 서브골: "적 Marine 제거마다 보상", "Medivac 보호 — **위협받으면 공격자에게
  즉시 사격 전환**, 추가 보상", "적 Medivac은 **전방 호위가 약해진 후** 제거, 큰 보상".

→ **우리와의 결정적 차이**: 우리는 LLM이 predicate **이름**을 직접 고르게 했다(1단계).
논문은 LLM이 **규칙을 서술**하고(2단계), 별도 모듈이 그라운딩한다. 논문 LLM의 출력은
조건(“if threatened”, “within a short time window”), 순서(“only after…”)를 포함한다.

### ③ Semantic grounding module (보상 측) — **존재만 명시, 메커니즘 미공개**
원문: "processed by a semantic grounding module and converted into computable reward
predicates and executable shaping functions f_j(s_t,a_t,s_{t+1}) … maps these
descriptions to **measurable events or predicates available in the training interface,
such as unit elimination, relative distance, focus-fire consistency, or ally protection**."
- 고정 라이브러리를 시사하는 근거: "predicates available in the training interface",
  복잡도 절에서 "semantic shaping adds O(K) team-level checks" (K=활성 서브골 수) →
  서브골당 검사 하나 = 라이브러리 술어 하나.
- 자연어→술어 매핑 방식(키워드 규칙? 2차 LLM 파싱? 코드 생성?)은 **어디에도 없음**.
  예시 규칙의 "within a short time window"가 살아남는지(시간창 파라미터가 있는 술어)
  버려지는지도 불명. Fig. 8의 조건부/순차 서브골도 마찬가지.
- 확실한 것: LLM은 보상 함수·시뮬레이터 내부를 보지 않고 "규칙 서술 + 가중치"만 낸다.

### ④ Mask parsing module (마스킹 측) — JSON→텐서, Eq.7은 **상태 의존(u_t) 갱신**
- "converts the LLM's structured JSON output into per-agent tensors: M_hard ∈{0,1},
  W_soft ∈ (0,∞)". 하드는 "prohibitions", 소프트는 "additive shaping of action scores".
- **Eq. 7**: `M_hard,t+1, W_soft,t+1 = Update(M_hard,t, W_soft,t, d_t, u_t, π_LLM)` —
  "when mask rules reference recent interaction — the joint action u_t". 즉 마스크 규칙이
  **직전 joint action을 참조**할 수 있다("focus-fire consistency"를 "직전에 때린 대상을
  계속 선호"로 구현했을 가능성). 우리는 매 스텝 무상태 재컴파일(u_t 미참조).
- Algorithm 1: L8 `Generate_Action_Masks(π_LLM)`은 refresh 블록 **밖** → 매 스텝 재생성
  (우리와 일치). 프롬프트 절: 하드 제약의 목적은 "**avoid infeasible or risky decisions**"
  (위험 회피) — 우리 LLM이 내는 stop/이동 금지는 이 의도와 다르다.

### ⑤ 결합 — Algorithm 1의 보상 저장 시점은 **수집 시점**
- L15–18: `r_shaping ← Compute_Shaping_Reward(s_t,a_t,s_{t+1},G,R)`;
  `r_total ← r_t + λ·r_shaping`; `Store_Experience(..., r_total, ...)`; L22: λ 감쇠는
  업데이트마다. → 논문 알고리즘은 **수집 시점 λ로 합성한 보상을 버퍼에 저장**한다.
  이것이 우리가 관측한 버퍼 오염(DIAG_A 붕괴)의 설계이고, arXiv 2607.04470이 지적한
  비정상성 문제다. 우리 `shaping_in_learner`(리플레이 시점 λ)는 **논문 알고리즘에서
  벗어난 개선**이며 보고 시 명시해야 한다.
- Eq. 5–6, ε-greedy 허용집합 균등 탐색, TD는 raw Q — 우리 구현과 일치 (리뷰 §0 확인).
- w_j ∈ [0,1], F_t = Σ w_j f_j, R = R_env + λ_t F_t — 일치.

## 2. 차이 요약표

| 항목 | 논문 | 우리 구현 | 성격 |
|---|---|---|---|
| Commander 출력 | 자유 CoT + JSON 내 **자연어 규칙 서술** + w_j | JSON 내 **predicate 이름** + w_j | 2단계 vs 1단계. 표현력·추론 여유 차이 |
| 그라운딩 | 별도 모듈, 방식 미공개 | `sanitize_guidance` 화이트리스트 | 우리는 매핑 오류가 없는 대신 표현력 상한이 어휘 |
| 술어 표현력 | 시간창·조건·순서 포함 시사 (예시·Fig.8) | 무조건 술어 8종 | 논문이 더 풍부 (추정) |
| 마스크 갱신 | Eq.7 상태 의존(u_t 참조) | 무상태 재컴파일 | 논문의 consistency 규칙 미구현 |
| 하드 제약 의도 | 위험/불가능 회피 | LLM이 stop·이동 금지 남발 | 프롬프트 프레이밍 차이 |
| 셰이핑 보상 저장 | 수집 시점 λ (Alg.1) | 리플레이 시점 λ | **우리가 의도적 개선** (보고 필수) |
| d_t 정보원 | 관측 가능 정보만 | 전체 상태 | 편차 (보고 필수) |
| F_update | 고정, 이벤트 트리거는 future work | 고정 (동적화가 본 연구) | 논문이 직접 열어둔 확장 |

## 3. 재현 충실도 판단

- **구조(Eq.1–2, 5–7, Alg.1 흐름)는 충실**. 논문 미공개 부분(③ 매핑 방식, 술어 목록,
  토큰 어휘, β·λ·F 값)은 자작 — 이 중 **③과 술어 표현력**이 가이던스 품질을 가장
  크게 좌우하며, 우리가 논문보다 좁게 만들었을 가능성이 높다.
- 두 가지 **의도적 편차**(학습시점 λ, 전체 상태 d_t)는 각각 "논문 알고리즘의 결함
  보정"과 "구현 편의"로 성격이 다르다. 후자는 관측 기반으로 바꾸는 것이 원칙상 맞다.

## 4. 조치 후보 (비용 순)

1. **프롬프트 프레이밍**: 하드 제약을 "infeasible or risky decisions 회피"로 한정 명시
   (논문 문구 그대로). M5(어휘 축소)와 상보적 — 0비용.
2. **술어 확장 3종**: `focus_fire_window(k)`(시간창 집중사격), `retaliate_for_type:<X>`
   (X를 때린 적 공격 보상), `kill_type_after:<X>,<Y>`(Y 전력 임계 이하 시 X 킬 보상)
   — Fig.8·본문 예시를 표현 가능하게. 작은 코드.
3. **Eq.7 u_t 의존 마스크**: "직전 스텝 공격 대상 유지" 소프트 선호 (consistency) —
   마이크로를 해치지 않는 유일한 마스크 후보.
4. **2단계 출력 옵션**: LLM이 predicate 이름 + 자연어 규칙 둘 다 내게 하고, 규칙 텍스트
   는 로깅·해석용 (그라운딩 방식을 논문과 맞추려면 자연어→술어 파서가 필요하나,
   화이트리스트 선택이 더 안전하므로 기본은 유지).
5. **d_t를 관측 기반으로**: 시야 밖 적 정보를 마스킹 — 재현 충실도용 (성능엔 불리할 수
   있음; 별도 그룹으로).
6. 보고서에 ⑤의 편차(학습시점 λ)를 "Alg.1은 수집 시점 저장이며 그대로 구현 시 붕괴,
   리플레이 시점 합성으로 보정"으로 명시.
