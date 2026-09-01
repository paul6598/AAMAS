# 관련연구 노트: LLM 호출(F_update)의 동적 스케줄링 (2026-08-29)

목적: LEHCA의 고정 F_update(매 F 스텝 Commander 호출)를 **적응적 호출 정책**으로
바꾸는 연구(A축)의 포지셔닝. 선행 노트 docs/related-work-adaptive-guidance.md의
확장. 4개 영역(LLM-guided RL/MARL · action advising · 계층 RL/재계획 주기 ·
비용 인식 LLM 호출)을 웹 검색으로 조사, URL 확인된 논문만 수록.
★ = 우리 설정에 가장 직접적. 연도는 arXiv 기준.

## 0. 한 줄 결론

"언제 LLM을 부를 것인가"는 단일 에이전트·행동 조언 수준에서는 이미 여러 답이
있다(불확실성 게이트, 정체 트리거, 학습된 게이트, 호출 비용 페널티). 그러나
**(a) 보상 셰이핑+마스크를 내는 coarse-timescale Commander를, (b) off-policy
value-decomposition MARL(QMIX 리플레이) 위에서, (c) 호출 주기 스윕과 함께**
다룬 논문은 없다. LEHCA 본문도 "adaptive Commander invocation strategies"를
future work로 명시하고 F_update ablation이 없다 — 갭이 논문 자체에 적혀 있음.

## 1. 직접 경쟁 (반드시 인용·차별화)

| 논문 | 트리거 | 설정 | 우리와의 차이 |
|---|---|---|---|
| ★★ **When2Ask** (Hu et al., RLC 2024, 2306.03604) | RL로 학습한 asking policy; "물었는데 같은 플랜이 돌아오면" 페널티 λ. 베이스라인: Always / Random 50% / 하드코딩(옵션 종료·100스텝 타임아웃) / Never | 단일 에이전트, MiniGrid/Habitat, LLM=플래너, test-time | 학습 루프가 아니라 실행 시 플랜 요청. 단일 에이전트. 리플레이/셰이핑 없음. **redundant-query 페널티는 그대로 차용 가능** |
| ★ **LLM-MARL** (Li, Campos, Wang, 2506.04251 v4 2025-11; 원문 확인) | 기본 **매 스텝** Coordinator 호출(에이전트별 서브골 1개 + LLM 제안 행동). 게이트 π_gate = "reward differences with/without LLM guidance"로 **지도학습한 2×64 MLP 이진 분류기**, 목적식 E[(R^LLM − R^noLLM) − α·C_query] (R^noLLM 산출법 미기술). rolling cache | **PPO + 공유 critic(on-policy)**, GPT-3.5 T=0.7, GRF/MAgent/SC2(맵 미명시), 3시드. Communicator·Memory 포함한 넓은 프레임워크 논문 | 게이트 결과는 "불필요 호출 43%↓, 성능 유지" 한 문장뿐 — 표·ablation·주기 스윕 없음. on-policy라 리플레이 staleness 문제 부재. **스스로 "outdated subgoal commitments"로 인한 hesitation을 실패 사례로 보고하고 "finer temporal grounding or recurrent querying"을 future work로 남김** → 우리 H0·τ 프레임의 직접 motivation. 비용 레짐도 다름(매 스텝 호출에서 43% 절감 vs LEHCA는 이미 1/200). 베이스라인으로 "보상차 지도 게이트" 재현 가치 있음 |
| ★★ **LLaPipe Advisor+** (Chang et al., 2507.13712) | 최근 에피소드 리턴에 선형회귀 → 기울기 < θ(=0.01)이면 호출 (**학습 정체 트리거**). 고정 주기 vs 적응 비용 모델 명시 | 단일 RL + LLM advisor, 데이터 준비 파이프라인 | 학습 루프 내 적응 호출로 우리와 구조가 가장 비슷. 도메인이 멀다. "정체 트리거" 베이스라인으로 구현 가치 |
| ★ **MIRA** (Nourzad & Joe-Wong, ICLR 2026, 2602.17930) | 롤아웃 utility가 여러 에피소드 연속 ~0이면 호출; **명시적 쿼리 예산**; LLM 출력을 메모리 그래프로 상각. 예산 ablation(0/10/20) + stale guidance 실험 있음 | 단일 에이전트 DoorKey | 예산 ablation·staleness 실험 프로토콜 참고. |
| ★ **Regime-Conditional Stabilisation** (2607.04470) | 매 에피소드 재질의 → PBRS 비정상성 + 버퍼 오염 → 붕괴. 처방: 페이즈 동결, EMA(α=0.2) | SMAC 3m 등, QMIX | 우리 G2(learner-λ) 발견의 독립 재발견. **"자주 부르면 해롭다"의 MARL 근거**. 그러나 해법이 정적 동결/EMA — 온라인 트리거 아님 |
| ★ **ASSCG** (2606.25509) | 매 프레임 **Query / Cache / Drop** 3-액션 게이트(RWKV), SFT+비용 인식 RL. chattering 억제 | 자율주행 fast-slow LLM 플래닝 | Query/Cache/Drop 분류가 우리 "refresh / 유지 / 무시"에 1:1. RL 학습 루프 아님 |
| ★ **Bayesian Partner Modelling** (2608.18490) | 팀원 행동이 추론된 스킬과 **모순**될 때만 재계획. periodic / event / LLM-judge 베이스라인 비교 | Overcooked, LLM 에이전트(RL 학습 없음) | 재계획 베이스라인 분류 체계 차용 |
| ★ **Learning When to Plan** (Paglieri et al., 2509.03581) | `<plan>` 토큰 발행 여부를 RL로, 토큰 비용 페널티. **fixed plan-every-k 스윕에서 중간 k가 최적("Goldilocks")**, 매 스텝 재계획은 불안정 | LLM 에이전트 | "자주가 항상 좋진 않다"의 근거. 우리 F 스윕의 해석 틀 |
| ★ **LLM-Guided Safe RL for energy topology** (2603.14018) | 고정 주기 f, **f 민감도 스윕(§4.7.2)**: 작은 f 진동, f=200 최적, 800–1000 너무 희소 | 단일 에이전트 | 고정 주기 U자형 곡선의 선례. 우리 스윕 결과와 대조 |

## 2. LLM-guided RL/MARL에서 호출 시점을 다루는 방식 (스펙트럼)

| 방식 | 논문 | 비고 |
|---|---|---|
| 매 스텝 + 캐시 | ELLM (ICML 2023, 2302.06692), LMGT (2409.04744), LLM4Teach (IJCAI 2024, 2311.13373; 초기 매 스텝, 가중치 선형 감쇠→0) | ELLM: 캐시 없으면 5M 스텝에 27시간 API |
| 고정 주기 k | LEHCA(F_update, ablation 없음), Hierarchical LLM+RL 2v2 (2606.20014; 2Hz), LLM-augmented obs hints (2510.08779; k=5 > k=10 BabyAI) | 주기 민감도 보고는 2510.08779·2603.14018뿐, 방향 상반 |
| 체크포인트/윈도우 후 off | LLM-GNCF (C&IS 2026; 첫 0.25M만 호출 후 학습된 그래프 재사용), SGRL (2509.22008; 2M마다), LLM-ALSO (2605.29293; 2개 체크포인트) | "적응"이라 해도 사실상 정적 |
| 확률 감쇠 + 재사용 예산 | Toral & Lazebnik (2509.08329; 상태 해시 캐시, 항목당 재사용 3회, 호출 확률 선형 감쇠) | 재사용이 수렴 가속하나 불안정 |
| 불확실성 게이트 (호출 여부) | ASK (IJCNN 2026, 2604.02226; MC-dropout), ASK+ (2607.02686; 정책 엔트로피) | 단일 에이전트 MiniGrid |
| 불확실성으로 **영향력** 조절 (호출은 매 스텝) | 2411.14457, ULPS (2606.06673) | B축(강도) 계열 |
| 학습된 게이트 | LaGR-SEQ (2308.13542; 2차 RL 에이전트가 호출 학습), LLM-MARL π_gate, When2Ask | |
| 한 번만 / 오프라인 | YOLO-MARL (2410.03997), MAESTRO (2511.19253), LEMAE (2410.02511), Motif (2310.00166), Eureka/Text2Reward | 주기 스펙트럼의 극단 = 우리 베이스라인 후보 (F→∞) |

## 3. 고전 뿌리: Action advising / teaching on a budget

"제한된 조언 예산에서 언제 물을 것인가"의 원형. LLM Commander = 교사로 서술 가능.

- **Torrey & Taylor (AAMAS 2013)** — Early / Importance (max_a Q − min_a Q > t) /
  Mistake-correcting (중요 상태 + 학생 의도 ≠ 교사) / Predictive. mistake-correcting이
  예산 효율 최고. 같은 예산도 **언제 쓰느냐**에 따라 결과가 크게 다름 — A축의 명제.
- **Amir et al. (IJCAI 2016)** — 학생 측 Ask-Important / Ask-Uncertain(Clouse 1996) /
  Ask-Unfamiliar; 공동 개시(학생이 주의 요청 → 교사가 판단)가 교사 부담 최소.
- **RCMP** (Da Silva et al., AAAI 2020) — 다중 Q-head 분산 = epistemic 불확실성으로
  요청. 딥 RL 표준 "ask when uncertain" 베이스라인.
- **Ilhan 계열** — Novelty-based asking (IEEE TG 2021, 2010.00381: *조언받은 상태* 대비
  novelty), Advice imitation (AAMAS 2021, 2104.08441), **Learning on a budget via
  teacher imitation** (CoG 2021, 2104.08440: 모방 모델이 불확실할 때만 질문, 임계값
  자동 튜닝), **Methodical advice collection & reuse** (ALA 2022, 2204.07254: 학생
  불확실 AND 교사 모델 불확실일 때만 질문). → **우리 캐시 = 교사 모방 모델**의
  가장 단순한 형태(키 일치 시 재사용). "예측 가능한 조언엔 비용을 내지 않는다".
- MARL: **AdHocTD** (AAMAS 2017; 방문수 기반 confidence, ask/give 예산 분리),
  **LeCTR** (AAAI 2019, 1805.07830; 팀원 학습 진전 보상으로 when/what to advise 학습),
  HMAT (AAMAS 2020), PSAF (2011.14281; 정책이 모두 움직일 땐 행동 조언이 취약 → Q 공유).
- Ask-for-help: Ask4Help (NeurIPS 2022, 2211.09960; 도움 비용 vs 보상), Xie et al.
  (NeurIPS 2022, 2210.10765; 비가역 상태 탐지), Min et al. (2502.04576; 개입 예산 C).
- 서베이: Da Silva & Costa (JAIR 2019), *Agents Teaching Agents* (JAAMAS 2020).

## 4. 계층 RL / 제어: "언제 다시 결정하는가"

| 트리거 계열 | 대표 | 우리 대응 |
|---|---|---|
| 고정 k | HIRO (c=10) | F_update |
| 학습된 종료·지속 + 전환 비용 | Option-critic + deliberation cost η (AAAI 2018, 1709.04571), **Lazy-MDP** (AAMAS 2022, 2203.08542; 기본=현 정책 유지, 개입에 페널티), TempoRL (ICML 2021), TACOS (NeurIPS 2024, 2406.01163; 상호작용 비용 명시) | "η보다 나아질 때만 refresh" |
| 상태 변화 이벤트 | **Event-triggered control** (DRL for ETC, CDC 2018, 1809.05152; 고전: ‖x − x_last‖ > 임계값), ET-MPC (2208.10302; 비싼 최적화 재풀이 vs 재사용), CPD-HRL (2510.24988; change-point) | cache_key 변화, phase 전환 |
| 플랜 staleness 점수 | **Adaptive Online Replanning w/ Diffusion** (NeurIPS 2023, 2310.09629; 현 플랜의 우도 하락 시 재계획) | "현 가이던스 유효성" 점수 |
| 실패/전제 위반 | LLM-Planner (ICCV 2023), DEPS (NeurIPS 2023), AdaPlanner, **DoReMi** (IROS 2024, 2307.00329; 플랜과 함께 전제조건 출력 → 위반 시 재계획), CoMuRoS (2511.22354) | 가이던스가 참조한 적 타입 전멸 등 "무효화" 이벤트 |
| 미래 불확실성 | Interval-aware RL (2603.22384) | |

## 4b. 멀티에이전트 시간 추상화: 팀 수준 상위 결정의 주기

단일 에이전트 τ 문헌(§4)의 MARL 판. 결론: **QMIX 계열 계층 MARL에서 팀 공유 상위
결정의 주기를 온라인으로 적응시킨 논문은 없다.** 전부 고정·동기 주기이며, 보고된
ablation은 모두 내부 최적(너무 짧아도, 길어도 나쁨)을 보인다.

| 논문 | 상위 결정 | 주기 | ablation | 종료 |
|---|---|---|---|---|
| ★ **COPA** (Liu et al., ICML 2021, 2105.08692) | 코치가 전략 벡터 z를 전 플레이어에 방송 | **T=4** | T ∈ {2,4,8,12,16,20,24}: **T=4 최적, "작을수록 좋다"는 직관 반박** ("코치는 에이전트가 시간적으로 일관되게 행동하게 할 때 가장 유용") | 동기 |
| | §3.4 게이트: 주기 T마다 새 z를 계산하되 ‖z_new − z_old‖ ≥ β일 때만 전송, Thm 1로 손실 상계 | | β ∈ {0,2,3,5,8}: 통신 25%→13%로 줄여도 성능 유지 | **고정 T 격자 위에서만 판단 — 주기를 늘릴 뿐 줄이지 못함, β 수동** |
| ★ **RODE** (ICLR 2021, 2010.01523) | 역할 선택기(QMIX 믹싱) | **c=5** | c ∈ {3,5,7,10} (App. D.1): "significant influence", 5–7 권장, 상호작용은 future work | 동기 |
| ★ **HAVEN** (AAAI 2023, 2110.07246) | 상위 매크로 액션 믹싱 | **k=3** | k ∈ {3,4,5}: k 커질수록 하락 | 동기 |
| **HMASD** (NeurIPS 2023) | 트랜스포머 코디네이터가 팀 스킬→개별 스킬 ("코치의 타임아웃") | k (값 미확인) | App. D: 너무 짧거나 길면 나쁨 | 동기 |
| HSD (AAMAS 2020, 1912.03558) | 스킬 선택 | t_seg=10 | {5,10,20}: 10·20 > 5 | 동기; 비동기는 future work |
| ALMA (NeurIPS 2022) | 서브태스크 할당 | N_t=3(SC2)/5 | 없음 | 동기 |
| FMH (1901.08492) | 매니저 서브골 | 8 | 정성 스윕 | 동기 |
| MASER (ICML 2022) | 버퍼 기반 서브골 | 에피소드당 1회 | 없음 | 동기 |
| ROMA (ICML 2020) | 역할 임베딩 | 매 스텝 (k=1 끝점) | — | — |

에이전트 수준 적응 종료 (팀 공유 신호는 아님):
- ★ **Han et al., Dynamic Termination** (PRICAI 2019, 1910.09508): 옵션 종료를 상위
  Q의 행동으로 두고 페널티 δ로 가격 매김 — 긴 옵션은 팀원 변화에 늦게 반응, 잦은
  전환은 방송된 의도를 신뢰 불가하게 만듦(**MARL 고유의 트레이드오프**). 비동기.
- **GMAH** (2408.11416): 서브골 달성 시 재생성 + 최대 주기 c 캡 — MARL 계층에서
  유일한 달성-트리거 refresh.
- DOC (AAMAS 2020, 1911.12825): 협동 MAS option-critic, 공동 옵션은 구성 옵션 중
  하나라도 종료하면 종료. IAD (2605.24343): Overcooked에서 학습된 β(z,s).
- 이벤트 트리거 통신(분산, 중앙 조정자 없음): ETCNet (TNNLS 2023, 2010.04978),
  ET-MAPG (2509.20338), AsynCoMARL (AAMAS 2025, 2502.00558).
- ETD-MAPPO (Jankowski, 2603.23722, arXiv 단독저자): 에이전트 수준 적응 frame-skip —
  정책 엔트로피 ≤ τ_H AND twin-critic 차이 ≤ τ_V이면 고정 N프레임 수면, SMDP식
  γ^N 부트스트랩. 임계값 환경별 수동 어닐링, 시드 수·게이트별 ablation 없음,
  TempoRL 등 학습 skip 비교 없음, LBF에선 절감 0%. 팀 수준 가이던스와 무관 —
  "에이전트 수준 불확실성 게이트" 사례로만 인용.

비동기 매크로 액션 형식 틀: **MacDec-POMDP** (Amato et al., AAMAS 2014 / JAIR 2019),
Xiao–Hoffman–Amato (CoRL 2019, 2004.08646), Mac-IAICC (NeurIPS 2022, 2209.10113),
**ToMacVF** (2507.10251; QMIX식 값 분해 + 비동기 매크로 액션, temporal IGM). Tang et al.
(1809.09332)은 동기 vs 비동기 종료를 설계 축으로 명시(비동기 3–5% 손실). 팀 수준
가이던스 refresh는 동기 모델에 해당 → 에이전트 타입별 비동기 refresh는 QMIX 설정에서
새롭다.

제어이론: periodic event-triggered control(고정 격자에서 트리거만 검사 = COPA 구조;
Nowzari et al., Automatica 2019)과 **self-triggered control**(제어기가 다음 갱신 시각을
스스로 계산; Heemels et al., CDC 2012) — 후자가 "Commander가 다음 F_update를 스스로
정한다"의 정확한 대응.

포지셔닝 문장: "value-decomposition MARL에서 팀 공유 가이던스(LLM Commander)의
refresh 주기를 학습/이벤트로 적응시키는 것"은 미개척. 최근접 = COPA(고정 T + skip
게이트), HMASD(고정 k 주기적 중앙 실행), Han et al.(가격 매긴 동적 종료, 에이전트 수준).

## 4c. 국면·전황 변화를 "감지"하는 연구 (regret 외 — A축 ① 트리거 후보의 뿌리)

우리 문제의 트리거는 "가이던스가 낡았는가"이며, 이는 결국 **변화 감지(change
detection)** 문제다. MARL/RL/게임 AI에서 변화를 감지하는 계열은 다섯 갈래.

| 계열 | 대표 | 무엇을 감지 | 통계량 | 우리와의 거리 |
|---|---|---|---|---|
| **① 비정상성 변화점 감지 (RL)** | Alegre, Bazzan, da Silva **AAMAS 2021** MBCD; Hadoux 2014 (HS3MDP); Padakandla 2020 (Context Q-learning + ODCP); Banerjee 2017 (QCD-MDP) | 환경 동역학 MDP 전환 | 동역학 예측 모델 앙상블의 로그우도 **CUSUM** — 검출 지연 최소화·오경보율 상한 (Lorden/Page 최적성) | 가장 이론적으로 정돈됨. **검출 지연 ↔ 낡음 비용, 오경보율 ↔ 호출 예산**으로 정확히 대응 |
| **② 상대 전략 전환 감지 (MARL)** | Hernandez-Leal 2016–17 MDP-CL/**DriftER** (concept drift, 고확률 검출 보장); Deep BPR+ (Zheng NeurIPS 2018); SAM (Everett & Roberts 2018); Bayes-ToMoP (Yang IJCAI 2019); OPS-DeMo (2024, running error decay); BADA (IJCAI 2024, Wasserstein 행동 거리, 임계값 없음) | 상대 정책 스위치 | 상대 모델의 예측 오차/사후확률의 급변 | 감지 대상이 "상대"이지 "아군 전황"은 아님. 통계량 설계는 차용 가능 |
| **③ 실행 감시 / 계획 수리 (plan execution monitoring)** | Fikes 1972 triangle tables; Pettersson 2005 서베이; Fox 2006 plan stability(수리 vs 재계획); Borrajo & Veloso 2024; LLM 계열 DoReMi 2023, Inner Monologue | 계획의 **전제조건 위반** | 기호적: 남은 계획의 precondition이 현 상태에서 거짓인가 | 우리 규칙은 기호적(attack_type:X, applies_to:type:Y)이라 **직접 적용 가능**: 참조 적 타입 전멸, 마스크 resolve 공집합 = 확실한 무효화. 노이즈 없고 비용 0 |
| **④ 게임 국면 인식 (RTS/MOBA)** | Tencent HMS (AAAI 2019): 게임 phase 모델링이 macro attention을 안내; Synnaeve & Bessière 2011 빌드 예측; Stanescu 2016 전투 상태 평가 CNN; Brood War 전략 스위칭 DQN | 개전/교전/추격 등 국면 | 지도학습 phase 분류기 또는 전투 결과 예측기 | phase가 macro 결정을 "언제" 갱신할지의 조건이 되는 선례. 다만 phase 라벨은 사람이 정의 |
| **⑤ 이벤트 조건부 행동 (MARL)** | Büchi, Flageat, Sebastián, Prorok 2026 "Events as Triggers for Behavioral Diversity" (이벤트 = 과업의 질적 변화, 이벤트 기반 하이퍼넷이 LoRA로 팀 정책 재구성); ETCNet/ET-MAPG(통신 이벤트 트리거); open ad hoc teamwork(GPL, 팀원 출입 이벤트) | 팀 구성·과업 조건의 질적 변화 | 이벤트는 대체로 **정의**되고(사전 지정), 반응이 학습됨 | "이벤트 → 상위 재구성"이라는 프레임이 동일. 우리 이벤트 = 유닛 사망/타입 전멸/병력 역전 |

**정리.** 규칙이 기호적이라는 우리 특성상 ③(전제조건 위반)이 가장 싸고 확실한
1차 트리거이고, 전제조건은 살아 있지만 상황이 바뀐 경우(체력·거리·병력비)는
①의 CUSUM을 요약 d_t의 특징 벡터 또는 "가이던스 모방 모델의 예측 오차"에 걸어
검출 지연/오경보를 원리적으로 조절한다. 두 층을 합치면 "확실한 무효화는 즉시,
통계적 변화는 지연-예산 트레이드오프로"라는 설계가 되고, 이는 S0 E1 로그로
오프라인 재생해 비교할 수 있다. ②·④의 학습 분류기는 라벨(=언제 갱신해야
했나)이 필요해 S0 이후 선택지.

### 4c-1. CUSUM/QCD를 RL에 적용한 연구 (트리거 이론의 직접 선례)

| 연구 | 무엇에 CUSUM을 걸었나 | 우리에게 주는 것 |
|---|---|---|
| Hadoux, Beynier, Weng 2014 (HS3MDP) | 관측 시퀀스의 우도비 CUSUM → 숨은 모드 전환 감지 후 모드별 정책 전환 | "모드 = 국면, 모드별 정책 = 가이던스" 구조 그대로 |
| Banerjee, Liu, How ACC 2017 (QCD-MDP) | 동역학/보상 변화의 quickest change detection; **두 임계값** 전략으로 "탐지 즉시 전환 vs 보상 손실" 트레이드오프를 명시 분석 | 탐지 후 *언제 전환할지*도 비용이라는 관점 — 우리는 전환 비용 = LLM 호출 |
| Alegre, Bazzan, da Silva **AAMAS 2021** (MBCD) | 동역학 예측 모델 앙상블의 **다변량 CUSUM**(로그우도비) → 고신뢰 변화점, 지연 최소·오경보 상한 | 가장 가까운 설계: 우리는 "가이던스 모방 모델"의 예측오차 또는 d_t 특징에 걸면 됨 |
| Padakandla et al. 2020 (Context Q-learning + ODCP) | 경험 스트림의 온라인 변화점 검출 → 컨텍스트별 Q 테이블 | 모델 없이 경험 통계량에 거는 대안 |
| Li, Shi, Wu, Fryzlewicz **NeurIPS 2025** (CUSUM-RL) | **최적 Q-함수의 정상성** 자체를 모델프리 검정, CUSUM으로 구조 변화점 → 정상 RL과 결합 | 검정 대상이 Q라는 점이 우리 "효과 공간" 관점과 부합. 오프라인 데이터로 가능 |
| Liu, Lee, Shroff AAAI 2018 (CUSUM-UCB / PHT-UCB); Cao et al. 2019 (M-UCB); Besson & Kaufmann JMLR 2022 (GLR-klUCB, 파라미터 프리) | 밴딧 보상 분포 변화 → 인덱스 리셋 | 리셋 = 갱신. GLR은 변화 크기 k 사전지식 없이 동작 → k를 정하기 어려운 우리에게 유용 |
| Arumugam, Fan, Liu 2025 (Option-Critic + CPD) | 트랜스포머 CPD를 옵션 **종료 신호**·옵션 발견에 사용(CUSUM 아님) | "변화점 = 옵션 종료 = 상위 재결정"이라는 매핑을 HRL에서 명시한 최신 선례 |
| Amiri & Magnússon CDC 2026 (SNS-MDP) | 감지 없이 잠재 마르코프 스위칭의 평균 MDP로 수렴 분석 | 대조군 사고: 감지 없이 평균으로 학습하면 무엇을 잃는가 |

**갭.** 위 전부 단일 에이전트이며, 감지 대상은 "환경 MDP의 변화"다. 우리는 (i)
멀티에이전트 팀, (ii) 감지 대상이 환경이 아니라 **외부 조언자의 가이던스
유효성**, (iii) 알람 비용이 명시적(LLM 호출)이라는 점이 다르다. 즉 CUSUM의
ARL₀를 호출 예산으로 직접 해석하는 첫 사례가 될 수 있다.

## 5. 비용 인식 LLM 호출 (프레이밍용)

- 캐스케이드/라우팅: FrugalGPT (2305.05176), Hybrid LLM (ICLR 2024), RouteLLM
  (2406.18665), **Online Cascade Learning** (ICML 2024, 2402.04513; 값싼 모델이 LLM을
  온라인 모방, "confidence 기반 defer는 불충분"), TREACLE (2404.13082; 이전 응답
  일관성), **Agreement-Based Cascading** (TMLR 2025, 2407.02348; 앙상블 불일치 시 승격).
- 캐시: GPTCache (NLP-OSS 2023; 임베딩 유사도), prefix caching (vLLM에서 이미 81% 적중).
- Fast-slow 에이전트: Talker-Reasoner (2410.08328), DriveVLM-Dual (CoRL 2024; 비동기
  저주파 VLM = F_update 구조), ASSCG.
- 프레이밍 포인트: 이 계열은 "호출을 건너뛰는" 4가지 신호(값싼 정책의 불확실성 /
  프록시 간 불일치 / 학습 정체 / 명시적 비용을 넣은 학습된 게이트)로 수렴한다.

## 6. 갭과 우리 포지셔닝

1. **Coarse-timescale Commander의 refresh 정책**은 미개척. 기존 적응 게이트는 전부
   스텝 단위 행동 조언(단일 에이전트) 또는 LLM-MARL(2506.04251)의 서브골 게이트.
   우리는 셰이핑 규칙+마스크라는 **팀 수준·규칙 수준 가이던스의 staleness**를 다룬다.
2. **주기 민감도 증거가 얇고 상반**: k=5>k=10 (2510.08779) vs f=200 U자 (2603.14018)
   vs 매 에피소드 재질의는 붕괴 (2607.04470) vs Goldilocks (2509.03581). SMAC/QMIX에서
   Commander 주기 스윕은 없음 → 우리 F 스윕 자체가 첫 데이터.
3. **Off-policy 리플레이와의 상호작용**은 2607.04470만 지적(해법은 정적 동결/EMA).
   refresh가 잦을수록 버퍼 내 보상 규칙이 이질화되는 문제를 **트리거 설계에 반영**
   (learner-λ 합성처럼 규칙 버전을 리플레이 시점에 재평가하거나, 규칙 변경 빈도에
   비용을 부과)하는 것이 차별점.
4. 논문의 프레이밍 제안: "Budgeted guidance refresh for LLM-guided MARL" — 호출 예산 B
   하에서 refresh 정책 π_refresh(상태 변화, 현 가이던스 유효성, 학습 신호)를 결정,
   평가는 **AUC_early vs 호출 수 Pareto** + 벽시계. 이론적 뿌리 = teaching on a budget.

## 7. 구현할 베이스라인·트리거 (문헌에서 도출)

베이스라인 (Bayesian Partner Modelling·When2Ask의 분류 차용):
- fixed-F (F ∈ {100, 200, 400, 800}; 진행 중), once/offline (F→∞; YOLO-MARL 대응),
  per-episode, EMA/freeze (2607.04470), Random-p.

트리거 후보 → 우리 로그의 신호:
| 트리거 | 문헌 | 우리 신호 |
|---|---|---|
| 상태 변화 이벤트 | ETC, cache_key | `cache_key` 변경, phase 전환, 유닛 사망 |
| 가이던스 무효화 | DoReMi, diffusion replanning | 현 룰의 `attack_type:X` 대상 전멸, 서브골 predicate가 연속 0 |
| 학습 정체 | LLaPipe Advisor+, MIRA | 최근 승률/리턴 기울기 |
| 학생 불확실성 / 중요도 | RCMP, Torrey-Taylor importance, ASK+ | `q_gap_mean`(작으면 불확실), 정책 엔트로피 |
| 교사-학생 불일치 | mistake-correcting, `mask_override_rate` | 틸트가 argmax를 바꾸는 비율이 0에 수렴하면 가이던스 불필요 |
| 중복 질의 페널티 | When2Ask λ | 같은 키에 같은 가이던스 반환 시 비용 |

주의(문헌 공통): 잦은 refresh = chattering/비정상성(ASSCG, 2509.03581, 2607.04470).
M5 덤프의 연속 변경률 0.98은 상태 변화와 LLM 비결정성을 분리해야 해석 가능 —
키별 캐시가 결정성을 보장하므로 트리거는 캐시 위에서 설계.

## 8. 참고 링크 (핵심만)

When2Ask https://arxiv.org/abs/2306.03604 · LLM-MARL https://arxiv.org/abs/2506.04251 ·
LLaPipe https://arxiv.org/abs/2507.13712 · MIRA https://arxiv.org/abs/2602.17930 ·
2607.04470 https://arxiv.org/abs/2607.04470 · ASSCG https://arxiv.org/abs/2606.25509 ·
Bayesian Partner https://arxiv.org/abs/2608.18490 · Learning When to Plan
https://arxiv.org/abs/2509.03581 · Energy Safe-RL https://arxiv.org/abs/2603.14018 ·
ASK+ https://arxiv.org/abs/2607.02686 · LaGR-SEQ https://arxiv.org/abs/2308.13542 ·
LLM-GNCF https://link.springer.com/article/10.1007/s40747-026-02356-7 ·
Torrey&Taylor https://www.ifaamas.org/Proceedings/aamas2013/docs/p1053.pdf ·
Amir 2016 https://www.ijcai.org/Abstract/16/119 · RCMP
https://ojs.aaai.org/index.php/AAAI/article/view/6036 · Ilhan budget
https://arxiv.org/abs/2104.08440 · SUA-AIR https://arxiv.org/abs/2204.07254 ·
LeCTR https://arxiv.org/abs/1805.07830 · Lazy-MDP https://arxiv.org/abs/2203.08542 ·
Deliberation cost https://arxiv.org/abs/1709.04571 · DRL-ETC
https://arxiv.org/abs/1809.05152 · DoReMi https://arxiv.org/abs/2307.00329 ·
Diffusion replanning https://arxiv.org/abs/2310.09629 · Online Cascade
https://arxiv.org/abs/2402.04513 · Agreement Cascading https://arxiv.org/abs/2407.02348
