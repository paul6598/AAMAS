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
| ★★ **LLM-MARL** (Li et al., 2506.04251) | 경량 게이트 π_gate가 매 스텝 호출 여부 결정, 보상 (R_LLM − R_noLLM) − α·C_query, 호출 캐시 | MARL(GRF, MAgent, SC2), Coordinator LLM이 에이전트별 서브골 | 가장 가까운 MARL 경쟁작. "불필요 호출 43% 감소, 성능 유지". 단 게이트 대상이 **스텝 단위 행동 서브골**이지 셰이핑/마스크 규칙이 아니고, off-policy 버퍼 staleness를 다루지 않음. 우리는 호출 주기 스윕·staleness·예산 Pareto로 차별화 |
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
