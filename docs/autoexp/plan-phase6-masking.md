# PHASE 6: 마스킹 이득 재현 캠페인 (2026-08-31, 사용자 지시)

목표: 논문 Table 5의 "마스킹 켠 완전 구성 > shaping-only" (2s3z AUC 0.455 vs 0.343,
+0.11)를 재현. 우리 현재: shaping-only 0.394±0.061 (OBS_v4, 관측 d_t, n=3),
완전 구성 β=0.1 0.342±0.040 (M3, 전체상태 d_t, 어닐 150k). **판정 참조 =
OBS_v4 0.394** (같은 관측 d_t 조건). 성공 = 완전 구성 3시드 평균 ≥ 0.44 (+0.05
이상), 논문 수준 = 0.455.

공통 설정 (v4 셰이핑 + 마스킹 on): use_action_masking=True use_masking_at_test=True
lambda_start=0.5 lambda_min=0 lambda_decay=0.9993 shaping_in_learner=True
dt_observable=True f_update=200, 2s3z 250k, AUC --frac 0.8. vLLM n020:8356.

## 논문 근거 조절 축 (본문 인용)
| 축 | 논문 문구 | 구현 플래그 |
|---|---|---|
| β 크기 | "β≥0 fixed coefficient controlling the strength of soft tilting", "moderate" | beta |
| 어닐 없음 | 논문엔 마스크 어닐 없음 (매 스텝 재생성만) | mask_anneal_t=0 |
| u_t 참조 | Eq.7 Update(…, u_t, …) "mask rules reference recent interaction" | mask_consistency_w |
| 하드 제약 범위 | "hard constraints … to avoid infeasible or risky decisions" | prompt_style=paper |
| CoT 단계 | "situation analysis, strategic planning, task decomposition" | prompt_style=paper |
| 추론 여유 | Fig.8 수준 추론 | llm_reasoning_effort=medium |
| 2단계 출력 | "natural-language reward rule descriptions … processed by a semantic grounding module" (p.6), Fig.8 | prompt_style=twostage |

## 라운드 1 (6 GPU 병렬, seed 0, 각 ~1.5h)
| ID | 그룹 | 오버라이드 | 검증 가설 |
|---|---|---|---|
| P1 | P6_b005 | beta=0.05 mask_anneal_t=0 | β 더 낮추면 이득? |
| P2 | P6_b01_twostage | beta=0.1 mask_anneal_t=0 prompt_style=twostage | **논문 방식 2단계**: 자유 서술 계획 → 그라운딩 모듈(2차 LLM 호출)이 JSON 변환 (사용자 지시: 출력 방식 자유) |
| P3 | P6_b01_noanneal | beta=0.1 mask_anneal_t=0 | 어닐 제거 효과 (M3 대비) |
| P4 | P6_b01_consist | beta=0.1 mask_anneal_t=0 mask_consistency_w=2.0 | Eq.7 u_t consistency |
| P5 | P6_b01_paperprompt | beta=0.1 mask_anneal_t=0 prompt_style=paper | 위험회피 한정 하드제약 + CoT |
| P6 | P6_b01_medium | beta=0.1 mask_anneal_t=0 llm_reasoning_effort=medium | 추론 품질 |

세션: P1=2, P2=34, P3=38, P4=40, P5=45, P6=46.

## 규칙
1. 라운드 1 판정: 참조 0.394. ≥0.42인 축을 "유효"로 표시. 유효 축 2개 이상이면
   결합 변형(P7)을 만들어 seeds 0-2 확증; 유효 축 1개면 그 축 seeds 1-2 확증.
2. 전부 <0.42면 라운드 2: β=0.05 위에 consistency/paperprompt/medium 결합 3런 +
   β=0.1 + mask_vocab=strategic 재시도.
3. 확증된 최선 설정이 3시드 ≥0.44면 "마스킹 이득 재현"으로 기록; 0.394±0.06 내면
   "β 무해까지만 재현, 이득 미재현(저자 β·프롬프트 필요)"으로 기록.
4. 매 판정 state.md 기록·커밋. 가이던스 덤프 results/guidance/*P6_*.jsonl로 forbid
   빈도·override 함께 기록 (paper 프롬프트가 forbid를 실제로 줄이는지 확인).
