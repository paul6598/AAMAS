# 관련연구 노트: 적응적 LLM 가이던스 스케줄링 (2026-08-28)

목적: LEHCA의 한계(고정 F_update, 고정 λ 감쇠, 항상-켜진 마스킹)를 개선하는
"적응적 가이던스 스케줄링" 연구의 포지셔닝. 우리 재현 캠페인 발견
(docs/final-report-20260827.md)이 실증적 motivation.

## 1. LEHCA 자체에 대한 외부 평가
- OpenReview 없음 (Scientific Reports는 OpenReview 미사용). 리뷰어 코멘트 접근 불가.
- 인용 논문: Semantic Scholar 기준 1편 (2608.07148, 제조업 MARL 참조 아키텍처 —
  서베이성, 기술적 비판 없음). 아직 후속·비판 논문 없음 → 우리가 첫 재현·비판이 됨.
- 저자 코드/supplementary 미공개 (메일 문의 권고 상태).

## 2. 직접 경쟁/인접 연구 (반드시 인용·차별화)

| 논문 | 핵심 | 우리와의 관계 |
|---|---|---|
| **Regime-Conditional Stabilisation of LLM-Augmented MARL** (arXiv 2607.04470) | 동적 LLM 보상 가중치 갱신이 PBRS 정상성 위반 + 리플레이 버퍼 오염 → 붕괴. 해법: 페이즈별 가중치 동결, EMA 스무딩. 베이스라인 역량에 따른 3-레짐(augmentative/essential/supplementary). SMAC 3m 등, QMIX. | **우리 G2 발견(버퍼 오염)의 독립 재발견** — 타당성 근거. 그러나 해법이 "안정화"이고 레짐을 **태스크별 정적**으로 부여. 우리는 레짐을 **온라인 판별해 개입을 조절** → 명확한 차별점. |
| **LLM-ALSO** (arXiv 2605.29293) | Critic LLM이 학습 실패를 진단 → Generator LLM이 보상 셰이핑 설정 제안 → 브랜치 검증 후 적용. 반복 적응. | 가장 가까운 "적응적 셰이핑" 연구. 다만 갱신이 **검증 브랜치 기반의 무거운 외부 루프**(비용 큼)이고, 호출 시점/강도의 온라인 스케줄링은 아님. 우리는 경량 신호 기반 온라인 조절. LEHCA 비교 없음. |
| **Uncertainty-Aware LLM Guidance for RL** (arXiv 2411.14457) | MC-dropout 엔트로피로 LLM 조언 불확실성 추정 → 가이던스 강도를 동적으로 조절. 단일 에이전트 Minigrid. | "강도 적응" 축의 선행. 단일 에이전트·소규모. 우리는 MARL + 학습 신호 기반 + 호출 시점까지. |
| **UG-CPPO** (불확실성 게이팅 LLM 주입, 트레이딩) | 프롬프트 앙상블 분산으로 LLM 신호 게이팅. | 게이팅 아이디어의 도메인 사례. |
| **LLM-guided incentive-aware reward design for coop MARL** (arXiv 2603.24324) | LLM이 실행 가능한 보상 프로그램 생성, 세대별 선택 (Overcooked, MAPPO). | 오프라인 보상 설계 계열 — 온라인 스케줄링 아님. |
| Hierarchical LLM planning + RL execution (arXiv 2606.20014) | LLM이 RL 스킬을 선택하는 계층 구조 (2v2 KotH). | 계층 구조 인접 연구; 계획 주기 미명시. |

## 3. 고전 문맥: Action Advising / Teaching on a Budget
- Torrey & Taylor (2013) "Teaching on a budget", Ilhan et al. (2019) 멀티에이전트 확장,
  "Learning on a budget via teacher imitation" (2104.08440), "Methodical advice
  collection and reuse" (2204.07254).
- 핵심 개념이 정확히 우리 문제: **제한된 조언 예산 하에서 언제 물을 것인가** —
  학생의 epistemic uncertainty 기반 요청, early advising(초반 집중), 조언 재사용
  (imitation model). 우리 연구는 이 프레임을 "LLM Commander가 교사"인 MARL로
  옮기는 것으로 서술 가능 → 이론적 뿌리 제공.

## 4. 포지셔닝 (한 문단)
기존 연구는 (a) LLM 가이던스의 **안정화**(2607.04470: 동결/EMA)나 (b) 가이던스
**내용**의 적응(LLM-ALSO: 진단→재설계)에 집중했다. 우리는 가이던스의 **개입
정책**(언제 묻고, 얼마나 강하게, 어느 채널로) 자체를 학습 신호로 온라인 조절하는
문제를 제기한다. 재현 실험은 고정 스케줄이 맵마다 정반대 처방을 요구함
(5m6m 유지 / 2s3z 차단 / 3s5z 억제)을 보였고, action-advising 문헌의 예산 제한
프레임이 자연스러운 이론적 토대다.

## 5. 리스크/열린 질문
- F_update 자체의 민감도 미측정 → 정적 스윕(100/400/800)으로 먼저 확인 (진행 중).
- 2607.04470과의 차별화를 실험으로 보여야 함: 그들의 정적 레짐 처방 vs 우리의
  온라인 판별 — 3s5z(해로운 레짐)에서 자동 차단이 killer demo.
- 스케줄러 신호 후보: TD 오차 추세, 승률 정체, 정책 엔트로피, 상태 변화량,
  Commander 자기평가. 오프라인 상관 분석부터 (wandb 로그에 전부 존재).
