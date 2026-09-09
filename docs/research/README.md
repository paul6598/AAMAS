# RSVP 연구 현황과 문서 안내

최종 갱신: 2026-09-09. 실행 상태는 매번 Slurm·Sacred·W&B 원자료로 다시 확인한다.

## 한 문장 요약

LEHCA는 LLM Commander의 고수준 지침을 보상 셰이핑과 행동 제약으로 MARL에 연결한다.
RSVP는 고정 주기 대신, 보유 지침이 강조하는 predicate의 잔여 셰이핑 가치를 예측하고
발급 시점 대비 지속적으로 낮아질 때 CUSUM으로 조기 갱신하는 shaping-only 방법이다.

현재 가장 큰 병목은 스케줄러가 아니라 **우리 LEHCA 지침 채널이 5m6m에서 QMIX를
안정적으로 이기지 못한다는 점**이다. 따라서 “동적 갱신이 우수하다”는 주장은 보류하고,
지침 내용의 유용성 → 고정 F 시간척도 → 동일 채널의 fixed/adaptive 비교 순서로 검증한다.

## 먼저 읽을 문서

| 순서 | 문서 | 역할 |
|---:|---|---|
| 1 | [draft-paper.md](draft-paper.md) | 문제 제기, LEHCA, RSVP 방법, 현재 증거와 한계 |
| 2 | [experiments-log.md](experiments-log.md) | 실행 ID·설정·수치·판정의 단일 원장 |
| 3 | [code-guide.md](code-guide.md) | 코드 지도와 실행 경로 |
| 4 | [lehca-grounding-audit-20260907.md](lehca-grounding-audit-20260907.md) | 원 논문 대비 접지·프롬프트·환경 의미 감사 |
| 5 | [lehca-backbone-validation-20260908.md](lehca-backbone-validation-20260908.md) | none/rule/shuffle/aligned로 학습 경로와 지침 내용을 분리한 감사 |

필요할 때만 읽을 문서:

- [related-work.md](related-work.md): 직접 경쟁, action advising, 시간 추상화, 변화 감지.
- [theory-cusum.md](theory-cusum.md): CUSUM의 고전 가정과 RSVP에 적용할 때의 한계.
- [retired.md](retired.md): 폐기·보류한 아이디어와 이유.
- [../archive-baseline-notes.md](../archive-baseline-notes.md): 8월 LEHCA 재현 캠페인의 압축 이력.
- [../shareboard.md](../shareboard.md): 현재 에이전트 간 전달 사항만 담는 짧은 보드.

## 현재 코드 기준

- 정식 방법명: **RSVP — Residual Shaping-Value Prediction for On-Demand LLM Guidance Refresh**.
- 구현: `algorithm/rsvp/`, 설정: `config/algs/rsvp.yaml`.
- `algorithm/vigil/`, `vigil_*` 이름은 이전 실행 ID와 호환을 위한 legacy bridge다.
- LEHCA: `algorithm/lehca/`; QMIX 기준선: `config/algs/qmix_paper.yaml`.
- 5m6m 본 비교의 공통값: Adam, lr .001, batch 128, buffer 5000, gamma .99,
  epsilon 1→.05/50k, 평가 10k마다 32 episode.
- 현재 RSVP 본선은 action masking을 끄고 learner-time shaping만 사용한다.
- `lambda_floor_frac=.4`는 전체 지평의 40%에 floor가 된다는 뜻이다. 300k 진단에서
  기존 1.2M 실험과 절대 시간축을 맞출 때는 `1.6`을 사용해 floor를 480k로 유지한다.

## 현재까지 확인된 사실

1. **경로 무해성:** 5m6m S0b에서 guidance-none RSVP 경로는 같은 seed의 QMIX와
   소수점까지 같았다. 러너·learner 경유 자체가 성능을 바꾸는 문제는 발견되지 않았다.
2. **지침 채널은 불안정:** 정렬·rule 셰이핑은 QMIX가 실패한 seed를 구제했지만 잘 학습한
   seed를 훼손했다. 평균 상향보다 분산 재배치에 가깝다.
3. **실제 LLM 내용의 추가 가치 미확인:** cache-off/temp-.2 actual LLM은 동일 guidance
   풀을 상태와 무관하게 재생한 shuffle보다 두 seed 모두 낮았다. 현재 접지에서 상태 맞춤
   내용이 기여한다는 근거가 없다.
4. **마스킹 위험:** 완전 구성은 여러 실험에서 shaping-only보다 나빴다. β와 forbid 규칙,
   test-time masking을 별도 요인으로 보며 RSVP 본선에서는 끈다.
5. **고정 F 민감도는 환경별:** 3s5z·Pursuit에서는 잦은 갱신이 해로운 패턴이 있었지만,
   RSVP가 동일 ceiling의 fixed를 일관되게 이긴다는 증거는 없다.
6. **CUSUM은 발화하지만 효용은 미입증:** early refresh와 국면 정렬은 관측됐으나,
   “그때 갱신해서 이후 외부 return이 좋아졌다”는 인과 연결은 아직 없다.
7. **Q-004 수정:** episode를 가로지르는 armed spell을 h 적응 회계에서 누락하던 결함은
   수정됐다. 수정 전후 RSVP 결과는 합산하지 않는다.
8. **QMIX 논문 격차:** 5m6m 원 논문 예산은 5M인데 기존 주요 비교는 1.2M이었다.
   구현 문제와 예산 차이를 분리하기 위해 qmix_paper 5M을 별도 실행한다.

역사적 정량표는 날짜별 보고서로 복제하지 않고 [experiments-log.md](experiments-log.md)에만
둔다. 논문에 인용할 때는 동일 코드 버전·seed·지평으로 다시 집계한다.

## 2026-09-09 검증 배치

라이브 상태와 정확한 ID는 [experiments-log.md §0](experiments-log.md)을 따른다.

- F25 shuffle, seed 0/1: guidance 교체 빈도 자체의 효과.
- LEHCA 원형 스타일 actual-LLM F25, seed 0/1: F50→F25가 원형 부진을 회복하는지.
- RSVP actual-LLM Fmax200, seed 0/1: Q-004 수정 후 조기 갱신과 비용/성능.
- 대기: qmix_paper 5M seed 0/1, shaping-only actual-LLM fixed F25 seed 0/1.

핵심 비교는 다음 세 층이다.

| 질문 | 비교 |
|---|---|
| 원형 LEHCA의 F가 병목인가? | mask-on LEHCA F50 vs F25, 동일 seed·0–300k |
| RSVP 스케줄러가 도움이 되는가? | shaping-only fixed F25 vs RSVP Fmax200 |
| 성능 차이가 단순 예산 문제인가? | qmix_paper 1.2M vs 5M |

## 판정 규칙

- AUC와 마지막 10%를 모두 보고하고, 비교 팔은 동일 seed·동일 지평으로 맞춘다.
- 1 seed 또는 초기 한 점으로 성공을 확정하지 않는다. seed가 엇갈리면 미결이다.
- refresh, 실제 LLM request, cache hit, 토큰/지연을 서로 다른 비용으로 보고한다.
- actual LLM, replay/shuffle, cache on/off, temperature, masking, lambda 시간축을 섞어
  하나의 평균으로 합산하지 않는다.
- RSVP가 fixed보다 좋지 않으면 파라미터 탐색보다 guidance/grounding/shaping 병목을
  먼저 수정한다.
- “동적 F 필요”는 동일 shaping 채널에서 budget-matched fixed/random/phase control을
  이긴 뒤에만 주장한다.

## 다음 의사결정

1. 현재 300k 배치 회수: F25, Fmax200, shuffle을 동일 seed로 비교한다.
2. shaping-only fixed F25와 RSVP Fmax200으로 scheduler-only 효과를 판정한다.
3. qmix_paper 5M으로 백본 예산 문제를 판정한다.
4. 둘 다 QMIX에 못 미치면 RSVP 튜닝을 중단하고 predicate 의미·grounding·shaping
   크기와 원 논문 구현 차이를 재설계한다.
5. 채널 이득이 확인될 때만 Fmax100, phase-aligned/permuted, budget-matched random과
   critic 예측/유지-갱신 분기 실험으로 확장한다.

## 운영 원칙

- Sacred의 `RUNNING` 문자열만 믿지 않고 Slurm, 프로세스, 마지막 `t_env`를 함께 본다.
- 기존 실행 ID와 raw 결과는 이름을 바꾸거나 삭제하지 않는다.
- 새 조건은 새 W&B group으로 기록하고, 실패·절단 런은 명시적으로 제외한다.
- 라이브 상태를 여러 문서에 복제하지 않는다. 수치는 experiments-log, 논문 서사는
  draft-paper, 에이전트 전달은 shareboard 한 곳에만 쓴다.
