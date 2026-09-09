# LEHCA 베이스라인 캠페인 압축 이력

2026-08-25~09-02의 날짜별 분석·자동실험 계획·중간 보고서를 이 문서 하나로 통합했다.
세부 원문은 Git 역사에서 복원할 수 있고, 이후 연구 수치는
[research/experiments-log.md](research/experiments-log.md)를 따른다.

## 재현 범위

- 공식 코드가 없어 LEHCA를 PyMARL/QMIX 위에 재구현했다.
- Commander는 `gpt-oss-20b`, 구조화 상태 요약, predicate reward shaping, 매 스텝
  action-rule grounding으로 구성했다.
- 논문에 공개되지 않거나 모호한 접지·캐시·lambda·F 세부값은 실험 설정으로 선택했다.
  따라서 “원 논문의 완전한 재현”이 아니라 원형 스타일 재구현으로 부른다.
- 원문 대비 차이와 현재 의미 손실은
  [research/lehca-grounding-audit-20260907.md](research/lehca-grounding-audit-20260907.md)에 있다.

## 핵심 결과

| 환경/구성 | 당시 결과와 판정 |
|---|---|
| 2s3z QMIX, 1M | AUC_early .332±.045, final .963±.018, n=5 |
| 2s3z shaping-only | AUC_early .454±.025, final .941, n=2; 논문 초기 AUC와 근접 |
| 2s3z 완전 구성 | AUC_early .250, final .536; masking이 유해 |
| 5m6m QMIX, 1.2M | AUC .127±.107, final .42±.30, n=3; seed 분산이 매우 큼 |
| 5m6m shaping-only | AUC .138, final .945, n=1; 단일 seed라 잠정 |
| 5m6m 완전 구성 | AUC .001, final 0; 붕괴 |

지표 정의와 코드 버전이 이후 라운드에서 바뀌었으므로 위 수치를 최신 결과와 합산하지
않는다. 특히 5m6m 원 논문 예산은 5M인데 당시 비교는 1.2M이었다.

## 진단 결론

1. 마스킹이 초반·후반 학습을 끌어내리는 현상이 반복됐다. `forbid stop`은 카이팅을
   막았고, soft tilt 크기가 당시 Q-gap보다 컸다. β를 .5→.1로 낮추면 일부 회복했지만
   shaping-only가 더 안정적이었다.
2. 탐색 행동까지 soft preference로 편향하는 실험은 기각됐다.
3. shaping reward는 수집 시점 lambda를 버퍼에 굳히지 않고 learner에서 현재 lambda로
   합성하는 편이 일관적이었다. 다만 과거 guidance를 새 guidance로 재계산하는 것은 아니다.
4. 2s3z F25/F50/F100 등 단일-seed 비교에서는 F 효과보다 seed 변동이 컸다.
5. 5m6m의 거친 cache key는 동일 Marine 구성에서 90% 이상 적중해 guidance와 masking을
   사실상 정적으로 만들었다. cache on 결과를 동적 LLM 효과로 해석하면 안 된다.
6. 가이던스 JSON 파싱보다 조건·표적·시간창이 고정 predicate로 손실되는 grounding이
   더 큰 문제였다.

## 당시 삭제된 문서

- `analysis-2s3z-20260826.md`, `analysis-20260827.md`, `final-report-20260827.md`
- `feedback-masking-20260829.md`, `probe-commander-20260829.md`
- `masking-fupdate-verdict-20260829.md`, `grounding-analysis-20260829.md`
- `autoexp/plan.md`, `autoexp/plan-phase6-masking.md`, `autoexp/state.md`
- `baseline-final-20260902.md`, `lehca-reproduction.md`

같은 문제를 다시 조사할 때는 이 요약 → 두 LEHCA 감사 문서 → experiments-log 순서로
확인한다.
