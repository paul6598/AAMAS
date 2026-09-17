# LEHCA 기반 경로 검증 — 2026-09-08

## 질문

LEHCA/RSVP가 QMIX를 이기지 못하는 현상의 앞단에서, 지침을 끄면 기본 QMIX 학습 경로와
동일한가? 이후 지침 내용과 셰이핑 경로를 어떤 대조군으로 분리할 것인가?

## 1. 코드 및 실제 배치 검증 결과

재현 스크립트: `analysis/verify_lehca_backbone.py`.
산출물: `results/diagnostics/backbone_20260908_1732/report.json`(CPU),
`results/diagnostics/backbone_cuda_20260908/report.json`(본실험 GPU 경로).

- 기존 5m6m QMIX Sacred142/153도 `lehca_q_learner`와 Adam을 사용한다. 현재 RSVP와
  optimizer가 달라서 생긴 비교는 아니다.
- 실제 5m6m 환경에서 seed0/1 각각 2 episode를 새로 생성했다. 지침·셰이핑·마스킹을
  모두 끈 `EpisodeRunner+BasicMAC`과 `LehcaRunner+LehcaMAC`의 state, observation,
  available action, 선택 action, 환경 reward, termination, filled tensor가 모두 정확히 같았다.
- 같은 실제 배치에서 두 MAC/learner를 같은 초기값으로 두고 8회 업데이트했다.
  loss, online parameter, target network의 최대 절대 차이는 모두 0이었다.
- `shaping_in_learner=True`가 만든 보상과 같은 보상을 명시적으로 넣은 대조 업데이트도
  parameter 차이 0이었다. sampled batch를 수정한 뒤 replay 원본 reward가 그대로임을 확인했다.
- RSVP critic 생성·학습은 전역 CPU torch RNG를 소비한다. CPU 정책 실행에서는 같은 seed의
  궤적을 바꿨다. 본실험처럼 정책이 CUDA에서 실행될 때 seed0/1의 초기 RSVP-off 궤적은
  QMIX와 정확히 같았다. 짧은 결정론 검사이며 장기 성능 동등성의 증명은 아니다.

따라서 현재 확인 범위에서 기본 QMIX 업데이트식이나 지침-off LEHCA 경로의 구현 오류를
우선 원인으로 지목할 증거는 없다. 다음 검사 대상은 지침이 만드는 셰이핑 신호다.

## 2. S0b 대조군

모든 팔은 RSVP runner, fixed F200, masking off, replay-time shaping, lambda40,
epsilon anneal50k, 5m6m 1.2M을 공유한다. none/rule/shuffle은 LLM을 호출하지 않는다.

| 팔 | 의미 | seed / Sacred | 상태 |
|---|---|---|---|
| none | 동일 실행 경로에서 guidance와 F를 0으로 둔 기반 | 0/256, 1/257 | 실행 중 |
| rule | 기존 수작업 Commander; damage/kill/focus/ally penalty | 0/255, 1/258 | 실행 중 |
| shuffle | 다른 seed의 LLM guidance를 상태와 무관한 순서로 재생 | 0/260, 1/259 | 실행 중 |
| aligned | enemy damage/kill만 양의 방향으로 강화 | 0/261 | 파일럿 실행 중 |
| actual LLM | 기존 fixed F200 cache-off/temp0.2 | 0/242·243, 1/245·249 | 완료 |

shuffle seed0은 기존 LLM seed1 로그, shuffle seed1은 기존 seed0 로그를 사용한다.
현재 상태와의 정렬은 깨되 LLM 출력의 predicate/weight 주변분포를 재사용하려는 대조다.
서로 다른 학습 궤적이므로 완벽한 교환 검정은 아니며 평가 seed의 미래 출력은 사용하지 않는다.

251–254는 보상 진단 로그를 추가하기 전에 약 2분 실행한 예비 런으로 중단했다.
판정에서 제외하고 삭제하지 않는다. 255–261부터 `|F|`, RMS, 비영 비율, clip 경계 비율,
`|lambda F|`, 외부 reward 대응 통계와 두 절댓값 누적량의 비율을 기록한다.

첫 10k의 seed0 rule 진단은 `|lambda F|/|r_env|=1.40`, 비영 F 비율 0.40,
episode 평균 lambda-weighted shaping sum -0.92였다. 초기 랜덤 정책 구간의 한 점이며
성능 판정은 아니다. 기존 rule은 패널티를 포함하므로 aligned 팔을 추가했다.

## 3. 판정 순서

1. none이 기존 동일 seed QMIX 곡선과 일치하는지 확인한다. 차이가 나면 full-run RNG와
   runner 부수효과를 재검사한다.
2. aligned도 none보다 나쁘면 셰이핑 크기, terminal reward의 상대 가중, lambda와 탐색을
   우선 조사한다.
3. aligned는 좋고 rule/actual LLM이 나쁘면 지침 구성과 grounding을 우선 조사한다.
4. actual LLM이 shuffle보다 좋으면 상태에 맞춘 지침 내용의 기여 신호다.
   차이가 없다는 결과를 n=2로 동등성이라고 선언하지 않는다.
5. 이 단계가 통과한 뒤 critic 예측력과 유지/갱신 분기 검증으로 이동한다.
