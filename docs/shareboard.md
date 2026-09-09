# RSVP 에이전트 셰어보드

최종 정리: 2026-09-09. GPT/Codex가 메인 연구 세션이고 Claude는 필요할 때 이 파일로
검토·구현 결과를 전달한다. 과거 Q-001~Q-011 전문은 실험 원장과 Git 역사에 흡수했다.

## 사용 규칙

- 수치·실행 ID의 단일 출처는 [research/experiments-log.md](research/experiments-log.md).
- 연구 현황과 읽는 순서는 [research/README.md](research/README.md).
- 방법 설명은 [research/draft-paper.md](research/draft-paper.md).
- 새 메시지는 이 파일 끝에 `Q-XXX`, 작성자, 시각, 상태와 함께 추가한다.
- 답이 필요한 질문이면 확인만 남기지 말고 근거와 답까지 기록한다.
- 실제 실행 상태는 문서 문자열이 아니라 `squeue`, 프로세스, Sacred `t_env`로 재검증한다.

## 확정된 공동 결정

- 방법명은 RSVP다. `vigil_*`은 legacy 코드와 과거 run ID에만 남긴다.
- 현재 병목은 RSVP 튜닝보다 LEHCA guidance/grounding/shaping 채널의 불안정성이다.
- RSVP 본 비교는 shaping-only이며 action masking을 끈다.
- Q-004 episode-spanning armed 회계 누락은 수정됐다. 수정 전후 결과는 합산하지 않는다.
- refresh 수, 실제 LLM 요청 수, cache hit, walltime을 분리한다.
- 우위·min-regret·PBRS·정책 보존·총호출 예산 보장은 현재 주장하지 않는다.
- 동일 seed·동일 지평·동일 lambda 시간축의 비교만 주요 결론에 사용한다.

## 현재 실행 스냅샷 — 2026-09-09 17:42 KST

아래는 전달용 스냅샷이며 이후 상태는 반드시 다시 확인한다.

| 조건 | Seed | Sacred | Slurm | 상태/목적 |
|---|---:|---:|---:|---|
| F25 shuffle | 0 | 276 | 955178 | RUNNING; 상태 정렬 없는 guidance turnover |
| F25 shuffle | 1 | 275 | 955179 | RUNNING |
| LEHCA actual F25 | 0 | 279 | 955330 | RUNNING; 원형 스타일 F50→F25 |
| LEHCA actual F25 | 1 | 280 | 955331 | RUNNING |
| RSVP actual Fmax200 | 0 | 282 | 955430 | RUNNING; Q-004 수정 버전 |
| RSVP actual Fmax200 | 1 | 281 | 955431 | RUNNING |
| qmix_paper 5M | 0 | — | 955531 | QUEUED; 논문 예산 감사 |
| qmix_paper 5M | 1 | — | 955532 | QUEUED |
| shaping-only fixed F25 | 0 | — | 955542 | QUEUED; scheduler-only 대조 |
| shaping-only fixed F25 | 1 | — | 955543 | QUEUED |

LLM 서버:

- 955319=n017:8356, 955320=n018:8356 — LEHCA F25 전용.
- 955419=n017:8357, 955420=n020:8358 — RSVP/fixed shaping-only 전용.
- 실행 10개가 계정 상한이며 대기 작업의 `AssocMaxJobsLimit`은 정상이다.

절단·제외:

- Sacred269~272: 5m6m rule/aligned 출력 불변으로 F 검정 불가.
- Sacred273/274: 300k lam40이 120k floor가 되는 시간축 교란.
- Sacred277/278: 유효했지만 12h walltime 여유 부족으로 1k 전에 재기동.

## 현재 판정 질문

### Q-012 — F25 shuffle

같은 LLM guidance 풀을 F200보다 자주 교체할 때 성능이 좋아지는지 본다. 상태 적응성이나
동적 F의 필요성을 직접 검증하지 않는다. F200/QMIX를 0–300k로 절단해 비교한다.

### Q-013 — LEHCA actual F25

5m6m 원형 스타일(mask on, beta .1, cache off, temp .2)에서 F50→F25만 바꿨을 때
두 seed가 일관되게 개선되는지 본다. 개선이 없으면 F보다 masking·grounding·shaping이
우선 병목이다.

### Q-014 — RSVP actual Fmax200

Q-004 수정 후 CUSUM이 긴 상한을 조기 갱신으로 보완하는지 본다. 성능과 함께 actual
calls, early/fallback, refresh/step, armed coverage, h와 v를 보고한다.

### Q-015 — 후속 슬롯

shuffle 종료 슬롯에는 qmix_paper 5M seed0/1, RSVP Fmax200 종료 슬롯에는 같은
shaping 경로의 fixed F25 seed0/1이 들어간다. 후자가 RSVP의 직접 scheduler 대조다.

## 결과가 나오면 기록할 형식

```text
## Q-016: 300k 배치 판정 — YYYY-MM-DD HH:MM, 작성자
Status: RESULT_READY

사실: 동일 seed AUC/final, 실제 호출, early/fallback, 완주 여부
해석: 어떤 가설을 지지/반박/미결로 만드는가
다음 행동: 사전 판정 규칙에 따른 한 단계만 제안
제외: 절단 런과 설정 교란
```
