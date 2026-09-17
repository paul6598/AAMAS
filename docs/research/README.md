# 연구 자료 안내

2026-09-17 공유 시점에 연구를 보류했다. 예약·실행 중인 실험은 없으며,
반복되는 상태 MD와 끝난 제출 계획을 종합 보고서로 통합했다.

| 자료 | 용도 |
|---|---|
| [연구 종합 보고서](research-summary.md) | 문제, 구현, LEHCA 재현 시도, RSVP 구체화, 최종 성능과 한계 |
| [코드·실행 안내](code-guide.md) | 환경, 주요 함수, 실행·분석·검증 방법 |
| [검증 방법](validation.md) | 실제 구현·실행한 metric과 미실행 제안의 구분 |
| [최종 결과 JSON](final-results.json) | Sacred356--380의 최종 설정·곡선·Slurm 완료 증거·원자료 SHA256 |
| [LLM 직접 검토](llm-output/llm-guidance-review-20260912.md) | 실제 출력16개와 원 API 요청·응답8개를 읽은 분석 |
| [관련연구](related-work.md) | 당시 문헌 조사 메모 |
| [실험 원장](archive/experiments-log.md) | 시점별 설정·실행 ID·판정의 전체 이력; 당시 미완료 표는 역사 기록 |

`llm-output/`의 다섯 파일은 기존 출력·원응답·직접 검토·프롬프트 재구성 기록으로 보존했다.
`archive/`에는 상세 backbone/grounding/trace 감사와 고정된 결과·제출 JSON을 남겼다.
실행 분석은 핵심 집계기3개와 별도 `analysis/audits/`만 남겼고 회귀 테스트는 `tests/`로 옮겼다.
GRF 소스는 공유본에서 제외했으며 Git `f138bf0`과 서버 백업에 보존했다.
현재 결과와 해석은 종합 보고서가 기준이며, 과거 계획·Markdown 초안은 유지하지 않는다.
LaTeX 작성물은 [AAMAS_draft/main.tex](../../AAMAS_draft/main.tex)만 편집 진입점으로 사용한다.

원자료는 서버의 `results/`와 `wandb/`에 그대로 있다. 한 번의 실행은 Sacred config,
동결 source manifest, guidance/validation 로그와 함께 식별한다. VIGIL은 RSVP의 과거
이름이며 기존 immutable 실험 ID와 import 호환 경로에만 남는다.
