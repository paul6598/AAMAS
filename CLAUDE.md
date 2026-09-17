# 프로젝트 안내

이 저장소는 LEHCA 재구현과 RSVP의 종료된 연구 자료다. 연구 상태와 최종 해석은
`docs/research/research-summary.md`, 실행법은 `docs/research/code-guide.md`를 먼저 읽는다.

- 새 실험 제출·진행 중 작업의 변경은 사용자의 명시적인 요청이 있을 때만 한다.
- `results/`와 `wandb/`의 원자료·checkpoint·동결 소스는 보존한다.
- 과거 실험 ID, VIGIL 명칭과 historical 설정을 현재 결과와 합산하지 않는다.
- 결과를 보고할 때 예측력, guidance 교체 진단, MRT 근접 효과, 전체 성능을 구분한다.
- 기존 실험에서 생성한 로그를 새 구현의 검증 결과로 읽지 않는다.
- 변경 후 관련 unittest, Python/shell 문법, 문서 링크와 `git diff --check`를 확인한다.
