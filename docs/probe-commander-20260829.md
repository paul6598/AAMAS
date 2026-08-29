# Commander 출력 프로빙 결과 (2026-08-29)

리뷰(docs/feedback-masking-20260829.md §3-1) 요청에 따라 gpt-oss-20b Commander의
실제 forbid/prefer 분포를 확인. 2s3z 시스템 프롬프트(현행), temperature 0.2,
reasoning_effort=low, 대표 상태 3종 × 3샘플.

## 결과

| 상태 | forbid (3샘플) | prefer |
|---|---|---|
| approaching, 풀피 | stop×3, move_north×2, move_south×2 | attack_type:Stalker×3, attack_lowest_health, attack_nearest, move_west×2, move_east×2 |
| engaged, 반피 | stop×3(중복 포함 4), move_north, move_south, **move_all** | attack_type:Stalker×3, attack_lowest_health×3, attack_nearest×2, move_west×2 |
| 아군 저체력 | stop×3, **move_all×2** | attack_type:Stalker×3, attack_lowest_health×3, move_west, move_north |

집계 (9샘플): forbid = {stop: 10, move_north: 3, move_south: 3, move_all: 3};
prefer = {attack_type:Stalker: 10, attack_lowest_health: 7, move_west: 6,
attack_nearest: 4, move_east: 3, move_north: 3, attack_type:Zealot: 1}.
prefer_weight는 대부분 2.0–3.0 (틸트 β·ln W = 0.35–0.55).
서브골은 9/9에서 focus_fire·kill_type·protect_type·retreat_low_health 포함.

## 해석 — 리뷰 가설 1-1 확인

1. **`stop`이 100% 금지된다.** SMAC 카이팅의 기본 원자(stop→적 접근 대기→사격→이탈)
   가 학습 내내 봉쇄됨. 2s3z Stalker, 3s_vs_5z 전체에 치명적.
2. **교전·저체력 상태에서 `move_all` 금지 (3/6)** — 후퇴 자체가 불가능. 동시에
   서브골로 `retreat_low_health`를 준다 → 두 채널이 정면 모순 (리뷰 지적 그대로).
3. 접근 단계의 `move_north/south` 금지는 적이 동쪽이라 "직진하라"는 뜻이지만,
   측면 기동·산개를 막는다.
4. prefer는 `attack_lowest_health`(7/9)·`attack_nearest`(4/9) 등 **스텝 단위 마이크로
   휴리스틱**에 편중 — 논문 예시("focus fire on damage dealers", "move south")보다
   훨씬 저수준. β·ln W ≈ 0.35–0.55는 초반 Q 격차를 압도할 크기.
5. temperature 0.2라 샘플 간 편차가 작음 → 캐시 무만료와 결합해 사실상 **정적 룰셋**.

결론: 마스킹 드래그의 인과 사슬 — "LLM이 stop/후퇴를 금지 → 카이팅 불가 → 초반
학습 지연, 카이팅 맵에선 붕괴" — 이 실제 출력으로 확인됨. 이는 **우리 프롬프트·
어휘가 유도한 결과**이므로 논문 마스킹 자체의 결함으로 일반화할 수 없다.

## 조치 (2026-08-29)

- 가이던스 JSONL 로깅(`results/guidance/<run>.jsonl`) + 마스크 통계 4종
  (`mask_forbid_frac`, `mask_override_rate`, `mask_fallback_rate`, `q_gap_mean`) 구현.
- `mask_vocab=strategic` 플래그: forbid는 `attack_type:*`/`attack_all`만 허용
  (stop·이동 방향 forbid 불허), `attack_lowest_health`/`attack_nearest` 제거.
- 진단: M4 = Rule-Commander + 마스킹(LLM forbid 없음), M5 = LLM + strategic 어휘 +
  마스킹. 둘이 QMIX 수준이면 "마스킹 해악은 어휘 문제"로 확정.
