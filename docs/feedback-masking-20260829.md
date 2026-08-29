# 피드백: 마스킹 ablation 부호 역전의 원인 추정과 확인 사항 (2026-08-29)

대상: 논문 Table 5는 마스킹 제거 시 AUC_early가 전 맵에서 하락(2s3z 0.4549→0.3425,
5m6m 0.0612→0.0446)한다고 보고하지만, 우리 재현에서는 마스킹이 전 맵에서 초반
드래그(DIAG_B 정체, G1 어닐 직후 폭발, H1 0.223 vs H6 0.388, 3s5z 0.021→0.49).
리뷰 세션 작성. 논문 본문(Eq. 5–7, 프롬프트 설계, Table 5/6), 코드
(`algorithm/lehca/controller.py`, `masking/compiler.py`, `commander/*`,
`env/semantic/sc2.py`, `src/learners/q_learner.py`), SMAC 소스, `results/logs`를 대조.
vLLM이 내려가 있어 Commander 실제 출력은 확인하지 못함 — 아래 추정은 그 전제 하의
정황 추론이며 §3의 확인 항목이 선행되어야 함.

## 0. 먼저 확정한 것: 메커니즘 구현은 논문과 일치

Eq. 5–6의 구현(`controller.py`)은 논문과 동일하다:
- `Q̃ = Q + β·log W_soft`, 하드 금지 액션은 −∞ (allowed = avail ∧ hard)
- ε-random 브랜치는 allowed 집합에서 균등 추출 (논문: "never samples outside the allowed set")
- TD 백업은 raw Q 사용, 마스크는 behavior policy에만 관여 (논문 명시와 일치)
- 빈 allowed 집합 시 avail로 fallback (논문의 non-emptiness 가정에 대한 실용적 처리)

따라서 부호 역전은 구조 차이가 아니라 **(a) 마스크 내용, (b) β·W 스케일, (c) 베이스라인
강도, (d) 논문이 명시하지 않은 세부**에서 나온다.

## 1. 원인 추정 (가능성 순)

### 1-1. 하드 마스크가 카이팅(kiting)을 막는다 — 마스크 "내용" 문제 [최유력]

SMAC에서 공격 액션은 **사거리 안**에서만 available이고(`get_avail_agent_actions`), 승리의
핵심 미세컨트롤은 "적에게서 멀어지는 이동"이다(2s3z의 Stalker, 3s_vs_5z 전체).
우리 요약문은 "Enemy centroid lies to the **east**"를 주고, 프롬프트는 공격 우선·전략적
일관성을 요구한다. LLM이 `forbid: ["move_west"]`류(후퇴 방향 금지)를 내놓으면 카이팅이
원천 봉쇄된다.

정황 증거:
- 3s_vs_5z(순수 카이팅 맵)에서 완전 구성 0.021 붕괴 → 마스킹만 끄면 0.49 회복
- G1: 마스크 어닐(300k) 직후 0.44→0.91 폭발 — 마스크가 브레이크였다는 직접 신호
- 논문의 마스크 예시는 "prioritize focus fire on enemy damage dealers", "move south" 같은
  **전략 수준**. 우리 어휘에는 `attack_lowest_health`, `attack_nearest` 같은 **스텝 단위
  마이크로 휴리스틱**이 있어 LLM이 사실상 매 스텝 컨트롤러 노릇을 하게 된다.
- 셰이핑 쪽에는 `retreat_low_health`(후퇴 보상)가 있는데 마스크가 후퇴를 금지하면
  두 채널이 서로 모순되는 신호를 준다.

### 1-2. β·log W의 절대 스케일이 Q에 비해 크다 [유력]

- W ∈ [1.1, 5], 기본 prefer_weight 2.0 → 틸트 = 0.5·ln 2 ≈ **0.35**, w=5면 **0.80**.
- 로그의 `q_taken_mean`: 2s3z 초반 0.04 → 1.9, 5m6m 0 → 1.6. QMIX 개별 Q_i는 믹서 때문에
  스케일 자체가 정해져 있지 않고, 같은 상태 내 액션 간 Q 차이는 보통 이보다 훨씬 작다.
- 즉 AUC 창(0–200k) 내내 greedy 브랜치는 **Q가 아니라 LLM 선호가 결정**한다.
  "Q + 작은 보정"이 아니라 "LLM 휴리스틱 + tie-break로서의 Q"에 가깝다.
- 논문 β는 미공개. 매우 작았다면(예: 0.05) 마스킹은 논문 표현대로 "moderate but
  systematic"한 미세 보정이 된다. 우리 β=0.5는 순전히 추측값이다.

### 1-3. 금지된 액션의 Q가 학습되지 않는 extrapolation error [구조적]

하드 마스크된 액션은 버퍼에 절대 들어가지 않지만, 학습기는 여전히 **env-available
전체**에 대해 max_a Q(double-Q도 live Q의 argmax)로 타깃을 만든다
(`q_learner.py` L70–80: `avail_actions` 기준 마스킹만 함). 한 번도 실행되지 않은 액션의
Q가 함수근사로 과대추정되면 그것으로 부트스트랩 → 값 폭주/불안정(BCQ류 문헌의
extrapolation error). 논문도 같은 설계("only the behavior policy is obtained from (5)–(6)")
이므로, 논문에서 forbid가 드물었다면 드러나지 않았을 것이다. shaping-only가 잘 되는
이유이기도 하다 — 모든 액션이 탐색되어 Q가 정상 학습된다.

### 1-4. 우리 QMIX가 이미 잘 탐색한다 [배경 요인]

논문 2s3z QMIX AUC 0.254 vs 우리 0.332. 마스킹은 "탐색 공간 가지치기"이고, 그 가치는
베이스라인의 탐색이 나쁠 때 크다. 우리 shaping-only(0.399)가 논문 shaping-only(0.3425)를
이미 넘으므로 마스킹이 보탤 여지는 작고, 잘못 자르면 손해만 남는다. 5m6m처럼 QMIX가
불안정한 맵에서만 가이던스 효과가 재현된 것과 일관된다.

### 1-5. 부수적: "dynamic" 마스크가 사실상 static

`llm_commander.py`의 캐시는 만료가 없다(5000개 FIFO, 히트율 ~63%). 동일 coarse state엔
학습 내내 같은 룰이 적용된다. 논문은 "reused **between refreshes**"라고만 했다.
셰이핑에도 똑같이 적용되므로 부호 역전의 주범은 아니지만, 마스크 오류가 있으면 그것이
1M 스텝 내내 고정된다는 점에서 1-1을 증폭한다. temperature 0.2와 결합하면 사실상
결정론적 정적 룰셋이다.

### 1-6. 통계적 주의

논문 Table 5의 마스킹 효과는 2s3z Δ0.11, 8m Δ0.036, 2m1z Δ0.12, 5m6m Δ0.017이고
**시드 표준편차 미보고**. 우리 AUC 시드 std는 0.04–0.09. 부호는 4개 맵에서 일관되므로
전부 노이즈로 치부할 수는 없지만, 5m6m/8m 수준의 효과는 우리 분산 안에 들어간다.
"논문은 훨씬 좋다"는 2s3z·2m_vs_1z에 한정된 얘기로 보는 것이 정확하다.

## 2. 구현상 실수/편차 점검 결과

명백한 버그는 찾지 못했다. 다만 다음은 논문 대비 편차이거나 결과에 영향을 줄 수 있는
설계 선택이다.

| # | 항목 | 위치 | 성격 |
|---|---|---|---|
| A | **Commander 출력이 어디에도 기록되지 않음** | runner/commander | 진단 불가의 근본 원인. 로그·sacred·wandb 어디에도 forbid/prefer 내용이 없다 |
| B | 스냅샷이 **privileged full state** 사용 (모든 적의 HP·위치, `env.enemies` 직접 접근) | `sc2.py: snapshot()` | 논문은 "same observable information available to MARL agents"(sight range 내). 마스킹 원인은 아니지만 재현 충실도 편차로 보고 필요 |
| C | `attack_lowest_health` / `attack_nearest` 토큰 | `sc2.py`, `base.py` | 논문 어휘에 없는 마이크로 휴리스틱. 전역 최저 HP 적이 사거리 밖이면 prefer가 무효 → 가이던스가 상태에 따라 켜졌다 꺼졌다 함 |
| D | 후퇴 방향을 forbid 가능 | `compiler.py` | 논문 프롬프트는 "hard constraints … to avoid infeasible or risky decisions" — 위험 회피용. 후퇴 금지는 그 반대 |
| E | 캐시 무만료 | `llm_commander.py` | 1-5 참조 |
| F | F_update=200 vs 에피소드 길이 ~50–100 | runner | 에피소드 경계를 넘어 가이던스가 유지됨. 새 에피소드 시작(approaching)에 이전 에피소드 중반(engaged)의 마스크가 적용될 수 있음. 논문도 같은 설계이나 F_update 값은 미공개 |
| G | 프롬프트의 "rolling win rate / steps elapsed" | `sc2.py: summary()` | 논문 d_t에 없는 정보. LLM이 승률 낮을 때 더 보수적/제한적 룰을 낼 가능성 (추측) |
| H | Eq. 7의 Update(M_t, W_t, d_t, **u_t**, π) | `compiler.py` | 논문은 이전 마스크와 최근 joint action을 입력으로 받는 stateful 갱신을 암시. 우리는 매 스텝 무상태 재컴파일. 실질 영향은 불명 |
| I | mask_anneal_t=150k로 마스크를 **급격히** 제거 | runner | 150k에서 behavior 분포가 불연속 변화. 논문에는 없는 장치. G1 폭발이 이 지점에서 나온 것은 마스크가 해로웠다는 증거이지 어닐이 좋다는 증거는 아님 |

## 3. 확인하면 좋을 것 (비용 순)

### 3-1. 실제 가이던스 확인 [즉시, 필수]

1. Commander 호출마다 `{t_env, cache_key, hit/miss, guidance JSON}`을 sacred run dir 또는
   `results/guidance/<run>.jsonl`로 덤프. 가장 중요. 이것 없이는 위 추정 전부 검증 불가.
2. vLLM 뜨면 오프라인 프로빙: 대표 상태(approaching 풀피 / engaged 반피 / 아군 저체력)
   각 3샘플씩 → forbid 토큰 분포. 후퇴 방향 forbid 빈도, `attack_*` prefer 편중도 확인.
   (2s3z용 프로빙 스크립트는 리뷰 세션 스크래치에 작성해 둠 — 서버만 뜨면 재사용 가능.)

### 3-2. per-step 마스크 통계 로깅 [작음]

runner/controller에 다음을 wandb로:
- `mask_forbid_frac`: available 액션 중 하드 금지된 비율 (에이전트 평균)
- `mask_override_rate`: β·log W 틸트가 greedy argmax를 **바꾼** 비율 (raw argmax ≠ tilted argmax)
- `mask_fallback_rate`: allowed 집합이 비어 avail로 fallback한 비율
- `q_gap_mean`: available 액션 중 (Q_max − Q_2nd) 평균 — 틸트 0.35–0.8과 직접 비교 가능

이 4개면 1-1(forbid_frac 높음)과 1-2(override_rate 높음, q_gap ≪ 틸트)를 정량 분리할 수 있다.

### 3-3. 성분 분리 진단 (250k, 시드 1개씩) [중간]

기존 DIAG_B는 하드+소프트를 함께 켠 상태라 구분이 안 된다.
| 런 | 설정 | 판별 |
|---|---|---|
| M1 | `beta=0` (하드만) | 하드 forbid 단독 해악 |
| M2 | forbid 무시 + β=0.5 (소프트만) | 소프트 틸트 단독 해악 |
| M3 | `beta=0.1` (하드+약한 소프트) | 스케일 문제인지 |
| M4 | **`commander=rule` + 마스킹 on** | RuleCommander는 forbid가 비어 있고 prefer만 준다. 이게 안 해로우면 LLM의 forbid가 주범 (논문 Table 6 Rule-Commander 2s3z AUC 0.392 > QMIX와 비교 가능) |

M4가 가장 싸고 정보량이 크다 — LLM 호출도 필요 없다.

### 3-4. 어휘/프롬프트 수정 후 재검증 [1-1 확인 시]

- forbid 허용 토큰을 전략 수준으로 제한(예: `attack_type:<X>` 금지, `stop` 정도)하고
  이동 방향 forbid는 아예 불허하거나 "적 방향으로의 이동만" 금지 가능하게.
- `attack_nearest`/`attack_lowest_health`를 어휘에서 제거하고 논문 예시에 가까운
  `attack_type:<X>`(=focus on damage dealers)만 남김.
- 캐시에 refresh-window 만료(예: 같은 키라도 N 스텝 지나면 재호출) 적용.
- 이 상태에서 마스킹 on/off ablation을 다시 돌려야 논문 Table 5와 비교 가능한 수치가 나온다.

### 3-5. 1-3(extrapolation) 직접 검증 [선택]

학습기에서 타깃 max를 **hard-allowed 집합**으로도 제한해 보는 변형(Q-learning 관점에서는
constrained-action MDP로 바꾸는 것이라 논문과 다른 알고리즘이 됨 — 본 연구용 아이디어로만).
또는 forbid된 액션의 Q 평균을 로깅해 실행된 액션 Q 대비 과대추정 여부만 확인.

## 4. 보고서 반영 시 문구 제안

현재 final-report의 "마스킹 드래그 — 4회 독립 입증"은 **관측으로서는 견고**하지만,
"논문 Table 5와 정면 배치"는 §1-6과 §2-A(가이던스 미기록) 때문에 "우리 마스크
내용·스케일 하에서"라는 조건을 달아야 한다. 논문의 마스크 어휘·β·forbid 빈도가 전부
미공개이므로, 현 시점 결론은 "하드 마스킹의 유익성은 마스크 내용에 극도로 민감하며,
논문이 명시하지 않은 그 세부가 재현의 병목"이 정확하다. §3-1·3-2를 채운 뒤에는
"LLM이 실제로 X를 forbid했고 그것이 카이팅을 차단했다"는 인과 서술로 격상 가능.
