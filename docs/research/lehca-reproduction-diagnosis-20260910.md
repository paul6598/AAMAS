# LEHCA 성능 재현 및 QMIX 순위 역전 진단 — 2026-09-10

현재 코드, Sacred 원시 결과, guidance 원문, 논문 본문과 공식 보충자료를 대조했다.
1--8절은 수정 전 코드에 대한 감사 기록이다. 그 결과에 따라 학습 코드와 설정을
수정했으며, 기존 결과 파일은 변경하지 않았다. 수정 상태는 9절에 분리해 기록한다.

**결론: 현재 결과는 논문과 조건이 같은 재현이 아니다. QMIX 쪽에는 확인된 탐색
스케줄 불일치와 미확인 환경 조건이 있고, LEHCA 쪽에는 평가 시 지침 갱신 누락과
실제 모순된 행동 제약이 있다. 보상 접지에도 전투 회피를 유도할 수 있는 설계가
있다. 어느 하나의 영향량이 전체 성능 격차를 설명한다고 확정할 단계는 아니다.**

## 1. 순위 역전은 실제지만, 맵과 지표를 구분해야 한다

5m_vs_6m에서 새 QMIX 5M 런 Sacred283/284는 로그의 `Finished Training`까지 완료했다.
Sacred `run.json`의 `RUNNING`은 종료 상태를 반영하지 않으므로 완료 판정에 사용하지 않았다.

| 5m_vs_6m | seed 수 | AUC, 0–1M | 최종 승률 |
|---|---:|---:|---:|
| 논문 QMIX, Table 4 | 5 | 0.0387 | 0.2950 |
| 논문 LEHCA, Table 4 | 5 | 0.0612 | 0.3672 |
| 현재 QMIX, Sacred283/284 | 2 | 0.0734 | 0.6141 |

현재 최종 승률은 4.5M–5M 안의 실제 평가 평균이다. 마지막 평가 시점은
4,996,566 / 4,992,276이다. 논문 표의 최종 승률 집계 창이 정확히 공개되지 않아
마지막 열의 집계 정의까지 동일하다고 단정하지 않는다. AUC는 논문처럼 첫 20%,
즉 **1M**까지 적분했다. 초기 y(0)=0을 가정하고 경계에서는 선형 보간했다.

QMIX seed별로 AUC는 0.01248 / 0.13435, 최종은 0.341875 / 0.886250이다.
**두 seed 평균은 논문 LEHCA보다 높지만 seed 편차가 매우 크다.** 기존 1.2M QMIX
142/153과 새 5M 런의 첫 120개 평가 시점·승률은 각각 정확히 같았다. 따라서 새
5M 런에서 설정이 우연히 바뀌어서 강해진 것은 아니다.

2s3z에서는 현재 QMIX 163/188/191의 0–200k AUC 평균이 0.3689다. 논문 QMIX
0.2536보다 높지만 논문 LEHCA 0.4549보다 낮다. 현재 최종 평균 0.9677도 논문
QMIX 0.9733보다 높지 않다. 따라서 “QMIX가 모든 맵·모든 지표에서 비정상적으로
강하다”는 해석까지 확대하면 안 된다.

## 2. QMIX 재현 조건: 확정 불일치와 미확인 조건

### 확정: epsilon anneal 300k → 50k

[공식 보충자료](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41598-026-54971-6/MediaObjects/41598_2026_54971_MOESM1_ESM.pdf)는
`epsilon_anneal_time=300000`을 명시한다. 현재 `config/algs/qmix_paper.yaml:7`과
`config/algs/lehca.yaml:8`은 50000이다. 감사 시점의 Sacred config **286개 모두
50000**이어서 300k 대조군이 아직 없다. `paper`라는 설정명은 전체 설정 일치를 뜻하지 않는다.

100k에서 현재 ε=0.05, 보충자료 설정 ε≈0.6833이다. 첫 300k 탐색률 적분은
38,750 대 157,500이다. 이는 초기 데이터 분포를 크게 바꿀 수 있지만, 어느 맵에서
얼마나 성능이 오르거나 내리는지는 두 알고리즘을 함께 바꿔 측정해야 한다.

### 미확인: 환경 보상과 SC2 버전

현재 `config/envs/sc2.yaml`은 `reward_sparse=False`, `reward_only_positive=True`,
`reward_scale=True`, `reward_scale_rate=20`, `reward_win=200`, `reward_death_value=10`이다.
설치된 SMAC `reward_battle()`은 적 체력/실드 감소와 처치에 보상한다. 따라서 현재
QMIX는 **피해·처치의 조밀한 보상**을 이미 받는다.

논문은 sparse/terminal reward를 반복해서 설명하지만, 보충자료는 보상을 환경 config에
위임하고 정확한 값은 주지 않는다. **논문이 실제로 reward_sparse=True였다는 증거는
없다.** 만약 원 실험 보상이 더 희소했다면 QMIX가 약하고 LEHCA 상대 이득이 커지는
현상을 설명할 수 있으므로 원 config 확인 우선순위가 높다. 결과를 맞추려고 임의로
sparse 모드만 켜서는 안 된다. 보상 스케일도 함께 달라져 λ 비교가 깨질 수 있다.

현재 실행 로그는 SC2 `B75689 (4.10)`을 확인해 준다. 원 논문 버전은 확인되지 않았다.
[SMAC 공식 저장소](https://github.com/oxwhirl/smac)는 SC2 버전 간 결과를 직접
비교하지 말 것을 명시한다. 논문이 4.6.2였다고 추정해서 확정 사실로 쓰지는 않는다.

보충자료는 target update를 steps로 설명하지만 PyMARL 및 현재 코드는 episodes로
계산한다. 원 코드가 PyMARL을 기반으로 했으므로 문서의 단위가 부정확할 가능성도 있다.
이 역시 확인할 사항이지 현재 코드를 즉시 steps로 바꿀 근거는 아니다.

## 3. LEHCA 평가에서 지침 갱신이 빠져 있다 — 확정

`algorithm/lehca/runner.py:104`는 `test_mode=True`이면 Commander 갱신을 바로
종료한다. `reset()`도 `state.guidance`를 지우지 않는다. 반면 `:147`에서는
`use_masking_at_test=True`이면 이 지침을 계속 적용한다.

그 결과 **마지막 훈련 에피소드에서 발급된 지침 하나를 다음 32개 테스트 에피소드에
걸쳐 사용한다.** 매 스텝 symbolic token을 현재 상태에 다시 매핑하는 것은 맞지만,
후퇴/전진 등 전략과 금지 목록 자체는 새 전투에 맞춰 갱신되지 않는다.

[논문 본문](https://www.nature.com/articles/s41598-026-54971-6)의 MARL policy learning
절은 reported evaluations에서도 fixed F_update에 따른 Commander 갱신을 설명한다.
따라서 현재는 해당 평가 절차를 충실하게 구현하지 못한다.

실제 `_maybe_refresh_commander()`에 32×70개 test step을 넣은 오프라인 프로브에서
호출은 0회, 이전 guidance는 그대로였다. 동일 객체의 train positive control에서는
호출 1회를 확인했다. SC2 롤아웃이나 성능 개선 실험은 아니다.

추가로 체크포인트 저장/복원은 신경망과 optimizer 위주이고 guidance를 저장하지 않는다.
새 프로세스에서 `evaluate=True`이면 guidance=None인 채로 갱신도 차단되어, 훈련 중
평가와 별도 체크포인트 평가가 서로 다른 정책이 될 수 있다.

이 문제는 mask-on LEHCA 평가에 중요하다. 하지만 Sacred244의 마지막 10%는
train 승률도 0이므로 **평가만 고치면 학습 붕괴 전체가 해결된다**고 볼 수 없다.

## 4. 모순된 LLM 제약이 실제 행동 금지로 적용된다 — 확정

프롬프트는 “never forbid all attack actions”를 명시하지만
`algorithm/lehca/commander/base.py`는 유효 token인 `attack_all` 금지를 통과시킨다.
서로 겹치는 forbid/prefer의 의미적 모순도 제거하지 않는다.

F25 seed1, `t_global=8250`의 실제 출력:

```json
{"applies_to":"all", "forbid":["attack_all"],
 "prefer":["attack_lowest_health"], "prefer_weight":2.0}
```

“가장 약한 적에게 집중사격” 전략 아래의 규칙인데 실행 결과는 모든 공격 금지다.
hard mask가 soft preference보다 우선한다. 현재 fallback은 **모든 행동**이 사라질
때만 작동하므로 이동/정지가 가능하면 공격을 전부 잃어도 복구하지 않는다.
실제 로그의 공격 금지 지침을 실제 compiler에 통과시키고 공격/이동 가능한 합성
availability에서 검사해 모든 공격 제거 및 fallback 미작동을 재현했다.

| 실행 | 유효 지침 | 전 아군 attack_all 금지 | move_all 금지와 방향 이동 선호 동시 존재 |
|---|---:|---:|---:|
| F50 seed0, Sacred244 | 24,001 | 154 | 1,396 |
| F50 seed1, Sacred250 | 6,569 | 40 | 426 |
| F25 seed0, Sacred279 | 12,003 | 18 | 433 |
| F25 seed1, Sacred280 | 12,003 | 50 | 667 |

동일 아군에 적용되는 `all/*/type:Marine` 규칙만 집계했다. 호출 횟수 기준이며
유해 행동의 점유 시간이나 성능 손실 비율은 아니다. 금지 자체가 항상 나쁘다는
뜻도 아니다. 여기서 확인한 문제는 **출력의 전략/선호와 실행 제약의 모순**이다.

## 5. 보상 접지와 프롬프트는 추가 병목 후보다

`llm_commander.py:50`은 enemy_damage/enemy_kill보다 focus_fire/protect_type/
retreat_low_health 등을 우선하라고 명시한다. 이것은 자체 설계한 프롬프트이며 원본
프롬프트와 동일하다는 근거가 없다.

현재 접지의 중요한 특징:

- `protect_type(Marine)`은 모든 Marine의 피해·사망에 패널티를 준다. 5 대 6 전투에서
  필요한 피해 교환까지 억제할 수 있다.
- `retreat_low_health`는 HP 30% 미만 유닛이 적 중심에서 멀어지면 보상한다.
  실제로 안전해졌는지, 나중에 전투에 복귀했는지는 조건에 없다.
- `focus_fire`는 피해가 없어도 같은 적을 겨냥한 행동만으로 양의 보상이 가능하다.
- `condition/target_id/time_window` 같은 목표 조건은 현재 schema에 보존되지 않는다.
- snapshot/summary에는 cooldown이나 개체별 전투 배치 정보가 충분하지 않다.
  특히 army separation <8을 “within typical attack range”로 설명하지만 설치 SMAC의
  `unit_shoot_range()`는 6이다. 군 중심 거리로 개별 공격 가능 여부를 표현할 수도 없다.

Sacred244는 1.2M 후반 train/test 승률 모두 0, 테스트 episode length 평균 70
(시간 제한), 외부 return 평균 약 0.0006이다. **적과 거의 교전하지 않는 행동으로
붕괴한 것과 일치하는 관측**이다. 패널티·마스킹·접근 정보 중 어느 것이 주원인인지는
궤적 및 통제 ablation 없이는 분리할 수 없다.

반대로 shaping 자체가 항상 해로운 것은 아니다. 같은 F200/lam40/mask-off/learner-time
경로의 1.2M 대조에서 다음 결과를 얻었다. 모든 행은 seed0/1이며 n=2의 기술통계다.

| 팔 | AUC 0–1M 평균 | 마지막 10% 승률 평균 |
|---|---:|---:|
| none = QMIX, 256/264 | 0.0734 | 0.4714 |
| rule, 262/263 | 0.1383 | 0.6641 |
| aligned damage/kill, 268/267 | 0.3387 | 0.5820 |
| 상태와 무관하게 섞은 LLM 지침, 265/266 | 0.3475 | 0.4727 |
| 실제 LLM F200, 242/249 | 0.1191 | 0.2786 |

정적 damage/kill만으로도 학습 속도가 개선되고 shuffle도 실제 LLM보다 높은 AUC다.
LLM의 현재 상태 추론이 이득의 필수 원인이라는 주장은 이 결과로 지지되지 않는다.
다만 이 대조는 출력 주변분포·경로·seed 편차가 완벽히 통제된 인과 검정은 아니다.

## 6. F를 줄이는 것만으로 회복되지 않았다

300k 종료 실험은 마지막 평가가 약 290k이므로, 공통 관측 범위 **0–290k**를 비교했다.
F50과 F25의 lambda floor는 모두 절대 480k로 맞춰져 있다. 아래 AUC는 논문의
AUC_early가 아니라 공통 290k 적분이다.

| LEHCA mask-on | seed0 AUC | seed1 AUC |
|---|---:|---:|
| F50, 244/250 | 0.0275 | 0.0133 |
| F25, 279/280 | 0.0000 | 0.0043 |

두 seed 모두 F25가 낮다. “지침이 너무 낡아서 실패하므로 F만 줄이면 해결된다”는
가설을 지지하지 않는다. 프롬프트·모델 설정은 같으나 LLM 서버와 실현 출력은 다르므로
엄밀한 결정론적 단일 변수 검정으로 보지는 않는다.

F25 mask-off/learner-time shaping 285/286도 300k까지 승리가 없다. 따라서 mask가
부진의 유일한 원인은 아니다. 다만 **동일 구간 QMIX도 두 seed 모두 승리 0**이어서
이 사실만으로 F25의 최종 성능 열세나 학습 불가능을 확정할 수 없다.

## 7. 우선순위가 낮아진 원인

- **QMIX 기본 구현/optimizer 분기:** guidance-off 256/264와 QMIX 142/153의
  120개 평가 시점·승률이 두 seed 모두 정확히 같다. 기존 GPU 배치 검사도
  지침-off LEHCA와 QMIX의 transition 및 8번 optimizer update가 동일했다.
  raw Q로 TD backup하는 방식은 논문 설명과도 맞는다.
- **단순 API/JSON 실패:** 최근 F25 런의 로그상 llm_failures=0이다. 의미가 잘못된
  valid JSON이 문제인 경우는 이 지표에 잡히지 않는다.
- **짧은 예산만의 문제:** QMIX는 5M으로 연장해도 논문보다 높은 평균 성능이다.
  LEHCA 완전 재현은 여전히 같은 5M/5seed로 확인되지 않았다.

## 8. 가장 효율적인 다음 검증

1. **학습 없이 평가 문제부터 분리:** 같은 LEHCA 체크포인트와 평가 seed에서
   frozen training guidance / fresh scheduled guidance / mask-off를 비교한다.
   fresh 평가는 별도 Commander·cache·시간축으로 수행하고 훈련 상태를 오염시키지 않는다.
   이것으로 stale mask 손실을 학습 실패와 분리한다.
2. **모순 검증을 접지 단계에 추가:** 동일 agent/action에 대한 forbid/prefer 충돌과
   모든 공격 금지를 검출한다. 현재 컴파일된 제약과 원 전략이 얼마나 자주 어긋나는지
   기록한다. 안전한 후퇴 등 의도된 제약까지 일괄 제거하는 식의 수정을 피한다.
3. **확인된 논문 설정 차이를 검증:** ε50k/300k × QMIX/고정 LEHCA의 2×2 비교.
   기존 ε50k 팔을 재사용하고 300k 두 팔을 추가한다. 새 LEHCA 수정이 들어가면
   ε50k 대응 팔도 같은 코드로 다시 맞춘다. 재현 확인은 5seed·논문 예산으로 수행한다.
4. **원 환경 config 확보:** reward_sparse, scale, SC2/SMAC/map 버전, target-update
   단위, 원 prompt/grounding 및 F/β/λ를 확인한다. 논문 설명만으로 빈칸을 추정한
   구현은 재구현 baseline으로 명시한다.
5. **보상 원인 분리:** 고정 damage/kill → protect/retreat 추가 대조로 회피형 패널티의
   영향을 확인한다. 검증 전에는 F/RSVP scheduler 탐색을 재현 문제의 해결책으로 삼지 않는다.

기존 강한 QMIX를 약화시키거나 삭제해서 논문 순위를 맞추는 것이 목적이 아니다.
공개 설정에 충실한 재현과 동일 조건에서의 새로운 비교를 각각 보고해야 한다.

## 9. 논문 정합성 재감사와 코드 수정 상태

사용자 지적대로 이전 프롬프트의 “환경이 damage/kill/win을 이미 보상한다”는 문장은
논문의 정보 경계와 **직접 충돌했다.** 논문은 Commander가 관측 기반 $d_t$와 prompt만
받고 true reward function, privileged simulator internals, training logs를 받지 않는다고
명시한다. shaping predicate의 실행 의미를 알려 주는 것은 Commander가 생성할 보조
신호의 출력 schema이므로 true environment reward 공개와 구분된다.

재감사에서 확인하고 수정한 항목:

| 항목 | 수정 전 | 현재 기본값/방어 |
|---|---|---|
| 환경 보상 정보 | SMAC·GRF prompt가 기존 환경 보상을 직접 서술 | 해당 문장 제거; task objective와 action schema만 제공 |
| 학습 통계 | `t_env`, rolling win rate를 summary에 포함 | 기본 차단; `commander_include_training_stats=False` |
| 관측 경계 | action token의 표적 선택이 full snapshot의 비가시 적도 사용 | visible enemy에만 grounding하고 env availability로 최종 교집합 |
| 고정 갱신 | 같은 coarse cache key면 예정 호출을 건너뜀 | LEHCA 기본 `llm_cache=False`; refresh마다 실제 호출 |
| 평가 지침 | 마지막 train guidance를 모든 test episode에 재사용 | 별도 Commander·cache·test 시간축으로 fresh refresh; train 상태 불변 |
| 탐색 설정 | ε anneal 50k | 공식 보충자료의 300k로 QMIX/LEHCA 동시 변경 |
| 모순 제약 | 광범위 hard forbid와 prefer 충돌 허용 | 위험 token 제거, forbid/prefer 충돌 제거, 전 공격/이동 범주 소거 방어 |
| 진단 가능성 | soft tilt 변화만 기록 | hard-mask argmax 변화와 action-category 제거율 추가 |

오프라인 회귀 테스트는 비가시 표적 차단, 보상 문구 제거, 제약 sanitizer/compiler,
평가 갱신 격리 및 training-stat 미전달을 검사한다. 과거 실제 `attack_all` 금지 출력은
현재 compiler에서 공격 범주를 전부 제거하지 않았고, 32×70 test-step probe는 별도
Commander를 3회 갱신하면서 train guidance를 보존했다.

다만 다음은 논문 공개물만으로 확정할 수 없어 임의로 맞추지 않았다.

- 정확한 환경 reward config와 SC2/SMAC/map 버전
- $F_{update}$, $\beta$, $\lambda$의 실제 수치 및 감쇠 곡선
- 원 prompt 전문과 grounding parser/규칙
- target update의 문서상 `steps`와 PyMARL식 episode 단위 중 실제 구현

따라서 현재 코드는 확인된 정보 누출과 실행 결함을 제거한 **paper-aligned
reimplementation**이지, 공개되지 않은 원 코드와 bitwise-equivalent한 reproduction은
아니다. 수정 전 결과와 수정 후 결과는 같은 실험군으로 합치지 않는다.

## 재현 산출물

- 스크립트: `analysis/diagnose_lehca_reproduction.py`
- 수치·프로브·소스 해시: `results/diagnostics/reproduction_20260910/report.json`
- 실행: `/home1/paul6598/miniconda3/envs/aamas/bin/python analysis/diagnose_lehca_reproduction.py`
- 회귀 테스트: `analysis/test_lehca_paper_alignment.py`
- 수정 전 결과 파일은 보존하며, 보고서 source hash/probe는 현재 코드 기준으로 다시 생성한다.

지표 주의: 기존 `analysis/agg_horizon.py`는 300k 이전 마지막 점까지만 적분한 뒤
300k로 나누고, `auc_early.py`도 early 경계 보간과 완주 검사가 없다. 기존 문서의
전 구간 평균 AUC를 논문의 첫 20% AUC와 직접 비교하지 않는다. 경계 보간 오차만으로
큰 역전을 설명할 수는 없지만, 비교 구간 혼동은 해석을 크게 바꾼다.
