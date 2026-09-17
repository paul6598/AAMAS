# RSVP 검증 지표와 실제 측정 범위

2026-09-17 인계 시점의 구현과 완료 결과를 기준으로 정리했다. 이전의 여러 검증 계획을
대체한다. 연구는 보류됐으며 이 문서는 추가 실험 예약을 의미하지 않는다.
최종 수치는 [종합 보고서](research-summary.md), 명령은 [코드 안내](code-guide.md)에 있다.

## 구분한 질문

| 질문 | 구현·측정한 지표 | 해석 범위 |
|---|---|---|
| 잔여 신호를 예측하는가? | 학습 전 prediction의 MAE/RMSE/R², baseline skill | 실제 후속 trajectory의 예측력 |
| 저하 뒤 유용한 지침으로 교체되는가? | staleness drop, replacement gain | 같은 trajectory에서 보조 신호가 복원되는지 |
| 지금 교체하는 판단이 유익한가? | 무작위 immediate/hold MRT, trigger와 control의 효과 차이 | 5개 후속 episode의 근접 효과 |
| 전체 학습과 호출 예산에 이점이 있는가? | 승률 AUC, final100k, train 호출 수 | 고정 주기·timer 대비 전체 성능 |

어느 한 지표가 양수여도 나머지 질문의 답이 자동으로 양수가 되지는 않는다.

## 1. 순차 예측 검증

완료 episode의 실제 predicate 벡터 `f[j,t]`에서 정답을 만든다.

```text
Y[j,T] = 0
Y[j,t] = f[j,t] + gamma_F * Y[j,t+1]
```

각 step에서 **해당 episode가 critic 학습 버퍼에 들어가기 전**의 prediction과 과거 target
평균 baseline을 저장한다. 종료 후 할인 접미합 target을 연결하고 나서 critic을 학습한다.
따라서 학습 loss를 그대로 예측 정확도로 부르는 방식과 구분된다. 종료·timeout 뒤의 값을
0으로 두므로 무한 지평 정답이 아니라 기록된 episode의 suffix 정답이다.

`baseline_skill = 1 - SSE(prediction) / SSE(과거 평균)`이며 양수는 과거 평균보다 작은
제곱오차를 뜻한다. 분모가 거의0이면 정의하지 않는다. R²의 분모는 평가 target의 분산이다.
전체 head, target variance gate를 통과한 head, 현재 지침의 가중합, 발급 시 선택한
head의 가중합을 따로 집계한다. 희소한 비활성 head가 많은 환경에서는 전체 MAE만으로
판단하지 않는다. `trusted()`는 warmup/target variance 조건이며 정확도 보증이 아니다.

최종 dense E1의 200k–1M 선택 head skill은 seed0=.4462, seed1=.5135였다.
이 결과는 실제 정책 아래 미래 신호를 예측한다는 증거다. 현재 guidance를 계속 유지했을
경우의 반사실이나 “교체하면 좋아진다”는 예측을 검증한 것은 아니다.

구현: `algorithm/rsvp/validation.py`, `analysis/summarize_rsvp_validation.py`.
집계: [prediction-final-20260917.json](archive/prediction-final-20260917.json).

## 2. 같은 trajectory의 교체 진단

shaping-only에서는 episode 도중 guidance 교체가 policy 행동을 직접 바꾸지 않는다.
같은 실제 predicate suffix에 이전·새 guidance의 weight를 모두 적용할 수 있어 추가
environment rollout 없이 진단한다. action masking이 켜진 log에는 적용하지 않는다.

성공한 early refresh 시점 t의 앞뒤 K=5 step이 **모두 같은 episode에 존재하는 사건**만 쓴다.
discount `d[k] = gamma_F^k / sum(gamma_F^i)`와 실제 shaping의 [-3,3] clipping을 사용한다.

```text
Vpost(w) = sum_k d[k] * clip(dot(w, f[t+k]), -3, 3)
Vpre(w_old) = sum_k d[k] * clip(dot(w_old, f[t-1-k]), -3, 3)
staleness_drop = Vpre(w_old) - Vpost(w_old)
replacement_gain = Vpost(w_new) - Vpost(w_old)
```

별도로 weight 합으로 나눈 normalized gain, 양의 gain 비율, 동일 weight 비율,
`drop>0 and gain>0` 비율, CUSUM score와의 순위상관을 집계한다.
같은 episode의 사건은 독립으로 취급하지 않고 episode cluster bootstrap으로 CI를 만든다.
이 CI는 한 seed 내부의 불확실성이며 seed 간 재현성 구간이 아니다.

최종 seed0/1의 drop은 .0950/.0385, gain은 .0066/-.0182였다.
새 weight가 같은 trajectory에서 만든 신호의 크기를 비교한 값이다. 음의 보호 penalty의
교육적 가치나 실행하지 않은 교정 행동은 평가하지 못하며, 더 큰 보조 reward가 더 높은
외부 return을 뜻하지 않는다. 성공한 refresh와 충분한 앞뒤 window만 골라지는 선택도 있다.

구현: `analysis/summarize_refresh_replacement.py`.
집계: [refresh-replacement-final-20260917.json](archive/refresh-replacement-final-20260917.json).

## 3. MRT: 즉시 적용과 지연 적용의 무작위 비교

MRT는 micro-randomized trial을 뜻한다. shaping-only RSVP에서 200k 이후 실제 trigger와
일부 eligible non-trigger에 대해 다음 과정을 수행했다.

1. candidate 생성 전에 독립 RNG로 apply/hold를 확률.5로 배정한다.
2. 두 팔 모두 LLM candidate를 요청한다. apply는 성공 candidate를 즉시 적용하고,
   hold는 이전 guidance를 유지한다. 실패도 배정된 팔에 남긴다.
3. decision이 포함된 episode는 outcome에서 제외한다. 다음5 training episode 동안
   early와 Fmax refresh를 모두 잠근다.
4. 이5 episode의 **환경 raw return 평균**과 win rate를 기록한다.
5. window 종료 후 두 팔 모두 공통 refresh를 요청한다. 실패하면 성공할 때까지 종료 갱신을 재시도한다.

block은 겹치지 않는다. assignment와 control probe RNG는 policy/critic RNG와 분리했다.
seed마다 trigger50, control50 block까지 측정했고 두 seed 모두 candidate 실패는0이었다.
MRT가 정상 scheduler에 개입하므로 그 run의 학습곡선을 비개입 RSVP seed와 합산하지 않는다.

**실제 control 추출:** episode 시작 시 확률.1로 probe를 뽑고, 해당 episode에서 최초로
`non-trigger, ready/armed, age >= min_interval`을 만족하는 시점에 control block을 연다.
이전 설계에서 제안한 학습 phase·생존 수·age 분포의 matching은 구현하지 않았다.
control도 trigger도 cap과 이전 block의 개입 이력에 영향을 받는다.

종합 보고서의 효과는 각 kind에서 `mean(return_apply)-mean(return_hold)`다.
`timing_selectivity_gain = trigger 효과 - control 효과`도 표시했다.
무작위 배정은 각 eligible population에서의 apply/hold 비교를 뒷받침한다. 다만 두 population이
완전히 matched되지 않았으므로 timing gain을 동일 상태 조건에서 CUSUM만의 가치로 읽지 않는다.

원 집계에는 Horvitz–Thompson 형태의 `return_itt_ipw`도 있다.

```text
IPW = mean(A * outcome / p - (1-A) * outcome / (1-p))
```

실현된 팔별 block 수가 다르면 이 비정규화 IPW와 팔 평균 차이는 크게 다를 수 있다.
두 추정량을 같은 수치로 혼용하지 않는다. 보고서 CI는 seed 내 kind/arm별 block을
50,000회 bootstrap한 팔 평균 차이의95% 구간이다. 시간에 따른 학습 추세를 보정한 CI나
seed 수준의 방법 불확실성 구간은 아니다.

trigger 효과는 seed0=-.3189, seed1=-.0355이며 두 CI가0을 포함했다.
근접 이득을 확인하지 못했지만 효과0의 등가성을 입증하지도 않았다. 특히 off-policy
replay에서 새 shaping transition이 5 episode 안에 충분히 학습되는지 보장되지 않아
장기 학습 이점에 대한 음의 결론으로 확대할 수 없다.

구현: `algorithm/rsvp/validation.py:RefreshTrial`, `algorithm/rsvp/runner.py`.
집계: [mrt-effect-summary-20260914.json](archive/mrt-effect-summary-20260914.json).
기존 summary의 seed1은 작성 시 gzip이 열려 있었지만, 종료 후 다시 읽어 동일한100개
완료 block과 효과를 확인했다. CLI의 `--trials-only`는 최종 block만 빠르게 집계한다.

## 4. 전체 학습·비용 비교

승률 AUC는 명목 지평으로 정규화한 사다리꼴 적분이며 y(0)=0을 사용했다.
완료 run의 마지막 평가 뒤에는 명목 지평까지 마지막 값을 유지한 근사다.
별도로 tail 외삽 없는 observed AUC를 export한다. final100k는 마지막100k 구간의
평가점 산술평균이고 호출은 마지막 training 누적 counter다. wall time, token, 평가 요청,
실패 재시도를 모두 포함한 총 LLM 비용과는 다르다.

RSVP·fixed200·gate-timer20·fixed136의 주요24run은 실제 config와 Slurm 완료를 확인해
[final-results.json](final-results.json)에 고정했다. F136은 같은 개발 seed의 RSVP 평균
호출률에서 선택한 후속 대조다. 두 seed 모두 양의 성능 차이가 있었지만 독립 확인 배치나
map 일반화를 측정하지 않았다. 이 전체 효과와 MRT의 짧은 근접 효과를 구분한다.

## 계획만 했거나 현재 측정하지 않은 것

환경을 복제해 긴 horizon을 비교하는 counterfactual refresh value, decision regret,
AUUC와 BMLG, phase-matched MRT와 phase 보정 회귀는 완료 측정치가 없다.
과거 soft-guidance의 같은 episode 즉시 outcome은 현재5-episode MRT와 다른 설계다.
계획 문구를 실제 수행 결과로 인용하지 않는다. 당시 논의는 [실험 원장](archive/experiments-log.md)에 남겼다.
