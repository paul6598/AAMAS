# LEHCA 재현과 RSVP: 연구 종합 및 인계 보고서

정리 기준: **2026-09-17**. 본 연구는 보류됐고 새 실험은 예약하지 않았다.
마지막 MRT seed1과 fixed F136 두 seed까지 모두 Slurm COMPLETED/exit0으로 종료됐다.
이 문서는 날짜별 미팅·계획·상태 보고를 대체한다. 당시 판단의 전체 이력은
[실험 원장](archive/experiments-log.md)에 보존했다.

## 1. 최종 요약

연구 질문은 **LLM 지침을 언제 다시 요청해야 하는가**였다. LEHCA의 고정 갱신 주기를
잔여 셰이핑 가치 예측과 누적 저하 검출로 대체하는 RSVP를 구현했다.

최종 증거는 다음처럼 분리된다.

- **LEHCA 기반선:** corrected dense 2s3z 300k에서는 4개 seed 중1개만 QMIX 평가 평균을
  넘었다. win-only sparse에서는 QMIX가 얻지 못한 일시적 승리를 만들었지만 유지되지 않았다.
- **예측력:** 최종 dense E1의 200k--1M 순차 평가에서 selected-head baseline skill은
  seed0=.4462, seed1=.5135였다. 미래 predicate 누적량은 과거 평균보다 잘 예측했다.
- **교체 진단:** 5-step replacement gain은 seed0=.0066, seed1=-.0182다. seed0의 CI는
  0을 포함하고 seed1은 음수이며 양의 교체 비율은34.2%/32.1%였다.
- **무작위 근접 효과:** MRT trigger에서 즉시 새 지침을 적용한 효과는 -.3189/-.0355이고
  두 CI가0을 포함한다. 5개 후속 episode에서 양의 갱신 이득을 확인하지 못했다.
- **전체 성능·호출:** 최종 fixed F136 비용 대조에서는 RSVP가 두 seed 모두 더 높았다.
  평균 AUC=.1765 vs .0291, final100k=.5438 vs .0516, 호출7,362.5 vs7,314회다.
  fixed F200 및 gate-timer와는 seed별 방향이 엇갈렸다.

따라서 **현재 자료에는 동일 호출 예산의 periodic 대조 대비 제한적인 양의 성능 신호가
있지만, RSVP trigger의 근접 인과효과나 일반적인 성능 우위는 입증되지 않았다.**
F136 완료 전의 “모든 fixed보다 이득이 없다”는 식의 해석을 최종 결론으로 사용하지 않는다.
두 개발 seed의 양의 결과를 확인 seed나 다른 환경의 재현으로 확대하지도 않는다.

## 2. 구현한 시스템

### LEHCA 재구현

공유 인프라는 PyMARL의 QMIX다. 공개된 논문·보충자료를 바탕으로 다음 경로를 구현했다.
공개되지 않은 prompt, grounding 세부 의미, beta/lambda 설정에는 로컬 구현 선택이 있다.

```text
관측 정보 → 텍스트 요약 → LLM Commander → JSON 지침 정제
                                     ├─ predicate 가중합 → 보조 reward → QMIX 학습
                                     └─ hard mask / soft preference → 행동 선택
```

공통 QMIX 값은 Adam, lr=.001, batch128, replay5000 episodes, gamma=.99다.
Commander는 gpt-oss-20b를 사용했고 재현 배치의 paper prompt는 temp=.2/cache off다.
현재 corrected 기본값은 beta=.1와 semantic dedup=True이며, beta=.1은 원 논문의 확인된
수치가 아니라 진단에 따른 구현 선택이다. full LEHCA는 train/test action guidance와
collection-time shaping을 사용한다. shaping-only RSVP와 그대로 scheduler 단독 비교하지 않는다.

`verify_lehca_backbone.py`에서 무지침 runner의 관측·상태·행동·보상·종료를 비교했고,
동일 배치8회 업데이트의 loss와 online/target parameter 차이는0이었다.
learner-time 합성과 명시적 reward 합성, replay 원본 불변도 확인했다.
이는 검사한 경로의 구현 일치이며 모든 장기 학습의 결정론적 일치를 보장하지 않는다.

근거: [backbone 감사](archive/lehca-backbone-validation-20260908.md),
[논문과 grounding 대응](archive/lehca-grounding-audit-20260907.md).

### RSVP 구체화

각 predicate j의 실제 전이 신호 f[j,t]에 대해 완료 episode의 MC target을 만든다.

`Y[j,t] = f[j,t] + gamma_F * Y[j,t+1]`, 종료 뒤 `Y[j,T]=0`.

TD bootstrap이 아닌 실제 trajectory의 할인 접미합이다. MLP는 상태 특징을 입력받아
predicate별 값을 함께 출력한다. hidden128 ReLU 두 층, 최대60k transition buffer,
head별 평균·표준편차 정규화 MSE를 사용한다. gamma_F=.8은 가까운 약5 step에 큰
비중을 두며 장기 전략의 수명 자체를 직접 예측하지 않는다.

현재 guidance의 선택 head를 가중 합성한다. 발급 특징 x_ref와 현재 특징 x_t를 동일한
현재 critic으로 평가해 0--1 가치 비율 v_t를 만들고,
`S_t=max(0,S_(t-1)+k-v_t)`가 h를 넘으면 최소 간격을 거쳐 early refresh한다.
Fmax 도달 시에는 강제로 갱신한다. 발급 gate는 warmup, head target variance,
선택 weight 비율, 작은 발급 기준값을 검사한다. `trusted=True`는 정확도 판정이 아니다.

learner-time shaping은 저장된 scalar F_t에 학습 시점의 lambda를 적용한다. 이전 guidance로
수집한 transition을 최신 guidance로 다시 계산하는 기능은 없다. 따라서 lambda stale 문제와
여러 guidance 보상이 replay에 섞이는 문제를 구분해야 한다.

구현 위치와 실행 명령은 [코드 안내](code-guide.md)에 있다.

## 3. LEHCA 재현과 RSVP 검증을 위해 시도한 것

아래는 서로 다른 버전·환경·지평의 탐색 이력이다. 표의 관측을 동일 실험군으로 합산하지 않는다.

| 질문·시도 | 주요 관측 | 남은 해석 |
|---|---|---|
| 초기 dense full/shape/mask 분해 | 초기 이득 후보와 masking 손실, 큰 seed 변동 | 초기 유리한 seed로 재현 성공을 선언할 수 없었음 |
| F100/F200/RSVP, 3s5z·5m6m·2s3z | F100보다 좋은 사례가 있으나 F200 추가 후 우위 약화 | 저빈도 갱신과 적응 시점의 기여를 구분해야 했음 |
| cache off·temp.2 실제 요청 | 5m6m seed0 개선, seed1 붕괴 | 캐시 없는 실제 요청에서도 일관 우위 미확인 |
| none/rule/shuffle/aligned | none은 QMIX와 일치, LLM이 shuffle보다 두 seed에서 낮음 | 현 접지에서 상태 맞춤 LLM 내용의 추가 가치 미확인 |
| actual LLM F25·QMIX 장기5M | F25가 보편적 개선을 주지 않았고 QMIX seed 차이가 큼 | 더 자주 호출하거나 오래 학습하는 것이 해결책은 아니었음 |
| 논문 epsilon50k→300k 정렬 | 5m6m 여러 팔이 낮아짐 | 탐색 일정 하나를 맞춰도 전체 재현은 해결되지 않음 |
| lambda·beta·reward flag·prompt·dedup 변경 | 결과 변화와 붕괴가 반복됨 | 여러 동시 변경을 개별 원인의 인과효과로 읽지 않음 |
| finite-lambda cutoff | 호출은 중단됐지만 자동 성능 회복 없음 | cutoff 전 감쇠도 달라 제거 시점 단독효과가 아님 |
| native sparse terminal ±1 | full만 일부 초기 승리, 다른 팔은 후반 실패 | sparse 조건만으로 안정적인 재현이 생기지 않음 |
| 의미 중복 검사와 sanitizer 교정 | 지침 약55--60%에 중복, weight 약1.3배 팽창 | legacy/corrected를 분리해 기록 |
| beta=.1 + dedup corrected dense gate | 300k n=4에서1개 seed만 QMIX 초과 | 안정적인 LEHCA 우위 미확인 |
| train-only soft guidance | dense seed0은 fixed, seed1은 RSVP가 좋음 | seed 평균 하나로 강건성을 주장할 수 없음 |
| win-only sparse full n=4 | QMIX는0, LEHCA는 transient wins 후 대부분0 | 탐색 신호는 생겼지만 학습이 유지되지 않음 |
| win-only soft RSVP/F200/F50 | 세 팔 모두 두 seed final100k0 | soft action channel과 호출 빈도 증가가 안정화하지 못함 |
| 4.8M-step trace 감사 | armed 발급 대부분은 짧은 early, 나머지는 F200 | 같은 issuance gate의 timer 대조를 추가 |
| 학습 전 prediction/종료 target 계측 | 선택 head skill 양수 | 예측 loss와 실제 별도 episode 예측력을 구분할 수 있게 됨 |
| 같은 trajectory의 old/new value | old value 감소는 보이나 new value 회복은 일관되지 않음 | 저하 예측과 유익한 교체 사이에 간극 |
| immediate-vs-delay MRT | trigger 효과 음수 방향, CI0 포함 | 5-episode 근접 이득 미확인 |
| 호출 예산 일치 fixed F136 | 두 seed 모두 RSVP가 더 높은 성능 | 제한적인 H3 양의 신호; CUSUM 고유의 인과 기여는 미확정 |

GRF에서는 bot trajectory shadow/replay·LOO로 gamma=.8의 예측 R²=.662,
gamma=.97에서 약-.16이라는 초기 기록이 있었다. SMAC 온라인 효용의 증거로 대체하지 않는다.
PettingZoo SISL Pursuit도 구현했다. 초기 가짜 catch semantic 오류는 수정하고 v2를 분리했다.
이 저장소의 Pursuit는 MPE가 아니며 MPE 결과는 없다.
GRF는 초기 탐색 결과만 보고서에 남겼고 인계 공유본의 구현에서는 제외했다.
해당 소스와 당시 분석기는 Git `f138bf0` 및 서버 정리 백업에서 확인할 수 있다.

## 4. 최종 실험 결과

주요24개 본실험은 [final-results.json](final-results.json)에 실제 config 요약, 평가 곡선,
config/info SHA256과 Slurm 완료 증거를 함께 제공한다. 제출 원장은
[campaign-manifests.json](archive/campaign-manifests.json)에 있다.

아래 AUC는 y(0)=0의 정규화 사다리꼴 적분이며, 명목 예산과 마지막 평가 사이에는 마지막
평가값을 유지한 근사다. JSON에는 tail 외삽 없는 observed AUC도 별도로 제공한다.
final100k는 명목 예산 마지막100k 안의 기록 평가 산술평균이다. train 호출은 마지막
주기 로그의 누적 counter이며 평가 요청·재시도·token·미기록 tail까지 포함한 총비용은 아니다.

### Dense E1와 호출 예산 대조: 2s3z, 1M

공통 조건은 shaping-only/test mask off, paper prompt/temp.2/cache off, dedup,
epsilon300k, learner-time lambda, floor 절대400k, 평가10k/32 episodes다.

| 방법 | seed0 AUC / final100k | seed1 AUC / final100k | 평균 AUC / final100k | 평균 train 호출 |
|---|---:|---:|---:|---:|
| RSVP Fmax200 | .2038 / .6375 | .1492 / .4500 | .1765 / .5438 | 7,362.5 |
| fixed F200 | .0646 / .0250 | .2804 / .7063 | .1725 / .3656 | 4,967.5 |
| gate-timer20 | .1618 / .5063 | .2299 / .7188 | .1958 / .6125 | 9,790 |
| fixed F136 | .0218 / .0813 | .0364 / .0219 | .0291 / .0516 | 7,314 |

RSVP의 평균 호출률에서 `1M/7362.5=135.82`로 F136을 선택해 동결했다. 평균 호출
차이는 약.66%이며 성능은 RSVP가 두 seed 모두 더 높다. 다만 동일 개발 seed의 호출률에서
선택한 후속 대조, n=2, 단일 map 결과다. 독립 확인 배치와 일반화를 검증하지 않았다.
F200 seed1은 RSVP보다 좋고 gate-timer 평균 성능도 더 높으나 호출 비용이 더 크다.
따라서 RSVP가 전체 trade-off의 유일한 우수점이라고 주장할 수 없다.

### Win-only sparse 2s3z: 300k

reward_sparse=True, reward_scale=False, sparse_win_only=True: 승리+1, 패배·timeout0.
full LEHCA는 F50, train/test masking, collection-time shaping, beta=.1/dedup이다.

| seed | QMIX AUC / final100k | full LEHCA AUC / final100k | LEHCA peak |
|---|---:|---:|---:|
| 0 | 0 / 0 | .0136 / .0031 | .1250 |
| 1 | 0 / 0 | .0094 / 0 | .1563 |
| 2 | 0 / 0 | .0345 / .0125 | .2188 |
| 3 | 0 / 0 | .0056 / .0031 | .0625 |

full이 작은 탐색 신호를 만든 것은 확인됐지만 학습이 유지되지 않았다. 300k는 epsilon
annealing 끝과 같으므로 이후 장기 성능을 측정한 결과는 아니다. native sparse와 비교하면
beta/dedup도 달라 패배 penalty 제거만의 인과 대조가 아니다.

Training-only soft guidance는 beta=.1, test mask off, learner-time lambda를 사용한다.

| scheduler | seed0/1 AUC | seed0/1 final100k | seed0/1 train 호출 |
|---|---:|---:|---:|
| RSVP Fmax200 | .0240 / .0010 | 0 / 0 | 1,972 / 1,943 |
| fixed F200 | 0 / .0042 | 0 / 0 | 1,457 / 1,454 |
| fixed F50 | .0021 / .0031 | 0 / 0 | 5,820 / 5,825 |

soft masking과 호출 빈도 증가는 이 조건의 학습 유지 문제를 해결하지 못했다.

### MRT: shaping-only dense 2s3z

MRT는 실제 refresh 후보에서 즉시 적용과 5 training episode 지연을1:1로 무작위 배정한다.
두 팔 모두 candidate를 생성하고, decision episode를 제외한 다음5 episode의 raw return을
측정한다. window 중에는 early/Fmax 갱신을 모두 막고, 종료 후 공통 refresh를 수행한다.
seed0/1 각각100 block(trigger50/control50)을 완료했으며 candidate 실패는0이었다.

| seed | trigger apply−hold [within-run bootstrap95% CI] | control apply−hold | timing gain [CI] |
|---|---:|---:|---:|
| 0 | -.3189 [-.7329,.0888] | +.3066 | -.6255 [-1.8200,.5622] |
| 1 | -.0355 [-.3639,.2960] | +.0543 | -.0898 [-.9558,.7828] |

효과는 양의 방향으로 재현되지 않았고 CI는0을 포함한다. 효과0의 등가성이나 유해성을
증명한 것은 아니다. 구현의 control은 armed와 age 조건을 만족하는 non-trigger의
episode 단위 확률 probe다. 설계 메모에서 구상한 학습 phase·생존 수 등의 matching과
phase 보정 분석은 실시하지 않았다. 단순 timing gain을 완전히 matched된 score 선택성의
증명으로 사용하지 않는다.

MRT는5 episode의 근접 효과다. off-policy replay에서는 새로운 shaping transition이
바로 학습 batch에 충분히 들어간다고 보장할 수 없어, 짧은 window가 장기 학습 효과를
포착하지 못할 수 있다. MRT run의 학습곡선은 비개입 RSVP seed와 합산하지 않는다.
상세 정의는 [검증 방법](validation.md)과 [효과 집계](archive/mrt-effect-summary-20260914.json)에 있다.

## 5. 실제 LLM 출력과 grounding 분석

[저장 출력16개](llm-output/llm-guidance-examples-20260912.md)와 실입력·원응답·sanitizer·
availability를 포함한 [pipeline8호출](llm-output/llm-pipeline-smoke-20260912.md)을 직접 읽었다.
전자는 sanitizer 후의 선정 사례로 입력 전문이 없고, 후자는 한 trajectory의4상태×2prompt
소표본이다. 모집단 오류율이나 모든 상태의 품질로 일반화하지 않는다.

- 단독 생존에 두 명 이상을 필요로 하는 focus_fire를 발급하는 사례가 있었다. 과거 trace에서
  early 직후 단독 생존 guidance의 약60--73%에 포함됐다.
- 아군 Zealot0/3 alive라는 실제 입력에서 Zealot 보호를 발급하는 상태 불일치가 있었다.
- 사거리 밖에서 공격 선호만 출력한 사례는 해당 시점의 actionable preference가0이었다.
- “공격 후 cooldown 중 kite”의 순서·조건이 schema에서 표현되지 않고 무조건 방향 이동으로 축약됐다.
- protect_type/ally_survive의 피해 penalty가 필요한 tanking과 충돌할 가능성이 있었다.
- paper_v2는 일부 상태 불일치를 줄였지만 raw4응답 중3건에 weight 범위 위반이 있었고,
  학습 성능은 검증하지 않았다. 최종 성능 배치도 paper_v2 결과가 아니다.

4.8M-step 과거 trace에서 armed 점유는 약4.2--8.8%였고 armed 발급 후 약98--99%가 early
갱신에 도달했다. armed 구간의 갱신 간격 중앙값은15--21step, unarmed는 전부200이었다.
발급 시 gate가 닫히면 다음 발급까지 재감시하지 않는다. 새 guidance 계수는 약98%에서
변하지만, 내용 변화가 좋은 replacement나 외부 성능을 보장하지 않는다.

이 관측은 Commander/grounding이 잔여 가치 저하 후 유효한 신호를 복원하지 못한다는
가능성과 일관된다. 충분히 좋은 guidance에서도 RSVP가 무효한지 검증한 oracle 실험은 아니다.
단일 원인을 확정하지 않고 양의 F136 성능 신호와 음의 근접 진단을 함께 보존한다.

근거: [직접 검토](llm-output/llm-guidance-review-20260912.md),
[trace 감사](archive/rsvp-mechanism-audit-20260912.md).

## 6. 과거 결과의 해석 범위

아래는 이전 집계에서 옮긴 과거 결과로 최종 E1과 설정·cache·탐색·정의가 다르다.
전체를 다시 집계한 최종 JSON의24run과 구분한다. 평균은 평가점 산술평균이며 AUC와 혼동하지 않는다.

| 실험 | 과거 관측 |
|---|---|
| 3s_vs_5z lam40 n=4 | RSVP 평균.409/final.920, F100 .282/.713, F200 .432/.902 |
| 5m_vs_6m lam40 n=4 | RSVP .198/.576, F200 .255/.564; seed별 역전 |
| 2s3z lam40 n=3 | RSVP .805/.939, F100 .763/.949 |
| Pursuit semantic v2 n=4 | final return RSVP15.29, F20013.86, F10012.05; 큰 분산 |
| cache-off5m6m n=2 | RSVP final.294, F200약.285--.298; seed0 개선·seed1 실패 |
| dense2s3z legacy1M | fixedF50이 RSVP보다 두 seed 모두 높음; QMIX4seed 중3seed final93% 이상 |
| corrected full dense2s3z300k n=4 | LEHCA 평가 평균.1115/.0927/.0885/.1792, QMIX .0875/.1927/.0948/.2188 |
| native sparse±1,2s3z1M | fullLEHCA final0/.1281, QMIX/RSVP/aligned는0; fixed는 미완료 취소 |
| train-only soft dense2s3z1M | seed0 fixed평균.2503/final.6938 vs RSVP.0694/.1656; seed1 fixed.1266/.2094 vs RSVP.2409/.7750 |

Pursuit226의 빈 Sacred info는 당시 local W&B history에서 복원했다. final 창이 다른
과거 수치를 최신 final100k와 섞지 않는다. 상세 이력은 실험 원장과 고정 snapshot을 참조한다.

## 7. 수정한 문제와 남은 한계

의미상 동일 predicate 중복, 강한 soft 편향, 평가 Commander의 train 상태 오염,
episode를 가로지르는 armed 구간의 h 적응 회계 누락, W&B seed/group, port 충돌, GPU OOM,
Slurm spool 작업dir 문제를 수정·분리했다. 기술 실패는 성능0으로 집계하지 않는다.
lambda_floor_frac=.4는1M에서400k, 300k에서120k이므로, 이전 corrected300k와 legacy1M의
차이를 dedup 단독 인과효과로 읽은 설명은 철회했다. 최종 E1은 floor 절대400k로 맞췄다.

남는 한계:

- LEHCA 원prompt/grounding/계수 미공개와 로컬 선택 때문에 동일 구현의 재현 실패로 단정할 수 없다.
- 양의 예측 skill은 실제 정책의 후속 trajectory 예측이며 guidance를 유지한 반사실이 아니다.
- replacement gain은 같은 trajectory의 clip된 보조 신호 차이로 penalty의 유용성이나 미실행 교정 행동을 포착하지 못한다.
- MRT CI는 seed 내 팔별 block resampling이며 seed 간 방법 불확실성의 구간이 아니다.
- F136은 같은 개발 seed에서 호출률을 선택한 비교로 n=2의 검증 결과에 한정된다.
- shaping/action channel, test 개입, 합성 시점이 다른 버전을 scheduler 단독효과로 해석하지 않는다.

## 8. 공유 범위와 종료 시점

공유 코드에는 LEHCA/RSVP, predicate/grounding, 실제 prompt, 검증 logger, 일반 실행기,
핵심 집계기3개와 별도 audits/·tests/를 남겼다. LLM 원문·원 API 응답과 검토는5파일을 유지했다.
끝난 예약·watcher·일회성 집계와 중복 MD는 로컬 backup으로 옮겼다.
LaTeX는 번호별 section의 미완성 초안으로 보존하며 최신 결론의 출처는 본 보고서다.

rawdata, W&B, checkpoint, 당시 source snapshot은 Git 밖에서 서버에 보존했다.
로컬 백업과 이동 원장은 `results/diagnostics/repository-handoff-20260917/`에 있다.
GRF와 보조 분석기 추가 축소의 백업은 `results/diagnostics/repository-trim-20260917/`에 있다.
이번 정리에서 새 성능 실험을 시작하지 않았다. 환경·실행·검증 방법은 [코드 안내](code-guide.md)에 있다.
