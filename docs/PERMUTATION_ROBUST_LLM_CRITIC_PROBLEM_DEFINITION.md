# 순열 강건 LLM Critic: 문제 정의와 연구 접근

> 상태: 초기 연구 방향 문서. 이 문서는 특정 architecture를 확정하지 않는다.  
> 범위: 1차적으로 **agent 순열**만 다룬다. Environment/object 정보는 고정된 문맥으로 취급한다.

## 1. 출발점

협력적 MARL에서 centralized critic은 한 시점 또는 한 궤적의 여러 agent 상태·action·보상을 함께 보고 team value나 agent별 credit을 추정한다. LLM-MCA처럼 LLM을 critic으로 쓰면 이 정보를 대개 다음처럼 prompt의 순차 block으로 넣는다.

```text
agent A: observation/action/history ...
agent B: observation/action/history ...
agent C: observation/action/history ...
```

그러나 homogeneous/exchangeable agent에서는 위 block의 나열 순서는 물리적 의미가 없다. 같은 환경 transition을 `[A,B,C]` 대신 `[C,A,B]`로 직렬화해도 critic의 의미 있는 답은 바뀌지 않아야 한다.

실제 raw LLM-MCA audit에서는 이 조건이 깨졌다. delayed cooperation 및 multi-event probe에서 agent block 순서만 바꿨는데 inverse-map한 individual credit뿐 아니라 팀 credit 총량도 달라졌다. 이 현상은 단순한 formatting failure가 아니라, critic target과 입력 representation 사이의 불필요한 symmetry mismatch다.

## 2. 정확한 문제

Agent feature/trajectory set을 $A=(a_1,\ldots,a_N)$, agent 이외의 규칙·환경 state·transition context를 $\xi$라 하자. Agent ordering을 바꾸는 permutation matrix를 $P$라 하면, 원하는 critic은 다음을 만족해야 한다.

### Global/team output: permutation invariance

$$
V(PA,\xi)=V(A,\xi).
$$

Team value, global reward prediction, team-level event representation은 agent의 입력 배열 순서를 바꿔도 같아야 한다.

### Agent-axis output: permutation equivariance

$$
C(PA,\xi)=P C(A,\xi).
$$

Agent별 credit, agent별 value, event participant distribution은 같은 physical agent에 대응하는 값이 유지되되 output row가 input row와 함께 이동해야 한다. 이는 “모든 agent에게 같은 값을 내는 invariance”와 다르다. Agent를 구분하는 state, action, capability, role은 보존해야 하며, 제거할 대상은 **임의의 storage/serialization 순서**뿐이다.

## 3. 왜 기존 prompt 방식이 실패하는가

일반 decoder LLM은 token content만 읽지 않는다. 각 token은 absolute/relative position encoding을 받고 causal mask에 의해 앞 token과 뒤 token의 다른 context를 본다. 따라서 agent block 순서를 바꾸면 같은 physical agent의 정보도 다른 position과 attention neighborhood를 받는다.

이 문제는 prompt에 agent 이름을 지우거나 `agent_0`을 익명 tag로 바꾸는 것만으로 해결되지 않는다. Agent state를 순차 block으로 넣는 한, block의 선후관계가 LLM 내부 계산에 남는다.

반대로 position encoding을 모두 제거하고 complete attention만 허용하면 또 다른 문제가 생긴다. Agent trajectory block 내부의 `t=0` state/action과 `t=1` outcome이 같은 agent에 속한다는 결속까지 잃을 수 있다. Set-LLM이 SetPE만으로는 서로 다른 set element 내부 token의 교환을 구분하지 못한다고 지적한 이유가 이것이다.

따라서 본 문제는 단순히 “LLM의 positional encoding을 없애는 일”이 아니다. 다음 두 요구를 동시에 만족해야 한다.

1. **교환 가능한 agent block 간의 순서**는 없애야 한다.
2. **각 agent 내부의 시간·field·행동과 결과의 결속**, 그리고 agent 사이의 협력 관계는 보존해야 한다.

## 4. 연구 목표와 비목표

### 목표

1. 한 번의 LLM/critic forward로 agent permutation invariant/equivariant input-output contract를 만족하는 구조를 만든다.
2. 순서 강건성이 실제 finite-precision LLM 실행에서도 유지되는지 adversarial reserialization으로 측정한다.
3. LLM이 규칙과 trajectory에서 event/participant relation을 유용하게 추출하는지 검증한다.
4. event와 agent representation을 사용해 순열 동변적인 individual credit 또는 shaped learning signal을 만들고, 이를 policy learning에 결합한다.
5. raw sequential LLM critic 및 강한 non-LLM/set baseline 대비, random/adversarial agent reserialization에서도 더 안정적인 credit과 더 높은 또는 적어도 동등한 task return을 실증한다.

### 현재 비목표

- Object/environment 요소의 순열성까지 주된 이론적·실험적 claim으로 만들지 않는다. 이들은 고정 문맥 $\xi$에 포함한다.
- 모든 환경·모든 policy optimizer에서 return이 높다는 **보편 수학 정리**는 주장하지 않는다. 대신 선택한 benchmark와 통제된 protocol에서 return 개선은 논문의 필수 empirical claim으로 둔다.
- Event-only representation에서 멈추지 않는다. Event representation을 credit/reward/policy update에 연결하는 allocator 또는 동등한 structural readout은 최종 알고리즘의 필수 요소다.

## 5. LLM을 왜 남기는가

순열성만 필요하다면 Deep Sets, Set Transformer, GNN 같은 non-LLM 모델이 더 간결한 baseline이 될 수 있다. LLM을 쓸 이유는 numeric scalar 자체가 아니라 다음 능력에 있다.

- 환경 규칙과 역할/능력 조건을 language-conditioned하게 해석
- 긴 trajectory에서 성공·실패·준비·협력 같은 사건 후보를 요약
- 환경마다 hand-coded reward rule을 새로 만들지 않고 transition의 의미론적 구조를 재사용

현재의 작업 가설은 LLM의 language-conditioned reasoning과 numeric credit calibration을 분리하는 것이다.

```text
LLM-backed encoder: rule + trajectory -> grounded event / participant representation
후속 구조 모듈:    event + agent state -> numeric credit 또는 densified training signal
```

이는 아직 확정된 전제가 아니다. LLM이 병렬 structured head를 통해 numeric credit을 직접 내는 방법, event를 거쳐 learned equivariant allocator가 credit을 내는 방법, 둘을 결합하는 방법을 비교해야 한다. 다만 순차 JSON generation은 agent-axis equivariance를 깨기 쉬우므로 direct-credit baseline으로 쓰더라도 shared parallel readout으로 바꾸어야 한다.

이 연구가 event extractor만 제시하고 끝나면 contribution은 불충분하다. Event가 실제 return 향상으로 이어지려면 (i) 어떤 agent/시간에 어떤 signal을 줄지 계산하는 allocator, (ii) 그 signal을 policy update에 연결하는 학습 규칙, (iii) raw/dense-reward 대조군 대비 utility 검증이 모두 필요하다.

## 6. 해결에 필요한 최소 설계 조건

아래는 특정 모델의 layer diagram이 아니라, 어떤 후보 방법이든 만족해야 할 조건이다.

### A. Agent content와 agent order의 분리

Agent의 observation, action, ability, role, inventory 등은 그 agent와 함께 움직이는 content여야 한다. 반면 input array index, block start offset, prompt 상의 선후관계는 learned feature가 되어서는 안 된다.

### B. Mixed set--sequence representation

Agent들의 집합은 unordered set이지만, 각 agent의 trajectory/history는 time-ordered sequence다. 따라서 position과 mask는 최소한 다음 두 종류의 구조를 표현해야 한다.

- 같은 agent 내부: time/field order 및 history membership 보존
- 서로 다른 agent 사이: block order 제거, 협력에 필요한 정보 교환 허용

### C. Permutation-compatible connectivity

Agent swap 뒤 attention/relation graph가 node 행·열과 함께 재배열되어야 한다.

$$
M(PA,\xi)=\widetilde P M(A,\xi)\widetilde P^\top.
$$

여기서 $M$은 mask 또는 relation graph이고 $\widetilde P$는 token/block 수준으로 확장한 agent permutation이다. 이 조건이 없으면 SetPE만 바꿔도 mask가 agent order를 복원할 수 있다.

### D. Output type에 맞는 readout

Global output은 permutation-invariant aggregation 또는 invariant query에서 나와야 한다. Agent별 output은 각 agent representation에 같은 shared head를 적용하거나 event--agent shared pairwise score로 나와야 한다. Autoregressive `agent_0_credit`, `agent_1_credit` generation은 이 조건에 맞지 않는다.

## 7. 아직 열어 두어야 할 접근 선택지

현재 시점에서 아래 중 하나를 성급히 “정답 architecture”로 확정하지 않는다.

| 접근 | 장점 | 핵심 위험/검증 질문 |
|---|---|---|
| Numeric agent node + set encoder | 구현·대칭성 검증이 단순함 | LLM의 token-level language prior를 얼마나 활용하는가? |
| Natural-language agent block + SetPE/SetMask | 규칙·서술적 role 정보를 직접 다룸 | block 내부 결속과 block 간 협력을 동시에 어떻게 mask할 것인가? |
| Local block encoding 후 agent-summary set interaction | Set-LLM association failure를 피하기 쉬움 | local/global bottleneck이 협업 reasoning을 제한하는가? |
| Equivariant pairwise relation/GNN stage | 직접적인 cooperation relation 모델링 | LLM과의 결합이 실질적으로 이득인가? |
| Event query/slot | agent-order-independent event proposal에 적합 | event가 해석 가능하고 downstream utility가 있는가? |
| Parallel equivariant direct-credit head | end-to-end critic/credit을 단순화 | LLM event abstraction보다 더 잘 작동하는가? |
| Event-conditioned equivariant allocator | semantic event와 numeric allocation을 분리 | allocator가 환경별 hand-crafted reward function으로 퇴화하지 않는가? |

가장 먼저 비교해야 할 것은 raw sequential LLM 대 “수정된 PE/mask를 가진 LLM”이며, 그 다음에 event module과 allocator의 유용성을 분리해서 검증해야 한다.

## 8. 단계별 연구 질문

### Phase 1 — symmetry contract가 실제로 성립하는가?

- 동일 transition을 agent order만 다르게 여러 번 넣었을 때 global output drift는 거의 0인가?
- agent output을 원래 environment axis로 inverse-map했을 때 full distribution/value drift는 거의 0인가?
- fp32 reference와 bf16 LLM에서 각각 drift가 어느 정도인가?
- raw LLM, canonical order, random-order augmentation, proposed mask/PE 구조의 차이는 무엇인가?

성공 기준은 return이 아니라 architecture-level permutation robustness다.

### Phase 2 — structure를 보존하면서 event를 추출하는가?

- input trajectory 안에서 non-zero transition (수집, delivery, 성공/실패 협력 등)을 event representation이 구분하는가?
- event participant output이 실제 변화에 관련된 agent와 일치하는가?
- block swap을 해도 event representation/participant distribution이 각각 invariant/equivariant한가?

여기서는 LBF의 apple disappearance 같은 자동 관찰 가능한 transition delta를 weak supervision으로 쓸 수 있다. Event label ontology를 처음부터 환경별로 hand-code하지 않는 방향을 우선 탐색한다.

### Phase 3 — credit/allocator가 sparse reward learning에 유용한가?

- Event representation 또는 direct structured critic output을 이용해 agent별 credit/shaped signal을 만든다.
- Allocator/readout은 agent permutation에 동변이고, total reward와의 calibration/conservation 조건을 만족하는가?
- 이를 policy update에 연결했을 때 team return, sample efficiency, variance가 raw LLM-MCA와 non-LLM/set critic 대비 개선되는가?
- 이 개선이 단순 dense hand-crafted reward보다 어떤 trade-off를 갖는가?

이 단계는 부가 실험이 아니라 최종 알고리즘 검증이다. 다만 Phase 1--2를 먼저 분리하는 이유는 return failure가 symmetry failure, event failure, allocator failure 중 무엇인지 식별하기 위해서다.

## 9. 평가 원칙

입력 재직렬화는 environment dynamics를 바꾸는 것이 아니라 critic call 직전에만 수행한다. Environment의 original agent axis는 유지하고, model output은 inverse-scatter하여 원래 axis에서 비교한다.

$$
\Delta_V=|V(PA,\xi)-V(A,\xi)|,
\qquad
\Delta_C=\left\|P^{-1}C(PA,\xi)-C(A,\xi)\right\|.
$$

Agent pointer/event output도 full probability distribution 기준으로 같은 방식으로 비교한다. `top_agent`의 argmax는 tie에 취약하므로 보조 지표로만 쓴다. Small-$N$에서는 all-permutation diagnostic, 큰 $N$에서는 random/adversarial permutation sampling을 쓴다.

Semantic utility는 transition/event grounding metric과 downstream RL metric을 분리해 보고한다. 순열 drift가 작다는 것만으로 event가 유용하거나 RL 성능이 좋다고 주장하지 않는다.

## 10. 현재 구현의 위치

현재 Qwen/numeric-node prototype은 다음을 확인한 **Phase-1 feasibility reference**다.

- Qwen에 structured numeric embedding, modified position/mask, parallel event/pointer head를 연결할 수 있음
- 단일 transition numeric node setting에서 agent reserialization drift가 raw sequential LLM-MCA보다 매우 작음
- LoRA backward가 현재 GPU 메모리에서 가능함

그러나 이것은 multi-token agent trajectory의 최종 architecture도, semantic event extractor도, allocator를 포함한 critic/RL algorithm도 아니다. 특히 raw structured node complete attention은 block-association 문제를 피할 필요가 없는 single-node setting의 baseline이다. 다음 설계·실험 결정은 Phase 1의 mixed set--sequence mask와 Phase 2의 grounded event task를 분리해 검증한 뒤, Phase 3에서 반드시 end-to-end credit/RL utility로 판정한다.

## 11. 당장 결정할 일

1. Agent trajectory를 어떤 최소 token/block 단위로 표현할지 정한다. 한 transition의 structured field인지, 짧은 $H$-step history인지부터 명확히 한다.
2. 그 representation에서 agent-internal association을 보존하면서 agent block order를 제거하는 PE/mask를 구현한다.
3. 먼저 synthetic swap test와 LBF transition grounding task로 표현력·대칭성을 함께 검사한다.
4. Direct equivariant credit head와 event-conditioned allocator를 최소한 하나씩 정의해, Phase 3의 end-to-end RL 대조 실험으로 연결한다.

이 순서는 “복잡한 architecture를 먼저 만들고 이유를 찾는 것”을 피하면서도, 최종 판정을 return 개선으로 두기 위한 연구 gate다.
