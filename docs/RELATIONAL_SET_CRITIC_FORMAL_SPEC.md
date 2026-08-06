# PEEL: Permutation-Equivariant Event LLM

> **문서 상태:** 연구 설계 명세 — 성능 주장 아님  
> **대상:** agent 순열 동변성을 우선하는 협력적 MARL LLM critic 및 event 표현  
> **보조 문서:** [연구노트](RELATIONAL_SET_CRITIC_RESEARCH_NOTE.md)

## 초록

대규모 언어 모델(LLM)을 다중 에이전트 강화학습(MARL)의 중앙집중형 critic으로 사용할 때, 동질 agent의 상태·행동·궤적은 대개 자연어 prompt의 임의 순서로 직렬화된다. Agent 나열 순서는 협력 과제의 의미와 무관하지만, decoder-only LLM은 이 위치를 이용하여 team value 또는 agent별 credit을 달리 출력할 수 있다. 이를 **직렬화 유발 agent 순열 편향(serialization-induced permutation bias)**이라 부른다.

본 문서는 **PEEL (Permutation-Equivariant Event LLM)**을 제안한다. 환경 규칙 text와 object state는 critic context로 유지하되, agent trajectory block에 대해서만 block 순번에 독립적인 positional encoding과 attention 구조를 적용한다. 순열 불변 global/event slot과 공유된 병렬 numeric head를 결합하여 team 출력에는 agent-order invariance, agent credit에는 agent-order equivariance를 부여한다. LLM은 처음부터 autoregressive JSON credit을 생성하는 대신, 규칙 조건부의 grounded latent event를 추출한다. Event에서 agent credit으로의 배분과 RL policy 학습은 별도의 후속 단계로 분리한다.

## 1. 문제 정의

협력 trajectory를 다음과 같이 둔다.

$$
x = \{(s_t, a_t, r_t, d_t)\}_{t=0}^{T-1},
$$

여기서 $N$명의 agent가 joint action $a_t$를 수행하고, 환경은 global reward $r_t$와 종료 표시 $d_t$를 준다. $P_A$는 교환 가능한 agent의 상태, 행동, 능력치, 궤적을 함께 재배열하는 순열 연산자이다. Object state $O$는 이 1차 연구에서 고정 순서의 environment context로 둔다.

중앙집중형 critic이 만족해야 할 조건은 다음과 같다.

$$
V(P_Ax,O)=V(x,O),
\qquad
C(P_Ax,O)=P_A C(x,O),
$$

여기서 $V$는 team scalar/value/event summary이고, $C\in\mathbb{R}^{N\times T}$는 agent-time credit tensor이다.

- 첫 번째 식은 **순열 불변성(permutation invariance)**이다. Agent 입력 순서가 바뀌어도 team value와 global event summary는 동일해야 한다.
- 두 번째 식은 **순열 동변성(permutation equivariance)**이다. Credit 값이 모든 agent에서 같아야 한다는 뜻이 아니라, 동일 physical agent의 credit이 재직렬화 뒤에도 그 agent에 붙어 있어야 한다는 뜻이다.

이 대칭성은 동일한 교환 가능 class 내부에만 요구한다. 서로 다른 action space, 역할, 물리 능력, 관측 구조를 가진 agent를 서로 바꾸는 것은 일반적으로 유효한 대칭이 아니다. Agent type이 $1,\ldots,K$라면 해당 대칭군은 전체 $S_N$이 아니라 $S_{n_1}\times\cdots\times S_{n_K}$이다. 반대로 strength, sensor range, carry 가능 여부, level처럼 환경적으로 의미 있는 능력 feature는 입력에 남겨야 한다.

Object도 sequence로 직렬화되므로 object-order bias는 장기적으로 해결할 가치가 있다. 그러나 이는 별도의 symmetry 축이다. 본 논문의 주 이론·metric·claim은 agent axis에 한정하며, 현재 object set 처리는 후속 확장을 위한 구현 선택일 뿐 object robustness를 주 결과로 주장하지 않는다.

## 2. 기존 LLM-MCA interface의 한계

LLM-MCA류 critic은 agent 위치·행동·이름·출력 key를 text sequence로 렌더링한 뒤, `agent_0_credit`과 같은 credit array를 autoregressive하게 생성한다. 이 방식에는 다음과 같은 순서 의존성이 있다.

- Agent/object block의 absolute 또는 RoPE position
- Causal attention과 좌에서 우로 진행되는 JSON 생성 순서
- 고정된 agent 이름 및 output key 순서
- 순서마다 다르게 발생할 수 있는 parser 실패, retry, fallback

따라서 “순서에 영향을 받지 말라”는 prompt instruction이나 단순한 익명화만으로는 위 대칭식을 보장할 수 없다. Canonical serialization도 계산량이 적은 control은 될 수 있지만, 동등한 feature를 가진 block의 tie 및 identity tracking 문제가 남는다.

임의 critic $C_0$에 대해 group average를 취하면 정확한 동변성을 얻을 수 있다.

$$
\bar C(x)=\frac{1}{|G|}\sum_{\pi\in G}\pi^{-1}C_0(\pi x).
$$

그러나 이는 $O(N!)$ LLM 호출을 요구하므로, small-$N$ diagnostic/control로만 적합하고 online critic의 주 방법은 아니다.

현재 Qwen 기반 raw LLM-MCA의 exhaustive small-$N$ probe에서는 agent/object block 재직렬화만으로 delayed cooperation credit의 평균 최대 agent drift가 $0.9$, 최대 team-total drift가 $2.0$이었다. Two-object trajectory에서는 각각 $1.05$, $3.0$이었다. 한-step 협력 사례의 zero drift는 일반적 robustness가 아니라 쉬운 prompt에서의 우연한 결과였다. 세부 조건과 raw 출력은 연구노트에 보관한다.

## 3. 배경 지식과 비교 방법

### 3.1 Set-structured learning

Deep Sets는 set input $X=\{x_i\}$에 대한 순열 불변 함수를 다음 형태로 표현할 수 있음을 보였다.

$$
f(X)=\rho\left(\sum_i\phi(x_i)\right).
$$

모든 원소에 같은 encoder $\phi$를 적용하고 commutative pooling을 하면 입력 순서가 사라진다. Agent별 output은 invariant global summary $g$와 각 원소 표현 $h_i$를 결합한 공유 head로 만들 수 있다.

$$
h_i=\phi(x_i),\qquad g=\rho\left(\sum_i h_i\right),\qquad c_i=\psi(h_i,g).
$$

Set Transformer는 attention으로 agent-object 및 agent-agent interaction을 모델링하면서도 set 대칭을 유지한다. 이는 “두 agent가 같은 object에 동시에 접근했는가” 같은 협력 관계에 특히 중요하다.

### 3.2 Permutation-aware MARL

Permutation Invariant Critic(PIC)과 후속 permutation-aware MARL 연구는 centralized critic이 agent observation/action을 고정 순서로 concatenate할 때 불필요한 slot bias가 생긴다는 점을 지적한다. 이 연구들의 핵심 교훈은 교환 가능한 agent index를 학습 feature로 취급하면 안 된다는 것이다. 본 연구는 여기에 pretrained causal LLM 및 language rule conditioning을 결합한다.

### 3.3 Set-LLM

Set-LLM은 unordered item을 causal LLM에 넣을 때 position encoding뿐 아니라 attention mask도 함께 바꾸어야 함을 보인다. Position만 제거해도 causal mask가 serialization order를 복원할 수 있기 때문이다. 본 연구는 이 원리를 MARL의 mixed set–sequence input으로 확장한다. 즉, rule text는 sequence이고, entity는 set이며, 각 entity의 history는 다시 sequence이다. 또한 본 연구에서는 globally invariant한 답만이 아니라 agent별 equivariant output이 필요하다.

### 3.4 비교 baseline

| 방법 | 대칭성 | LLM 호출 수 | 역할 |
|---|---:|---:|---|
| Raw sequential LLM-MCA | 보장 없음 | 1 | 직접 baseline |
| Canonical serialization | tie-free 상황의 일관성 | 1 | 저비용 control |
| Random-order training | 분포적 완화 | test 시 1 | augmentation control |
| $K$-permutation average | 근사 동변성 | $K$ | 계산량–robustness 비교 |
| All-permutation average | 정확한 대수적 동변성 | $N!$ | small-$N$ diagnostic |
| PEEL | architecture-level 목표 | 1 | 제안 방법 |

## 4. Stage 0: 환경 contract와 event interface

Stage 0의 목적은 “LLM이 생성한 scalar credit이 곧바로 policy를 개선하는가”를 묻는 것이 아니다. 먼저 **규칙과 transition을 보고, 어떤 사건이 일어났고 누구/무엇이 관련되었는지를 순열 안전하게 표현할 수 있는가**를 독립적으로 검증한다. 따라서 이 단계에는 allocator, shaped reward, policy gradient, RL return이 없다.

### 4.1 Adapter가 제공해야 하는 최소 contract

각 environment adapter는 환경마다 다른 raw observation을 아래의 공통 transition record로 변환한다.

$$
\mathcal D_t = (p, X_t, R_t, a_t, \Delta_t, r_t, d_t).
$$

| 항목 | 표기 | 역할 | 모델에 주는가? |
|---|---|---|---|
| task/rule text | $p$ | 목표, action 의미, type/capability 의미 | 입력 |
| entity state set | $X_t$ | agent/object의 물리적·범주적 상태 | 입력 |
| relation graph | $R_t$ | same entity, same time, visibility, possession 등 | 입력 metadata/mask |
| joint action | $a_t$ | 각 agent가 실제 수행한 행동 | 입력 |
| transition delta | $\Delta_t$ | $s_t\rightarrow s_{t+1}$에서 실제로 변한 사실 | target/diagnostic |
| team reward | $r_t$ | 환경의 원래 global reward | target/diagnostic |
| termination | $d_t$ | 종료·성공·time limit 여부 | target/diagnostic |

여기서 entity state는 agent와 object의 두 set으로 나눈다.

$$
X_t=\{x^A_{i,t}\}_{i=1}^{N}\cup\{x^O_{j,t}\}_{j=1}^{M_t}.
$$

Agent feature에는 local/global position, action, capability, inventory, observation-derived state가 들어갈 수 있다. Object feature에는 type, position, active/requested 상태, level, goal 관계 등이 들어간다. **원래 array index나 `agent_0`이라는 저장 slot은 feature가 아니다.** 반면 role, strength, sensor range처럼 환경에서 관측 가능한 차이는 feature로 반드시 남긴다.

`agent_ids`, `object_ids`는 model input이 아니라 adapter의 외부 bookkeeping이다. 이 ID는 transition 전후에 같은 physical entity를 매칭하고, model output을 원래 environment axis로 inverse-scatter하며, grounding label을 평가하는 데에만 쓴다.

### 4.2 입력과 target을 분리하는 이유

Event extractor의 입력은 원칙적으로 $(p,X_t,R_t,a_t)$ 또는 짧은 history window이며, $\Delta_t$, $r_t$, $d_t$는 모델이 예측하거나 설명해야 할 결과다.

```text
입력:  현재 state + 관계 + joint action + task rule
모델:  grounded latent events
target: 다음 state의 변화 + reward + done
```

Delta를 input에 넣으면 “food가 사라졌다”는 답을 이미 본 상태에서 event를 만들게 된다. 이는 event understanding보다 결과 복사를 학습시키는 누설(leakage)이 된다. 반대로 $s_t$만 보고 $s_{t+1}$ 전체를 복원하는 것은 정적인 배경까지 반복 예측하게 만든다. Delta는 두 상태의 차이만 보므로, event가 설명해야 하는 변화를 밀도 높은 학습 신호로 제공한다.

Delta는 반드시 단순 좌표 차이만을 뜻하지 않는다. Adapter는 환경에 따라 다음과 같이 typed change로 표현한다.

| 변화 유형 | 일반적 표현 | LBF 예시 | RWARE 예시 |
|---|---|---|---|
| 연속/격자 속성 변화 | $x' - x$ | robot 위치 한 칸 이동 | robot 위치·방향 변화 |
| 상태 전환 | before/after categorical flag | food active $1\rightarrow0$ | shelf requested $1\rightarrow0$ |
| 생성/삭제 | active flag + null state | 수집된 food의 disappearance | 새 requested shelf 등장 |
| 관계 변화 | edge add/remove | 두 robot이 load 조건을 만족 | robot이 shelf를 들기 시작 |
| 종료 | success/failure/time-limit flag | 모든 food 수집 | delivery 완료/episode 종료 |

### 4.3 Event slot의 의미

모델은 한 transition 또는 trajectory window에 대해 $K_E$개의 **교환 가능한 event slot**을 출력한다.

$$
\mathcal E = \{e_k\}_{k=1}^{K_E},\qquad
e_k=(z_k,\alpha_k^{\mathrm{time}},p_k^A,p_k^O,v_k,u_k).
$$

| 출력 | 의미 | 순열 요구 조건 |
|---|---|---|
| $z_k$ | event 의미를 담는 latent embedding | agent/object 재정렬에 불변 |
| $\alpha_k^{\mathrm{time}}$ | event의 시점 또는 span attention | time축은 유지, agent 순열에는 불변 |
| $p_k^A\in[0,1]^N$ | 참여/영향 agent pointer | agent와 함께 동변적으로 재배열 |
| $p_k^O\in[0,1]^M$ | 관련 object pointer | 현재 object serialization의 index-aligned grounding |
| $v_k$ | event impact: 긍정/부정·크기·불확실성 | 불변 |
| $u_k$ | optional event confidence/existence | 불변 |

Slot $k$가 사전에 “성공”, “실패”, “접근”이라는 고정 이름을 가질 필요는 없다. 여러 slot 중 하나는 실제 변화와 연결되고, 나머지는 null/no-event를 표현할 수 있다. 따라서 event slot의 나열 순서도 nuisance다. Label이 존재한다면 bipartite matching loss를 써서 slot label 대응을 결정하고, label이 없으면 reconstruction/prediction objective가 slot usage를 유도한다.

중요하게도 $p_k^A$는 아직 numeric credit이 아니다. 이는 “event $k$에 누가 관련되었는가”라는 grounded relation이다. 한 agent가 event의 필요조건을 준비했지만 즉시 reward를 받지 않았을 수 있고, 여러 agent가 event에 참여해도 최종 allocator의 credit 비율은 달라질 수 있다. 이 분리를 통해 LLM이 text reasoning에 강한 event/role identification을 담당하고, 민감한 scalar credit 배분은 이후 구조적 module이 담당하게 한다.

### 4.4 Stage-0 학습 objective

Event slot을 단순한 자유 latent로 두면, 모든 정보를 하나의 slot에 넣거나 reward=0만 예측하는 퇴화가 가능하다. 따라서 최소한 다음 objective를 함께 사용한다.

$$
\mathcal L_{\mathrm{stage0}}
=\lambda_\Delta\mathcal L_{\Delta}
+\lambda_r\mathcal L_r
+\lambda_d\mathcal L_d
+\lambda_{\mathrm{future}}\mathcal L_{\mathrm{future}}
+\lambda_{\mathrm{perm}}\mathcal L_{\mathrm{perm}}.
$$

| Loss/평가 | 질문 | 예시 |
|---|---|---|
| $\mathcal L_\Delta$ | event가 실제 entity 변화를 설명하는가? | food disappearance, shelf movement 예측 |
| $\mathcal L_r$ | event summary가 immediate reward를 예측하는가? | successful load의 reward 예측 |
| $\mathcal L_d$ | success/failure/termination을 구분하는가? | 모든 object 완료 여부 |
| $\mathcal L_{\mathrm{future}}$ | reward=0인 준비 행동이 미래 성공과 연결되는가? | $H$ step 내 success/return prefix 예측 |
| $\mathcal L_{\mathrm{perm}}$ | 순열 후 동일 event/동변 pointer가 나오는가? | inverse-mapped pointer consistency |

$\mathcal L_{\mathrm{future}}$가 특히 중요하다. Sparse MARL에서 대부분의 transition은 $r_t=0$이므로, delta와 immediate reward만으로 학습하면 “한 칸 움직임” 같은 저수준 변화만 잘 표현하고 enabling action을 놓칠 수 있다. Future-success label은 환경이 준 trajectory만으로 만들 수 있으며, 예를 들어 $\mathbb{1}[\sum_{\tau=t}^{t+H}r_\tau>0]$ 또는 discounted return prefix를 사용할 수 있다.

### 4.5 Grounding과 permutation 평가

Stage 0에서 event가 유의미하다고 주장하려면 최소한 다음을 분리해 측정한다.

1. **Delta coverage:** event representation만으로 changed attribute, create/delete, relation change를 예측할 수 있는가.
2. **Agent/object grounding:** pointer의 상위 entity가 실제 changed entity, action 수행자, 또는 transition의 필요조건과 일치하는가.
3. **Predictive utility:** event summary가 raw state보다 적은 정보량으로 future success/reward/done을 예측하는가.
4. **Permutation robustness:** 같은 physical record를 reserialize한 뒤 $z_k,v_k$는 slot matching 후 같고, $p_k^A,p_k^O$는 inverse-map 후 같아야 하는가.
5. **Null-event behavior:** 변화가 없는 transition에서 event slot이 임의의 agent/object에 과도하게 집중하지 않는가.

LBF의 load success/failed load/food disappearance, RWARE의 shelf pickup/delivery/request 변화는 이러한 평가를 위한 자동 diagnostic이다. 이는 환경별 reward rule을 손으로 agent credit으로 분해하는 것이 아니라, adapter가 이미 관측 가능한 state transition을 annotation-free target으로 노출하는 것이다.

### 4.6 Stage 0의 종료 조건

다음 조건을 만족하기 전에는 allocator나 RL 성능 실험으로 넘어가지 않는다.

- Raw sequential LLM 또는 단순 structured baseline보다 낮은 permutation drift
- Agent/object pointer가 random pointer보다 의미 있게 높은 grounding score
- Delta 및 future-success prediction이 단순 current-state/reward baseline보다 개선
- Reward=0 transition에서도 future success와 관련된 event signal이 존재
- Event slot 수, null event, variable entity cardinality에 대해 명확한 failure analysis 가능

이 종료 조건은 “return이 아직 낮다”는 이유만으로 event extractor를 기각하지 않게 하고, 반대로 event representation의 실패를 allocator나 RL optimizer 문제로 오인하지 않게 한다.

## 5. 모델 아키텍처

```text
순차 rule / task text
unordered agent trajectory blocks {tau_i}
unordered object trajectory blocks {o_j}
                     |
       mixed set-sequence tokenizer + SetPE
                     |
       PEEL set-attention LLM backbone
              |                    |
  invariant global/event slots    per-agent block states
              |                    |
     global/event heads       shared parallel pointer/credit heads
```

### 5.1 Token과 metadata

Agent token에는 physical state, action, capability/type, time feature를 넣는다. Object token에는 object type, state, time feature를 넣는다. Rule text는 일반 language token으로 처리한다. Structured token이 갖는 metadata는 다음과 같다.

| Metadata | 용도 | Learned feature 여부 |
|---|---|---|
| token/entity type | robot, apple, shelf, state, action, event 구분 | 가능 |
| time ID | block 내부 temporal order | 가능 |
| block membership | same-entity mask와 output gather | **bookkeeping만** |
| physical attribute | 위치, level, visibility, carrying, capability | 가능 |
| original array index/agent 이름 | 저장 위치 또는 tag | **불가** |

`block_id`는 “두 token이 같은 entity block에 속하는가”를 mask에서 판정하고 output을 원 환경 agent 축으로 scatter하기 위한 key다. 모델 embedding에는 넣지 않는다. 따라서 physical agent와 output의 대응은 유지하면서도 “prompt의 첫 번째 block”이라는 인공 정보는 제거할 수 있다.

### 5.2 Mixed positional encoding

Rule text는 통상적인 sequential position을 유지한다. 반면 모든 homogeneous agent block은 같은 temporal origin을 사용한다. 즉, 어느 block이 먼저 tensor에 놓였는지와 무관하게 모든 $t=0$ structured token은 같은 time position을 갖고, 모든 $t=1$ token도 마찬가지다. Object block에도 같은 원리를 적용한다. Agent index나 block start offset에 따른 absolute/RoPE position은 주지 않는다.

이는 `rule text: sequence`, `entities: set`, `entity history: sequence`이라는 MARL 입력 구조에 맞춘 SetPE이다.

### 5.3 PEEL Set Attention

Attention graph는 entity array index가 아니라 환경 관계에서 만들어야 한다. 가능한 edge family는 다음과 같다.

- 같은 entity의 temporal history
- 공간 거리, adjacency, visibility, communication range
- object possession 및 같은 task/region 관계
- 모든 entity에 대칭적으로 연결된 global/event bridge

순열 $P$에 대해 mask는 반드시 다음 조건을 만족해야 한다.

$$
M(Px)=P M(x)P^\top.
$$

즉, 순열 전후의 relation graph는 동일하고 node row/column만 함께 재배열되어야 한다. 현재 Stage-0 reference와 Qwen 구현의 기본값은 모든 structured node를 직접 연결하는 complete set graph다. 이는 global critic에 불필요한 locality bottleneck을 두지 않는 가장 단순한 대칭 baseline이다. 거리·visibility·task edge를 이용한 sparse relation graph는 이후 ablation으로 추가하며, 같은 class의 모든 원소에 동일한 relation predicate를 적용해야 한다.

### 5.4 병렬 output head

Agent block representation을 $h_i$, invariant global/event representation을 $g$라 하자. Team output과 agent output은 다음과 같이 계산한다.

$$
V=\rho(g), \qquad C_i=\psi(h_i,g).
$$

모든 agent에 같은 shared head $\psi$를 사용한다. Event participant/object pointer도 event query와 agent/object node 사이의 비교를 병렬로 계산한다. `agent_0_credit` 뒤에 `agent_1_credit`을 생성하는 autoregressive JSON decoder는 사용하지 않는다. Environment adapter는 block-to-original-agent map을 보관하고, model output $[N,T]$를 original environment axis로 inverse-scatter한다.

### 5.5 순열 동변성의 근거

Shared entity encoder, block-order-independent SetPE, 위의 conjugate mask 조건, row-wise shared attention/MLP, shared output head가 모두 성립하면 layer induction으로 다음을 얻는다.

$$
H(P_Ax)=P_AH(x),\qquad g(P_Ax)=g(x).
$$

따라서

$$
C(P_Ax)=P_AC(x),\qquad V(P_Ax)=V(x)
$$

가 성립한다. 이는 exact arithmetic에서의 architecture statement다. BF16, fused attention, reduction order는 작은 수치 drift를 만들 수 있으므로, formal proof와 별도로 finite-precision audit을 보고해야 한다.

## 6. LLM adaptation

제안 구조는 prompt wrapper만으로 구현되지 않는다. Pretrained decoder-only LLM은 ordinary sequence position과 causal attention 분포에서 학습되어 있다. 제안 모델은 custom structured position ID, 4-D relational attention mask, structured numerical projection, event query, 병렬 numeric head를 직접 forward에 제공한다.

현실적인 첫 adaptation recipe는 다음과 같다.

1. Base causal LLM 대부분은 freeze한다.
2. Attention Q/K/V/O projection에 LoRA를 적용한다.
3. Structured embedding, event query, numeric/pointer head를 학습한다.
4. Permutation-augmented transition/event pretext dataset에서 학습한다.
5. Event representation이 검증된 뒤 learned event-to-credit allocator와 RL objective를 추가한다.

현재 Qwen2.5-7B 구현은 rank-2 LoRA가 A10 24GB에서 실행 가능한지를 확인했다. 3584차원 BF16 raw dot product가 softmax pointer assignment를 불안정하게 만들 수 있어, pointer는 cosine-normalized bounded logit을 사용한다.

## 7. Event-to-credit allocator: 후속 단계

Event extractor가 바로 신뢰 가능한 scalar reward를 생성할 필요는 없다. Grounded event impact를 credit으로 바꾸는 generic shared allocator의 한 형태는 다음과 같다.

$$
a_{e,i}=\operatorname{softmax}_i q(z_e,h_i,g),
\qquad
c_{i,t}=\sum_e w_{e,t}\,a_{e,i}\,v_e.
$$

$q$가 shared function이고 event/agent pointer가 함께 순열되므로 이 allocator 역시 equivariant하다. Allocator의 structural constraint는 parameter sharing, optional conservation, no index feature처럼 일반적이어야 한다. LBF 전용의 rule-based reward function이 되어서는 안 된다. 단순 rule-based allocator는 debugging control로만 사용한다.

## 8. 평가 protocol

핵심 robustness test에서는 환경과 policy의 agent order를 바꾸지 않는다. 동일한 physical trajectory를 고정하고 **critic input serialization만** 여러 방식으로 섞는다. Opaque stable tag는 data block과 함께 이동하며, raw text baseline의 output은 parse 뒤 original environment axis로 inverse-scatter한 뒤 비교한다.

보고할 지표는 다음과 같다.

- Global value/event impact drift
- Inverse-mapped agent credit 및 agent-pointer drift
- Top-contributor rank/sign agreement와 parser validity
- Event-slot matching consistency
- Structured model의 finite-precision drift
- Event extraction 검증 이후의 downstream RL return

$N\le4$에서는 모든 **agent** 순열을 열거하고, 더 큰 $N$에서는 random/adversarial agent permutation sample을 사용한다. 다음 네 condition을 비교한다.

1. fixed canonical order train/test
2. fixed-order train, randomized serialization test
3. random-order augmentation train/test
4. proposed relational architecture train/test

LBF의 3–4 homogeneous agent cooperative setting은 primary stress domain이다. RWARE는 typed object, 더 많은 entity, 다른 transition graph에서의 scale/transfer evidence를 제공한다. Climbing은 row/column role이 본질적으로 비대칭이므로 homogeneous permutation claim의 주 benchmark가 아니다.

## 9. 기대 기여점

1. **문제와 benchmark:** LLM centralized MARL critic의 serialization-induced permutation bias를 inverse-mapped randomized reserialization test로 정의하고 정량화한다.
2. **아키텍처:** Set-LLM의 position/mask 원리를 mixed MARL set–sequence input과 parallel equivariant agent output으로 확장한다.
3. **Grounded event interface:** LLM의 규칙 이해 능력을 direct numeric credit 생성보다 event 추출·participant/object grounding에 먼저 사용한다.
4. **평가 분리:** symmetry, event grounding, credit fidelity, RL utility를 서로 다른 실험 축으로 분리한다.

## 10. 한계와 열린 조건

- Formal symmetry가 유용한 event semantics나 RL return 향상을 보장하지는 않는다.
- Sparse random rollout만으로는 LLM event extraction을 학습·평가하기 부족하다. Non-zero event와 delayed enabling action이 포함된 targeted trajectory data가 필요하다.
- Variable-cardinality padding/mask와 RWARE-scale batch 지원은 아직 구현 과제다.
- Global event bridge와 relation graph는 충분한 interaction 정보를 전달하면서도 비대칭 edge를 다시 도입하지 않아야 한다.
- Raw baseline의 변동에는 position bias뿐 아니라 generation/parser instability도 포함된다. 둘 다 실용적으로 중요하지만 가능한 한 분리해 보고해야 한다.

## 참고문헌

- Zaheer et al., *Deep Sets*, NeurIPS 2017.
- Lee et al., *Set Transformer*, ICML 2019.
- Liu et al., *Permutation Invariant Critic*, AISTATS 2020.
- Foerster et al., *Counterfactual Multi-Agent Policy Gradients*, AAAI 2018.
- Xie et al., *Set-LLM*, 2025.
