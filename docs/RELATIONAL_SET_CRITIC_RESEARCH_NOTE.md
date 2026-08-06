# PEEL (Permutation-Equivariant Event LLM) — Research Note

이 파일은 [formal specification](RELATIONAL_SET_CRITIC_FORMAL_SPEC.md)의 작업용 보조 노트다. **PEEL**은 *Permutation-Equivariant Event LLM*의 약칭이다. formal 문서는 문제 정의·배경·모델·평가 설계를 보존하고, 이 노트에는 대화에서 확정되거나 기각된 설계 선택, 새 가설, 실제 구현/실험 결과를 날짜순으로 축적한다. 의미 있는 결정과 결과만 기록하며, 단순 질의응답이나 자동 job 상태 변화는 기록하지 않는다.

## 운영 원칙

- 대화에서 나온 아이디어는 **가설**, **결정**, **관찰 결과**, **기각 이유**를 구분해 기록한다.
- 아직 실행하지 않은 항목을 성능 결과처럼 쓰지 않는다.
- 큰 설계 변경은 formal specification에도 반영하되, 이 로그에는 변경 이유와 날짜를 남긴다.
- formal specification은 안정된 문서이고, 이 노트는 이를 보완하는 living document다.

---

## 2026-07-30 — exact averaging의 비용과 online 주 방법

### 논의

Reynolds/group average는 모든 agent 순열에 대해 critic을 평가하므로 agent 수가 $N$일 때 $N!$번의 LLM 호출이 필요하다. 2-agent LBF에서는 두 번으로 가능하지만, LLM critic이 이미 training wall-clock의 병목인 상황에서 online learning의 기본 방법으로 쓰기에는 비현실적이다.

### 결정

- **All-permutation symmetrization은 online 주 방법이 아니라 offline gold-standard/diagnostic으로 한정한다.**
- 특히 2-agent LBF에서는 raw critic의 순열 편향을 측정하고, 후술할 one-call 방법이 정확히 구현되었는지 검증하는 control로 사용한다.
- online training의 1차 주 방법은 **one-call canonical serialization**으로 둔다. 원래 LLM-MCA와 동일하게 trajectory당 LLM 호출은 한 번이다.

### one-call canonical serialization

1. 원래 agent index와 인간 친화적 이름(Alice/Bob)을 prompt에서 순서 정보로 사용하지 않는다.
2. episode 전체에서 각 agent의 label-free record를 구성한다. 가능한 key는 type/capability, initial state, 그리고 state-action history다. key에는 원래 array index를 넣지 않는다.
3. 이 key의 lexicographic order로 agent를 canonical slot `slot_0`, `slot_1`, ...에 배치하고, state/action/progress 및 output schema를 같은 순서로 직렬화한다.
4. LLM이 canonical slot별 credit을 출력하면, canonicalization 때 저장한 inverse permutation으로 실제 environment agent axis에 되돌린다.

입력 agent 순열이 달라도 canonicalizer가 동일한 physical trajectory를 동일한 prompt로 바꾸면, deterministic LLM decoding 하에서 global output은 invariant이고 inverse-mapped agent output은 equivariant다. 이는 LLM 내부가 permutation-invariant라는 뜻은 아니며, **input quotient/canonical representative를 이용해 interface 수준에서 대칭을 강제**하는 방법이다.

### 남은 위험과 검증

- 두 agent의 전체 canonical key가 완전히 같은 tie는 canonical order만으로 해소할 수 없다. tie group은 credit을 동일하게 배분하거나, 그 group 내부에 한해서만 averaging해야 한다.
- canonical key가 episode의 미래 action까지 사용하므로, trajectory를 모두 받은 뒤 credit을 주는 현재 offline/batch LLM-MCA에는 적합하다. online step critic으로 전환하면 prefix-only key로 다시 설계해야 한다.
- 위치 기반 canonical order는 정책이 exploit할 수 있는 새로운 convention을 만들 수 있다. raw prompt, canonical prompt, exact 2-permutation average를 같은 trajectory에서 비교해 convention 효과와 symmetry 효과를 분리한다.
- canonicalization, parser retry/fallback, grounded filter까지 포함한 전체 pipeline에 대해 equivariance unit test를 만든다.

### 다음 구현 단위

1. `Trajectory`의 agent axis를 permute/inverse-permute하는 순수 utility와 unit test.
2. LBF serializer의 anonymous canonical-slot 모드.
3. raw/canonical/swap-average가 동일 trajectory에서 만드는 prompt와 credit을 JSONL로 비교하는 offline diagnostic script.

---

## 2026-07-30 — averaging 이후의 연구 방향

### 정리: group average는 기여가 아니라 diagnostic이다

모든 순열에 대한 averaging은 [Janossy Pooling](https://arxiv.org/abs/1811.01900)의 기본 구성과 직접 연결된다. Janossy Pooling은 순서 민감 함수의 모든 순열 평균으로 invariant function을 만들고, 계산량을 줄이기 위해 canonical ordering, 제한된 interaction order, stochastic permutation을 다룬다. 따라서 LLM critic에 all-permutation average 또는 sampled average를 붙이는 것만으로는 주된 방법론 기여가 되기 어렵다.

이 방법의 역할은 유지한다.

- 2-agent LBF에서 **exact symmetry gold standard**를 제공한다.
- raw LLM의 order bias가 얼마나 큰지, one-call 방법이 올바른지 검증한다.
- 이후 제안 방법이 성능/대칭성 양쪽에서 비교할 명확한 control이 된다.

그러나 online LLM-MCA의 principal method는 여전히 1회 호출이어야 한다.

### 아이디어 1 — LLM semantic event extractor + equivariant allocator

#### 가설

LLM이 agent별 numeric credit array를 autoregressively 직접 생성하는 것이 order bias와 noisy reward의 주요 원인일 수 있다. LLM의 장점은 숫자 배분 자체보다 trajectory에서 협업의 원인·사건·역할을 자연어/구조로 해석하는 능력에 있다.

#### 제안 구조

1. LLM은 한 trajectory를 보고 agent 이름이 아닌 물리적 state/action/좌표에 anchored 된 **event 또는 role constraint**를 생성한다.
   - 예: “t=12에서 level-2 apple의 서로 다른 loading cell에 도달해 동시에 load한 두 robot의 공동 행동이 성공 원인이다.”
   - 예: “t=7의 boundary-hit action은 진전이 아니다.”
2. deterministic 혹은 작은 learned **permutation-equivariant allocator**가 event를 trajectory의 agent records와 매칭하여 $C_{i,t}$를 계산한다.
3. allocator는 shared local map, invariant set aggregation, 동일한 agent-wise rule로 구현한다. 따라서 numeric credit은 input order와 무관하게 agent에 연결된다.

#### 기대 효과와 검증 질문

- LLM 호출은 기존과 같은 trajectory당 1회다.
- LLM의 output order는 event list의 presentation 문제로 축소되고, credit allocation은 별도 구조가 보장한다.
- 단, LLM event가 `first robot`, `Alice` 같은 slot-based 표현을 쓰면 다시 대칭이 깨진다. prompt/schema가 좌표, object id, action, time, 필요한 role cardinality만 허용하도록 해야 한다.
- LBF의 rule-based oracle과 비교하여 LLM이 실제로 geometric progress를 넘어 collaborative load/failure 같은 semantic event를 더 잘 식별하는지 확인해야 한다. 그렇지 않으면 LLM은 불필요하다.

### 아이디어 2 — set-aware LLM teacher에서 equivariant student critic으로 distillation

#### 가설

LLM을 online numerical critic으로 호출하는 방식은 비용, parser failure, reward nonstationarity를 동시에 만든다. LLM의 semantic prior는 offline annotation teacher로 사용하고, RL에는 작은 permutation-equivariant student critic을 사용해야 할 수 있다.

#### 제안 구조

1. LLM이 offline trajectory에 credit 또는 event/role annotation을 생성한다.
2. student는 `agent trajectory records: set`을 입력으로 받아 invariant global head와 equivariant per-agent/per-timestep credit head를 출력한다.
3. 학습 손실은 teacher supervision 외에 permutation consistency, event/counterfactual correctness, global-return calibration을 포함한다.
4. RL rollout과 update 중에는 student만 실행하므로 LLM 호출이 없다.

이 방향은 “LLM-MCA 성능을 그대로 재현한다”보다 “LLM semantic knowledge를 대칭적인 MARL credit function으로 전이한다”라는 새로운 문제 정의에 가깝다.

### architecture-level 장기 방향

Set-LLM을 그대로 agent별 JSON 생성에 쓰는 것은 충분하지 않다. 장기적으로는 agent block은 set, 각 block 내부의 time history는 sequence, object block도 set으로 보는 mixed-symmetry encoder가 필요하다. 마지막에는 autoregressive token generation이 아니라 agent slot representation마다 공유된 numeric head를 적용해야 한다.

$$
\{\text{agent history}_i\}_{i=1}^N
\longrightarrow \{h_i\}_{i=1}^N
\longrightarrow \{C_{i,t}\}_{i,t}.
$$

이 구조의 global head는 invariant pooling에서, individual head는 shared equivariant map에서 나온다. Set-LLM의 SetPE/SetMask는 input-side guarantee의 출발점이며, output-side equivariance는 별도로 설계·증명해야 한다.

### 공부 및 문헌 순서

1. [Deep Sets](https://arxiv.org/abs/1703.06114): invariant/equivariant set function의 기본 표현.
2. [Janossy Pooling](https://arxiv.org/abs/1811.01900): averaging/canonicalization/random-permutation의 이론적 위치와 한계.
3. [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html): interaction-aware set representation.
4. [PIC](https://proceedings.mlr.press/v100/liu20a.html) 및 [Hao et al., ICLR 2023](https://mlanthology.org/iclr/2023/hao2023iclr-boosting/): MARL의 invariant critic/equivariant network.
5. [COMA](https://ojs.aaai.org/index.php/AAAI/article/view/11794), [Shapley Counterfactual Credits](https://arxiv.org/abs/2106.00285), STAS: structural·temporal credit의 correctness criterion.
6. Set-LLM: mixed set-text input에 대한 architecture-level LLM invariance.

### 현재 우선순위

첫 구현은 여전히 canonicalization + exact 2-agent diagnostic이다. 그러나 이를 최종 방법으로 고정하지 않는다. 이 실험으로 raw LLM order bias의 크기와 canonical prompt가 잃는 정보를 먼저 측정한 뒤, event-extractor allocator와 distillation 중 어느 방향이 실제 LLM semantic advantage를 보이는지 선택한다.

---

## 2026-07-30 — LLM event/role extraction과 equivariant credit allocation의 분리

### 핵심 가설

sparse global reward가 0인 transition에서도 LLM은 trajectory의 의미를 이용해 유의미한 사건을 식별할 수 있다. 반면 agent별 real-valued credit을 autoregressively 출력하게 하면 position bias, scale inconsistency, parser failure, agent label bias가 한꺼번에 credit signal로 들어온다.

따라서 LLM은 **credit 숫자 생성기**가 아니라 **semantic event/role extractor**로 쓰고, 숫자 credit은 별도의 grounded permutation-equivariant allocator가 내도록 분리한다.

### 제안 pipeline

```text
anonymized + canonicalized trajectory
    → LLM: structured event graph
    → verifier/grounder: trajectory에서 event 사실성 확인
    → equivariant allocator: event × agent records → per-agent dense credits
    → RL learner
```

1. **Input canonicalization.** LLM에 들어가는 agent block은 label-free canonical serialization을 사용한다. event/allocator 분리만으로는 LLM input의 agent-order bias를 제거할 수 없으므로 이 단계가 필요하다.
2. **LLM structured event graph.** LLM은 agent name이나 slot index가 아니라 time, object의 물리적 attribute, action, 상대 위치, 필요한 역할 수(cardinality)로 사건을 기술한다.
3. **Grounded verification.** 생성 event가 실제 state/action trajectory와 양립하는지 프로그램으로 확인한다. 확인할 수 없는 event는 버리거나 `unknown`으로 둔다.
4. **Equivariant allocation.** 검증된 role predicate와 실제 trajectory를 matching하여 실제 agent에 event credit을 배분한다. 모든 agent에 shared rule/head를 쓰고, tie matching은 동일 분배 또는 symmetric aggregation으로 처리한다.

### event schema의 초안

LLM output은 numeric credit 대신 아래처럼 제한된 JSON event를 목표로 한다.

```json
{
  "event_type": "cooperative_load",
  "time_span": [6, 9],
  "object": {"kind": "apple", "initial_position": [4, 5], "level": 2},
  "outcome": "completed",
  "roles": [
    {"name": "loader", "time": 9, "action": "load",
     "relation": "adjacent_to_object", "cardinality": 2},
    {"name": "approacher", "time_span": [6, 8],
     "relation": "moves_toward_object", "cardinality": 2}
  ],
  "causal_relation": "approacher_enables_loader"
}
```

`initial_position`, object type/level 등은 LBF에서 agent index보다 안정적인 physical anchor다. LLM은 “Alice가 잘했다”가 아니라 “이 apple에서 이 조건을 만족한 두 loader가 기여했다”를 말한다. event list 자체의 출력 순서는 allocator가 set으로 aggregate하므로 numeric credit에 영향을 주지 않게 한다.

### allocator의 두 가지 구현 후보

**A. Deterministic event compiler (첫 prototype)**

- verifier가 role predicate를 만족하는 agent들을 찾는다.
- event type/outcome에 대해 고정된 credit template을 적용한다.
  - successful cooperative load: 필요한 loader에게 event budget을 동등 분배.
  - verified approach: 해당 role을 만족한 agent에게 이전 window의 작은 positive budget 분배.
  - failed load/boundary hit: predicate를 만족한 agent에 small negative budget.
- matching이 여러 개이면 모든 valid matching의 allocation을 평균내거나 동등하게 나눈다.

이는 LLM의 숫자 scale 문제를 완전히 제거하고, allocator의 equivariance를 증명/단위검증하기 쉽다. 단, template가 과도하게 hand-crafted oracle이 되지 않도록 LLM 없는 heuristic baseline과 분리해야 한다.

**B. Learned equivariant allocator (장기)**

- 각 agent trajectory record와 검증된 event set을 shared encoder/Set Transformer 또는 SAQA로 처리한다.
- shared per-agent head가 $C_{i,t}$를 출력하고, global pooling head가 team consistency/return을 예측한다.
- LLM은 event type, causal link, role cardinality 같은 discrete semantic condition을 제공하고, 숫자 크기와 temporal spreading은 data에서 학습한다.

이 버전은 fixed template의 표현력 한계를 줄이지만, oracle event·heuristic event·LLM event를 각각 넣는 ablation으로 LLM event 자체의 가치를 검증해야 한다.

### 중요한 한계

- LLM의 event 내용 자체가 canonicalized input에 대해서도 틀리거나 불안정할 수 있다. allocator는 **같은 event graph에 대한 numeric symmetry**를 보장하지만, semantic hallucination을 자동으로 해결하지 않는다.
- 따라서 event grounding rate와 event-level precision/recall이 핵심 중간 지표다.
- shaped credit이 원래 task의 optimal policy를 보존한다는 보장은 별도 문제다. global reward anchoring, reward-scale calibration, potential-based shaping 가능성을 분리해서 검토한다.

### 필수 ablation

1. environment sparse reward only.
2. deterministic geometric/event heuristic (LLM 없음).
3. direct numeric LLM-MCA.
4. LLM event → allocator, grounding 없음.
5. LLM event → grounded deterministic allocator.
6. oracle event → 같은 allocator (event extraction upper bound).
7. LLM event → learned equivariant allocator.

이 비교가 있어야 성능 향상이 LLM의 event reasoning, grounding, allocator 구조 중 어디에서 왔는지 분리할 수 있다.

---

## 2026-07-30 — 우선순위 재설정: invariant event-based densification을 먼저 분리

### 결정

agent별 allocator의 설계는 아직 reward-function handcrafting, credit conservation, zero-reward shaping, policy invariance 문제가 얽혀 있다. 이를 event extraction과 동시에 해결하려 하지 않는다.

첫 번째 방법론 단위는 다음으로 좁힌다.

> **LLM이 sparse cooperative trajectory에서 검증 가능한 event graph를 추출하고, 이 event graph를 사용해 permutation-invariant team-level dense shaping signal 또는 global critic representation을 만드는가?**

이 단계에서는 team signal을 모든 agent에 공유해도 된다. 목표는 LLM이 reward 0 transition에서 cooperative progress/failure event를 유의미하게 찾는지, 그리고 그 semantic signal을 순열불변으로 만들 수 있는지 확인하는 것이다. agent별 dense credit/equivariant allocator는 event graph가 유효하다는 증거가 나온 뒤의 두 번째 문제로 둔다.

### world-centric serialization 가설

agent 목록을 `Alice, Bob, ...` 순서로 나열하는 대신, centralized critic이 보는 world state를 물리적 좌표/관계 중심으로 serialize한다.

```text
t=7
  cell (2,3): homogeneous robot, action=right
  cell (2,4): homogeneous robot, action=down
  cell (4,5): apple(level=2)
```

grid coordinate의 row-major order는 agent storage slot이 아니라 환경의 물리적 구조에 의해 정해진 canonical order다. 동질 agent의 identity를 swap해도 occupancy, position, action-at-position이 같은 world description은 변하지 않을 수 있다. **그러나 이것은 LLM 자체의 permutation invariance를 보장하지 않는다.** 이는 input에서 한 종류의 agent-index ordering을 제거하는 canonical encoding일 뿐이다. LLM은 여전히 cell line/token position, spatial scan order, output decoding order에 민감할 수 있다.

RWARE 등에는 grid serialization 대신 typed relational graph를 쓸 수 있다.

```text
nodes: anonymous agent-at-time, shelf, goal, event
edges: adjacent, carrying, moves-toward, loads, delivers, blocks
attributes: time, position, action, object type/state
```

LLM은 grid/graph에 기반해 time·position·object attribute에 anchored 된 event graph를 출력한다. event list 출력 순서는 이후 deterministic parser/graph builder가 set으로 aggregate할 수 있다. 다만 LLM이 input representation에 따라 **다른 event graph 자체를 생성하는 문제**는 남는다.

### 순열성과 graph 구조의 연결

- graph encoder/GNN/Set Transformer가 typed physical relation graph를 입력으로 받을 경우, graph node relabeling에 대해 invariant global head와 equivariant node head를 설계할 수 있다.
- 그러나 raw text LLM이 그 graph를 읽는 단계는 graph encoder가 아니다. Set-LLM식 attention/position modification, group-equivariant architecture, 또는 명시적인 symmetrization 없이는 해당 단계의 invariance를 주장할 수 없다.
- identity를 완전히 삭제한 grid encoding은 team-level event extraction에는 쓸 수 있지만, individual credit에는 agent trajectory track을 다시 연결해야 한다. 이 mapping이 순열동변적이라는 별도 설계·증명이 필요하다.

따라서 event graph는 LLM semantic extraction과 future equivariant allocator를 연결하는 intermediate representation(IR) 후보이지만, LLM input symmetry의 해법 자체로 간주하지 않는다.

### 다음 검증 질문

1. world-centric serialization이 agent-list serialization보다 raw LLM event output의 swap consistency를 경험적으로 높이는가? (보장으로 주장하지 않는다.)
2. zero-return trajectory에서 LLM event가 verified progress, cooperative preparation, failed collaboration, waste를 얼마나 정확히 찾는가?
3. event graph 기반 team shaping/global representation이 environment sparse reward보다 학습을 개선하는가?
4. 같은 graph를 LLM 없이 geometric rules로 구성했을 때보다 LLM이 제공하는 추가 정보가 있는가?

agent별 allocator 연구는 위 질문에 긍정적인 증거가 나온 뒤, event graph의 agent-event incidence를 입력으로 하는 equivariant node head로 진행한다.

---

## 2026-07-30 — 핵심 문제의 재정의: serialization이 아닌 LLM의 mixed set-sequence architecture

### 문제의 본질

agent 정보와 environment object 정보는 동등한 entity들의 집합인데, 표준 decoder-only LLM은 이를 하나의 token sequence로 받는다. 따라서 agent block, object block, output key가 sequence의 서로 다른 absolute/relative 위치와 causal context를 갖게 된다. 이 위치 차이가 raw LLM critic의 order variability를 만든다.

agent 이름을 익명화하거나 input order를 canonicalize하는 것은 이 문제를 근본적으로 없애지 않는다. 전자는 label bias 일부만 제거하고, 후자는 특정 serialization을 고정하는 preprocessor일 뿐이다. 물리 cell/graph serialization도 유용한 empirical baseline일 수 있으나 LLM 자체의 set symmetry를 보장하지 않는다.

### 중심 연구 목표

> **LLM critic이 fixed text와 temporal sequence는 보존하면서, homogeneous agent trajectory blocks 및 exchangeable object blocks를 set으로 처리하도록 attention mask, positional encoding, output head를 설계할 수 있는가?**

입력은 다음 mixed structure로 본다.

$$
x=\left(p,\;\{\tau_i\}_{i\in\mathcal A},\;\{o_j\}_{j\in\mathcal O}\right),
$$

여기서 $p$는 environment instruction/global reward 같은 ordered text, $\tau_i$는 agent $i$ 내부에서는 시간 순서가 있는 trajectory, $o_j$는 object record다. 요구 대칭군은 모든 entity에 대한 하나의 $S_N$이 아니라 homogeneous agent class 및 exchangeable object class 각각의 product group이다.

### Set-LLM에서 가져올 원리

Set-LLM은 (1) sequential position encoding 제거, (2) causal mask 제거/prefix attention 사용, (3) set position encoding(SetPE), (4) set attention mask(SetMask)를 결합해 set-item permutation invariance를 보장한다.

본 문제에서는 agent block 내부의 time/action token order는 보존해야 하지만, block 간 agent order는 제거해야 한다. 따라서 각 agent trajectory block이 같은 SetPE origin을 공유하고, block 내부에는 temporal position을 유지하는 mixed position scheme이 필요하다. object block에도 같은 원리를 적용한다.

### Set-LLM을 그대로 쓰면 부족한 이유

Set-LLM의 기본 목표는 set-text input에서 하나의 invariant answer를 내는 것이다. LLM-MCA에는 두 output이 필요하다.

- team value/event representation: agent/object permutation에 invariant.
- per-agent credit: agent permutation에 equivariant.

따라서 autoregressive하게 `agent_0_credit`, `agent_1_credit`을 순서대로 생성하는 것은 피해야 한다. 대신 input의 각 agent block에 연결된 query/readout token 또는 hidden state $h_i$를 만들고, 모든 agent에 공유된 numeric head를 병렬 적용한다.

$$
\{\tau_i\}\n+\xrightarrow{\text{set-aware LLM}} \{h_i\}
\xrightarrow{\text{shared head}} \{C_{i,t}\},
\qquad
\{h_i\}\xrightarrow{\text{invariant pooling}} V.
$$

이 구조에서 set-aware LLM layers가 block permutation-equivariant이고 shared credit head가 agent마다 같은 map을 적용하면 $C(\pi x)=\pi C(x)$가 된다. pooling global head는 $V(\pi x)=V(x)$를 만족한다.

### event extraction의 위치

LLM event extraction은 이 architecture의 첫 application/head가 될 수 있다. set-aware LLM이 invariant team event graph 또는 event representation을 생성하면, 이를 team-level sparse reward densification에 먼저 사용할 수 있다. 이후 agent-block hidden states와 event graph를 연결하는 equivariant credit head/allocator를 추가한다.

즉 연구의 우선순위는 다음과 같다.

1. mixed set-sequence LLM backbone의 symmetry 보장 및 adaptation/fine-tuning.
2. invariant event extraction과 verified team-level densification.
3. equivariant individual credit head/allocator.

all-permutation average, canonical serialization, world-centric text는 1번의 필요성과 효과를 검증하는 baseline/diagnostic으로 둔다.

---

## 2026-07-30 — architecture 후보: PEEL

### 제안

Set-LLM을 그대로 적용하는 대신, MARL centralized critic의 mixed set-sequence input과 two-level output에 맞춘 **PEEL (작업명)**을 고안한다.

```text
environment instruction / global reward text (ordered sequence)
agent trajectory blocks                         (set of sequences)
object trajectory blocks                        (set of sequences)
     ↓ SetPE + relational set attention mask
agent/block representations + global latent tokens
     ├─ invariant global/event head
     └─ parallel shared per-agent credit head
```

이 구조의 기여 후보는 세 부분의 결합이다.

1. **Mixed positional encoding.** agent/object block 내부에는 time/action order를 표현하는 temporal position을 유지한다. 그러나 동등한 block들의 시작 position은 SetPE처럼 공유하여 block order가 embedding에 들어가지 않게 한다.
2. **Relational set attention.** SetMask를 단순히 agent block 간 완전 차단으로 쓰면 협업 관계를 읽기 어렵다. 따라서 $K$개의 shared global latent/event token을 둔다. 모든 global token은 agent/object block 전체를 대칭적으로 읽어 invariant context를 만들고, 각 agent block/query는 자신의 local history와 global latent를 읽는다. block 간 정보 교환은 이 symmetric bridge를 통해 이뤄진다.
3. **Non-autoregressive equivariant readout.** `agent_0_credit`, `agent_1_credit`을 순서대로 text generation하지 않는다. 각 input agent block에 묶인 query/hidden state $h_i$에 동일한 numeric head를 병렬 적용해 $C_{i,t}$를 낸다. global pooling/readout은 $V$ 또는 team-level event representation을 낸다.

### mask/readout의 개념적 조건

agent block permutation $P$에 대해 backbone은

$$
H(Px)=P H(x), \qquad G(Px)=G(x)
$$

를 만족해야 한다. 여기서 $H=\{h_i\}$는 agent block representation, $G$는 global latent representation이다. 공유 credit head $\psi$와 global head $\rho$를 사용하면

$$
C_i=\psi(h_i,G), \qquad V=\rho(G)
$$

이므로 $C(Px)=PC(x)$ 및 $V(Px)=V(x)$가 나온다. 이 증명은 attention mask, SetPE, global latent의 connectivity가 permutation action과 commute한다는 것을 보여야 한다.

### event extraction head

event를 autoregressive JSON list로 출력하면 event 순서도 다시 nuisance가 된다. 대안은 $K_e$개의 learned event query/slot을 병렬로 두고 event type, time span, outcome, object relation을 예측하는 것이다. event slot의 순서는 set prediction loss(예: bipartite matching)로 무시한다. object pointer는 object node에 대한 equivariant attention distribution으로 표현한다.

이 head는 우선 invariant team-level event representation 또는 verified event proposal을 만들고 sparse reward densification에 사용한다. agent별 credit head는 그 다음 단계다.

### 왜 pure Set Transformer와 다른가

pure Set Transformer만 쓰면 numerical MARL model이 되며 LLM의 pretrained language/causal reasoning을 사용했다는 장점이 약해질 수 있다. 이 architecture는 environment rule text, novel task description, event semantics를 LLM backbone이 처리하게 유지하면서, structured entity axis에만 symmetry를 강제한다. LLM이 정말 필요한지는 rule-text 변화, unseen layout/task, zero-reward event detection에서 pure Set Transformer/GNN baseline보다 나은지로 검증한다.

### 학습 및 baseline

- Set-LLM mask/position modification은 pretrained causal LLM의 inference-only wrapper가 아니라 adaptation/fine-tuning을 요구하는 방법으로 본다.
- loss는 critic/event supervision 외에 permutation consistency audit/loss를 보조적으로 둔다. architecture가 맞다면 이 loss는 symmetry를 학습시키는 주 수단이 아니라 implementation/finite-precision drift를 점검하는 역할이다.
- 비교: raw causal LLM-MCA, canonical serialization, permutation average diagnostic, Set-LLM without relational bridge, pure Set Transformer/GNN, 제안 model.

---

## 2026-07-30 — 긴 trajectory를 다루는 LLM critic 문헌과 temporal hierarchy 필요성

### 문헌 관찰

긴 trajectory를 LLM에 넣어 per-timestep dense numeric credit을 직접 생성하는 문제에 대해, 확인한 LLM critic/long-horizon agent 논문들은 공통된 표준 해법을 갖고 있지 않다.

| 방법 | trajectory 처리 | critic output | 시사점 |
|---|---|---|---|
| LLM-MCA | full trajectory 또는 trajectory batch를 centralized critic에 제공 | agent별 timestep numeric credit | long-context/chunking 정책을 명시적으로 해결하지 않음; 현재 재현에서 드러난 병목과 동일 |
| [Prospector](https://aclanthology.org/2024.findings-emnlp.879/) | 하나의 complete trajectory를 fixed context budget 안에서 score | expected total reward로 trajectory ranking | critic max context length를 1024로 둠; dense credit 대신 episodic ranking으로 문제를 축소 |
| [LogicGuard](https://arxiv.org/abs/2507.03293) | full trajectory를 offline/periodically 분석 | compact LTL constraints | 긴 history를 계속 numeric array로 반환하지 않고, 압축된 symbolic feedback으로 전환 |
| [LLM-ALSO](https://arxiv.org/abs/2605.29293) | sparse-return metrics와 compact behavior evidence를 stage별 진단 | shaping configuration proposal | raw full trajectory 대신 summary/short-horizon validation을 사용 |
| [KLong](https://arxiv.org/abs/2602.17547) | early context를 보존하고 later context를 progressive truncation; overlapping sub-trajectories | SFT/RL training sample | LLM actor 연구이지만 overlap chunking의 직접적 참고점 |
| [TRACE](https://arxiv.org/abs/2607.13988) | 모든 prefix transition의 progress를 batch forward로 평가 | turn-level reward | frozen reference model의 gold-answer predictability를 사용; generic MARL에는 gold answer가 없지만 prefix value를 batch 처리한다는 계산적 아이디어는 유용 |
| [SALT](https://arxiv.org/abs/2510.20022) | 같은 task의 여러 trajectory를 graph로 구성 | step-level advantage | LLM critic은 아니지만 raw long sequence 대신 trajectory graph에서 step quality를 계산 |

### 해석

LLM-MCA는 full episode를 직접 다루는 드문 사례이지만, context length·array generation·long-horizon temporal reasoning의 해법을 공개적으로 분리하지 않는다. Prospector도 full trajectory의 scalar ranking을 위해 context length를 1024로 제한한다. 나머지 최근 연구는 장기 trajectory를 다음 중 하나로 바꾼다.

1. episodic scalar/ranking;
2. compact symbolic constraint 또는 summary;
3. overlap된 segment;
4. prefix/state-value sequence;
5. trajectory graph.

따라서 proposed PEEL도 agent/object symmetry만 해결하고 full $T$ trajectory를 무제한 attention에 넣는 구조로 끝나면 실용성이 부족하다.

### 확장 가설 — hierarchical temporal set critic

trajectory를 $W$ 길이의 overlapping temporal chunk로 나눈다. 각 chunk에서는 agent blocks가 set이고 chunk 내부 time은 sequence다. set-aware local backbone은 다음을 출력한다.

$$
z^{(m)}_{\mathrm{global}},\qquad \{h^{(m)}_{i,t}\}_{i,t},\qquad \text{verified event proposals}^{(m)}.
$$

chunk summary $z^{(1)},z^{(2)},\ldots$는 time-ordered sequence 또는 trajectory graph node로 상위-level critic에 들어간다. 상위 critic은 long-range causal relation을 요약한 global temporal context를 각 local chunk로 되돌려준다. 최종 credit은 local agent-time representation과 global temporal context를 함께 읽어 계산한다.

```text
agent/object set × short temporal chunk
  → local PEEL critic
  → chunk event/global summaries (time sequence or graph)
  → temporal hierarchy / trajectory graph critic
  → global temporal context
  → local parallel agent credit heads
```

이 구조에서 agent permutation은 모든 chunk의 agent axis에 동일하게 작용하므로 local/upper-level agent representation은 equivariant, chunk/global summary는 invariant하게 유지할 수 있다. chunk order는 실제 time order이므로 permutation 대상으로 취급하지 않는다.

### 우선 실험

1. full trajectory, disjoint chunk, overlapping chunk의 event/credit consistency 및 cost를 비교한다.
2. local chunk만 본 model과 upper-level temporal summary를 추가한 model을 비교해 delayed reward/long causal dependency에서의 차이를 본다.
3. chunk 경계에서 approach event와 eventual cooperative load의 credit이 끊기는지를 event-level metric으로 측정한다.

---

## 2026-08-03 — 첫 permutation-safe event shaping control 결과

이 control의 source는 현재 공유 repository에서 제외하고 local archive에 보관한다. 아래 내용은 후속 allocator 설계의 판단 근거를 남기기 위한 과거 기록이다.

LLM 호출 없이 구조적 대칭만 분리해 검증하는 `event_shaping` control을 추가했다. LBF global state에서 남아 있는 각 food에 대해, **동등한 agent들의 unordered set** 중 food level을 만족하는 subset과 서로 다른 loading-adjacent cell의 minimum joint travel cost를 exact enumeration으로 계산한다. 이 remaining-task cost의 potential difference를 team shaping으로 만들고 모든 agent에 동일하게 나눈다.

따라서 team shaping scalar는 agent reindex에 invariant이고, output credit tensor는

$$
C(Px)=P C(x)
$$

를 정확히 만족한다. state/action agent axis를 swap하는 unit test도 통과했다.

### smoke result

`Foraging-8x8-2p-2f-coop-v3`, 80 iterations, 8 episodes/iteration, 20 update steps/iteration, 25-step horizon에서 다음 three short runs를 수행했다.

| critic | seed / scale | final eval return |
|---|---:|---:|
| equal-split true reward RNN-IQL | 0 | 0.00 |
| invariant team-event shaping | 0 / 0.10 | 0.00 |
| invariant team-event shaping | 1 / 0.25 | 0.00 |

event shaping은 sparse reward 이전에도 nonzero dense signal을 일관되게 만들었다. 그러나 team signal을 agent들에게 균등 공유하는 것만으로는 cooperative collection policy가 이 짧은 budget에서 학습되지 않았다. 이는 단순한 global densification만으로 structural credit assignment가 해결되지는 않으며, 다음 단계에서 agent-conditional but equivariant relational readout/allocator를 검토해야 한다는 evidence다. 이 결과는 성능 결론이 아니라 early smoke failure이며, 더 긴 multi-seed experiment 전에 shaping scale, terminal convention, and agent-specific readout을 분리하는 진단이 필요하다.

### follow-up control — Shapley structural allocator

equal sharing의 직접적인 다음 control로 exact small-$N$ Shapley allocator를 구현했다. characteristic value는 하나의 global remaining-task cost reduction이며, 모든 agent subset $S$에 대해 $v(S)$를 계산하고 agent $i$에는 $v(S\cup\{i\})-v(S)$의 Shapley-weighted average를 준다. 따라서 allocator가 새 개별 reward function을 정의하지 않고 one team potential을 분배한다. 계산량은 $O(2^N)$이므로 LBF small-$N$ diagnostic 전용이다. `shapley_allocation(Px)=P\,shapley_allocation(x)` unit test를 통과했고, 160-iteration LBF two-seed smoke를 시작했다. 이 control의 목적은 성능을 주장하는 것이 아니라 **어떤 agent-conditional allocation이 필요한지**와 learned relational allocator의 supervision/inductive bias 후보를 분리하는 것이다.

### future metric note

최종 평가에는 return 외에 permutation robustness metric을 둔다. 동일 trajectory $x$와 random agent permutation $P$에 대해 team output drift $|V(Px)-V(x)|$, inverse-mapped credit drift $\lVert P^{-1}C(Px)-C(x)\rVert$, 그리고 tolerance 내 consistency rate를 측정한다. 마지막 지표는 Set-LLM의 adversarial accuracy와 유사한 “permutation adversarial consistency”로 사용할 수 있다.

---

## 2026-08-03 — event representation은 LBF taxonomy에 고정하지 않는다

`approach/load/apple` 같은 LBF-specific event label을 architecture의 입력·출력 contract로 삼지 않는다. 환경마다 필요한 최소 adapter는 raw observation을 다음 generic transition graph로 바꾸는 부분이다.

```text
task/rule text
state entities:     class + numeric/categorical attributes (+ optional geometry)
relations:          generic typed edges (visibility, distance, ownership, communication, ...)
joint action:       per-agent action tokens
transition delta:   changed entity attributes / created-deleted entities / termination
global reward:      observed team signal
```

LLM event module은 $K$개의 **exchangeable latent event slots**를 병렬로 출력한다. 각 slot은 pre-defined event name 대신 다음 structure를 갖는다.

```text
event embedding z_e
time/span attention over trajectory
equivariant participant pointer over agents
equivariant related-entity pointer over objects
signed/uncertainty-aware event impact
```

event slots 자체의 order도 set-prediction/matching loss로 nuisance로 처리한다. event type은 필요할 때만 text-conditioned open-vocabulary label 또는 learned prototype으로 붙이며, LBF-specific supervised labels는 debugging/evaluation auxiliary로만 사용한다.

학습의 generic signal은 named event label보다 transition-delta reconstruction, next-state/reward/termination prediction, contrastive future prediction, TD objective가 된다. allocator는 event impact와 participant/time pointer를 credit으로 변환하므로 task-specific event rule을 직접 작성하지 않는다. LBF/Climbing/RWARE 등은 서로 다른 `EnvironmentAdapter`만 제공하고, PEEL/event-slot/allocator는 공유하는 방향을 목표로 한다.

---

## 2026-08-03 — 구현 순서: allocator/RL 이전에 event representation을 검증

full RL model을 바로 만들거나 rule-based allocator를 더 튜닝하지 않는다. 먼저 structured, permutation-safe LLM이 generic transition event를 추출할 수 있는가를 독립 과제로 검증한다. Latent event만 제안하고 return만 보면 해석과 실패 원인을 알 수 없으므로, event quality의 observable proxy를 함께 둔다.

### Stage 0 — minimal contract

모든 environment adapter의 contract는 `(task text, entities/relations at s_t, joint actions, transition delta, global reward/done)`로 한정한다. model은 $K$ event slots의 `(event embedding, time attention, agent pointer, entity pointer, impact)`를 반환한다. allocator와 RL policy는 아직 붙이지 않는다.

### Stage 1 — event extraction pretext/evaluation

환경별 named event label이 아니라 다음 generic objective로 event representation을 확인한다.

1. **transition-delta coverage:** event slots만 보고 changed entity attributes, create/delete, termination을 재구성할 수 있는가.
2. **grounding:** event slot의 agent/entity pointer가 실제 state change 및 joint action에 관련된 node를 가리키는가.
3. **predictive utility:** event summary가 raw state보다 적은 token으로 next-state/reward/return prefix를 예측하는가.
4. **permutation robustness:** $E(Px)=E(x)$ (slot matching 후)와 agent/entity pointer의 inverse-permuted drift를 측정한다.

LBF의 food disappearance, Climbing의 joint action/payoff, RWARE의 object movement/delivery는 named training target이 아니라 transition delta가 자동으로 제공하는 diagnostic example이다.

### Stage 2 — architecture ablation

같은 dataset/objective로 raw sequential LLM, canonical serialization, Set-LLM, PEEL을 비교한다. 이 단계에서 LLM backbone의 rule-text/generalization benefit과 set structure의 symmetry benefit을 분리한다. model은 우선 small HuggingFace causal LLM + LoRA + parallel event heads로 시작한다.

### Stage 3 — allocator, 그 다음 RL

event representation이 Stage 1의 delta/grounding/prediction 및 permutation metric에서 raw LLM보다 의미 있게 낫거나 안정적일 때 event-conditioned learned equivariant allocator를 붙인다. 마지막에만 dense credit을 decentralized policy training에 연결한다. 이렇게 하면 RL 성능 실패가 event extraction, allocation, policy optimization 중 어디에서 왔는지 분리할 수 있다.

---

## 2026-08-03 — 제안 model을 드러내는 environment/evaluation suite

단일 environment return은 “구조가 순열 안전한가”와 “RL이 잘 학습됐는가”를 섞어 버린다. 논문 실험은 다음 세 축으로 분리한다.

| 축 | primary setting | 검증할 주장 |
|---|---|---|
| exact permutation stress | 같은 recorded trajectory의 agent/object reserialization | raw LLM의 position bias vs proposed exact invariance/equivariance |
| cooperative credit/densification | LBF cooperative, 3–4 homogeneous agents | delayed reward와 event-to-agent credit의 utility |
| scale/transfer | RWARE tiny/small, 4–6 homogeneous agents | 더 많은 agent/object와 다른 transition graph에서도 같은 architecture가 작동 |

### 1. permutation stress protocol (모든 environment에 공통)

새 environment를 만들 필요 없이 recorded trajectory $x$를 만든 뒤, 매 trajectory마다 $K$개의 random agent permutation과 object permutation을 input serialization에 적용한다. model output은 original indexing으로 inverse-map하여 비교한다.

$$
\Delta_V=|V(Px)-V(x)|, \qquad
\Delta_C=\lVert P^{-1}C(Px)-C(x)\rVert.
$$

pointer/event output도 동일하게 inverse-map하고 slot matching 후 비교한다. raw sequential LLM, canonical serialization, Set-LLM, proposed relational model을 같은 trajectory에서 비교한다. $N=2,3,4,6$ scaling과 adversarially selected permutations를 보고, proposed method가 inference-time permutation ensemble 없이 near-zero drift를 내는 것이 핵심 figure가 된다.

### 2. LBF는 primary cooperative benchmark

LBF는 homogeneous agent, interchangeable food object, spatial relation, level-threshold cooperation, delayed sparse collection reward를 동시에 제공한다. primary training은 2-agent 설정만으로 끝내지 않고 3/4-agent cooperative variants를 포함한다. agent 수가 커져 order의 possible permutations가 증가할수록 raw serialization bias를 stress할 수 있다. event/credit evaluation에는 food disappearance, successful/failed joint loading, and relevant agents가 transition delta로 자동 관찰된다.

### 3. RWARE는 generalization/scale benchmark

RWARE는 agent 수를 4–6 이상으로 키울 수 있고 robot/shelf/goal/requested-shelf라는 typed entity graph와 긴 sparse delivery event를 제공한다. LBF와 다른 object dynamics에서 같은 adapter contract와 event heads가 동작하는지를 보인다. primary LBF result가 선행되어야 하며, RWARE는 return 비교보다 event representation/pointer/permutation metric 및 transfer evidence에 우선 사용한다. 현재 wrapper에는 `next_state` snapshot을 추가해야 transition-delta event target을 만들 수 있다.

### 제외/보조 환경

Climbing은 두 agent가 payoff matrix의 row/column이라는 본질적으로 asymmetric role을 가지므로 homogeneous-agent permutation claim의 primary evidence로 쓰지 않는다. Overcooked는 compositional long-horizon event demonstration으로 유망하지만 external dependency와 두-agent 제한이 있으므로, LBF/RWARE prototype 이후 optional third domain으로 둔다.

### randomized reserialization protocol

모든 critic call의 environment-side record는 original environment indexing을 유지한다. LLM/model input 직전에만 independent random permutations $P_A$ (agents), $P_O$ (homogeneous objects)를 sampling하여 blocks를 재배열한다. Raw text baseline에는 mapping을 보존하기 위해 opaque stable tags를 block과 output key에 함께 붙인다.

```text
original environment map: agent 0 -> tag A, agent 1 -> tag B, agent 2 -> tag C
one critic call serialization: {tag B: [...], tag C: [...], tag A: [...]}
raw model output:             {tag B: credit_B, tag C: credit_C, tag A: credit_A}
environment wrapper:          tag-to-original inverse scatter
```

제안 model에서는 tag/index가 learned embedding이 아니라 mask/scatter bookkeeping에만 존재한다. Raw sequential baseline에서 tag는 text token이므로 remaining position/content bias를 그대로 측정한다.

평가는 (a) fixed canonical order train/test, (b) fixed-order train + random-order test, (c) random-order augmentation train/test, (d) proposed architecture train/test의 네 condition을 비교한다. 동일 trajectory를 $K$회 reserialize해 global drift, inverse-mapped credit/pointer drift, top-contributor agreement, event-slot matching consistency, parse-validity를 보고, $N\le4$에서는 all permutation diagnostic도 추가한다. RL condition에서는 policy/environment agent order는 바꾸지 않고 critic input만 call마다 shuffle한 뒤 credit을 inverse-scatter하여, performance drop의 원인을 critic order sensitivity로 한정한다.

### implementation note from first Qwen forward smoke

Frozen Qwen with custom SetPE/4D mask가 forward path를 실행하는 것은 확인했다. 다만 bf16 7B Qwen의 raw unnormalized pointer logits에는 reserialization 후 non-negligible numeric drift가 관찰되었다. 이는 architecture proof와 mixed-precision finite-precision behavior가 다름을 보여준다. 다음 audit은 probability/normalized output, fp32 small-backbone reference, and bf16 Qwen을 모두 보고 absolute and relative drift를 분리해야 한다. 큰 model에서 claimed robustness는 “theoretical symmetry”만이 아니라 이 randomized protocol에서의 empirical drift로 주장한다.

### 2026-08-03 implementation smoke results

1. **Native PEEL reference.** Generic LBF transition adapter + parallel event slots + agent/object pointer/delta heads를 구현했다. LBF 3-agent and 4-agent random-transition delta pretraining에서 agent delta MSE가 각각 약 `0.02`, `0.01` 수준으로 감소했다. Random agent/object reserialization 뒤 inverse-map한 output drift는 $0$–$2.3\times10^{-5}$였다. 이는 pretrained LLM result가 아니라 architecture/interface symmetry reference다.
2. **Qwen forward + LoRA feasibility.** Qwen2.5-7B-Instruct에 structured numeric embeddings, time-only SetPE, custom 4D relation mask, and parallel event heads를 연결했다. Q/K/V/O rank-2 LoRA one-step backward pass는 A10 24GB에서 peak 약 `14.6 GB`로 실행됐다. Raw 3584-dim dot-product pointer는 bf16 reorder noise가 softmax assignment까지 증폭될 수 있었고, cosine-normalized bounded pointer로 바꾼 integrated LoRA smoke에서는 participant probability drift가 관찰된 checkpoints에서 최대 약 `0.002`로 안정화됐다. 이 값은 small smoke이며 formal benchmark value가 아니다.
3. **Raw LLM-MCA audit harness.** raw text input의 agent/object blocks를 tag와 함께 재배열하고 parser output을 inverse-scatter하는 code를 추가했다. Random sparse rollout에서는 Qwen이 all-zero credit을 내어 order test 자체가 무의미해지는 failure를 확인했다. A hand-constructed successful 3-agent cooperative-load transition에서는 Qwen credit `[1,1,0]`이 두 cyclic reorder에도 identical after inverse-map이었다. 이 one-step easy case는 adversarial test가 아니므로 raw baseline robustness evidence로 쓰지 않는다; delayed/multi-object/success-failure contrast trajectory suite가 필요하다.

### 2026-08-03 — raw LLM-MCA의 multi-event permutation failure 확인

한-step cooperative load만으로 raw sequential critic의 순서 민감성을 판단할 수 없어서, 같은 물리 trajectory를 agent cyclic permutation과 food reverse permutation으로 재직렬화하는 세 개의 hand-constructed probe를 만들었다. 모든 output은 stable tag로 원 agent 축에 inverse-scatter한 뒤 비교했다. 모델은 Qwen2.5-7B-Instruct 기반 기존 LLM-MCA critic이며, prompt/rule/credit parser는 바꾸지 않았다.

| probe | canonical credit | reordered 결과 요약 | max agent drift | max team-total drift |
|---|---|---|---:|---:|
| one-step cooperative load | `[1], [1], [0]` | 두 cyclic reorder 모두 동일 | `0` | `0` |
| delayed 3-step cooperation | `[0,0,1], [0,0,1], [0,0,0]` | 한 reorder는 all-zero, 다른 reorder는 두 agent에 `[0,0,2]` | `1` | `2` |
| two-object sequence | `[0,0], [1,1], [0,0]` | reordered output이 agent/time 배분과 총량 모두 변경 | `2` | `3` |

따라서 one-step 대칭 사례의 zero drift는 raw LLM의 일반적 robustness가 아니라 쉬운 prompt의 우연한 결과다. delayed/multi-object context에서는 **입력 순서만 바꿔도 individual credit뿐 아니라 output credit의 팀 합계까지 바뀐다.** 이는 논문에서 raw LLM-MCA / canonical serialization / random-order augmentation / proposed PEEL critic을 비교해야 하는 직접적인 empirical motivation이다.

동시에 cosine-normalized pointer를 쓴 Qwen PEEL encoder는 128-step LoRA delta smoke에서 participant probability drift가 `0–0.00391` 범위로 유지됐다. 이는 아직 random transition-delta objective의 numerical feasibility result일 뿐, semantic event extraction 또는 RL improvement evidence가 아니다.

### 2026-08-03 — exhaustive permutation audit 및 자동 experiment queue

각 handcrafted probe에 대해 identity를 제외한 agent ordering 5개와 object ordering 2개, 총 10개 reserialization을 모두 평가했다. delayed cooperation의 raw LLM-MCA max agent drift는 `1.0`, mean drift는 `0.9`, max team-total drift는 `2.0`이었다. two-object sequence에서는 각각 `2.0`, `1.05`, `3.0`이었다. 따라서 해당 baseline은 일부 adversarial permutation에서만 실패하는 것이 아니라, small-$N$ exhaustive probe의 대부분에서 credit이 변한다.

반복 실행을 위해 autopilot controller와 queue를 한때 사용했다. 현재는 공유 repository를 간결하게 유지하기 위해 해당 일회성 orchestration source를 local archive로 옮겼다. 완료/실패와 generic metric extraction을 자동화하더라도, research direction을 바꾸는 allocator/RL 단계는 review gate 뒤에만 새 실험으로 진행하고 의미 있는 결과만 이 문서에 반영한다는 원칙은 유지한다.

### 2026-08-03 — 동일 adversarial probe에서 Qwen PEEL의 finite-precision audit

Raw LLM-MCA와 같은 delayed 및 two-object handcrafted trajectory에 대해, proposed Qwen PEEL encoder의 agent/object block을 exhaustive하게 재직렬화했다. Delayed 사례는 3 timestep × 10 permutation, two-object 사례는 2 timestep × 10 permutation으로 평가했다. Cosine-normalized participant/object pointer probability의 최대 inverse-mapped drift는 각각 `0.001953125`, `0.00390625`이었다. 같은 probe에서 raw LLM-MCA의 agent credit max drift는 각각 `1.0`, `2.0`이고 team-total drift는 `2.0`, `3.0`이었다.

이 비교는 proposed model의 event/credit 값이 의미 있다는 증거가 아니다. 현재 Qwen event head는 random/delta smoke만 거친 상태이므로, reward/delta scalar의 BF16 drift와 semantic quality를 성능 지표로 해석해서는 안 된다. 다만 raw sequential prompt가 크게 달라지는 adversarial reserialization에서 **parallel structured participant distribution이 수치적 noise 수준으로 유지됨**을 확인한 architecture-level result다. 다음 병목은 symmetry가 아니라 targeted non-zero event data를 이용한 grounding/predictive utility 검증이다.

### 2026-08-03 — 기본 attention graph를 complete structured set으로 단순화

초기 prototype의 radius-based agent–object edge와 inactive-object gate는 순열 불변/동변성을 위한 필요조건이 아니며, global critic의 비국소 object reasoning을 불필요하게 제한한다는 점을 확인했다. 기본 mask는 rule text 내부의 causal order와 `structured node -> rule text` 방향만 유지하고, agent/object/event를 포함한 모든 structured node 사이에는 complete attention을 허용하도록 변경했다.

따라서 순열성과 직접 관련된 mask 조건은 block 순번을 이용하지 않고 structured subgraph가 순열에 대해 함께 재배열된다는 점뿐이다. Geometry/visibility/task relation에 의한 sparse edge는 primary architecture가 아니라 이후 ablation으로 둔다.

### 2026-08-04 — 이론적 주장 범위와 event–allocator 역할 분리

제안 architecture만으로 downstream RL return이 반드시 더 높다고 일반적으로 증명할 수는 없다. Return은 environment, policy optimization, data coverage, event representation의 semantic quality에 의존한다. 대신 논문의 이론적 중심 주장은 다음으로 한정한다.

1. Shared encoder, block-order-independent SetPE, permutation-compatible mask, shared parallel head를 만족하면 team output은 invariant, agent output은 equivariant하다.
2. Exchangeable ground-truth credit $C^\star$에 대해 Reynolds/group symmetrized predictor $\bar C(x)=|G|^{-1}\sum_{\pi\in G}\pi^{-1}C_0(\pi x)$는 squared loss의 convexity에 의해 permuted raw predictor의 평균 risk보다 나쁘지 않다.
3. 제안 one-call architecture는 이 expensive group average를 직접 실행하는 방법이 아니라, 그 symmetry-respecting function class를 구조적으로 parameterize하는 방법이다.

LLM의 역할은 autoregressive JSON scalar credit 생성이 아니라 rule-conditioned event/role/participant/object relation extraction으로 둔다. Numeric credit은 event impact와 grounded pointer를 입력으로 받는 별도 shared equivariant allocator가 계산한다. 따라서 raw LLM-MCA가 한 출력에 섞는 natural-language reasoning, agent correspondence, numeric calibration, conservation, JSON validity를 두 module로 분리한다.

Stage 0에서 LLM-backed backbone의 외부 interface는 event set과 training auxiliary prediction으로 제한한다. 실질적 allocator 입력은 event tensor $\mathcal E=\{z_e,\alpha_e,p^A_e,p^O_e,v_e\}$뿐 아니라 각 agent-time contextual representation $H^A=\{h_{i,t}\}$도 포함하는 것이 적절하다. $H^A$는 user-facing event가 아니라 allocator가 event impact를 현재 agent state와 결합하기 위한 equivariant latent state다. 이후 allocator는 $C=A_\phi(\mathcal E,H^A)$를 계산한다.

### 2026-08-04 — 1차 연구 scope는 agent 순열성으로 한정

Agent와 object 모두 prompt sequence에서 position bias를 받을 수 있지만, 두 symmetry 축을 동등한 주제로 전개하면 핵심 문제인 homogeneous agent credit correspondence가 흐려진다. 따라서 formal claim, adversarial metric, architecture proof, main experiment는 agent permutation $P_A$에 우선 한정한다. Object는 event grounding과 critic context에 필요한 state/environment information으로 유지하며, object order robustness는 후속 extension 또는 ablation으로 둔다.

현재 implementation이 object node에도 set-style encoding/pointer를 사용하는 것은 agent-order robustness와 양립하는 forward-compatible choice다. 그러나 논문 본문에서는 object permutation invariance/equivariance를 보장하거나 실증한다는 주장을 하지 않는다. Agent reserialization을 할 때 object serialization은 고정하고, agent output만 original environment axis로 inverse-scatter하여 평가한다.

### 2026-08-05 — 현재 structured event extractor의 agent 순열 보장 원리

현재 주장하는 것은 **agent feature 행의 순열**에 대한 구조적 성질이다. Agent feature matrix를 $A\in\mathbb{R}^{N\times d}$, agent 순서를 바꾸는 permutation matrix를 $P$로 쓰면 재직렬화된 입력은 $A'=PA$이다. Agent 이외의 environment state, task rule, transition context는 하나의 고정 문맥 $\xi$로 묶는다. 이 절에서는 그 문맥 내부 구조나 순열성을 논하지 않는다.

$$
V(PA,\xi)=V(A,\xi),\qquad
Z(PA,\xi)=Z(A,\xi),\qquad
C(PA,\xi)=P C(A,\xi).
$$

여기서 $V$는 global/team scalar, $Z$는 event slot set, $C$는 agent-axis output (agent pointer, agent transition prediction, 이후 allocator credit)을 뜻한다. 이는 학습 데이터에 우연히 나타나기를 기대하는 성질이 아니라, 다음 세 설계 선택에서 나오는 architecture-level property다.

#### 1. Agent index position을 representation에서 제거한다

일반 decoder prompt에서는 첫 agent와 마지막 agent가 다른 token position/RoPE position을 받는다. 이 차이가 동일한 agent state라도 prompt 순서에 따라 달라지는 직접 원인이다. Structured agent node에는 agent-specific absolute position이나 `agent_0` index embedding을 넣지 않는다. 모든 agent는 동일한 agent type embedding과 동일한 SetPE position을 받고, 오직 해당 agent의 state/action/role/ability feature만 받는다. 따라서 $a_i$와 $a_j$를 교환하면 node representation도 내용과 함께 행 위치만 교환된다.

Agent의 능력이나 역할은 제거하는 것이 아니라 node feature의 일부로 넣는다. 예를 들어 ability vector $u_i$를 $a_i$에 포함하면, 순열 시 $u_i$도 agent state와 함께 이동하므로 동변성은 유지된다. 반대로 능력을 `agent_0 is strong`처럼 agent index에 연결한 text serialization으로 넣으면 보장이 깨진다.

#### 2. Structured attention mask가 agent 순번을 구별하지 않는다

Rule text 내부에는 자연어 순서를 위해 causal mask를 유지한다. Agent node와 event query가 참여하는 structured computation에는 causal order나 agent-index-specific edge를 두지 않고 complete attention을 적용한다. Structured token 전체에 대해 agent 행만 재배열하는 확장 permutation을 $\widetilde P$라 하면, mask와 type/position encoding은 $\widetilde P$와 교환 가능하도록 구성된다.

Self-attention은 개념적으로 다음과 같다.

$$
\operatorname{Attn}(X)=
\operatorname{softmax}\left(
\frac{(XW_Q)(XW_K)^\top}{\sqrt d}+M
\right)XW_V.
$$

이때 현재 mask/SetPE 조건에서는

$$
\operatorname{Attn}(\widetilde P X)
=\widetilde P\operatorname{Attn}(X).
$$

이다. 즉 self-attention은 agent 내용을 새 position에 맞춰 다르게 해석하지 않고, 입력 행이 이동한 만큼 해당 representation도 이동시킨다. Feed-forward layer, residual, layer norm은 token-wise shared operation이므로 이 성질을 보존한다. 여러 layer를 쌓아도 structured encoder는 agent 축에 대해 equivariant하다.

#### 3. Output head가 agent axis를 보존하는가, 집계하는가에 따라 성질이 갈린다

Agent별 transition head는 같은 shared MLP를 모든 agent node에 적용한다.

$$
\widehat\Delta_i=h_\Delta(h_i),
\qquad
\widehat\Delta(PA,\xi)=P\widehat\Delta(A,\xi).
$$

따라서 `predicted_agent_delta`는 agent 순열 동변이다. Event slot $z_k$와 agent representation $h_i$ 사이의 pointer score도 shared pairwise function으로 계산한다.

$$
s_{k,i}=\frac{(W_Qz_k)^\top(W_Kh_i)}{\sqrt d},
\qquad
s_k(PA,\xi)=P s_k(A,\xi).
$$

Softmax는 agent 축 원소의 순서를 바꿀 뿐이므로 `agent_pointer_logits` 및 agent pointer probability 역시 동변이다. JSON inspection의 `top_agent`는 이 pointer distribution의 argmax를 사람이 읽기 좋게 표시한 값일 뿐이다. 물리적으로 같은 agent가 input에서 다른 row로 이동하면 `top_agent.index`도 그 row 이동을 따라야 한다. Exact tie에서는 argmax index가 불안정할 수 있으므로, 평가는 top-1 index보다 inverse-mapped full probability drift를 우선 사용한다.

반대로 team reward는 agent representations를 permutation-invariant pooling으로 집계한 global latent에서 계산한다.

$$
g=\operatorname{Pool}(\{h_1,\ldots,h_N\}),
\qquad
\widehat r=h_r(g),
\qquad
\operatorname{Pool}(PH)=\operatorname{Pool}(H).
$$

따라서 `predicted_team_reward`는 invariant하다. Event query $e_1,\ldots,e_K$는 agent index에 의존하지 않는 fixed learned query다. 각 query가 agent set을 cross-attention으로 읽을 때 agent key/value의 순서만 바뀌므로 event slot representation과 그 shared impact head의 출력도 변하지 않는다.

$$
z_k(PA,\xi)=z_k(A,\xi),\qquad
v_k(PA,\xi)=v_k(A,\xi).
$$

| output | 요구 성질 | 현재 구현 원리 |
|---|---|---|
| `predicted_team_reward` | invariant | set pooling 뒤 global head |
| event embedding / `event_impact` | invariant | agent-index-independent event query가 agent set을 read |
| `agent_pointer_logits` | equivariant | event--agent shared pairwise score |
| `predicted_agent_delta` | equivariant | shared agent-wise prediction head |

#### 보장에 필요한 경계 조건

1. Rule text는 agent permutation과 독립적으로 고정되어야 한다. 현재 LBF rule text의 Alice/Bob/Carol은 generic environment description이며 structured agent row와 name의 대응을 주지 않는다. Agent별 state/ability를 순차 text block으로 다시 주입하면 이 보장은 사라진다.
2. 이 절의 formal claim은 agent permutation에 한정한다. $\xi$ 내부의 환경 정보는 agent reserialization 때 고정하며, 그 내부 요소의 순서에 대한 보장은 현재 main proof/metric의 대상이 아니다.
3. 위 식은 실수 연산의 구조적 등식이다. 실제 Qwen bf16/GPU attention은 reduction order와 rounding 때문에 작은 finite-precision drift를 낼 수 있다. 따라서 empirical report에는 architecture proof와 함께 inverse-mapped probability/logit drift를 별도로 제시해야 한다.

### 2026-08-05 — SetPE + complete mask의 block-association failure와 수정 방향

SetPE만으로 agent block의 시작 position을 공유한 뒤 raw block token 전체에 complete/prefix attention을 허용하는 것은 충분하지 않다. 이는 Set-LLM의 Figure 4가 지적한 failure와 같다. 서로 다른 block의 같은 SetPE position token을 교환해도, complete graph에서는 각 token이 보는 neighborhood가 같아져 모델이 “어느 $t=0$ action과 어느 $t=1$ state가 같은 agent history에 속하는가”를 복원하지 못할 수 있다. 순열 불변성은 유지되지만, block 내부의 **소속/결속 정보**가 사라져 표현력이 떨어진다.

현재 Qwen smoke는 각 agent를 한 개의 numeric node로 압축한 single-transition reference이므로 block 내부 token association이 없다. 따라서 complete structured mask는 그 좁은 setting에서는 valid symmetry baseline이다. 그러나 multi-step trajectory 또는 natural-language agent block을 실제 입력으로 쓰는 본 architecture에 그대로 확장하면 안 된다.

수정할 mixed set--sequence mask의 기본 구조는 다음과 같다.

```text
agent i의 local history tokens  <-->  agent i의 local readout q_i
                                      |
agent j의 local history tokens  <-->  agent j의 local readout q_j
                                      |
                          shared global/event latent slots g_1,...,g_K
```

1. **Local SetMask.** 같은 agent block 내부의 token끼리는 temporal/field order를 유지하며 읽는다. 서로 다른 agent block의 raw history token 사이 edge는 기본적으로 닫는다. 이 mask의 same-block membership은 learned embedding이 아닌 bookkeeping metadata다.
2. **Block summary/readout.** 각 agent마다 local history를 읽는 동일한 readout query $q_i$를 둔다. $q_i$는 해당 block과 shared global latent만 읽는다. 이로써 $h_i$는 “agent $i$의 history”라는 결속을 보존한 representation이 된다.
3. **Symmetric global bridge.** $K$개의 global/event latent는 모든 agent block readout을 대칭적으로 읽고, 모든 $q_i$가 다시 이 latent를 읽는다. 협업 정보는 raw token mixing이 아니라 이 bridge를 통해 전달된다. Latent는 agent index가 아닌 set 전체를 읽으므로 invariant하다.
4. **필요 시 equivariant pairwise relation stage.** global bottleneck만으로 부족하면 raw token complete attention을 되살리는 대신, block summary 수준에서 $m_i=\sum_j f(h_i,h_j,r_{ij})$ 형태의 shared relation/message module을 둔다. $r_{ij}$는 거리·visibility·같은 time 등 environment predicate에서 만들며 index가 아니라 state 관계에만 의존해야 한다. 이 연산은 agent permutation에 equivariant하면서도 pairwise cooperation을 직접 모델링한다.

따라서 완성형 backbone의 원칙은 **block 내부는 sequence/local association을 보존하고, block 사이는 set-equivariant bridge 또는 relation message로만 교환한다**이다. Complete mask는 single-node-per-agent Stage-0 reference의 ablation으로만 남긴다. 이 변경은 symmetry proof를 약화시키지 않는다. Mask가 agent block permutation에 대해 $M(P_Ax)=P_AM(x)P_A^\top$를 만족하고 query/head가 공유되면, local/readout/global/relation module 모두 같은 동변성 induction을 만족한다.
