# PEEL: Permutation-Equivariant Event LLM

협력적 MARL에서 LLM을 centralized critic/credit estimator로 사용할 때, 동질 agent의 **입력 직렬화 순서**가 바뀌면 credit output이 달라지는 문제를 다루는 연구 레포지터리입니다.

현재 연구의 목표는 다음과 같습니다.

- global critic/value는 agent 입력 순서에 대해 permutation-invariant
- agent별 credit은 동일 physical agent를 따라 permutation-equivariant

여기서 $A$는 agent trajectory/state/action set, $P$는 agent permutation, $\xi$는 고정된 environment/rule context입니다. PEEL은 *Permutation-Equivariant Event LLM*의 약칭입니다.

## 현재 상태

- 기존 **LLM-MCA**와 **LLM-TACA** baseline을 재현·분석했습니다.
- 동일한 LBF trajectory에서 agent block 순서만 바꿔도 raw LLM-MCA credit이 달라지는 small adversarial probe를 구현했습니다.
- Qwen2.5-7B에 structured numeric input, SetPE/attention mask, parallel output head를 연결한 초기 **PEEL** prototype을 구현했습니다.
- 현재 prototype은 **순열 구조와 forward/backward feasibility**를 확인하는 단계입니다. Semantic event extraction, learned allocator, end-to-end MARL return 개선은 아직 검증되지 않았습니다.

연구 문제와 설계 방향은 [문제 정의](docs/PERMUTATION_ROBUST_LLM_CRITIC_PROBLEM_DEFINITION.md), 구체 명세는 [PEEL formal specification](docs/RELATIONAL_SET_CRITIC_FORMAL_SPEC.md)를 참고하십시오.

## 구조

```text
algorithms/
  common/             공용 RL/trajectory/permutation utility
  llm_mca/            LLM-MCA baseline
  llm_taca/           LLM-TACA baseline
  peel/               현재 PEEL Qwen/native prototype
  ddqn/, rnn_iql/,
  mappo/              non-LLM RL baseline

envs/                 LBF, Climbing, RWARE wrapper
experiments/          permutation audit, PEEL 학습, event output inspection
tests/                baseline 및 PEEL regression test
scripts/              LBF/Climbing/RWARE/vLLM 재현 실행 script
docs/                 문제 정의, formal specification, 연구노트
paper/                참고 논문 PDF
```

## 핵심 파일

| 목적 | 파일 |
|---|---|
| 기존 LLM-MCA 구현 | `algorithms/llm_mca/` |
| 현재 PEEL Qwen prototype | `algorithms/peel/qwen.py` |
| native symmetry reference | `algorithms/peel/model.py` |
| trajectory → structured input adapter | `algorithms/peel/transition.py` |
| raw LLM-MCA 순열 audit | `experiments/audit_raw_llm_permutations.py` |
| Qwen-PEEL structured output audit | `experiments/audit_qwen_peel_cases.py` |
| event interface 입출력 확인 | `experiments/inspect_event_extractor.py` |
| native / Qwen PEEL 학습 | `experiments/train_peel_event.py`, `experiments/train_qwen_peel_event.py` |

## 빠른 검사

프로젝트 환경이 준비된 뒤 다음으로 non-LLM symmetry reference와 permutation utility를 검사할 수 있습니다.

```bash
conda activate permute
PYTHONPATH=. python tests/test_peel.py
PYTHONPATH=. python tests/test_permutation_data.py
```

## 문서

- [문제 정의 및 연구 방향](docs/PERMUTATION_ROBUST_LLM_CRITIC_PROBLEM_DEFINITION.md)
- [PEEL formal specification](docs/RELATIONAL_SET_CRITIC_FORMAL_SPEC.md)
- [지속 연구노트](docs/RELATIONAL_SET_CRITIC_RESEARCH_NOTE.md)

## Git에 포함하지 않는 항목

실행 결과, W&B run, local checkpoint/cache는 `.gitignore`로 제외됩니다. 정리 전의 로컬 산출물은 이 작업 공간의 `local_artifacts/`에 보존되어 있으며 Git에 포함되지 않습니다.
