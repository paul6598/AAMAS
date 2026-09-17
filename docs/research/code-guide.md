# 코드·환경·실행 안내

2026-09-17 인계용 정리. 연구는 보류됐고 실행·예약 중인 작업은 없다.
일반 실행기는 API 서버나 Slurm 작업을 자동으로 시작하지 않는다.

## 환경과 의존성

실제 SMAC 본실험은 Python3.10.20, torch2.5.1+cu121, numpy2.2.6, sacred0.8.7,
SC2 4.10과 SMAC commit `d6aab33f76abc3849c50463a8592a84f59a5ef84`을 사용했다.
핵심 Python 의존성은 [requirements.txt](../../requirements.txt)에 기록했다. PyPI의
동명이인 `smac` 패키지를 설치하지 않도록 MARL 환경의 Git commit을 지정했다.
새 장비의 전체 설치는 이번 정리에서 실행하지 않았으므로 CUDA wheel, SC2 실행 파일,
SMAC 맵과 라이브러리 경로는 장비에 맞게 준비한다.

```bash
conda create -n aamas python=3.10
conda activate aamas
pip install -r requirements.txt
export SC2PATH=/path/to/StarCraftII
```

LLM 서버는 별도 환경을 사용했다: Python3.11.15, vLLM0.25.1,
`openai/gpt-oss-20b`, max model length8192. 서버 환경을 활성화한 GPU job에서
아래 명령을 실행한다. 기존 클러스터에서는 FlashInfer JIT에 nvcc가 필요해
`module load cuda/13.1.1`을 사용했다. 실행기는 conda나 cluster module을 강제하지 않는다.

```bash
conda activate vllm
bash scripts/serve_llm.sh openai/gpt-oss-20b 8355
```

Pursuit는 실제 학습 결과가 있는 선택적 환경이며 SMAC-only 설치에서 필수는 아니다.
당시 패키지는 `pettingzoo==1.27.0`, `gymnasium==1.3.0`, `pygame==2.6.1`이었다.
GRF의 초기 탐색 코드는 공유본에서 제외했고 Git `f138bf0`에 보존했다.
MPE 구현이나 MPE 실험은 포함하지 않는다.

## 실행 흐름과 읽는 순서

```text
main.py → run.py → runner.run() → episode batch → replay → learner.train()
                      ├─ semantic summary → Commander → sanitized guidance
                      ├─ grounded predicate → shaping
                      └─ 선택적 action masking

RSVP: predicate별 MC suffix target → ValueCritic → issuance gate → CUSUM/Fmax
```

| 경로·함수 | 역할 |
|---|---|
| `main.py`, `run.py:run_sequential` | Sacred 설정, 수집·리플레이 학습·평가·모델 저장 |
| `algorithm/lehca/runner.py:run` | LEHCA 전이 수집과 보상·행동 지침 통합 |
| `algorithm/lehca/runner.py:_maybe_refresh_commander` | 고정 F 갱신과 평가용 Commander 분리 |
| `env/semantic/sc2.py:summary` | 관측 가능한 정보를 LLM용 텍스트로 요약 |
| `algorithm/lehca/commander/llm_commander.py:__call__` | API 요청, 재시도, JSON 추출·정제 |
| `algorithm/lehca/commander/base.py:sanitize_guidance` | 어휘·범위 검사, 의미상 동일 subgoal 중복 제거 |
| `algorithm/lehca/shaping/predicates.py:compute_shaping` | grounded 전이 신호의 가중합과 clipping |
| `algorithm/lehca/masking/compiler.py:build_masks` | 행동 token을 현재 행동 인덱스에 접지 |
| `algorithm/lehca/controller.py:select_actions` | hard mask와 Q + β log W soft preference |
| `algorithm/lehca/learner.py:train` | QMIX/Adam 학습과 lambda 스케줄 |
| `algorithm/rsvp/predlib.py` | 환경별 head library, predicate 벡터, 특징 추출 |
| `algorithm/rsvp/critic.py:add_episode` | 완료 episode의 MC 할인 접미합 타깃 생성 |
| `algorithm/rsvp/critic.py:train`, `predict` | 정규화 MSE 학습과 원 단위의 가치 예측 |
| `algorithm/rsvp/runner.py:_maybe_refresh` | 가치 비율, issuance gate, CUSUM/timer, 요청 실패 처리 |
| `algorithm/rsvp/validation.py:ValidationLog` | 현재 episode 학습 전 prediction과 종료 후 target 기록 |
| `algorithm/rsvp/validation.py:RefreshTrial` | 무작위 immediate/hold block, refresh lockout, 후속 outcome |
| `env/__init__.py:_WinOnlyRewardEnv` | native sparse의 음의 reward를0으로 바꾸는 opt-in wrapper |

주요 함수에 한국어 역할 주석을 유지했다. `trusted()`는 target variance 검사이며 예측
정확도 검사가 아니다. `shaping_in_learner=True`는 저장된 scalar F에 현재 lambda를
곱하는 옵션으로, replay를 최신 guidance로 relabel하는 기능은 아니다.

## 일반 실행

아래 명령은 저장소 root에서 할당된 compute 작업 안에서 실행한다. API는 이미 준비돼
있어야 한다. `AAMAS_PYTHON`으로 Python executable을 지정할 수 있고,
`SEEDS`는 일반 실행기의 순차 seed 목록, `EXTRA`는 Sacred override다.

```bash
SEEDS=0 bash scripts/run_qmix.sh 2s3z False qmix_demo
SEEDS=0 bash scripts/run_lehca.sh 2s3z False lehca_demo llm http://localhost:8355/v1
SEEDS=0 bash scripts/run_ablation.sh 2s3z True False False shape_demo http://localhost:8355/v1
SEEDS=0 bash scripts/run_rsvp_sc2.sh 2s3z vf False rsvp_demo http://localhost:8355/v1
```

일반 YAML과 demo는 역사적 캠페인의 모든 override를 자동으로 맞추지 않는다. 정확한
비교를 재실행하려면 다음 배치 명령 또는 Sacred 실제 config를 사용한다.

## 최종 실험 조건의 재실행

E1 공통값: dense 2s3z, epsilon300k, paper prompt/temp.2/cache off,
shaping-only, learner-time lambda, 절대 floor400k, dedup, Fmax200, test masking off,
평가10k/32 episodes다. fixed 기본 period는200이고 F136은 명시한다.

```bash
bash scripts/run_rsvp_validation.sh audit 0 http://localhost:8355/v1 1000000 shape
FIXED_PERIOD=200 bash scripts/run_rsvp_validation.sh fixed 0 http://localhost:8355/v1 1000000 shape
FIXED_PERIOD=136 bash scripts/run_rsvp_validation.sh fixed 0 http://localhost:8355/v1 1000000 shape
bash scripts/run_rsvp_validation.sh gate_timer 0 http://localhost:8355/v1 1000000 shape

# MRT 개입 run: 일반 RSVP 성능 seed와 합산하지 않음
bash scripts/run_rsvp_validation.sh trial 0 http://localhost:8355/v1 1000000 shape

# win-only 300k: qmix/full LEHCA 또는 training-only soft guidance
bash scripts/run_winonly_sparse.sh qmix 0
bash scripts/run_winonly_sparse.sh lehca 0 http://localhost:8355/v1
bash scripts/run_winonly_sparse.sh rsvp 0 http://localhost:8355/v1
bash scripts/run_winonly_sparse.sh fixed50 0 http://localhost:8355/v1
bash scripts/run_winonly_sparse.sh fixed200 0 http://localhost:8355/v1
```

`USE_WANDB=True`로 계측 배치의 W&B를 켤 수 있다. 계정 entity는 본인이 지정한다.
`RSVP_EXPERIMENT_TAG`/win-only GROUP으로 새로운 실행 이름을 정할 수 있다.
동일 명령도 LLM 응답과 학습 stochasticity로 기존 곡선과 완전히 같지는 않을 수 있다.
full LEHCA와 soft RSVP는 masking/test/보상 합성 경로까지 다르므로 scheduler 단독효과
비교로 해석하지 않는다.

## 원자료와 분석

Git의 [final-results.json](final-results.json)은 주요24개 본실험의 평가 곡선, 설정 요약,
완료 증거, config/info SHA256을 포함한다. 파일명과 Sacred ID는 immutable 식별자다.
Slurm이 완료됐어도 과거 Sacred run.json은 RUNNING으로 남는 사례가 있으므로 평가
step이나 Sacred 잔류 상태만으로 완료를 판단하지 않는다.

서버에 별도 보존된 경로:

- `results/sacred/<id>/config.json`, `info.json`: 실제 설정과 학습·평가 scalar.
- `results/guidance/`, `results/validation/`: 지침 JSONL과 prediction/target/MRT gzip JSONL.
- `results/source_snapshots/<campaign>/`: 당시 실행 코드, source-manifest.json, dirty diff.
- `results/diagnostics/<campaign>/`: 제출 원장, launch provenance, server log.
- `results/models/`, `wandb/`: 정책 checkpoint와 로컬 W&B 자료.

원자료가 없는 clone에서는 제공된 JSON과 LLM 출력 문서를 읽을 수 있다. 아래 재집계
명령은 서버 원자료를 가져온 뒤 사용한다. 해당 분석 자체는 학습이나 LLM 호출을 하지 않는다.

```bash
python analysis/summarize_experiments.py --runs 357 356 379 380 \
  --jobs 976091 976094 982780 982781 --output results/comparison.json
python analysis/summarize_rsvp_validation.py results/validation/<audit>.jsonl.gz --start 200000
python analysis/summarize_rsvp_validation.py results/validation/<trial>.jsonl.gz --trials-only
python analysis/summarize_refresh_replacement.py results/validation/<audit>.jsonl.gz \
  --horizon 5 --gamma .8 --bootstrap 10000 --output results/replacement.json
```

`--jobs`를 생략하면 로컬 Sacred의 명시적 COMPLETED만 완료로 인정한다. Slurm이 없는
장비에서 과거 RUNNING 메타데이터를 step 수만으로 완료로 바꾸지 않는다.
`analysis/`에는 위의 결과·예측력·교체 효과 집계기3개를 남겼다.
`analysis/audits/`의 `verify_lehca_backbone.py`, `audit_lehca_grounding.py`,
`smoke_llm_pipeline.py`는 기반선 재현과 실제 LLM 출력 확인에 사용한 도구다.
backbone 및 grounding의 rollout 옵션은 SC2를 시작하고, pipeline 점검은 SC2/LLM
요청을 만든다. 순수 오프라인 집계와 구분해 명시적으로 실행한다.
과거 중복·상태 의존성·trace·보상 진단의 일회성 분석기는 Git `f138bf0`에 보존했다.

## 검증과 논문 초안

```bash
python -m unittest discover -s tests -p 'test_*.py'
python tests/check_pursuit.py
bash AAMAS_draft/build.sh
```

LaTeX는 XeLaTeX/BibTeX와 packages.tex의 한글 폰트를 필요로 한다. main.tex가 section
파일을 조립하고 build/에 PDF·중간 파일을 모은다. 초안은 미완성 역사 자료로 보존했다.

추가 축소 후 현재 경로의 회귀 테스트32개와 Pursuit 검사가 모두 통과했다.
Python77파일 AST, YAML16개·공유 JSON7개 파싱, Bash9개 문법 및 문서 링크도 확인했다.
SMAC/Pursuit 환경 등록과 GRF를 로드하지 않는 import를 확인했고, 환경 선택 외 RSVP
메서드와 SMAC/Pursuit 특징 계산 함수가 이전 commit과 같은지도 검사했다.
LaTeX PDF 빌드는 최초 인계 정리에서 확인했으며 이번 축소에서는 TeX 소스를 변경하지 않았다.
예전 구현의 AST 전체가 같음을 요구하던 migration 검사는 이후 기능 변경에 적용되지
않아 제외했으며, VIGIL 설정·import 호환과 grounding 회귀 검사는 유지했다.
일반 실행기·ablation·최종 배치의 인자 전달도 stub으로 확인했다.
새 성능 실험이나 LLM 호출은 실행하지 않았다.

## 정리와 보존

끝난 제출·monitor·private-server 예약 wrapper와 cohort 고정 일회성 집계기는 공유 코드에서
정리했다. 일반 진입점, 실제 검증 계측, 그 회귀 테스트, 중요한 감사 재현기는 유지했다.
VIGIL 호환 및 공유 registry의 COMA/QTRAN/VDN, 선택적 Pursuit는 유지했다.
정리 전 authored source/document 백업과 이동 원장은 서버의
`results/diagnostics/repository-handoff-20260917/`에 있다. raw 실험 자료를 삭제하지 않았다.
`before-cleanup.tar.gz`의 SHA256은
`406594ca371f4700f0e4afb4fda02c7e330f24009b7c6bbe8caaf29bdbe9cf2a`다.
GRF와 보조 분석기를 추가로 제외하기 전 소스와 이동 원장은
`results/diagnostics/repository-trim-20260917/`에도 보존했다.
