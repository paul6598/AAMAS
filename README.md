# LEHCA 재구현 및 RSVP 연구 코드

LLM 지침을 QMIX의 보상·행동에 연결하는 LEHCA 재구현과, 잔여 셰이핑 가치 예측으로
지침 갱신을 결정하는 RSVP(Residual Shaping-Value Prediction)의 실험 코드다. **2026-09-17 기준 연구를 보류하고 공유·인계용으로
정리했다. 모든 제출 실험은 종료됐으며 자동 예약 작업은 없다.**

[연구 종합 보고서](docs/research/research-summary.md)부터 읽으면 연구 질문, 재현 과정,
최종 결과와 한계를 확인할 수 있다. 동일 호출 예산의 fixed F136보다 RSVP가 두 seed에서
높은 성능을 보였지만, 무작위 MRT에서 갱신의 근접 이득은 확인되지 않았다.
LEHCA의 안정적인 QMIX 대비 우위도 재현하지 못했다. 일반적인 성능 우위나
갱신 시점의 인과적 유효성을 확정한 구현은 아니다.

## 읽는 순서

1. [연구 종합 보고서](docs/research/research-summary.md): 무엇을 시도했고 무엇이 남았는가.
2. [코드·환경·실행 안내](docs/research/code-guide.md): 주요 함수, 설치 조건과 재실행 방법.
3. [검증 방법](docs/research/validation.md): 예측력, replacement gain, MRT의 정의와 해석.
4. [실제 LLM 출력과 직접 검토](docs/research/llm-output/llm-guidance-review-20260912.md).
5. [최종 결과 JSON](docs/research/final-results.json) 및 [실험 원장](docs/research/archive/experiments-log.md).

## 구성

```text
main.py, run.py      Sacred 진입점과 수집·학습·평가 루프
algorithm/lehca/     Commander, 지침 정제, shaping, masking, QMIX learner
algorithm/rsvp/      predicate library, 가치 예측기, scheduler, 검증 계측
algorithm/src/       PyMARL 기반 공유 네트워크·리플레이·학습 인프라
algorithm/vigil/     RSVP의 옛 이름에 대한 import 호환 모듈
config/             알고리즘·환경 설정
scripts/            일반 실행 및 최종 실험 재실행 스크립트
analysis/           결과·예측력·교체 효과의 핵심 집계기 3개
analysis/audits/    backbone·grounding 감사와 실제 LLM pipeline 점검
tests/              회귀 테스트와 Pursuit predicate 검사
env/                SMAC, 선택적 PettingZoo Pursuit 및 의미 인터페이스
docs/research/      공유용 보고서, LLM 원문, 검증 방법, 실험 원장
AAMAS_draft/        섹션별 LaTeX 초안; 미완성 과거 연구 자료
```

`results/`, `wandb/`, 참조 논문 PDF와 LaTeX 빌드 생성물은 Git에 포함하지 않는다.
서버의 원자료·checkpoint·동결 소스는 보존했다. Git에는 최종 곡선·설정 요약·SHA256과
실제 검토한 LLM 요청·응답을 함께 제공한다.

GRF의 초기 탐색 구현과 일회성 분석기는 공유본에서 제외했다. 과거 결과는 연구 보고서에
남겼으며, 소스는 Git의 `f138bf0` 및 서버 정리 백업에 보존돼 있다. Pursuit는 실제 학습
결과가 있어 선택적 환경으로 유지했다.

## 빠른 실행

실험 환경은 Python 3.10.20, PyTorch 2.5.1+cu121, SC2 4.10, SMAC의 고정 commit을
사용했다. [requirements.txt](requirements.txt)는 관측된 핵심 패키지 버전을 기록한 것이며,
새 장비의 설치 전체를 검증한 lockfile은 아니다. StarCraft II와 SMAC 맵은 별도로 설치한다.

```bash
conda create -n aamas python=3.10
conda activate aamas
pip install -r requirements.txt
export SC2PATH=/path/to/StarCraftII

# SMAC 설치 확인용 짧은 실행: LLM·W&B 없이 동작
python main.py --config=lehca --env-config=sc2 with \
  env_args.map_name=3m commander=rule use_cuda=False \
  t_max=400 test_nepisode=4 use_wandb=False

# 회귀 테스트: 환경 프로세스나 LLM 요청을 시작하지 않음
python -m unittest discover -s tests -p 'test_*.py'
```

LLM 실험은 할당된 GPU 작업 안에서 실행하며 이미 준비된 OpenAI 호환 API를 사용한다.
별도의 vLLM 환경에서 `bash scripts/serve_llm.sh openai/gpt-oss-20b 8355`로 서버를 기동한다.

```bash
# 한 seed만 실행; SEEDS를 생략하면 일반 실행기는 여러 seed를 순차 실행
SEEDS=0 bash scripts/run_qmix.sh 2s3z False qmix_demo
SEEDS=0 bash scripts/run_lehca.sh 2s3z False lehca_demo llm http://localhost:8355/v1

# 최종 dense E1 설정을 재실행
bash scripts/run_rsvp_validation.sh audit 0 http://localhost:8355/v1 1000000 shape
FIXED_PERIOD=136 bash scripts/run_rsvp_validation.sh fixed 0 http://localhost:8355/v1 1000000 shape
```

W&B는 기본 공유 예제에서 꺼져 있다. 사용할 경우 `wandb_entity`, `wandb_project`를 본인
계정에 맞춘다. 주요 함수에는 한국어 역할 주석이 있다. 기존 YAML 기본값은 캠페인별
override와 다를 수 있으므로 보고서의 조건과 Sacred의 실제 config를 함께 확인한다.

## 출처와 재현 범위

공유 인프라는 [oxwhirl/PyMARL](https://github.com/oxwhirl/pymarl)을 기반으로 한다.
LEHCA는 논문·보충자료를 바탕으로 재구현했으며, 공개되지 않은 prompt·grounding·계수의
구현 선택이 포함돼 있다. 원 논문의 코드나 수치에 대한 동일 구현을 보장하지 않는다.
구체적인 대응과 차이는 [grounding 감사](docs/research/archive/lehca-grounding-audit-20260907.md)에 남겼다.

LaTeX 초안은 `bash AAMAS_draft/build.sh`로 빌드한다. 결과는 `AAMAS_draft/build/`에
모인다. 초안의 수치·미완성 섹션·과거 학회 template 정보는 최종 연구 보고서를 대체하지 않는다.
