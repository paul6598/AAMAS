# HANDOFF — 새 세션 온보딩 (2026-08-29)

이 문서는 어느 Claude 세션이든 5분 안에 같은 출발선에 서게 하는 요약이다.
**먼저 이 문서 → docs/final-report-20260827.md → docs/related-work-adaptive-guidance.md
순으로 읽을 것.** 상세 타임라인은 docs/autoexp/state.md (의사결정 로그).

## 프로젝트 한 줄
LLM×MARL 연구(AAMAS 투고 목표). 1단계로 LEHCA(Bai et al., Sci. Rep. 2026)를 pymarl
위에 재구현·재현 완료. 2단계(진행 중): LEHCA의 한계(고정 F_update·고정 λ·상시 마스킹)
를 개선하는 **적응적 가이던스 스케줄링** 연구.

## 세션 역할 분리 규칙
- **A. LEHCA 베이스라인 세션**: `algorithm/lehca/`·`config/algs/{lehca,qmix_paper}.yaml`
  동결. 변경은 기본값-off 플래그로만. 진단·재실행·리뷰 대응 담당.
- **B. 연구 세션**: 새 방법은 `algorithm/<새이름>/`에 격리, LEHCA 코드는 import만.
  공용 인프라(`algorithm/src`, `env/`, `run.py`) 수정 시 docs/autoexp/state.md에 기록.
- **C. 리뷰 세션**: 결과 교차검증, docs/analysis-*.md·feedback-*.md로 피드백.
- 기준선 태그: git `lehca-baseline-v4` (재현 캠페인 종료 시점 코드).
- 진단 완료 후 권장 기본: `beta=0.1`, `dt_observable=True`, `shaping_in_learner=True`.

## 확정 설정 (LEHCA 베이스라인)
- **v3 (완전 구성, final 재현)**: beta=0.5 lambda_start=0.5 lambda_min=0
  lambda_decay=0.998 shaping_in_learner=True mask_anneal_t=150000 use_masking_at_test=True
- **v4 (shaping-only, 초반 가속 재현; ablation 라인)**: v3 + use_action_masking=False
  lambda_decay=0.9993
- QMIX: `--config=qmix_paper` (Table 2 값). 논문 명시값·ε-anneal은 절대 변경 금지.
- **d_t 정보원**: 2026-08-29 17:05부터 기본 `dt_observable=True`(아군 시야 내 적만).
  그 이전 모든 결과(v3/v4/H/M/F)는 전체 상태 d_t → 비교 시 `dt_observable=False` 명시.

## 핵심 수치 (FINAL 확정 9/2, docs/baseline-final-20260902.md; AUC_early = 예산 첫 20%)
| 맵 | QMIX AUC/final | LEHCA(shaping-only) AUC/final | 완전구성(마스킹 on) | 논문 |
|---|---|---|---|---|
| 2s3z (1M) | 0.332±0.045 / 0.963 (n=5) | **0.454±0.025 / 0.941** (n=2) | 0.250 / 0.536 (유해) | QMIX 0.254, LEHCA 0.455, w/o-mask 0.343 |
| 5m_vs_6m (1.2M) | 0.127±0.107 / 0.42±0.30 (n=3, v2) | 0.138 / **0.945** (n=1) | 0.001 / 0.000 (붕괴) | QMIX ≈0.187, LEHCA 0.295 |
(구버전 참조: 5m_vs_6m 구 QMIX 0.194±0.137 / LEHCA v3 0.309 — 전체상태 d_t 시절)
| 3s_vs_5z | 0.116 / 0.568 (n=1) | v3 0.017/0.021, v4 0.073/0.492 | ✗ |
| 27m_vs_30m | 0.033 / 0.141 (n=1) | v4 0.013/0.073 | ✗ |

## 2026-08-29 진단 결론 (docs/masking-fupdate-verdict-20260829.md)
- **마스킹 해악 = β 스케일**: β=0.5→0.1이면 마스킹 on에서도 shaping-only 수준
  (M3 0.342, M6 0.402 vs H6 0.388). 어휘 축소는 무효(M5 0.143). **권장 β ≤ 0.1.**
- **F_update 민감도 약함(2s3z)**: F100 0.367, F200 0.388, F400 0.313, F800 0.352,
  F∞ 0.326 — 본 무대는 5m6m.
- **관측 d_t 비용 0**: OBS_v4 0.394 vs 전체상태 0.388 → `dt_observable=True` 확정.
- 논문 완전구성 AUC 0.455는 미달성(β=0.1 완전구성 0.342); 저자 회신(β/F/술어) 대기.

## 핵심 발견 (연구 motivation)
1. 마스킹이 우리 어휘·β 하에서 초반 드래그 (4회 일관) — 프로빙으로 인과 확인:
   LLM이 stop 100%·후퇴 forbid → 카이팅 봉쇄 (docs/probe-commander-20260829.md).
   틸트 β·lnW(0.35–0.55) ≫ 초반 Q 격차(~0.05). 진단 M4/M5 진행 중.
2. 셰이핑 보상은 리플레이 시점 λ로 합성해야 함(shaping_in_learner) — 아니면 버퍼
   오염으로 후반 붕괴. 독립 재발견: arXiv 2607.04470.
3. 가이던스 가치는 학습 단계·맵에 따라 부호가 바뀜(5m6m 유익 / 2s3z 초반만 / 3s5z
   유해) → 고정 스케줄로는 불가능, 적응 스케줄러의 근거.
4. 벽시계 오버헤드 2s3z +40–70%, 5m6m +13% (LLM 호출).
5. 우리 QMIX가 논문 QMIX보다 전 맵에서 강함 (베이스라인 효과).

## 인프라
- conda `aamas`(학습), `vllm`(서빙). SC2PATH=/gpfs/home1/paul6598/StarCraftII.
- GPU: tmux 세션 2,34,37,38,40,45,46,47,48에서 `cd ~/AAMAS && srun --pty -p gpu6 -c 2
  --gres=gpu:1 /bin/bash` (2일 한도). 노드 이름은 매번 바뀜 — squeue로 확인.
- vLLM: `export VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_HOME=/usr/local/cuda-12.7;
  HF_HUB_OFFLINE=1 vllm serve openai/gpt-oss-20b --port 8356 --max-model-len 8192
  --gpu-memory-utilization 0.9` → llm_api_base=http://<노드>:8356/v1
- 학습 명령 템플릿: `SC2PATH=... python main.py --config=lehca --env-config=sc2 with
  env_args.map_name=<맵> use_wandb=True wandb_group=<그룹> seed=<S> t_max=<T>
  commander=llm llm_api_base=... llm_model=openai/gpt-oss-20b test_interval=10000
  test_nepisode=32 <오버라이드>`
- 지표: `python analysis/auc_early.py <wandb그룹> [--frac f]` (frac: T_early/t_max).
- 가이던스 덤프: results/guidance/<unique_token>_<group>_s<seed>_p<pid>.jsonl (2026-08-29 15:55
  이후 런만; 그 전 M4/M5는 M4M5_mixed_*.jsonl에 혼재). 마스크 통계: wandb mask_* / q_gap_mean.
- wandb: project AAMAS-LEHCA, entity joonhuk6598-university-of-seoul.

## 미결/진행 중 (2026-08-29)
- F_update 정적 스윕 (100/400/800 vs 200) — 세션 2/38/40.
- M4(rule commander+마스킹), M5(LLM+strategic 어휘+마스킹) — 세션 46/45.
- 저자 문의 메일(supplementary·코드) 발송 여부 사용자 결정.
- 5M 완주·시드 확장은 보류 (사용자 결정 사항).

## 하지 말 것
- results/, wandb/, paper/*.pdf 커밋 금지 (공개 레포).
- 기준선 설정 변경 후 "재현 수치"로 보고 금지 — 변형은 반드시 새 그룹명·플래그.
- 판정 지표는 last-10% 평균·AUC_early로 통일 (최근 몇 포인트 눈대중 금지).
