# 자율 재현 실험 계획 (2s3z)

**목표**: 논문 주장 재현 — LEHCA가 QMIX보다 (a) 초반 수렴 빠르고(AUC_early↑),
(b) 최종 승률 같거나 높음(2s3z에서 QMIX ~0.93 대비 ≥0.9).

**관측된 문제** (2026-08-25): 기본 설정 LEHCA는 초반은 빠르나(70k에서 0.36)
최종 0.43–0.65로 QMIX(0.93)보다 크게 나쁨. 가이던스가 후반 수렴 방해 추정.

## 진단 매트릭스 (각 1시드, t_max=500k, 2s3z, wandb 그룹 DIAG_*)

판정 기준: **500k 시점 test_battle_won_mean 최근 5개 평균**.
참조점: QMIX@500k ≈ 0.85–0.9 (seed0 로그 기준).

| ID | 그룹 | 오버라이드 | 가설 |
|----|------|-----------|------|
| A | DIAG_A_shaping_only | use_action_masking=False | 마스킹이 문제인지 분리 |
| B | DIAG_B_masking_only | use_reward_shaping=False | 셰이핑이 문제인지 분리 |
| C | DIAG_C_no_testmask | use_masking_at_test=False | 테스트시 하드마스크가 카이팅 방해? |
| D | DIAG_D_lambda_to_zero | lambda_min=0.0 lambda_decay=0.999 | 잔존 셰이핑(λ_min=0.05)이 정책 왜곡? |
| E | DIAG_E_weak_guidance | beta=0.25 lambda_start=0.3 lambda_min=0.0 | 가이던스 전반 약화 |

## 의사결정 규칙

1. A·B 먼저 (순차, 2:0에서). 결과에 따라:
   - A 좋음(≥0.8) & B 나쁨 → 마스킹 문제 → C 실행 (마스킹은 켜되 테스트만 오프).
     C도 나쁘면 beta 축소(D' = beta=0.25) 시도.
   - A 나쁨 & B 좋음 → 셰이핑 문제 → D 실행.
   - 둘 다 나쁨 → E 실행 (+ f_update=400도 후보).
   - 둘 다 좋음 → 상호작용 문제 → C와 D 결합 변형 실행.
2. 어떤 변형이 500k에서 ≥0.8 달성하면 → 그 설정으로 **본 실험 5시드 1M**
   (그룹 LEHCA_2s3z_v2) 시작. AUC_early 우위(초반 가속)는 유지되는지
   analysis/auc_early.py로 확인.
3. 본 실험 완료 → analysis로 QMIX vs LEHCA_v2 비교표 작성, docs/에 결과 기록.
4. 논문 충실도 노트: 변경한 값은 전부 논문에 명시 없는 자유 파라미터
   (supplementary 페이월)이므로 "재현 시 튜닝 필요"로 기록. 논문 명시값
   (Table 2: lr, batch, buffer, gamma, LLM 설정)은 절대 변경하지 않는다.

## 인프라 규칙

- vLLM gpt-oss-20b가 죽으면 세션 37에서 재기동:
  `export VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_HOME=/usr/local/cuda-12.7 && export PATH=$CUDA_HOME/bin:$PATH && HF_HUB_OFFLINE=1 vllm serve openai/gpt-oss-20b --port 8355 --max-model-len 8192 --gpu-memory-utilization 0.85`
- Slurm 잡 만료(2일 한도) 시 tmux 페인은 로그인 셸로 돌아옴. 재할당:
  `srun --partition=gpu3 --gres=gpu:1 -c 2 --time=2-00:00:00 --pty bash`
  (gpu6도 가능). 재할당 후 conda activate aamas, 실험 재시작 (state.md 참조).
  잡 만료 시각: n064(37, 2:0) 8/26 ~18:00, n016(34) 8/26 ~06:45 KST 무렵.
- 대기 중 잡 895288/895290 (tmux 38/40)이 할당되면 진단 병렬화에 사용.
- QMIX(34) 5시드는 그대로 완주시킴 (베이스라인 재현 양호). 만료로 끊기면
  남은 시드만 이어 돌림 (`for SEED in <남은 시드>` 수정 실행).
- 학습 명령 템플릿 (AAMAS 루트에서):
  `SC2PATH=/gpfs/home1/paul6598/StarCraftII python main.py --config=lehca --env-config=sc2 with env_args.map_name=2s3z use_wandb=True wandb_group=<GROUP> seed=<S> t_max=<T> commander=llm llm_api_base=http://n064:8355/v1 llm_model=openai/gpt-oss-20b test_interval=10000 test_nepisode=32 <OVERRIDES> 2>&1 | tee results/logs/<GROUP>_seed<S>.log`
  (vLLM 노드가 바뀌면 llm_api_base의 호스트도 그 노드명으로 변경)


## PHASE 3: hard-map 검증 (2026-08-26 오후 추가)

2s3z 결론: QMIX 0.96 vs LEHCA_v2 0.85-0.90 — v2로 붕괴는 해결했으나 QMIX 미달.
2s3z는 QMIX 단독으로도 포화되는 맵이라 가이던스 headroom이 없음. 논문의 격차가
큰 hard map에서 LEHCA>QMIX 재현 여부가 핵심 검증.

1. v2 5시드 + AUC_early 표 완료 후 (analysis/auc_early.py LEHCA_2s3z_v2 / QMIX_2s3z):
2. **5m_vs_6m** 착수 (t_max=5M, 우선 seed 0-1만):
   - LEHCA v2 설정 (그룹 LEHCA_5m6m_v2), QMIX (그룹 QMIX_5m6m)
   - GPU 배치: 2:0(v2 완료 후)과 38/40 할당분 활용. QMIX는 LLM 불필요하므로
     아무 GPU나, LEHCA는 vLLM(n064:8355) 접근 가능해야 함.
   - 5M 런은 잡 2일 한도를 초과 → 만료 시 재할당 후 남은 시드/이어서 재시작
     (pymarl은 mid-run 체크포인트 재개 미설정이므로 시드 단위로만 재시작).
3. 27m_vs_30m: 38/40 등 GPU 여유 생기면 seed 0-1 착수 (같은 요령).
4. 판정: hard map에서 LEHCA_v2가 QMIX보다 초반 가속 + 동등 이상 최종 승률이면
   논문 핵심 주장 재현 성공으로 기록. 아니면 f_update/predicate 스케일 등
   자유 파라미터 추가 탐색 여부를 사용자에게 보고 후 결정.


## PHASE 2.5: QMIX 갭 해소 (2026-08-26 저녁 추가 — hard map보다 우선, 사용자 지시)

2s3z에서 v2(0.85-0.90) vs QMIX(0.96) 갭 해소가 최우선. 남은 개입 요인은
(a) 학습 내내 켜져 있는 train-time 마스킹, (b) 버퍼에 남은 수집 시점 λ의 셰이핑 보상.
이를 겨냥한 구조적 처방 2개 (코드 구현·스모크 테스트 완료, 기본값 off):

| ID | 그룹 | 오버라이드 (v2 설정 위에 추가) | 겨냥 |
|----|------|------|------|
| G1 | DIAG_G1_maskanneal | mask_anneal_t=300000 | 300k 이후 마스킹 완전 오프 → 이후는 순수 QMIX |
| G2 | DIAG_G2_learnshape | shaping_in_learner=True | 리플레이 시점의 현재 λ로 보상 합성 → 버퍼 오염 제거 |

실행 규칙:
1. v2 5시드 완료 후 2:0에서 G1부터 (500k, 1시드). n064 만료(~02:15) 전 1런 가능.
2. 만료 시 재할당(gpu3/gpu6, --time=2-00:00:00) 후 G2. 38/40 할당되면 병렬.
3. 판정: 500k에서 QMIX@500k(~0.85) 동등 이상이면 성공. G1·G2 모두 좋으면 결합.
4. 성공 설정으로 LEHCA_2s3z_v3 5시드 (1M) → QMIX와 최종 비교.
5. v3도 갭이 남으면: f_update=400, predicate 스케일 0.5×, lambda_start=0.3 등
   잔여 후보를 1런씩. 그래도 안 되면 "2s3z 포화맵에선 격차 미재현"으로 기록하고
   PHASE 3(hard map)으로 — hard map이 논문 주장의 본진임을 사용자에게 보고.
PHASE 3(5m_vs_6m/27m)은 38/40 할당 시 G-실험과 병렬로 시작 가능 (QMIX_5m6m부터).


## 인프라 정책 갱신 (2026-08-26 17:40, 사용자 지시)
- 만료된 tmux 세션은 해당 세션에서 `srun --pty -p gpu6 -c 2 --gres=gpu:1 /bin/bash`
  로 재할당 → conda activate aamas (서빙이면 vllm) → cd /gpfs/home1/paul6598/AAMAS → 작업 재개.
  srun 전 셸이 지워진 디렉토리에 있으면 getcwd 에러 → 먼저 cd ~/AAMAS.
- 모든 가용 GPU를 재현에 투입 (세션 34 제한 해제됨).
- 표준 vLLM 서버: n026:8356 (A10, 세션 45, ~8/28 17:00 만료). n064:8355은 8/27 02:15 만료 예정.
  A10에서 gpt-oss-20b 서빙+LEHCA 학습 모두 검증 완료.
- 현재 배치: 2:0=LEHCA_v2 잔여시드(n064), 34=DIAG_G1(n039), 46=DIAG_G2(n026), 45=vLLM(n026).


## 리뷰 피드백 반영 (2026-08-26 17:45, docs/analysis-2s3z-20260826.md)

판정 기준 개정:
- 성공 지표는 (a) AUC_early ≥ QMIX(0.332), 목표 0.45 (논문 주장의 본질은 1.79×
  초반 가속이지 final 우위가 아님 — final Δ0.005는 시드 노이즈 이하), (b) final
  last-10% 평균 ≥ 0.92 (QMIX@500k 5시드 평균; 기존 0.85는 seed0 단독이라 낮았음).
- 모든 승률 판정은 "최근 5포인트"가 아닌 last-10% 평균으로 통일.

실험 트리 개정:
- G2 (진행 중): 성공 = 100-200k 함몰 소멸 + 500k에서 0.92 근접.
- G1 (진행 중): 함몰(150-300k)보다 anneal(300k)이 늦어 함몰은 못 잡을 것으로 예상.
  애매하면 mask_anneal_t=150000 변형 1런.
- **G3 (G2 성공 시 즉시)**: G2 + beta=0.5 + lambda_start=0.5 (기본 강도 복원).
  붕괴 원인이 제거된 상태에서 강한 가이던스로 AUC_early 우위가 복원되는지가
  재현의 관건. G3가 AUC ≥ 0.40 & final ≥ 0.92면 그 설정이 v3.
- v3 확정 시: masked-test 평가 편차를 없애기 위해 use_masking_at_test=True로
  되돌릴 수 있는지 G3에서 확인 (논문 프로토콜은 test도 masked greedy).
  안 되면 편차로 명시하고 보고.
- 5M 런(5m_vs_6m 등) 착수 전 save_model=True + checkpoint_path 로드 재개 경로 점검
  (pymarl 기본 기능, 미검증). 시드 통째 재시작 방지용.

기록 정정: v2 final은 last-10% 평균 0.832 (0.800/0.831/0.866) — state.md의
"0.85-0.90"은 낙관적이었음. v2 AUC_early 0.25로 QMIX(0.33)보다 낮음 (β 약화 부작용).

사용자 액션 후보(보고만): supplementary/저자 코드 확보가 최대 병목 —
교신저자 zhuwei929@hotmail.com 문의 권고됨.


## 계획 개정 (2026-08-27 01:45, 사용자 지시: 3시드 + hard map 신속 확인)

- v3는 **3시드로 축소**: 34의 for루프(seeds 0,1,2)는 seed 2 시작 시점에 중단(Ctrl-C),
  46의 for루프(seeds 3,4)는 seed 4 시작 시점에 중단 → 유효 시드 = 0, 1, 3.
- 프리된 GPU는 즉시 hard map 검증 투입. **단축 프로토콜**: t_max=1200000
  (5M 예산 맵의 논문 AUC_early 구간=첫 1M을 커버; final 비교는 필요 시 연장).
  test_interval=10000 test_nepisode=32, 시드 0부터, 맵당 LEHCA(v3 설정)와 QMIX 각
  최대 3시드 (우선 1-2시드로 신호 확인).
- 실행 순서 (GPU 프리 순서대로): ① QMIX_5m6m seed0 (46, ~05:30 프리 예상)
  ② LEHCA_5m6m_v3 seed0 (34, seed1 완료 후) ③ 이후 시드/3s_vs_5z/27m_vs_30m 확장.
  27m은 스텝당 비용 큼 — 마지막, 1시드 신호 먼저.
- LEHCA hard map 오버라이드 = v3 설정 그대로 (mask_anneal_t=150000는
  1.2M 런에서 12.5% 지점 — 유지).
- 논문 참조점(Table 4): 5m_vs_6m LEHCA final 0.6423/AUC 0.2950, QMIX final
  0.5912/AUC 0.0387 (AUC 7.6×!) — 초반 가속 재현 검증에 최적 맵.


## PHASE 4: AUC_early 재현 집중 (2026-08-27 09:30, 사용자 지시: LEHCA 성능 재현 집중)

목표: 논문 LEHCA의 초반 가속 재현 — 2s3z AUC_early 0.455 (우리 v3 0.22, G3 s1은
0.317 달성 사례 있음). 문제의 본질은 시드별 느린 출발(slow-start)의 분산.

H-시리즈 (250k 초단축런, AUC는 --frac 0.8 → T_early=200k, 각 ~1h):
| ID | 그룹 | 오버라이드 (v3 위에) |
|----|------|---------------------|
| H1 | DIAG_H1_slowdecay | lambda_decay=0.9993 |
| H2 | DIAG_H2_densecmd  | f_update=100 |
| H4 | DIAG_H4_strong    | lambda_start=1.0 lambda_decay=0.9993 |
| **H5** | DIAG_H5_explorebias | explore_soft_bias=True (+v3 그대로) — **최우선 실행** |

설계 개선 (09:20, 사용자 지적 반영 — 이후 모든 LEHCA 런에 적용됨):
- 프롬프트: "환경이 이미 데미지/킬 보상하니 협조 술어(focus_fire/protect_type/
  kill_type/retreat) 우선" 지침 추가 (중복 술어 억제).
- retreat_low_health 버그 수정: 적 중심에서 멀어지는 이동만 보상.
- explore_soft_bias 플래그 신설: ε-랜덤 샘플을 W_soft 비례로 편향 — 논문 초록의
  "masking guides exploration" 해석. 초반(ε≈1) 가이던스 무반영 문제의 직접 처방.
- 주의: 프롬프트/술어 수정은 무조건 적용이므로 H-런들은 v3 런과 미세하게
  비교조건이 다름 (v4 확정 시 최종 설정으로 일괄 재검증).

규칙:
1. GPU 프리 순서대로 H1/H2/H4 각 seed0 실행 (34 프리 시부터; 38/40/46은 진행 중인
   1.2M 런 완주 후 투입). 3s_vs_5z LEHCA 확장은 보류 (QMIX_3s5z s0만 완주시킴).
2. 판정: AUC(200k) 상위 후보를 seeds 1,2로 확증 (3시드 평균 ≥ 0.35면 성공 방향,
   ≥0.45면 논문 수준). ε-anneal 변경은 confound라 금지.
3. 확정 설정 = v4 → 2s3z 3시드 1M (final 유지 확인) + 5m_vs_6m 재도전.
4. 각 H 결과와 결정은 state.md에 즉시 기록.


## 2차 리뷰 반영 (2026-08-27 09:40, docs/analysis-20260827.md)

정정: 5m6m 논문 참조점은 AUC 0.0612(LEHCA) vs 0.0387(QMIX) = **1.58×**
(이전 기록 "7.6×"는 Table 4 win rate 열 오독). 우리 1.18×(각 1시드)는 근접.
2s3z 비율 상한 인식: 우리 QMIX AUC 0.332 기준, 논문 절대값 0.455 달성 시에도
1.37×가 상한 — 보고서에 절대값·비율 병기.

H-시리즈 우선순위 개정 (2카드 최우선, 250k 런):
| ID | 그룹 | 오버라이드 (v3 베이스) | 성격 |
|----|------|------|------|
| H5 | DIAG_H5_explorebias | explore_soft_bias=True | 마스킹을 탐색에 활용 (논문 구성 유지) |
| H6 | DIAG_H6_nomask | use_action_masking=False lambda_decay=0.9993 | 마스킹 제거 + 셰이핑 연장 (ablation 라인 — 성공해도 "마스킹 없는 LEHCA"로 보고) |
그다음 H1(slowdecay)/H2(densecmd)/H4(strong). 판정: AUC(200k, --frac 0.8),
QMIX 0.332 초과가 1차 목표, 0.40+면 성공.

운영: v3 seed2/4는 GPU 여유 생기면 채움(n=3 취약성 보완). 3s_vs_5z 참조점은
Table 4에 없음 — Fig 4 커브만 (논문 QMIX는 1.5M까지 ~0; 우리 QMIX가 강할 가능성 높음).
5m6m final 비교는 5M 완주 필요 — 현 1.2M 수치엔 "1.2M 시점" 라벨 필수.


## PHASE 5: 새 연구 착수 — F_update 민감도 스윕 (2026-08-28 15:20)

목적: "적응적 가이던스 스케줄링" 연구의 첫 확인 실험. 호출 주기 자체가
성능(AUC_early)에 영향을 주는지 정적 스윕으로 측정. 관련연구 노트:
docs/related-work-adaptive-guidance.md.

설정: v4(shaping-only) 베이스 + f_update ∈ {100, 400, 800}, 2s3z, 250k, seed0,
AUC(200k, --frac 0.8). 참조: f_update=200 = DIAG_H6 (0.388±0.061, n=3).
배치: 2=F100, 38=F400, 40=F800 (n027/n026, 2일 할당). vLLM=37(n026:8356).
그룹명: SWEEP_F100 / SWEEP_F400 / SWEEP_F800. 로그 results/logs/sweepF*.log.
판정: AUC가 F_update에 둔감하면 "언제" 축은 비용 절감용, "강도/게이팅" 축이 주
기여. 민감하면 최적 주기가 맵/단계별로 다른지 후속(5m6m 스윕).
