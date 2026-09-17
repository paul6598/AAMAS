# 실험 로그 — 적응 갱신 스케줄링 (research track)

이 문서는 당시 설정·실행 ID·판단을 보존한 역사 원장이다. 아래의 현재 상태·대기·향후 계획은
각 작성 시점의 기록이다. 2026-09-17 최종 상태와 결론은 [종합 보고서](../research-summary.md),
실제 측정 정의는 [검증 방법](../validation.md), 실행 안내는 [코드 안내](../code-guide.md)를 따른다.
정리한 과거 계획·중복 보고의 원문은 서버 정리 백업에 보존했다.
2026-09-03 압축 정리(운영 서사 축약, 수치·판정·섹션 번호 보존 — 전문은 git 히스토리).

## 0. 현재 상태 (2026-09-09 17:42 스냅샷)

- 메인: GPT. 정식 방법명 **RSVP — Residual Shaping-Value Prediction**. 현재 질문과
  판정 순서는 [README.md](../README.md), 에이전트 전달은 ../shareboard.md（당시 파일 `../shareboard.md`; 원문은 서버 정리 백업）.
- 실행 학습 6개: F25 shuffle Sacred276/275, LEHCA actual F25 Sacred279/280,
  RSVP actual Fmax200(Q-004 수정) Sacred282/281. actual 팔은 seed별 전용 vLLM 4개를 쓴다.
- 대기 4개: qmix_paper 5M Slurm955531/955532, shaping-only actual fixed F25
  Slurm955542/955543. 계정 동시 상한 10개가 반환되면 제출 순서대로 시작한다.
- 라이브 상태는 이 스냅샷을 믿지 말고 `squeue`, 프로세스, Sacred 마지막 `t_env`로
  다시 확인한다. 상세 설정·제외 런·판정은 바로 아래 §0b--0e에 있다.
- 원본 `vigil_*` 식별자와 기존 raw 결과는 추적성을 위해 보존한다. 코드 버전·mask·cache·
  temperature·lambda 시간축이 다른 런은 합산하지 않는다.

## 0a. S0b 재기동 (2026-09-08 20:20, Codex 착수 → Claude 인계 확인)

- 사유: 9/8 할당 10개 만료(21:40~23:24) vs S0b(Sacred255~261, 5m6m 1.2M) 잔여 4~8h.
  Codex가 240k AUC_early 회수본 저장(results/diagnostics/s0b_240k_recovery_20260908.md),
  완주 가능한 **none s0(Sacred256, tmux37, Slurm934810)만 보존**(ETA 22:30, 만료 23:24),
  나머지 6런 + vLLM(s38 n039:8356, s48 n038:8357) 반납.
- 신규 할당 7개(Slurm949368~949374, 12h, 만료 9/9 08:10): tmux `s0b3_*` 7세션,
  Sacred262~268 = rule s0/s1, none s1, shuffle s0/s1(풀 교차: s0←s1 로그, s1←s0 로그),
  aligned s1/s0. save_model=True·200k 체크포인트, wandb `s0b3_<arm>_F200_lam40_rsvp_path`.
  ETA 4.5h → 9/9 00:45 완주 예상. 감시 워처 가동(Claude 세션).
- 255/257~261은 절단 런 — 판정 제외, 삭제 안 함. LLM 실팔은 기존 F200 nc_t02 완료 런 재사용.
- 현재 vLLM 서버 없음 — LLM 호출 런을 다시 띄우려면 서버부터.

## 0b. S0c F25 시간척도 진단 (2026-09-09 16:19 투입, 16:26 교란 제거 후 재기동)

- 목적: 논문에 수치가 공개되지 않은 `F_update`를 200에서 25로 줄여, 5m6m 전투 중
  갱신이 가능한 시간척도에서 상태 의존 가이던스의 효용이 나타나는지 검사한다.
- 공통: 5m6m, fixed F25, lam40, epsilon anneal 50k, masking train/test off,
  `t_max=300k`, seed0/1. 기존 QMIX 및 S0b F200 곡선을 재사용한다.
- 최초 6런: Sacred269/270 aligned s1/s0, 271/272 rule s0/s1, 273/274 shuffle s1/s0.
  Slurm955140~955145, tmux `s0c_f25_*`, 6h 할당(9/9 22:18 만료).
- W&B: `s0c_<arm>_F25_lam40_rsvp_path_seed<k>`. 모델은 100k 간격 저장하며,
  PID가 포함된 `unique_token`으로 동시 실행 체크포인트 충돌을 방지한다.
- **16:23 설계 감사:** 5m6m에서 aligned는 항상 같은 damage/kill subgoal을 반환한다.
  rule도 양 팀 unit type이 Marine 하나뿐이고 Medivac이 없어 항상 같은 subgoal 집합을
  반환한다. masking off이므로 action rule도 작동하지 않는다. 따라서 두 팔은 F25/F200의
  학습 신호가 동일해 동적 갱신 필요성을 검정할 수 없다. Slurm955140/141/144/145를 약
  4분에 중단했고 성능 판정에서 제외한다.
- **16:24 추가 감사:** 최초 shuffle 런도 `t_max=300k, lambda_floor_frac=0.4`라서 shaping
  lambda가 120k에 floor에 도달한다. 비교 대상 S0b F200(1.2M, lam40)의 floor 시점은
  480k이므로 F 외에 lambda 시간축까지 달라지는 교란이 생긴다. Sacred273/274 및
  Slurm955142/143을 약 10k 전에 중단하고 성능 판정에서 제외했다.
- **현재 유효 실행:** shuffle F25 s0/s1을 `lambda_floor_frac=1.6`으로 재기동했다
  (Sacred275=s1, 276=s0; Slurm955178/179; tmux `s0c2_f25_shuffle_s0/s1`;
  W&B `s0c2_shuffle_F25_lam40_rsvp_path_seed<k>`). 이 설정은 300k까지 S0b F200의
  1.2M-lam40 lambda 궤적과 같은 절대 시간축을 사용하고, floor 시점도 480k로 맞춘다.
  따라서 공통 0--300k 구간에서 F25와 F200을 비교한다. 실제 LLM 호출은 없으며,
  완료된 LLM 지침 풀을 seed별로 결정론적으로 shuffle하여 재생한다.
- 이 실험은 상태 적응 효용이 아니라 같은 종류의 LLM guidance 풀의 **교체 빈도/비정상성**
  증가가 학습을 돕는지 또는 교란하는지를 측정한다. 완주 F200 전체 AUC/final이 아니라
  QMIX와 F200 곡선을 300k에 절단한 동일 horizon 지표로 판정한다.
- 동적 갱신 필요성의 올바른 후속 대조는 출력 분포·보상 크기를 맞춘
  `static / deterministic phase-aligned / phase-permuted` 세 팔이다. phase-aligned가 두
  대조군을 이길 때만 실제 LLM F25 및 RSVP 갱신 타이밍 검증으로 진행한다.

## 0c. LEHCA 원형 스타일 F25 실제 LLM 진단 (2026-09-09 16:50 투입)

- 문제의식: §0b shuffle은 guidance turnover 대조군일 뿐, 사용자가 제기한 “원래 LEHCA의
  짧은 F가 성능을 회복하는가”를 직접 검정하지 못한다. 따라서 실제 LLM Commander를
  쓰는 LEHCA 원형 스타일 F25를 본 진단으로 추가했다.
- 유효 런: Sacred279=s0 / 280=s1, Slurm955330/955331, tmux
  `s0c3b_lehca_f25_s0/s1`, W&B `s0c3_lehca_default_F25_b01_nc_t02_lam40_seed<k>`.
  두 런 모두 SC2 시작, API 200 응답, cache miss 상태의 유효 guidance가 t=0,25,...에서
  기록되는 것을 확인했다. seed별 전용 vLLM은 Slurm955319(n017:8356),
  955320(n018:8356)이며 전체를 18h로 할당했다.
- 설정: 5m6m, LEHCA runner, actual gpt-oss-20b, fixed F25, cache off, temperature 0.2,
  default prompt, shaping+train/test action masking on, beta=0.1, collection-time shaping,
  epsilon50k, 300k, seed0/1. `lambda_floor_frac=1.6`으로 기존 원형 스타일 F50-lam40
  1.2M의 절대 floor 시점 480k를 유지한다. 100k 간격 체크포인트.
- 대조: 기존 F50 원형 스타일 Sacred244=s0(약1.196M), 250=s1(324k)에서 공통
  0--300k 구간을 절단해 같은 seed로 비교한다. 따라서 이 비교에서 실질적 처치는
  F50→F25이다. QMIX 0--300k는 보조 기준이다.
- 판정: F25가 F50보다 두 seed에서 일관되게 좋아야 “기존 부진이 지나치게 긴 F 때문”이라는
  가설을 지지한다. F25도 동급/열세면 update 빈도보다 action masking·grounding noise·
  shaping 설계가 병목이다. 이 실험은 고정 F25 검정이며 RSVP의 동적 F 필요성을 직접
  증명하지 않는다.
- 인프라: vLLM 0.25.1에서 폐기된 `--disable-log-requests`를 제거하고, FlashInfer JIT용
  `module load cuda/13.1.1`을 `scripts/serve_llm.sh`에 추가했다. 실패한 서버 시도와
  Sacred 구성검사 실패는 성능 런에 포함하지 않는다. 최초 유효 Sacred277/278도 초기
  처리량상 12h 할당이 경계선이라 각각 1k 미만에서 선제 중단했으며 판정에서 제외한다.

## 0d. RSVP actual-LLM Fmax200, Q-004 수정 버전 (2026-09-09 17:18 투입)

- 목적: 같은 5m6m·300k·actual LLM 조건에서 고정 LEHCA F25와 현재 RSVP를 함께
  측정한다. RSVP가 최대 간격 200으로 호출을 절약하면서 CUSUM 조기 갱신으로 성능을
  유지하는지가 핵심이다. Fmax=100--200 후보 중 비용 대비 적응성을 가장 강하게
  검정하는 상단 200을 선택했다(min interval=10).
- 유효 런: Sacred282=s0 / 281=s1, Slurm955430/955431, tmux
  `s0d_rsvp_fmax200_s0/s1`, W&B `s0d_rsvp_Fmax200_q004_nc_t02_lam40_seed<k>`.
  두 런 모두 SC2, API 200, cache miss 유효 guidance 기록을 확인했다.
- 전용 vLLM: Slurm955419=n017:8357(seed0), 955420=n020:8358(seed1).
  서버·학습 모두 18h 할당. 현재 전체 Slurm 사용량은 shuffle 2 + LEHCA 학습 2 +
  RSVP 학습 2 + vLLM 4 = 10개다.
- 설정: current RSVP runner(Q-004 fix 포함), scheduler=vf, Fmax200, sched_k=.6,
  h=3, min interval=10, gamma=.8, eps_frac=.25, warmup20ep, target early=.15/ep,
  cache off, temp.2, shaping-only(mask train/test off), learner-time shaping,
  lam40 절대축(floor=480k), epsilon50k, seed0/1, 300k.
- 비교: 성능 AUC/final 외에 actual calls, refresh/ep와 refresh/step, early/fallback,
  CUSUM armed coverage, h/v를 보고한다. 이전 actual-RSVP Sacred236/248은 Q-004 수정 전
  코드이므로 참고만 하고 새 런과 합산하지 않는다.
- 판정: RSVP가 LEHCA F25에 비해 훨씬 적은 호출로 동급 성능이면 비용-성능 근거,
  성능까지 높으면 masking/과잉 갱신의 해악을 줄인 근거다. 둘 다 QMIX보다 낮으면
  동적 스케줄러가 아니라 guidance/shaping 채널 자체가 병목이라는 결론을 우선한다.

## 0e. 후속 슬롯 자동 충전 실험 (2026-09-09 17:39 예약)

- 현재 계정 동시 실행 상한 10개를 모두 사용 중이다. 완료 슬롯을 놀리지 않도록 후속
  4개를 Slurm 대기열에 제출했다(`AssocMaxJobsLimit`은 정상적인 대기 사유).
- **논문 예산 QMIX 감사:** Slurm955531/955532, tmux `s0e_qmix5m_s0/s1`, seed0/1,
  `qmix_paper`, 5m6m, **5M t_env**, 24h, W&B `s0e_qmix_paper_5M_budget_audit`.
  shuffle 두 런이 끝나면 먼저 시작할 예정이다. 기존 1.2M 결과와 0--1.2M을 비교하고,
  5M final/AUC로 논문과의 격차가 단순 예산 차이인지 판정한다.
- **스케줄러-only fixed F25:** Slurm955542/955543, tmux `s0f_fixed_f25_s0/s1`,
  seed0/1, RSVP shaping 경로, scheduler=fixed F25, actual LLM, cache off/temp.2,
  masking off, learner-time shaping, 300k, lam40 절대축, 18h. W&B
  `s0f_fixed_F25_rsvp_path_nc_t02_lam40`. Fmax200 학습 두 런이 슬롯을 반환하면 기존
  n017:8357/n020:8358 서버를 이어받는다.
- fixed F25는 §0d와 scheduler만 핵심적으로 달라, 현재 mask-on LEHCA F25와 RSVP의
  비교에 있던 masking/runner 교란을 제거한다. 따라서 RSVP Fmax200 대 fixed F25의
  성능·호출 수 차가 본 방법의 가장 직접적인 검정이다.

## 0f. LEHCA 논문 정합성 수정 후 원인분해 (2026-09-10 13:50 시작)

- 목적: true environment reward/training-stat 누출, 비가시 target grounding, stale test
  guidance, fixed-refresh cache skip, 모순 hard mask 및 ε50k를 수정한 뒤 LEHCA가 QMIX보다
  낮았던 원인을 shaping과 masking으로 분리한다. 수정 전 런과 합산하지 않는다.
- 공통: commit `1b0e9b9` 이후 reward/mask 진단 계측 추가 tree, SC2 4.10,
  ε anneal 300k, F50, gpt-oss-20b, paper prompt, cache off, temp0.2,
  training stats off, observable $d_t$, fresh isolated test Commander, 32 test episodes/10k.
- 5m6m, 1.2M, seed0/1 네 팔: QMIX / full LEHCA / mask-off(shaping-only) /
  shaping-off(mask-only). 2s3z, 300k, seed0 sanity 두 팔: QMIX / full LEHCA.
- Sacred: 287=2s3z qmix s0, 288/289=5m6m qmix s0/s1,
  291/292=5m6m full s1/s0, 294=2s3z full s0가 유효 시작됐다. 최초 290(shaping-off s0),
  293(mask-off s0)은 vLLM 한 대에 2--3 clients를 붙인 처리량이 48h 제한을 넘길 위험을
  시작 4분/t=25에 확인해 회수했으며 판정에서 제외한다.
- 재배치: vLLM 960649(n020)/960650(n031)에 960687(n023)을 추가했다. 새 mask-off s0는
  Sacred295/Slurm960690으로 API 200 및 유효 지침을 확인했다. 남은 960693=mask-off s1,
  960694=shaping-off s0, 960695=shaping-off s1은 각각 2s3z qmix/full 및 5m6m qmix
  종료 dependency 뒤 자동 시작한다. 총 10개 실험 조건은 유지하며 tmux 이름은 Slurm
  job name과 동일하다.
- 시작 검증: 모든 actual-LLM 런에서 SC2 연결 및 API 200을 확인했고 guidance JSONL에
  cache miss의 유효 출력이 기록됐다. 첫 출력의 forbid는 비어 있으며 보상함수·학습
  통계 문구는 입력에 없다.
- 판정: 5m6m AUC/final 외에 교전·사상·episode length, shaping magnitude 및
  $|\lambda F|/|R_{env}|$, hard/soft argmax override, attack/move category 제거율,
  호출/failure와 guidance 구성 분포를 함께 본다. 2s3z에서도 full이 QMIX보다 낮으면
  환경 특수성보다 guidance/grounding 자체를 우선 병목으로 판정한다.
- **17:30 오프라인 중간 감사:** 상세 수치와 재현법은
  offline-evidence-analysis-20260910.md（당시 파일 `archive/offline-evidence-analysis-20260910.md`; 원문은 서버 정리 백업）에 분리했다.
  동일 seed QMIX에서 ε50k→300k가 0–1M 승률 AUC를 약 0.0734→0.00047로 붕괴시켜
  ε 단독 교정은 faithful reproduction이 아니었다. 첫 100k에서 full LEHCA는 QMIX보다
  적 처치(1.00 vs 1.20)와 아군 사망(3.66 vs 4.70)이 모두 적고 episode가 길다
  (46.2 vs 32.1). hard override·LLM failure는 0에 가깝지만 soft override 13.4%,
  40–100k `|λF|/|r_env|` 42.7%라, 현재 우선 병목은 API/전면금지가 아니라
  **생존·후퇴 shaping과 soft bias에 의한 전투 소극성**이다. mask-off가 일부만 회복해
  shaping-off 완주가 가장 중요한 다음 판정이다.

## (구) 현재 상태 (2026-09-05 02:30)

- **9/4 16:05 만료 사건**: 전 할당(10개, vLLM 포함) walltime 만료 — 세션 부재 중이었으나
  밤 체인 런은 **전원 만료 전 완주**(데이터 손실 0). 9/4 16:05~9/5 02:00 유휴.
- **재구축(9/5 02:10)**: 신규 할당 10개(n029×4+n038×4+s34/s38, 만료 9/7 02시).
  vLLM: s2=n029:8356·s46=n038:8356. 투입: s40/s45 3s5z qmix s1·s2, s37 2s3z qmix s1,
  **s48 Pursuit Stage 1 qmix s0**, (vLLM 대기) s49 2s3z F100_lam40 s1+s2 체인,
  s50 2s3z vigil_lam40 s1+s2 체인, s34 2s3z qmix s2, s38 5m6m RSVP200_lam40 s3.
- **G2 판정: MMM2 탈락 확정** — F100_lam40 s0 = 0.033/0.285 vs qmix 0.196/0.566:
  셰이핑 채널 자체가 유해(RSVP 전멸도 채널 탓). 해악 경계 기록으로 강등, qmix 시드
  확충 취소.
- lam40 시드 현황: 5m6m 5팔 n=3 ✓ · 3s5z 3팔 n=3(qmix n=1→s1·s2 진행) ·
  2s3z 시드 확장 진행. 결과 집계 §8p.

## (구) 현재 상태 (2026-09-03 저녁 기준)

- **본선 = lam40 라운드** (λ 지평 비례 floor_frac 0.4, 분모 재평가 러너, 캐시 on/temp 0):
  5m6m 4팔(F100·F200·vigil_Fmax200·vigil_Fmax600, s45/49/46/48) + 3s5z 2팔(F100·RSVP,
  s50/34) 시드0 진행, MMM2 F100_lam40 s0(s38), 3s5z qmix s0(s40), 2s3z qmix s0 완료.
- **스케줄 계획 v3 (9/3 17:30 수립, 사용자 보고)**: 18시~ 6슬롯 오픈 → 시드1 일괄
  (s45 5m6m F100·s46 RSVP200·s48 RSVP600·s49 F200·s50 3s5z F100·s34 3s5z RSVP);
  19:30 s40 → 5m6m qmix s2 → 22:30 2s3z F100_lam40 s0 → 05:00 2s3z vigil_lam40 s0;
  01시 시드1 완주 → 시드2 일괄(07:30 완주 예정); **07:30부터 옛 할당 신규 LLM 런
  금지 → 재할당 사이클**(srun 반납→재요청, vLLM 2대 우선 → 새 주소로 이후 런);
  ~11시 s38 MMM2 판정 → 16:05 만료 무사고 통과 → 오후 잔여(3s5z qmix s1·2,
  2s3z 확장, Pursuit Stage 1, MMM2 조건부). 게이트: G1 lam40 시드 종합→2라운드,
  G2 MMM2 존치, G3 F200 거취, G4 Pursuit Stage 2.
- **리스크**: 옛 srun 8개(vLLM 2대 포함) walltime 48h → **9/4(금) 16:05 일괄 만료** —
  오전에 vLLM 재할당·훈련 런 보호 필요. MaxJobs=10/MaxSubmit=20.
- **wandb 삭제 대상(스트레이·폐기)**: GRF기 hxqysxrp·vntp152m·y7h7o9mr(+RSVP s2 조각),
  구시맨틱 3cjpw5vs·3b0t0exk·fe4m1p76, 구λ 중단 aehke9gd·07a58bag·ovuqn4wj·phsxr1rk·
  3l32cb6l·ubsyyme0 — 사용자 UI 삭제.
- **초안 반영 대기**: GPT 리뷰 문구 일괄(§8h) + 비용 축 per-step/총횟수 재정의(§8j·8m)
  + λ 스케줄 교체·λ창/워밍업 겹침 분석(§8k) + 희귀 서브골 커버리지 한계(§8k) +
  Fmax600 프레임 수정(§8m) + 발화 적합성 소절(§8n) + 검출 해상도 한정·리셋 불일치
  서사(9/3 당시 시간척도 기준으로 소급 판정).

## 1. SMAC 2s3z 진단 프로브 (2026-08-31, temp 0.2, 휴리스틱 아군, 9 ep)
휴리스틱 AI 궤적 + 5스텝마다 shadow LLM 2회(캐시 off)로 오프라인 재생 가능 로그 수집
(`probe_phase.py`/`analyze_probe.py`, results/vigil/probe_2s3z_heur_s0.jsonl).
결과: 같은 상태 2샘플 hard 불일치 **0.31 ≈ D_stale(0.25–0.37)** — temp 0.2 접지
노이즈가 낡음을 가림. 서브골은 에피소드 내내 고정(2s3z에 국면 전환 없음).
→ **전환 결정**: 에피소드 내 급변(소유권 ν)이 내생하는 GRF로 이동, temp 0 전제.

## 2. GRF 5_vs_5 프로브 (2026-08-31 ~ 09-01)
### 2a. 실행 기록 (압축)
- 빌드: conda env `grf`. 엔진 100 steps/s, LLM 3.5s/호출.
- 봇 대 봇 교착 → **하이브리드 봇 정책**(비소유: 공 최근접 1명 공 방향, 소유: 캐리어
  골 방향·x>0.7 슛)으로 ν 5–18회/ep 확보. 프로브 난이도 0.6.
- 런: `probe_grf_5v5_d06_t0`(10 ep, temp 0) / `..._t02`(3 ep, temp 0.2 대조).

### 2b. 결과 — temp 0, 10 ep (995 shadow 시점, ν 평균 9.1/ep)
- 노이즈 바닥선(같은 상태 2샘플): hard 0.008 / prefer **0.014**; 규칙 문자열 일치 65%,
  서브골 Jaccard 0.96 → temp 0에서 접지 노이즈 소멸(SMAC temp 0.2: 0.31).
- 낡음: D_stale Δ=10 0.071 → Δ=150 **0.167**(노이즈 12배). ν 정렬: 전환 창 0.139 vs
  무이벤트 창 0.045 (3배).
- **국면 모순율**(보유 가이던스가 현 국면과 정반대 지시): 신선 **7.8%** → Δ=100
  **47%** → Δ=150 **54%**. 소유권 평균 ~110스텝마다 전환.
- temp 0.2 대조(3 ep): 노이즈 0.059(4배), 규칙 일치 1%, 낡음 Δ=100 0.152 —
  신호/잡음 temp 0 12× vs 0.2 2.6×. temp 0 채택 근거.
- 판정: **C1 ✓**(낡음≫노이즈, ν 정렬), **C2 프록시 ✓**(모순율 8→54%),
  "빈도가 아니라 타이밍" ✓(동일 호출대 이벤트 트리거가 고정 F 지배 — §4 표),
  **C3 ✗**(φ-CUSUM은 이산 소유권 신호 이상을 못 뽑음 — CUSUM의 남는 가치는
  h로 예산 연속 조절). 서브골 적용가능비율 트리거(frac<0.7): 13.1호출/0.069/11 —
  도메인 이벤트 없이 손 이벤트 전선 도달(상세 draft §5.3, retired §3·5).

### 2c. 신선 가이던스의 상태 적합성 (temp 0, 7 ep, 796 시점)
국면 인식 ours→ATTACK 99%/theirs→DEFEND 98%, 서브골 국면 일관성 99–100%, 상황별
행동 선호 96–100% 적절. 약점: 박스 안 슛 강조 62%, 상대 소유 시 carrier 선택자 오용
73%(공집합 접지·무해), 세트피스 인지 45%. 다양성 ours 86종/theirs 191종(붕괴 아님).
→ "제때 다시 물으면 유효한 답이 있다"의 근거. 스크립트 analyze_guidance_quality.py.

## 3. 재생표의 계산 방법 (analyze_probe.py `replay`)
같은 로그를 스케줄별 재생: 트리거 발화 → 다음 shadow 시점의 기록된 LLM 출력을 채택.
호출/ep = 채택 횟수. 낡음 = 보유 g vs 신선 g\*를 같은 상태로 컴파일한 마스크 차이 칸
비율(effect.py). ν = 소유 팀이 공을 잃은 첫 스텝(상대 정착 시). 지연 = ν 후 첫 갱신.
오경보 = 직전 20스텝 무ν 갱신. 갱신이 10스텝 격자라 지연 최솟값 ≈5 — 상대 비교용.

## 4. 재생표 (GRF, temp 0, 10 ep) — 호출/ep | prefer 낡음 | 지연 | 오경보
| 스케줄 | 호출 | 낡음 | 지연 | 오경보 |
|---|---|---|---|---|
| F=ep | 1.0 | 0.157 | 543 | 1.0 |
| F=200 | 5.0 | 0.126 | 102 | 4.0 |
| F=100 | 10.0 | 0.111 | 45 | 8.0 |
| F=50 | 20.0 | 0.077 | 23 | 16.8 |
| F=25 | 33.9 | 0.057 | 15 | 28.7 |
| F=10 | 99.5 | 0.000 | 4.6 | 83 |
| 손 이벤트(전이 전부) | 28.2 | 0.037 | 5.0 | 15.7 |
| 손 이벤트(반대 팀 정착) | 10.6 | 0.071 | 15.9 | 4.3 |
| φ-CUSUM k0.1 h16 | 31.7 | 0.041 | 7.7 | 23 |
| 단발 임계값 h2 | 29.8 | 0.037 | 5.0 | 17.6 |
| 서브골 값 CUSUM (W10,k0.1,h2) | 1.8 | 0.135 | 418 | 1.5 |
| 손 표 적용 비율 < 0.7 | 13.1 | 0.069 | 11.1 | 4.4 |
| (1−frac) CUSUM k0.2 h3 | 10.6 | 0.074 | 22 | 4.7 |
| 학습 적용확률 < 0.5 | 35.3 | 0.034 | 68 | 23 |
| 학습 적용확률 CUSUM k0.3 h4 | 27.0 | 0.042 | 12 | 19 |
| 학습 적용확률 CUSUM k0.5 h2 | 17.5 | 0.056 | 107 | 10.7 |
| 학습 적용확률 CUSUM k0.5 h4 | 11.1 | 0.074 | 139 | 8.2 |

## 5. 신선 가이던스 적합성 → §2c.

## 6. 가이던스 진척 CUSUM (`gp`, 09-01) — 폐기 경로 (retired §5)
정의: 창 평균 발화 f̄_j를 컨텍스트 테이블 기대값 m̂_j(c)로 정규화(LOO), 단측 CUSUM.
GRF: gp k0.5 h2 = 29.8/0.042/10.6 (손 이벤트 동급), gp k0.3 h8 = 11.0/0.090/35
(저호출대에서 적용가능성 계열에 미달 — 창 확인 지연). SMAC 2s3z: 거의 안 욺(서브골이
전투 내내 보상 생성 = 정체 없음). → 컨텍스트 정의·게이트 등 설계 자유도 과다로 폐기,
잔여가치 크리틱으로 교체(§7).

## 7. 잔여가치 크리틱 오프라인 검증 (`vf_replay.py`, 09-02, GRF 10 ep 에피소드-LOO)
멀티헤드 MLP(14특징→128×128→9헤드), MC 타깃(할인 접미합).
- γ=0.97(~33스텝): **실패** R² −0.16 (미래 셰이핑이 소유권 뒤집힘에 지배 — 환원 불가).
- γ=0.8(~5스텝): R² **0.662** — ball_progress 0.94, keep 0.71, defensive_shape 0.63,
  press 0.61, regain 0.30; 발화 없는 헤드는 "신호 없음" 분류(가드 대상).
- 재생: vf k0.7 h2 = 12.4호출/낡음 0.086/지연 27.6 — F=100(10/0.111/45) 대비 낡음
  −23%·지연 −40%로 **고정 F는 이김**, 이벤트·적용가능성 계열(10.6–13.1/0.069–0.074)
  에는 미달. 원인: 통계적 확인 지연 + 저가치 발급 스펠의 판단 불가.
- 판정: 잔여가치 정식화는 "고정 F 우위"까지 오프라인 성립 → 최종 판정은 학습 비교로.

## 8. 1라운드 (구λ·캐시 on) — 투입과 교훈 (09-02, §8~8g 통합 요약)
- 16:23 GRF(500k)+MMM2 6런 투입 → 17:01 wandb 규칙 개편 재기동 → 17:12 MMM2→MMM
  (계산 예산; 21–30h/시드) → 밤 GRF·MMM 완주/조기종료.
- **캐시 관찰**: temp 0 + 거친 키로 GRF 갱신의 ~92% 캐시 적중(실호출 130 vs 적중
  1,487/25k) vs MMM 21–26% → 비용 축 논의 촉발. LEHCA 논문은 캐시 미구현(미래 확장
  언급뿐) — 재현 측 추가임을 공개.
- **GRF 판독 철회 사건**: "RSVP 종반 우위(last3)" 주장 → 분해 결과 **GRF는 학습되지
  않음**(train score_left 0.000 전 런, test 요동 = 평가 노이즈 SE 0.23–0.38) → 철회.
  문헌 확인: GRF 5v5에서 QMIX·QPLEX **100M 스텝에도 승률 0%**(GRF-MARL, arXiv
  2309.12951; PPO 계열만 성공, 128CPU+2×A100). 우리 qmix 2시드 평평 — 재현 버그 아님.
- **MMM**: F200 s0 조기종료 #1(250k, 0.969×3) · qmix s0 조기종료 #2(460k, ≥0.95×3,
  w2egmke1) — **가이던스 없이 포화** → 셰이핑 기여 여지 없음(경계 조건).
- **h 컨트롤러 결함·수정**: 게이트 닫힘 구간에서 h가 바닥(0.5) 고착 → hair-trigger.
  수정: 무장 에피소드만 적응 증거로, 바닥 0.5→2.0 (신규 런부터).
- **테스트베드 재편(21:50, 사용자 승인)**: 삼박자(학습 가능 ∧ 셰이핑 기여 ∧ 국면
  전환) 기준 GRF ①탈락·MMM ②탈락 → **SMAC 포트폴리오**: 5m_vs_6m(주 무대 — FINAL
  앵커: 셰이핑 final 0.42→0.945)·MMM2(파일럿)·2s3z(무해성)·3s_vs_5z(해악 경계).
  GRF는 §2–5 측정·재생 근거 전용으로 회수(봇 궤적 기반이라 유효). vLLM 분산:
  5m6m→n023, 기타→n039.
- **RSVP 예산 재설계**: RSVP = F_max 2F_base + te 0.15/ep(F100 동률 설계점) +
  Fmax600(효율 설계점, 9/2 사용자 F_max 완화 지시).
- 운영 사건(요지): SEEDS 루프 중복 기동 2건·워처 사망 수회 → 마커·통합 워처 체계로
  정비. 스트레이 wandb 런은 §0 삭제 목록.

### 8h. GPT 초안 리뷰 검증 + 러너 드리프트 수정 (09-02 22:30)
- 33항목 코드 대조: (i) "β=0 하드마스크 잔존" — 우리는 `use_action_masking=False`로
  모듈 off, **실험 무사**(초안 표기만 수정). (ii) R² 누수 — 이미 에피소드 LOO.
  (iii) **v_t 분모 크리틱 드리프트 — 실결함 확인**: 발급 시점 스칼라 고정이던 분모를
  x_ref 저장 + 매 스텝 현재 크리틱 재평가로 수정(runner.py). (iv) 자체 발견:
  protect류 predicate ≤0 — 게이트가 방어하나 "protect 위주 가이던스 상시 F_max 퇴화"
  한계 명시 필요. (v) Li et al. 게재처 정정: **Ann. Statist. 53(3) 2025**.
  Wang 2026(2607.13048)·Goel 2026(2608.18490) 실존 확인.
- s48 회수 → 5m6m vigil_Fmax600_te0.15 s0 투입(수정 러너 첫 적용).

### 8i. 신 시맨틱 통일 재시작 + 야간 자동 스케줄 (09-02 22:55, 사용자 승인)
구 시맨틱 RSVP s0 3개(각 ~25분) 폐기·재시작(5m6m·MMM2·2s3z — 분모 재평가 러너 통일).
통합 워처(night_watch.sh, 5분 폴링, 통지 전용) + 빈 슬롯 우선순위 큐 운영 개시.

### 8j. 1라운드 시드0 완주 결과 (09-03 11:20; AUC=전평가 평균/final=last 10%)
- **5m6m 구λ (1.2M)**: qmix 0.027/0.109 (3.1h) · F100 0.370/**0.521**(refresh
  0.254/ep, 6.4h) · F200 0.282/0.370 (0.126/ep, 8.3h) · vigil_Fmax200
  0.079/**0.513** (0.344/ep = early 0.071+fallback 0.273, h 2→11.7, 3.8h) ·
  vigil_Fmax600 0.273/0.404 (**0.062/ep** = early 0.019+fallback 0.043, 6.6h).
  판독: ①채널 확인(qmix≪F100) ②RSVP200 초반 게이트 닫힘 잠식→종반 F100 동급
  ③**per-ep 회계는 ep_len 내생성으로 왜곡**(RSVP200 fallback 0.273/ep = 초반 긴
  에피소드 ~55스텝 탓; per-step은 1/200+α < F100) → 비용 축 per-step/총횟수로
  ④Fmax600: F100의 1/4 예산에 AUC 0.273 — 효율 후보. 전부 n=1(5m6m 분산 ±0.30).
- **MMM2 구λ**: qmix 0.196/0.566/last3 0.677 (**학습 가능**, 문헌 저역대) vs
  vigil_Fmax200 **0.000/0.002 전멸** → 채널 유해 vs 스케줄러 분리 위해 F100 대조 투입.
- **2s3z 구λ RSVP**: 0.799/0.909 — 앵커(셰이핑만 0.941/qmix 0.963) 대비 무해성 유지.
- 벽시계 분해(9/3 오후 추가): 벽시계 ≈ 에피소드 수(=업데이트 수) 지배 — F200 50.2k
  에피소드 0.59s/ep=8.3h vs RSVP200 22.9k(ep_len 52) 0.60s/ep=3.8h. 실호출은 5m6m
  캐시 88–97%로 200–400회(수십 분)뿐 — _runtime 그래프로 오버헤드 판독 금지, 논문
  비용은 에피소드당 비용·실호출 수로. 캐시 off 추정(실측 1.6–2.5s/호출): F100 ~2배
  (11.7–14.5h), RSVP600 +15–25%(7.4–8h) — 캐시 off에선 비용 그림 역전.

### 8k. λ 스케줄 지평 비례화 + lam40 라운드 개시 (09-03 12:05, 사용자 지시)
- 실측: 구 스케줄은 1.2M의 **첫 8%(~100k)에 감쇠 종료**(0.34@20k→0.054@100k) +
  업데이트 기반이라 맵별 env-step 궤적 비일관.
- 새 스케줄: `λ(t)=max(λ_min, λ_start·(λ_min/λ_start)^p)`, `p=t_env/(0.4·t_max)`
  — 어떤 t_max든 40% 진행도에서 바닥(1.2M→480k, 2M→800k). 구현: state.set_lambda_
  progress + learner 분기(lambda_floor_frac>0; 0=레거시 — lehca.yaml 원형 불변).
- **lam40을 본선으로**(사용자: 구 결과에 얽매이지 않음): 구λ 세트는 예비 기록 강등,
  가이던스 런 6개 재시작(그룹 `_lam40`), qmix는 λ 무관 공용.
- 초안 대기 추가: **희귀 서브골 커버리지 한계** — 정책이 안 만드는 이벤트의 헤드는
  V̂≈0(행동정책 기준 참값이나 발급 후 셰이핑 학습 효과 과소평가) → 게이트 무장 거부
  → F_max 퇴화만. RSVP 이득은 "정책이 이미 만드는 이벤트"에 국한(실측 무장률 ~57%).

### 8l. qmix-first 큐 재편 + 인프라 (09-03 12:20, 사용자 지시)
MaxJobs=10 도달·유휴 셸은 GPU 없음·--overlap은 CPU 2개라 실익 없음 → 빈 슬롯 큐를
qmix-first로 재편(§0 큐 v2). walltime 리스크 §0. qmix 테이블: 5m6m s0✓ s1✓(0.80 —
시드 분산 실증) · MMM2 s0✓ · 2s3z s0✓(0.938) · 3s5z s0 진행 · MMM 포화 기록✓.

### 8m. 갱신원 분해 — CUSUM vs F_max 만기 (09-03 15:30)
CUSUM 비중(early/refresh): 5m6m RSVP200_lam40 **28%(24→31→35% 상승)** ·
RSVP600_lam40 21% · 3s5z 16%(하강) · 구λ 4런 21–30% — 전 런 만기 과반.
판독: 본선 팔은 크리틱 성숙과 함께 상승(기대 방향, 성능 우위와 동시).
**Fmax600 "CUSUM 주도" 의도 미실현(21%)** — 무장률 ~60% + 유효 가이던스는 저하가
없어 안 쏨(정상 동작; te는 상한). → 프레임 "극소 호출(F100의 1/4~1/5) 동급 성능"으로.
3s5z 0.82 refresh/ep는 긴 에피소드(~150스텝) per-ep 왜곡.

### 8n. CUSUM 발화 적합성 사후 검증 (09-03 15:50~16:10, 사용자 제안)
- 방법 1(가이던스 로그, temp 0 "같은 답=같은 상황"): CUSUM발 동일답 **0.0~0.1%**·키
  변경 ~100%·서브골 Jaccard 0.21–0.37 vs 만기발 동일답 1–12%·Jaccard 0.16–0.29 —
  낭비 0, 더 큰 전환을 골라 쏨. 발화 시 보유 평균 83/중앙값 71스텝(F_max 200).
- 방법 2(rule-based, cache_key의 실상태 필드): 주요 전환(Δ아군≥2 ∨ Δ적≥2 ∨ 시야)
  일치율 — 5m6m RSVP600 **99.0%**/RSVP200 **96.7%**/3s5z **78.1%** vs 만기
  78.1/78.4/53.9% (+18~24pp, LLM 해석 무관).
- 정성 샘플: 발화 = 전력 붕괴(A 5:x→1:0)·시야 상실 순간, 달성 불가 서브골 제거.
- 단서: 기저 변화율 자체가 높음(판별력은 마진), 무장 스펠 선택 편향, 다수가 패배
  진행 포착("승기 포착"과 구분 서술).
- **검출 지연 측정 준비**: 러너에 phase 로거 추가(results/phase/*.jsonl — cache_key
  변화 시각 = 초안 §3 ν 라벨 구현, 다음 런부터). 논문 §5 "발화 적합성" 소절 제안.
- (9/3 오후 논의) 검출 해상도: D_detect ≈ 10–15스텝(h≈4, k=0.6, min_interval 10) —
  그보다 짧은 미세 국면은 설계상 transient로 무시. 5m6m의 유효 낡음 단위는 붕괴·
  리셋 불일치·정책 성장(≥70스텝). 조건 일반화는 이후 테스트베드 시간척도로 판정.

### 8o. lam40 시드0 완주 — 6팔 결과 (09-03 18:30; AUC=전평가 평균/final=last 10%)
| 런 | AUC | final | refresh/ep | run |
|---|---|---|---|---|
| 5m6m RSVP200_lam40 | **0.419** | 0.487 | 0.157 | ag34hvp9 |
| 5m6m RSVP600_lam40 | 0.288 | 0.391 | **0.057** | 1dt18uat |
| 5m6m F100_lam40 | 0.252 | 0.354 | 0.246 | a0a37tek |
| 5m6m F200_lam40 | 0.091 | 0.578 | 0.262 | 0jerkkab (늦깎이 급등) |
| 3s5z F100_lam40 | 0.393 | **0.979** | 1.674 | ld697pq7 |
| 3s5z vigil_lam40 | 0.389 | 0.956 | **0.897** | 1fk5crqz |
- 판독(전부 n=1): ① 5m6m — **RSVP 두 팔이 F100을 AUC·final·호출 모두에서 지배**
  (RSVP200 AUC +66%·호출 −36%; RSVP600은 호출 23%로 우위). 구λ의 "초반 잠식"이
  lam40에서 AUC 1위로 역전 — λ 창/워밍업 겹침 해소 효과와 부합. F200은 slow-start
  후 final 최고(0.578) — 5m6m 분산 경고 유지, 시드 필수. ② 3s5z — **동급 성능
  (0.96~0.98), RSVP 호출 절반** = 효율 프레임 교과서 사례. 셰이핑-only는 유해는커녕
  qmix 앵커(0.568) 압도 → 구 "3s5z 유해" 이력은 마스킹+구λ 조합 문제로 재해석.
- 조치: 6팔 전부 SEEDS='1 2' 자동 체인 재투입(01시 시드1 완주 → 시드2 자동).
  다음: s40 3s5z qmix(~19:20) → 5m6m qmix s2 && 2s3z F100_lam40 && vigil_lam40 3연쇄.

### 8p. lam40 라운드 시드 집계 (09-05 02:20; AUC=전평가 평균/final=last 10%, mean±std)
| 맵 | 팔 | n | AUC | final |
|---|---|---|---|---|
| 5m6m | qmix | 3 | 0.184±0.112 | 0.421±0.304 |
| 5m6m | F100_lam40 | 3 | 0.087±0.117 | 0.131±0.158 |
| 5m6m | F200_lam40 | 3 | 0.309±0.156 | **0.578±0.064** |
| 5m6m | RSVP200_lam40 | 3 | 0.222±0.157 | 0.539±0.210 |
| 5m6m | RSVP600_lam40 | 3 | 0.099±0.134 | 0.145±0.174 |
| 3s5z | qmix (2M) | 1 | 0.442 | 0.923 |
| 3s5z | F100_lam40 | 3 | 0.231±0.128 | 0.662±0.235 |
| 3s5z | vigil_lam40 | 3 | **0.403±0.059** | **0.926±0.022** |
| 2s3z | qmix / F100_lam40 / vigil_lam40 | 각 1 | 0.808 / 0.801 / 0.804 | 0.938 / 0.988 / 0.969 |
| MMM2 | qmix / F100_lam40 | 각 1 | 0.196 / 0.033 | 0.566 / 0.285 |
- **판독 (G1)**: ① **5m6m은 시드 분산이 팔 차이를 삼킴** — n=3으로 분리 불가
  (F100 시드1·2 붕괴 ≈0, RSVP600도 시드1·2 붕괴; F200·RSVP200만 전 시드 생존).
  시드0의 "RSVP 지배"는 유지되지 않음. 주 무대로서의 5m6m 재고 필요(n≥5 또는
  무대 전환 — 사용자 결정 대기). ② **3s5z가 가장 깨끗한 서사**: F100(고정 주기)은
  qmix보다 **해로움**(0.662 vs 0.923), RSVP는 해악을 중화해 qmix 동급(0.926±0.022,
  저분산) + 호출 절반 — "타이밍 개선이 나쁜 채널의 해악을 제거"라는 harm-reduction
  결과. qmix n=3 확충으로 확정 예정. ③ 2s3z 무해성 유지. ④ MMM2 채널 유해 확정(G2).

### 8q. Pursuit Stage 1 통과 + 야간 시드 보강 (09-05 07:00)
- **Pursuit Stage 1 (qmix 단독 1M, yrzm7x8y): 통과** — test return −46.8(무작위 수준)
  → 490k에 −0.3 → 1M에 **+15.0**, 단조 상승·미포화. QMIX 학습 가능 확인
  (당시 Stage 1 ✓). qmix s1 확인 런 투입(s48). **Stage 2(시맨틱
  인터페이스·predicate·커맨더 프롬프트, 코딩 1-2일) 착수 가능 — G4 사용자 결정.**
- 야간 완주: 2s3z qmix s1 0.981(9n12hvvs)·s2 1.0(1sje3dzl) → **2s3z qmix n=3**
  (0.938/0.981/1.0). 5m6m RSVP200_lam40 s3 마지막 관측 0.763(l6ts0guh) — 생존
  시드 추가로 n=4.
- 투입: s34 3s5z F100_lam40 s3 · s37 5m6m F200_lam40 s3 · s38 3s5z vigil_lam40 s3 ·
  s48 pursuit qmix s1. 진행: 3s5z qmix s1·s2(~09시), 2s3z lam40 체인 ×2.

### 8r. Codex 원시 로그 재검증 — 완주 현황 및 해석 정정 (2026-09-05 18:20 KST)

#### 검증 범위와 지표

- 현재 lam40 라운드와 비교 가능한 QMIX **41런**을 오프라인 분석.
  `analysis/archive/audit_20260905.py` 실행 결과, 모든 평가 win-rate의 **시점·값 전체**가
  Sacred `info.json`과 로컬 W&B `run-*.wandb` 사이에서 일치했다.
- AUC는 기존 §8p와 같은 **전 평가 승률 산술평균**이다(시간 적분 또는 AUC_early 아님).
  final은 **각 런 마지막 평가 시점 t_last의 90% 이후 평가 평균**. t_max의 90%와
  경계가 달라질 수 있어 명시한다. ±는 시드 간 population std(ddof=0), 신뢰구간 아님.
  본 라운드 내 정의를 바꾸지 않았으며, 중단된 구λ/짧은 런은 혼합하지 않았다.
- 완주 판단은 마지막 평가가 예산의 98% 이상인 것과 학습 세션 종료 메시지를 함께 확인.
  Sacred `run.json`의 RUNNING 표기는 실제 프로세스 생존 증거로 사용하지 않는다.
- 18시 점검: tmux **34,37,38,40,45,48,49,50 모두 Finished Training 후 셸 대기**.
  마지막 런은 각각 3s5z F100 s3, 5m6m F200 s3, 3s5z RSVP s3,
  3s5z QMIX s1/s2, Pursuit QMIX s1, 2s3z F100/RSVP s2.
  Slurm GPU 할당 10개는 살아 있으나 이것이 학습 10개 가동을 뜻하지 않는다.
  당시 잔여 약 32시간(9/7 02시대 만료); 새 학습은 이번 검토에서 실행하지 않았다.

#### 전체 집계

| 맵 | 팔 | seeds | AUC mean ± std | final mean ± std |
|---|---|---|---|---|
| 2s3z | F100 lam40 | 0,1,2 | 0.763 ± 0.027 | 0.949 ± 0.031 |
| 2s3z | QMIX | 0,1,2 | 0.806 ± 0.004 | 0.968 ± 0.022 |
| 2s3z | RSVP200 lam40 | 0,1,2 | 0.805 ± 0.002 | 0.939 ± 0.032 |
| 3s_vs_5z | F100 lam40 | 0,1,2,3 | 0.282 ± 0.142 | 0.709 ± 0.222 |
| 3s_vs_5z | QMIX | 0,1,2 | 0.493 ± 0.041 | 0.943 ± 0.014 |
| 3s_vs_5z | RSVP200 lam40 | 0,1,2,3 | 0.409 ± 0.052 | 0.920 ± 0.022 |
| 5m_vs_6m | F100 lam40 | 0,1,2 | 0.087 ± 0.117 | 0.131 ± 0.158 |
| 5m_vs_6m | F200 lam40 | 0,1,2,3 | 0.255 ± 0.164 | 0.564 ± 0.060 |
| 5m_vs_6m | QMIX | 0,1,2 | 0.184 ± 0.112 | 0.421 ± 0.304 |
| 5m_vs_6m | RSVP200 lam40 | 0,1,2,3 | 0.198 ± 0.142 | 0.576 ± 0.192 |
| 5m_vs_6m | RSVP600 lam40 | 0,1,2 | 0.099 ± 0.134 | 0.145 ± 0.174 |
| MMM2 | F100 lam40 | 0 | 0.033 ± 0.000 | 0.275 ± 0.000 |
| MMM2 | QMIX | 0 | 0.196 ± 0.000 | 0.569 ± 0.000 |

Pursuit QMIX 1M: s0/s1의 final **return = 15.288 / 13.169**, 평균 **14.228 ± 1.060**.
final 승률(모든 evader를 제한시간 전에 포획)은 **0.2125 / 0.0750**,
final 평가 episode length는 **479.74 / 492.84**(상한 500).
따라서 두 시드에서 학습 가능성은 관측되지만 포획 성공률은 낮고, LLM 셰이핑의
유용성이나 가이던스 수명은 아직 검증하지 않았다. §8q의 단일 최신값은 final 평균과 다르다.

#### 해석 — 사실과 가설 구분

1. **3s5z: F100 대비 회복은 있지만 QMIX 개선은 아니다.**
   F100 vs RSVP는 s0–3 공통으로 AUC 차이 +0.1267, final +0.2105.
   그러나 QMIX와 공통 s0–2만 비교하면 RSVP의 AUC 차이는
   **−0.0536 / −0.0155 / −0.2029**(평균 −0.0907), final 차이 평균 −0.0172.
   QMIX보다 학습이 느린 현상이 세 공통 시드에서 일관적이다. final이 비슷하다는
   관측은 정식 동등성/비열등성 증명이 아니다. 또한 F100 vs Fmax200 비교만으로는
   **적응 타이밍 효과와 단순한 저빈도 갱신 효과를 분리할 수 없다.**
2. **5m6m: F200은 필수 기준선이며 현재 RSVP이 이를 지배하지 않는다.**
   s0–3에서 RSVP−F200 AUC = +0.3273/−0.3484/−0.2417/+0.0344,
   final = −0.0911/−0.1875/+0.1615/+0.1641.
   평균 final 차이는 +0.0117뿐이고 AUC는 −0.0571. 효과 방향이 시드에 따라 뒤집힌다.
   “분산이 크다”와 “RSVP 우위가 이미 입증됐다”는 별개이며 후자는 지지되지 않는다.
   F100/RSVP600의 s1–2 붕괴는 유지되어 해당 팔의 무조건적 시드 확대 우선순위는 낮다.
3. **2s3z: AUC 유지, final의 무해성은 아직 확정 아님.**
   RSVP−QMIX 공통 시드 AUC 평균 −0.0003, final 평균 −0.0292.
   특히 s1 final 차이 −0.0844를 숨기고 “무해성 확인”이라고 쓰지 않는다.
   포화 sanity 맵으로 남기고 추가 대규모 탐색은 보류 권장.
4. **MMM2: 현 F100 lam40 설정의 큰 손해를 관측(n=1).**
   §8p의 final 0.285/0.566 대신 원시 이력의 동일 정의는 **0.2750/0.56875**.
   “해당 구성은 유해해 보인다”는 맞지만 한 seed·한 F로 모든 셰이핑 채널 또는
   모든 스케줄에 대한 유해성을 확정할 수 없다. 현 예산에서 중단 우선순위는 유지.
5. **Pursuit는 Stage 2 파일럿 후보이지 RSVP 성공 무대 확정이 아니다.**
   긴 episode만으로 전략 수명 조건이 증명되지 않는다. 고정 셰이핑의 기여 및
   실행 가능한 목표 표현을 먼저 검사해야 한다.

#### 비용과 검출기 가동률

- Guidance JSONL을 시작 시각·맵·seed 및 W&B 캐시 히트 누적치로 유일하게 매칭.
  다음 refresh는 **전체 성공 갱신 이벤트**, LLM calls는 **마지막 기록 시점의
  commander n_calls**다. 후자는 성공 HTTP 응답 후 증가하며 JSON 재시도 포함 가능,
  전체 요청 시도/토큰/GPU 비용과는 다르고 마지막 미기록 tail도 있을 수 있다.
- 3s5z s0–3: F100 vs RSVP의 평균 refresh **20,001.5 → 10,752.5 (−46.2%)**;
  기록 LLM calls **763.5 → 689.25 (−9.7%)**. 따라서 “LLM 비용 절반”은 부정확하다.
  RSVP의 평균 갱신 간격은 약 **186 training steps**로, 고정 F200 대조군이 특히 중요.
  early 이벤트는 전체 refresh의 약 **17.1%**다.
- 5m6m s0–3: F200 vs RSVP 평균 refresh **6,001 → 7,522.5 (+25.4%)**;
  기록 LLM calls **346.25 → 348.25**. 현재 동일 ceiling 기준 비용 절감도 입증되지 않았다.
- 2s3z s0–2: F100 vs RSVP 평균 refresh **10,001 → 5,724.7 (−42.8%)**;
  기록 LLM calls **2,771.3 → 1,846.7 (−33.4%)**. 성능과 함께 보고할 비용 절충 사례.
- RSVP200의 발급 단위 armed 비율 `sum(ok)/sum(ok+low_vref+no_heads+warmup)`:
  3s5z **74.6–79.4%**, 5m6m **31.2–63.8%**, 2s3z **77.8–79.2%**.
  나머지는 거의 low_vref이고 이 런들에서 no_heads는 0.
  이는 **발급 비율**이며 timestep 가동률 C_time과 다르다.
  특히 5m6m은 detector가 닫힌 기간과 fallback 영향을 분리해야 한다.
- 실제 코드에서 `_ep_armed`는 새 armed 발급 때만 True이고 매 train episode 종료마다
  False가 된다. 이전 episode의 guidance/reference는 유지되므로 **기존 armed spell로
  감지한 다음 episode가 h 예산 창에서 누락될 수 있다**. 코드 경로는 확인했으나
  실측 발생률·성능 영향은 UNKNOWN. 셰어보드 Q-004에서 최소 재현을 요청한다.
- phase cache-key 전환은 상태 요약 변화의 proxy이지 “가이던스가 쓸모없어지는 정답 시점”
  이 아니다. 현재 로그만으로 CUSUM drift separation이나 정확한 T_valid 분포를
  입증했다는 표현은 금지. 후속 계획은 셰어보드 Q-003 참고.

#### 시드별 추적표

Pursuit 행의 AUC/final도 이 표에서는 **승률**이며 return은 위 별도 수치를 사용한다.

| 맵 | 팔 | seed | Sacred / W&B run | AUC | final |
|---|---|---|---|---|---|
| MMM2 | QMIX | 0 | 140 / jcefvg4z | 0.1965 | 0.5687 |
| 5m_vs_6m | QMIX | 0 | 142 / cn9pu349 | 0.0266 | 0.1094 |
| 5m_vs_6m | QMIX | 1 | 153 / cuynb64z | 0.2469 | 0.8333 |
| 3s_vs_5z | RSVP200 lam40 | 0 | 157 / 1fk5crqz | 0.3887 | 0.9563 |
| 3s_vs_5z | F100 lam40 | 0 | 158 / ld697pq7 | 0.3932 | 0.9781 |
| 5m_vs_6m | F100 lam40 | 0 | 159 / a0a37tek | 0.2523 | 0.3542 |
| 5m_vs_6m | RSVP200 lam40 | 0 | 160 / ag34hvp9 | 0.4188 | 0.4870 |
| 5m_vs_6m | RSVP600 lam40 | 0 | 161 / 1dt18uat | 0.2883 | 0.3906 |
| 5m_vs_6m | F200 lam40 | 0 | 162 / 0jerkkab | 0.0914 | 0.5781 |
| 2s3z | QMIX | 0 | 163 / cejs1f9c | 0.8084 | 0.9375 |
| 3s_vs_5z | QMIX | 0 | 164 / 2s7zdlhv | 0.4422 | 0.9234 |
| MMM2 | F100 lam40 | 0 | 165 / vdt4c9ct | 0.0331 | 0.2750 |
| 3s_vs_5z | F100 lam40 | 1 | 170 / v2lhim53 | 0.0795 | 0.4172 |
| 3s_vs_5z | RSVP200 lam40 | 1 | 171 / 3u4qmkqz | 0.4804 | 0.9047 |
| 5m_vs_6m | F200 lam40 | 1 | 172 / fbc43ln4 | 0.3831 | 0.5000 |
| 5m_vs_6m | F100 lam40 | 1 | 173 / vb7g4rvd | 0.0003 | 0.0026 |
| 5m_vs_6m | RSVP200 lam40 | 1 | 174 / s5g2wsfk | 0.0346 | 0.3125 |
| 5m_vs_6m | RSVP600 lam40 | 1 | 175 / 44znvcn8 | 0.0076 | 0.0365 |
| 5m_vs_6m | QMIX | 2 | 176 / 9s5tb2xx | 0.2792 | 0.3203 |
| 5m_vs_6m | F100 lam40 | 2 | 177 / obttfdu2 | 0.0083 | 0.0365 |
| 2s3z | F100 lam40 | 0 | 178 / ioyim2eb | 0.8006 | 0.9875 |
| 5m_vs_6m | RSVP600 lam40 | 2 | 179 / ktu3oajr | 0.0008 | 0.0078 |
| 5m_vs_6m | RSVP200 lam40 | 2 | 180 / 9pvjf0su | 0.2107 | 0.8177 |
| 3s_vs_5z | F100 lam40 | 2 | 181 / 3npviwp1 | 0.2209 | 0.5813 |
| 3s_vs_5z | RSVP200 lam40 | 2 | 182 / 0h1f49d4 | 0.3394 | 0.9172 |
| 5m_vs_6m | F200 lam40 | 2 | 183 / 0h2awdyx | 0.4523 | 0.6563 |
| 2s3z | RSVP200 lam40 | 0 | 184 / gf4ihc2i | 0.8041 | 0.9688 |
| pursuit | QMIX | 0 | 185 / yrzm7x8y | 0.0350 | 0.2125 |
| 3s_vs_5z | QMIX | 2 | 186 / 9amcc07i | 0.5422 | 0.9531 |
| 3s_vs_5z | QMIX | 1 | 187 / 9mlxafpf | 0.4959 | 0.9531 |
| 2s3z | QMIX | 1 | 188 / 9n12hvvs | 0.8000 | 0.9781 |
| 2s3z | RSVP200 lam40 | 1 | 189 / engvblz1 | 0.8075 | 0.8938 |
| 2s3z | F100 lam40 | 1 | 190 / lhrfv8e0 | 0.7356 | 0.9469 |
| 2s3z | QMIX | 2 | 191 / 1sje3dzl | 0.8084 | 0.9875 |
| 5m_vs_6m | RSVP200 lam40 | 3 | 192 / l6ts0guh | 0.1279 | 0.6875 |
| 3s_vs_5z | F100 lam40 | 3 | 193 / fnt0gzos | 0.4345 | 0.8609 |
| 5m_vs_6m | F200 lam40 | 3 | 194 / d1q0caxu | 0.0935 | 0.5234 |
| 3s_vs_5z | RSVP200 lam40 | 3 | 195 / ajcvi115 | 0.4267 | 0.9016 |
| pursuit | QMIX | 1 | 196 / lqkiy6gm | 0.0238 | 0.0750 |
| 2s3z | RSVP200 lam40 | 2 | 197 / 7t8s7flm | 0.8044 | 0.9531 |
| 2s3z | F100 lam40 | 2 | 198 / 0ptdm85r | 0.7538 | 0.9125 |

재현: 저장소 루트에서
`/home1/paul6598/miniconda3/envs/aamas/bin/python analysis/archive/audit_20260905.py`.
코드는 로그를 읽고 JSON을 stdout에 출력할 뿐 서버 접속·실험 파일 수정은 하지 않는다.

### 8s. 코덱스 계획 실행 + Pursuit Stage 2 구현 완료 (09-05 저녁, 사용자 승인)
- **P1 투입(코덱스 Q-003)**: 3s5z fixed F200_lam40 s0-3(각 2M, s40/45/49/50) +
  3s5z qmix s3(s37) + 5m6m qmix s3(s34) — "같은 상한의 고정 F200으로 충분한가"가
  판정 질문. F200 ≥ RSVP이면 SMAC 적응 타이밍 우위 주장은 접는다(사전 등록).
- **P0 진단**: runner에 opt-in per-step 트레이스(sched_trace — v 분자/분모 원값·S·h·
  armed·발화 사유; 의사결정 불변) 추가, 100k 진단 런 2개(diag_trace_3s5z/5m6m,
  s48/s38). Q-001 실측(전환 전후 E[k−v] 부호)·Q-002(분모 floor 진입률)용.
- **Q-004 재현 확정**: 에피소드 관통 armed 스펠이 h 적응 창에서 누락(2케이스 mock
  재현) — 1줄 패치 제안, 본선 적용은 승인 대기(보드 참조).
- **Pursuit Stage 2 구현 완료**: shaping/pursuit.py(14헤드: approach/encircle/
  blockade×4섹터 + catch/tag_pressure — 델타형은 **부호 있는 퍼텐셜 차**로 왕복
  해킹 원천 차단), env/semantic/pursuit.py(섹터 요약·캐시 키), commander/pursuit.py
  (셰이핑-only 프롬프트, sector→unit_type 매핑), predlib/러너/레지스트리 연결.
  **유닛 테스트 16/16 통과**(텔레스코핑·sanitize·헤드 해석). LLM 파이프라인 스모크
  통과 — 실제 가이던스 예: "encircle SW + blockade SE to funnel evaders" (키
  pursuit|c0|E2222|P3211, 25스텝 주기 갱신 정상).
- 다음: diag 완주 슬롯(s48/s38) → **pursuit lehca-shape_F100_lam40 s0·s1**(1M,
  코덱스 P3 순서: QMIX vs 고정 셰이핑 먼저, RSVP는 채널 이득 확인 후).

### 8t. P1 판정 — 3s5z 적응 타이밍 우위 철회 (09-06 02:40, 사전 등록 규칙)
| 팔(n=4) | AUC | final |
|---|---|---|
| qmix | 0.495±0.035 | 0.945±0.013 |
| F100_lam40 | 0.282±0.142 | 0.713±0.221 |
| F200_lam40 | 0.432±0.087 | 0.902±0.037 |
| vigil_lam40 | 0.409±0.052 | 0.920±0.022 |
- 공통 시드 차(RSVP−F200): AUC −0.037/−0.014/+0.046/−0.088(평균 −0.023),
  final +0.094/−0.026/+0.049/−0.044(평균 +0.018) — 잡음 안. F200이 비용 더 낮고
  동급 → **코덱스 Q-003 사전 등록대로 SMAC 적응 타이밍 우위 주장 철회**. P2(예산
  매칭·무작위) 스킵(조건 불성립). 유리한 k/h/λ 사후 탐색 금지 준수.
- 남는 실증: 고정 주기 취약성(F100 0.713 vs F200 0.902 — ΔF 하나로 0.19), 발화
  적합성(§8n), 채널 부호 맵 의존(§8p). 갈림길 = **최적 F의 환경 의존성**: pursuit
  F100/F200 파일럿(진행 중)이 방향을 정함 — 뒤집히면 "환경 간 안전 기본값" 서사,
  아니면 측정·진단 논문 프레임으로.
- 런: F200 s0-3 = uccxg70z/dcif6wyn/u0m0spaq/contit0j, qmix s3 = ncer0jmw.
- Pursuit 채널 중간: F100 s0 return 19.7(+37% vs qmix 14.2±1.1) / s1 13.4(동급)
  — 미확정, s2 + qmix s2 진행 중.

### 8u. Codex 종합 감사 — 성능·후회·비용·검출기 및 Pursuit semantic 결함 (09-06)

**데이터 기준: 2026-09-06 20:00–20:08 KST.** 이후 완주 런은 포함하지 않는다.
상세 수치·정의·신규 run ID·재현 명령은 이 절과 연결된 원시 로그에 둔다.
§8r/§8t와 수치가 다른 항목은 부록의 final 시간창 정의 및 공통 seed 범위를 먼저 확인한다.

- 이전 41런과 신규 14런을 합친 완주 55런 검토. 신규 런은 Sacred와 로컬 W&B의
  평가 시점·return·승률 전체 이력 대조 일치. 미완주 Pursuit은 공통 600k 비교로만 분리.
- 판독: F100 대비 회복 및 갱신 감소는 지지되나, F200 대비 일관된 우위·효율은 미입증.
  final 맵 평균의 작은 격차를 시드별 위험·학습곡선 전체·미평가 환경의 강건성으로
  확대할 수 없다. min-regret/비열등성 서사 확정은 보류 권고.
- Pursuit은 공통 시드에서 F100도 QMIX 대비 AUC 개선이 있어 “고정 채널 이득 0,
  RSVP만 이득 창출” 해석을 정정한다. return과 전원 포획 성공률의 순위도 다르다.
- **신규 결함:** semantic snapshot이 개체 수 대신 점유 칸 수를 세어 실제 포획 없는
  `catch` 셰이핑이 발생한다. 격리된 실제 환경 100스텝 재현으로 확인(부록 §7).
  외부 return이 잘못 기록됐다는 뜻은 아니며, 성능 영향의 크기는 아직 미측정이다.
- CUSUM 조기 갱신의 실제 발생은 확인. 유효한 교체 시점 검출은 미입증.
  floor/clip/가동률 및 Q-004 회계 누락은 초기 SMAC 진단 범위로 한정해 보고한다.
- 권고: semantic 회귀 테스트/수정안 → Q-004 분모 정리 → 변경 버전 분리·조건 고정 비교.
  기존 결과는 보존. cache-off와 temperature 변경을 한꺼번에 진행하지 않는다.
- 이번 작업은 분석·재현·문서화이며 학습 코드/config 변경 및 학습 실행·종료는 미수행.
  Claude 확인 요청은 셰어보드 Q-005（당시 파일 `../shareboard.md`; 원문은 서버 정리 백업）에 기록했다.

### 8v. 코덱스 감사 대응 + 사용자 승인 2건 (09-06 저녁)
- **Pursuit 가짜 catch 결함**(Q-005): 독립 재현(stay-100, 가짜 catch 37회/실포획 0)
  → env/semantic/pursuit.py `grid_entities`로 개체 수 기반 수정, 회귀 테스트 18/18 +
  실환경 재검 0회. SMAC 무관. 결함 라운드(pursuit 가이던스 런 전부)는 보존·비합산.
- **주장 수위 조정**(감사 수용): min-regret 확정 보류(pursuit qmix s3=22.9 →
  qmix n=4 15.7±4.4, RSVP 우위 통계력 부족), 텔레스코핑 주장은 비할인·고정 가중치
  합으로 한정, pursuit F100 채널 이득 정정.
- **사용자 승인(9/6)**: ① Q-004 패치 적용(runner.py — ready 에피소드 무장 집계;
  이후 RSVP 런은 새 버전) ② **pursuit-v2 라운드**: 수정 시맨틱 + 패치 러너, 그룹
  접미 `_v2`, F100/F200/RSVP × 시드0-3(12런, 2런 체인 × 6슬롯), 타 조건 동결.
  qmix는 시맨틱 미사용이라 재런 불요(n=4 보유: 15.3/13.2/11.4/22.9).
- 야간 계획: 결함 라운드 잔여 5런 완주 회수 → 02:03 만료 전 재할당 사이클
  (vLLM 2대 우선) → v2 12런 투입.

### 8w. Codex LEHCA 접지 감사·고정 상태 반복 프로브 (09-07 12:44)

사용자 요청에 따른 실측. 상세 정의·원자료·재현·후속 순서는
[lehca-grounding-audit-20260907.md](lehca-grounding-audit-20260907.md).

- 실제 SMAC 5m6m heuristic 8에피소드/157전이: Marine typed kill/damage는 기본
  predicate의 정확한 배수. 일반 최대 6기저, 이번 전이 행렬 rank5(retreat 미발생).
  합성 상태에서 진형의 summary 소실, 방향의 cache key 소실, 목표 조건 sanitize 소실 재현.
- 기존 완주 F200/V200 8런의 54,094회 발급 기록: 고유 유효 보상 계수 135–162개/런,
  직전과 동일 계수 갱신 3.35–12.98%. “항상 같은 보상” 가설은 지지되지 않음.
- cache off/temp0.2, 6상태×3프롬프트×2회: 36/36 반환, 48 HTTP, 미해결 head0.
  default/twostage의 동일 상태 반복 보상 RMS(0.241/0.519)가 상태별 평균 간 차이
  (0.164/0.380)보다 큼. 소표본 기술 통계이며 학습 성능/유의성 판정 아님.
  paper는 2/12개 응답에 subgoals가 없어 shaping-only에서 보상0.
- 자유 계획의 팀 체력50% 후퇴 등이 실제 개별 HP30% predicate로 축소되는 사례 확인.
  twostage llm_calls12 대 실제 HTTP24: 프롬프트 간 비용 비교 시 기존 로거 주의.
- 공식 보충자료 ε anneal300k 대 현재50k 확인. 현재 Sacred236/237은
  F200/F50, masking, λ, replay 방식이 달라 직접 스케줄러 비교 불가.
- 우선순위: 현재 RSVP과 일치하는 fixed F200 팔 → QMIX 포함 ε50k/300k 비교 →
  비-LLM 정적 보상 대조/조건부 접지 확장. 신규 장기 학습은 시작하지 않았으며,
  기존 두 런은 보존. 학습 소스·config 수정 없음. Claude 확인 요청은 Q-006.

### 8x. RSVP 명칭 전환 및 첫 통제 대조군 실행 (09-07 13:41)

- 사용자 승인으로 코드/registry/config/script/문서/연결된 그림을 RSVP로 전환.
  현재 계획·판정은 [README.md](../README.md)에 통합했다. 이전 명칭의 raw ID·결과 파일과
  진행 중인 런을 덮어쓰지 않았다. 모든 기존 방법 논리/수치는 명칭 변경과 분리한다.
- `analysis/test_rsvp_migration.py`: 네 테스트 통과. before.tar와 AST 대조로
  러너·critic·predicate·commander 실행 로직 동일 확인. 실제 SMAC smoke501스텝 정상 종료.
- 대조군242는 원본236 config에서 name/runner/wandb_run 식별자와 scheduler만 변경.
  실실행 Sacred config에서도 정확한 diff를 재검증했다. 원본 기준 te0.15 등 나머지는 유지.
- 242는 학습 업데이트 t3231, 성공 guidance29건(t_global5600) 확인. 성능은 아직 미판정.
  24h 또는1.2M의 단일 seed0 실행; 추가 시드 자동 체인 없음.
- 새 그래프는 로컬 Sacred snapshot에서 생성, 개명 전 그림은 미삭제. Pursuit 그림은
  수정 semantic v2만 포함하며 미완주(<98% 평가예산) 시드는 제외했다. 원본11:40 표와
  새 그림의 시점/포함 시드가 다르므로 동일 집계로 인용하지 않도록 문서에 경고했다.

### 8w. λ 스케줄 통일 결정 + 캐시off·temp0.2 3팔 구도 (09-07 오후, 사용자 지적)
- **원문 확인**(paper/LEHCA.pdf, Eq.2 부근): λ_t는 "progressive decay schedule …
  larger early, gradually reduced" 한 문장뿐 — 함수형·감쇠율·바닥 **미명시**. 따라서
  구λ(×0.9995/update)도 재현 측 추측이며 원형에 구λ를 유지할 충실성 근거 없음.
- **결정**: 전 팔 λ = lam40 통일(지평 비례, 논문의 정성 서술과 정합). lehca.yaml에
  기본값-off 플래그 `lambda_floor_frac: 0` 추가(베이스라인 동결 규칙 준수), 원형 런은
  0.4 오버라이드. 구λ 원형 체인(s45, 1.5h) 중단 — 폐기 런 `0thrgvkn`(삭제 대상).
- **5m6m 캐시off·temp0.2 3팔** (전부 lam40): ① lehca-orig_b01_t02_nocache_lam40
  (β0.1 마스크 on, F100 — 논문 명시 조건만 RSVP와 다름; s45 s0+s1 체인)
  ② lehca-shape_F200_lam40_nc_t02 (마스크 off, fixed F200 — 스케줄러 A/B 대조군;
  s40 s0+s1) ③ rsvp_Fmax200_lam40_nc_t02 (vf; s34 s0+s1). 구λ 초반-가속 가설은
  기존 교차 데이터(2s3z: FINAL 구λ shaping-only AUC_early 0.454 vs lam40 F100 0.359)
  로 참고하고 별도 런은 보류.

### 8x. Pursuit v2 라운드 완주 — 전 팔 n=4 (09-07 16:10; return AUC/final)
| 팔 | AUC | final |
|---|---|---|
| qmix | −8.33±2.99 | 14.97±4.70 |
| F100_lam40_v2 | −10.14±2.74 | 12.17±2.54 |
| F200_lam40_v2 | −8.84±4.96 | 13.86±6.01 |
| **RSVP_Fmax200_lam40_v2** | **−7.53±2.96** | **15.49±5.43** |
- 순서(두 지표 모두): RSVP ≥ qmix > F200 > F100. 차이는 전부 1σ 이내 — 순위 확정
  불가(사전 경고대로). 방향은 3s5z와 같은 **harm-reduction 형태**: 고정 F100 셰이핑이
  qmix보다 낮고(AUC −1.8/final −2.8), RSVP가 같은 채널로 qmix 수준 이상 회복
  (vs F100: AUC +2.6/final +3.3). "고정 주기는 잃고 RSVP는 잃지 않는다"가 두 환경에서
  같은 부호 — 확정 주장은 아니나 서사의 일관성 증거.
- 런: F100_v2 s3=teyr4ael(13.97), F200_v2 s3=4pp3uom3(13.51), RSVP_v2 s1=ovjkwx9d(11.58)·
  s3=leckv90s(15.99). 결함 라운드는 비합산 참고.

### 8y. 5m6m 캐시off·temp0.2 3팔 + 2s3z n=5 결과 (09-08 16:40, Claude 회수 보고)
| 5m6m 팔 | n | AUC_early | AUC | final | 실호출/런 |
|---|---|---|---|---|---|
| qmix | 4 | 0.014 | 0.235±0.131 | 0.458±0.271 | 0 |
| lehca-orig(β0.1 마스크, F100, t0.2, 캐시off, lam40) | 1 | 0.013 | 0.129 | **0.000** | 23,853 |
| F200_lam40_nc_t02 | 2 | 0.003 | 0.148±0.135 | 0.271±0.156 | 5,964 |
| rsvp_Fmax200_lam40_nc_t02 | 2 | 0.004 | 0.199±0.199 | 0.294±0.294 | 7,694 |
| (참고, 캐시on·t0) F200_lam40 / rsvp_lam40 | 4/4 | 0.012/0.005 | 0.255/0.199 | 0.564/0.576 | 346/348 |
- 판독: ① 원형(마스킹 on) 붕괴 재현 — FINAL의 완전구성 붕괴(0.000)와 동일, lam40으로도
  구제 안 됨. ② temp0.2·캐시off 팔은 같은 스케줄의 temp0·캐시on 팔보다 final이
  절반 이하(0.27~0.29 vs 0.56~0.58)이고 qmix(0.458)에도 미달 — **가이던스 확률성이
  학습을 해치는 방향**(§5.1 접지 노이즈 0.059 vs 0.014와 정합; temp0+메모이제이션이
  사실상 안정화 장치였음). n=2·rsvp 시드 간 0.0/0.59로 분산 극단 — 확정 불가.
  ③ 실호출: 원형 23.9k(F100·캐시off) vs rsvp 7.7k vs F200 6.0k.
- GPT 메인 세션 런: `lehca-shape_F200_lam40_nc_t02_rsvp` s0(5oj3rvan)·s1(rdv2kjtp) —
  RSVP 러너 경유 fixed 대조군(집계는 GPT 문서 참조).
- **2s3z n=5**: qmix(n=3) AUC_early 0.353±0.056 / AUC 0.806 / final 0.968 ·
  F100 0.367±0.036 / 0.778 / 0.956 · rsvp **0.402±0.106** / 0.792 / 0.948. 시드 보강으로
  rsvp 초반 우위가 0.426→0.402(std 0.06→0.11)로 축소 — qmix 대비 +0.05, 잡음 안.
- 인프라: 할당 10개 만료 9/8 21:40~23:24. 진행 중 런은 s45 lehca-orig seed1(120k/1.2M)
  뿐 — 만료로 절단 예정(재할당 여부는 사용자/GPT 결정). 나머지 세션 유휴.

### 8z. 2s3z λ 스케줄 가설 재분석 — 기존 wandb 런만 사용 (09-08 21:30, Claude)
질문: 초기(n=2)에 본 2s3z 초반 우위(+0.12)가 시드 운인지, 구λ→lam40 교체로 사라진 것인지.
AUC_early(첫 20%·사다리꼴, 1M 완주 런만). 구λ = ×0.9993/update(마스크 off, LLM 셰이핑,
FINAL_shaping F100 s0/s1 + LEHCA_v4 F200 s0/s1/s2 합산).

| 팔 | n | AUC_early | 시드쌍 차(−qmix) | 승 | t(Welch) |
|---|---|---|---|---|---|
| qmix | 5 | 0.332±0.050 | — | — | — |
| **구λ shaping-only** | 5 | **0.421±0.082** | **+0.083** | 4/5 | 2.06 |
| lam40 F100 | 5 | 0.358±0.038 | +0.025 | 4/5 | 0.91 |
| lam40 RSVP | 5 | 0.389±0.115 | +0.057 | 3/5 | 1.02 |

시드별 구λ: s0 0.278/0.478, s1 0.447/0.429, s2 0.471 (qmix s0 0.407·s1 0.272·s2 0.330).
- 판독: 구λ 셰이핑의 초반 우위는 n=5에서도 +0.08(4/5 승, p≈0.07)로 남고, 같은 채널을
  lam40으로 돌린 F100은 +0.025로 소실. **λ 교체가 초반 가속을 지웠다는 가설과 정합**
  (혼입: 구λ 런은 F100/F200 혼합·구 코드, 교차 시드가 아닌 동일 시드 대조는 3개).
- 전 구간 AUC 0.812 vs qmix 0.797, final 0.942 vs 0.963 — 초반 이득이 후반까지 가진
  않음(2s3z 포화). LEHCA 주장(초반 가속)의 형태 그대로.
- 함의: "채널이 죽었다"의 절반은 자초(λ 재설계). 구λ(전면 집중형) 또는 하이브리드
  (초반 집중 + 지평 비례 꼬리)로 되돌리면 2s3z에서 채널 우위는 회복 가능성 있음.
  5m6m·3s5z는 초반 창이 없어 별개. → 사용자/GPT 결정 항목: 2s3z shaping-only 구λ
  n=4 신규 실행(하루) 또는 위 재분석으로 갈음.

### 8aa. S0b 대조군 완주 판정 — 5m6m 1.2M, 같은 시드 짝 비교 (09-09 07:30, Claude 집계)
공통: RSVP 러너·fixed F200·마스킹 off·lam40·ε 50k. none/rule/shuffle/aligned는 LLM 무호출.
실제 LLM = 기존 F200 nc_t02(temp 0.2·캐시 off) 완료 런(LEHCA 경로/RSVP 경로 두 벌).
shuffle 풀 = 그 nc_t02 RSVP 경로 런의 guidance 로그(시드 교차). Sacred 256·262~268.

| 팔 | s0 AUC / final | s1 AUC / final | 평균 AUC / final |
|---|---|---|---|
| qmix (기존) | 0.027 / 0.109 | 0.247 / 0.833 | 0.137 / 0.471 |
| none | **0.027 / 0.109** | **0.247 / 0.833** | 0.137 / 0.471 |
| rule | 0.272 / 0.479 | 0.168 / 0.849 | 0.220 / 0.664 |
| shuffle | 0.474 / 0.526 | 0.260 / 0.419 | 0.367 / 0.473 |
| aligned | 0.480 / 0.664 | 0.265 / 0.500 | 0.373 / 0.582 |
| 실제 LLM nc_t02 (LEHCA 경로) | 0.283 / 0.427 | 0.012 / 0.115 | 0.148 / 0.271 |
| 실제 LLM nc_t02 (RSVP 경로) | 0.271 / 0.443 | 0.019 / 0.154 | 0.145 / 0.298 |
| (참고) F200 캐시on·t0 | — | — | 0.255 / 0.564 (n=4) |

판정 순서(lehca-backbone-validation §3) 결과:
1. **none = qmix 소수점까지 동일**(두 시드) → 러너 경로 부수효과 없음 확정. 절단 런
   s0b2와 재기동 s0b3의 초반 곡선도 일치(shuffle s0 AUC_early 0.092=0.092) → 결정성 확인.
2. **aligned ≥ none**: s0(qmix가 못 배우는 시드)에서 0.109→0.664로 구제, s1(잘 배우는
   시드)에서는 AUC 동급·final 0.833→0.500 하락. 셰이핑 채널은 '죽은 시드 구제·산 시드
   훼손' = 분산 축소 방향이지 상향 이동이 아님.
3. rule도 같은 형태(s0 구제 0.479, s1 final 유지 0.849). 손 규칙이 aligned보다 s1 final이
   높음.
4. **실제 LLM < shuffle, 두 시드 모두**(AUC 0.283 vs 0.474, 0.012 vs 0.260). 같은 출력
   분포를 상태와 무관한 순서로 재생한 쪽이 상태 정렬된 원본보다 낫다 → 상태 맞춤
   내용은 기여가 없거나 음(−). 캐시on·t0 F200(0.564)은 shuffle·aligned 수준 → 캐시가
   가이던스를 정적 prior로 만들어 준 것이 오히려 성능을 지켰다는 §8y 판독과 정합.
- 한정: n=2, 5m6m 시드 분산 극단(qmix final 0.109~0.833). 순위 확정 아님. 단 "LLM <
  shuffle"은 두 시드 부호 일치 + §8y(캐시 off가 캐시 on보다 나쁨)와 독립 증거로 수렴.
- 셰이핑 크기 진단(wandb train/, 첫 240k = early / 마지막 240k = late 평균):
  |λF|/|r_env| = rule 0.37/0.05, shuffle 0.32/0.05, aligned 0.31/0.04 (s0·s1 거의 동일);
  F RMS 0.26~0.68, 비영 비율 rule s1 0.22 → 나머지 0.5~0.8. 즉 셰이핑은 초반에도
  환경 보상의 1/3, λ 바닥 이후 5% 수준의 작은 섭동이고, 팔 간 **크기는 같으므로**
  shuffle/aligned vs 실제 LLM의 차이는 내용(부호·타이밍) 차이다. 작은 섭동으로도
  s0 구제·s1 훼손이 일어난 것은 5m6m 초반 탐색이 섭동에 민감하다는 뜻(보상 중복
  가설과 정합: 채널이 주는 건 정보가 아니라 탐색 교란).
- 채널 우위 조건 탐색 우선순위 재정렬 근거: 5m6m에서는 어떤 셰이핑도 qmix 평균을
  유의하게 넘지 못하고(aligned 0.582 vs 0.471, n=2), LLM 내용은 손 셰이핑보다 못함.
- 운영 검증: Slurm949368~949374는 모두 `COMPLETED`, exit code 0이며 W&B도 exit code 0으로
  동기화됐다. 마지막 기록 평가는 1.191~1.193M이고 이후 episode에서 1.2M을 넘어 정상
  종료했으므로 표는 완주 런으로 취급한다. Sacred `run.json`의 `RUNNING` 표시는 종료 시
  갱신되지 않는 기존 로거 현상이다. 동시 시작 런의 모델 디렉터리가 초 단위
  `unique_token` 때문에 충돌한 사실을 발견해 `run.py` 토큰에 PID를 추가했다. 이번 성능
  로그에는 영향이 없으며, 이후 체크포인트 경로는 런별로 분리된다.

### 8ab. 발화 타이밍 대 동일예산 무작위 — Q-001 희소전환 재설계 (09-09 18:10, Claude, GPU 미사용)
스크립트: scratchpad/timing_vs_random.py. 대상: trace+phase 로그가 모두 있는 4런
(5m6m 3런, Pursuit 1런, 각 10만 스텝). 무작위·주기 베이스라인은 **CUSUM과 같은 후보군**
(armed AND since≥min_interval)에서 같은 개수로 추출, 200회 부트스트랩.

**(1) §8n 발화 적합성 주장 철회.** §8n은 cache_key 전체 변화를 국면 전환으로 삼았는데
그 라벨은 10만 스텝 중 6만 회(60%) 발생한다. 라벨이 조밀하면 어떤 스케줄도 높은 일치를
얻는다. 아군 병력만 쓰는 엄격한 라벨(시야 무관, 에피소드 리셋 제외)로 다시 재고
동일예산 무작위를 대조로 넣으면 CUSUM 우위가 사라진다.

| 5m6m 런 | 라벨 밀도 | w=±5 CUSUM | 동일예산 무작위 | 주기 | z |
|---|---|---|---|---|---|
| p121435 | 아군변화 35.5% | 0.618 | 0.868±0.026 | 0.888 | −9.67 |
| p1825112 | 23.7% | 0.889 | 0.985±0.009 | 0.989 | −10.72 |
| p71815 | 40.2% | 0.901 | 0.872±0.023 | 0.910 | +1.26 |

3런 중 2런에서 CUSUM이 무작위보다 **유의하게 나쁘고** 1런만 근소 우위다. 따라서
"CUSUM 발화가 국면 전환과 잘 맞는다(78–99%)"는 §8n 문장은 근거로 쓸 수 없다.

**(2) 원인은 데이터 품질이 아니라 구조.** 5m6m은 이벤트가 검출기 분해능보다 조밀하다.

| 런 | 에피소드 길이 | 아군변화 간격 | 아군사망 간격 | min_interval |
|---|---|---|---|---|
| p121435 | 26.6 | 2.8 | 5.9 | 10 |
| p1825112 | 35.4 | 4.2 | 12.1 | 10 |
| p71815 | 23.5 | 2.5 | 5.1 | 10 |

이벤트가 2.5~12스텝마다 나는데 검출기는 최소 10스텝 잠금이 있다. **어떤 스케줄러도
정의상 이벤트에 정렬될 수 없다.** 시간척도 조건 H_V < D_detect < T_valid이
5m6m에서 위배된다는 정량 확인이며, §8y의 24스텝 에피소드 문제와 같은 뿌리다.

**(3) Pursuit에서는 방향이 맞다(단 검정력 부족).** 500스텝 에피소드, 포획 티어 변화는
평균 1,057스텝 간격(전체 스텝의 0.09%)으로 희소하다. 발화 69회.

| w | CUSUM | 동일예산 무작위 | 주기 | 만기 | z |
|---|---|---|---|---|---|
| ±5 | 0.029 | 0.011±0.012 | 0.014 | 0.006 | +1.58 |
| ±25 | 0.087 | 0.047±0.026 | 0.029 | 0.047 | +1.54 |
| ±50 | 0.145 | 0.101±0.036 | 0.043 | 0.102 | +1.25 |

네 창 전부 CUSUM > 무작위 > 주기 순서이고 주기 대비는 2~3배지만, 발화 69회로 z≈1.3
수준이라 유의하지 않다. 같은 런의 조밀 라벨(적 섹터 변화, 24.6%, 간격 4.1스텝)에서는
z≈0으로 판별력이 사라진다 — (2)의 조밀도 설명과 일치.

**결론.** 타이밍 주장은 (a) 5m6m에서는 검증 불가, (b) Pursuit 희소 이벤트에서만 검증
가능하며 현재 방향만 맞고 검정력이 없다. 후속: Pursuit 런에 `sched_trace=True`를 켜서
트레이스를 4~8런 확보(추가 비용 거의 없음). 이것이 "적응 타이밍" 주장의 유일한 실증
경로다.

### 8ac. F25 주기 스윕(300k) + QMIX 5M 예산 감사 (09-10 11:30, Claude 집계)
Sacred 269~286. 전부 5m6m. 스크립트: scratchpad/agg300k.py(모든 팔을 300k 지평에서 재평가).

**(1) F25 스윕, 300k 지평, n=2** — 어제 S0b(F200)와 같은 대조군을 주기만 8배 빠르게.

| 팔 | 주기 | AUC@300k | final@300k | 시드별 final |
|---|---|---|---|---|
| shuffle | F200 | 0.083±0.099 | 0.214±0.228 | 0.38 / 0.05 |
| aligned | F200 | 0.058±0.073 | 0.193±0.228 | 0.35 / 0.03 |
| shuffle | F25 | 0.040±0.009 | 0.182±0.096 | 0.25 / 0.11 |
| RSVP Fmax200 | vf | 0.037±0.053 | 0.161±0.228 | 0.32 / 0.00 |
| 실제 LLM | F200 | 0.014±0.020 | 0.099±0.140 | 0.20 / 0.00 |
| qmix | — | 0.028±0.034 | 0.094±0.111 | (n=4) |
| rule | F200 | 0.011±0.016 | 0.016±0.022 | 0.03 / 0.00 |
| LEHCA 원형(마스킹 β0.1) | F25 | 0.002±0.003 | 0.005±0.007 | 0.00 / 0.01 |
| none | F200 | 0.000 | 0.000 | 0.00 / 0.00 |
| **실제 LLM** | **F25** | **0.000** | **0.000** | 0.00 / 0.00 |

- **실제 LLM이 F25에서 두 시드 전멸(0.000)**, 같은 주기의 shuffle은 0.182. §8aa의
  "LLM < shuffle"이 **두 번째 주기에서 재현**됐다(합계 4시드, 부호 일치).
- 갱신을 빠르게 하면 비-LLM 가이던스(shuffle 0.214→0.182, 무해)와 달리 LLM 팔만
  0.099→0.000으로 무너진다 → 해악이 주기와 상호작용하는 쪽은 LLM 내용이다.
- 원형(마스킹 on)은 F25에서도 붕괴(0.005). 세 번째 주기에서의 재확인.
- **한정**: 300k는 5m6m에서 아무도 못 배우는 구간이고 n=2다. 실제로 rule은 300k에서
  0.016이지만 1.2M에서는 0.664(§8aa)다. **300k 순위는 1.2M 순위를 예측하지 못한다.**

**(2) QMIX 5M 예산 감사 (n=2, 완주)** — 우리 1.2M 지평이 QMIX를 불리하게 잘랐는가?

| 시드 | 1.2M 시점 | 5M best | 5M last-10% |
|---|---|---|---|
| 0 | 0.062 | 0.562 | 0.342 |
| 1 | 0.938 | 1.000 | 0.886 |

- 예산을 4배 늘려도 **시드 간 격차가 0.54로 유지**된다(0.342 vs 0.886). 1.2M에서 본
  분산은 지평 절단 artifact가 아니라 **맵의 내재 성질**이다.
- seed0은 best 0.562에서 last-10% 0.342로 후퇴 — 5M에서도 불안정.

**결론: 5m6m을 주 무대에서 내린다.** 독립적인 두 이유로 판별 불가 맵이다.
(a) §8ab — 이벤트 간격 2.5~12스텝 < 검출기 최소 간격 10스텝이라 타이밍 주장 검증 불가.
(b) 본 절 — QMIX 자체의 시드 분산이 5M에서도 0.54라 n=2~4로는 어떤 팔도 구분 불가.
5m6m 결과는 "해악 경계"(LLM < shuffle, 원형 붕괴)의 부호 증거로만 인용하고, 성능 순위
주장에는 쓰지 않는다.

### 8ad. paper-alignment / lambda-cutoff 중간 스냅샷 — 소극성은 ε300k 공통 현상 (09-11 00:40, Claude)
진행 중 런의 중간 집계(완주 아님). 스크립트: scratchpad/agg_pa.py, cutoff_ba.py.
전 팔 공통: 5m6m, ε 담금 300k, F50, 1.2M 목표.

**(1) 300k 공통 지평 행동 지표 (n=1~2, 승률은 전 팔 0)**

| 팔 | 구성 | 적 처치 | 아군 사망 | ep 길이 | 리턴 |
|---|---|---|---|---|---|
| qmix | — | 0.60 | 3.31 | 49.7 | 3.11 |
| mask-off | 셰이핑만 | 0.63 | 3.08 | 52.5 | 2.97 |
| shaping-off | 마스킹만 | 0.57 | 1.95 | 55.1 | 2.07 |
| full | 둘 다 | 0.46 | 1.85 | 59.7 | 1.80 |

셰이핑만 켠 쪽이 qmix에 가깝고 마스킹만 켠 쪽이 full에 가깝다 — 9/10 진단 문서의
"셰이핑이 주범" 가설과 **반대 방향**. 단 mask-off 두 시드가 0.98/4.25 대 0.28/1.90으로
정반대라 팔 평균이 시드 하나에 끌린다. shaping-off는 s0가 4만에서 죽어 사실상 n=1.

**(2) 320~462k 창을 맞춘 비교 — 결정적**

| 런 | 적 처치 | 아군 사망 | ep 길이 | 리턴 | 승률 |
|---|---|---|---|---|---|
| qmix s0 | 0.01 | 0.10 | 69.9 | 0.13 | 0.000 |
| qmix s1 | 0.03 | 0.19 | 69.9 | 0.69 | 0.000 |
| full s1 | 0.04 | 0.31 | 69.7 | 0.24 | 0.000 |
| **mask-off s0 (셰이핑만)** | **2.69** | 4.98 | **30.5** | **7.95** | **0.004** |
| rsvp λ→0 s0 | 0.00 | 0.01 | 69.9 | 0.00 | 0.000 |

- **소극성은 가이던스 탓이 아니다.** 이 창에서 **QMIX도 똑같이 얼어 있다**(처치 0.01~0.03,
  ep 69.9 = 상한 70). ε300k에서는 전 팔이 긴 소극 구간을 통과한다. QMIX는 이후 회복해
  1.19M까지 평균 0.93~1.18을 찍지만(창 320k~1193k), 462k까지는 전멸 상태다.
  → 9/10 진단 문서의 "40~60k부터 LEHCA가 전투를 덜 한다"는 관찰은 유지되나, "소극성 =
  셰이핑 효과"라는 인과 귀속은 이 대조로 **성립하지 않는다**.
- **셰이핑만 켠 s0는 유일하게 이 구간을 탈출**한다: 처치 2.69, ep 30.5, 리턴 7.95,
  첫 승리(0.004). 같은 창의 QMIX 대비 처치 90배·리턴 12배. 단 mask-off s1은 0.00으로
  완전 동결 — **양봉(bimodal)**.
- **λ 컷오프는 구제 수단이 아니다.** λ가 300k에서 정확히 0이 된 뒤 (320~511k) 처치 0.00·
  사망 0.01·ep 69.9로 오히려 가장 깊이 동결됐다. 다만 같은 창의 QMIX도 동결이므로
  컷오프가 동결의 **원인**이라고 주장할 근거도 없다. 확정 가능한 것은 "이미 소극
  상태에 든 정책은 셰이핑을 제거해도 전투로 돌아오지 않는다"이다.

**(3) 2s3z에서 처음으로 채널이 QMIX를 상회** (29만 지점, n=1)

| 팔 | 승률 AUC | 적 처치 | 아군 사망 |
|---|---|---|---|
| full | **0.1292** | 2.45 | 4.44 |
| qmix | 0.0875 | 2.08 | 4.60 |

적을 더 죽이면서 아군을 덜 잃는다. 캠페인 전체에서 가이던스가 QMIX를 넘은 유일한
지점이고, §8z의 2s3z 구λ 초반 우위와 같은 맵이다. **2s3z 시드 확충이 최우선 후속.**

**해석 (Claude).** 5m6m ε300k 설정에는 흡수 상태 두 개가 있다. 전투 어트랙터(처치↑,
ep↓, 리턴↑)와 소극 어트랙터(양측 생존, ep=상한, 리턴≈0). `reward_only_positive=True`라
소극 상태는 잃는 것이 없고 ε도 300k에서 0.05로 떨어져 탈출 압력이 없다. 가이던스는
평균을 내리는 것이 아니라 **어느 어트랙터에 빠지는지의 분산을 키운다**(mask-off s0 대
s1). 이 관점은 §8aa의 "죽은 시드 구제·산 시드 훼손", §8ac의 5M에서도 유지되는 0.54
시드 격차와 정합한다. 판정에는 시드 수가 필수이며 n=1~2 순위 주장은 금지.

### 8ae. paper-alignment 완주 판정 — 마스킹 유해·셰이핑은 분산 증폭기 (09-11 17:00, Claude)
5m6m, ε 담금 **300k**(논문 정렬), F50, 1.2M. 구간별 승률/적처치. 스크립트: scratchpad/agg_full.py.

| 런 | 0–300k | 300–600k | 600–900k | 900–1200k |
|---|---|---|---|---|
| qmix s0 | .000/0.60 | .000/0.01 | .001/1.29 | .000/1.43 |
| qmix s1 | .001/0.59 | .001/0.51 | .000/1.10 | .000/1.85 |
| full s0 (셰이핑+마스킹) | .000/0.39 | .000/0.07 | .000/0.56 | .000/1.15 |
| full s1 | .000/0.52 | .000/0.03 | .000/0.02 | .000/0.02 |
| **mask-off s0 (셰이핑만)** | .000/0.98 | **.009/2.93** | **.049/3.59** | **.069/3.74** |
| mask-off s1 | .000/0.28 | .000/0.00 | .000/0.00 | .000/0.00 |
| shaping-off s1 (마스킹만) | .000/0.57 | .000/0.02 | .000/0.23 | .000/0.91 |
| λ→0 fixed s0/s1 | .000/0.44·0.40 | .000/0.00 | .000/0.00 | .000/0.07·0.03 |
| λ→0 rsvp s0/s1 | .000/0.51·0.37 | .000/0.00 | .000/0.00 | .000/0.11·0.10 |

**(1) ε300k는 5m6m을 전 팔 붕괴시킨다.** 1.2M 승률이 거의 전부 0.000이다. ε50k 라운드의
qmix final 0.458(n=4, §8y)과 비교하면 논문 정렬 설정 자체가 이 맵을 망친다. 9/10 진단
문서의 판단(ε300k 결과를 재현 성공/실패 판정에 쓰지 말 것)을 완주 데이터가 지지한다.

**(2) 마스킹은 유해하다.** 900–1200k 적처치 평균: mask-off(셰이핑만) 1.87 >
qmix 1.64 > shaping-off(마스킹만) 0.91 > full 0.59 > λ→0 0.08. 셰이핑을 고정한 채
마스킹만 켜면(full vs mask-off) 1.87→0.59, 마스킹만 단독으로도 qmix 아래(0.91 < 1.64).
MMM2 채널 유해 판정·LEHCA 원형 붕괴와 같은 방향이며 이번엔 같은 시드 짝 대조다.

**(3) 셰이핑은 평균이 아니라 분산을 바꾼다.** mask-off s0는 캠페인 전체에서 **유일하게
유의한 승률**(0.069)에 도달하고 적처치 3.74로 qmix(1.43~1.85)의 2배 이상이다. 반면
mask-off s1은 300k 이후 0.00으로 완전 동결. 같은 설정 두 시드가 정반대 어트랙터로 간다.
§8aa("죽은 시드 구제·산 시드 훼손"), §8ad(어트랙터 두 개) 해석과 일치.

**(4) λ→0 컷오프는 유해하다 (n=4, 전원 동일).** 300k에서 λ를 정확히 0으로 끊은 네 런
모두 300–900k 구간 적처치 0.00으로 동결되고 끝까지 0.03~0.11에 머문다. floor 0.05를
유지한 mask-off(1/2 탈출)보다 나쁘다. **"40% 이후 LLM 제거" 아이디어는 이 설정에서
기각**이다. §8ad의 "이미 소극 상태에 든 정책은 셰이핑 제거로 돌아오지 않는다"가 n=4로
확인됐다. 단 혼입: 컷오프 팔은 코사인 스케줄이라 300k 이전 λ 궤적도 다르다.

**(5) Pursuit 컷오프 1차 시도 실패.** 묶음 스크립트가 GPU 1개에 트레이너 3개+vLLM을
올려 fixed_zero·rsvp_zero 두 런이 60,500스텝에서 동시 사망(fixed_floor만 생존, 100k 진행).
주 세션이 `run_lambda_cutoff_one.sh`(GPU당 1런)로 재설계해 16:30 재투입, 현재 자원 대기.
sched_trace=True는 새 스크립트의 rsvp_zero 팔에 유지됨(9/10 Claude 추가분).

**함의.** 채널 분해가 끝났다: 마스킹 = 순해악, 셰이핑 = 고분산 양날, λ 완전 제거 = 해악.
지금까지 "가이던스가 QMIX를 못 이긴다"로 뭉뚱그렸던 것이 **"마스킹이 깎고 셰이핑은
도박"** 으로 분리된다. 후속 우선순위: (a) mask-off(셰이핑만) 시드 확충 — 승률 0.069을
낸 유일한 구성, (b) 2s3z 시드 확충(§8ad에서 full 0.129 > qmix 0.088), (c) 분산의 원인
규명(어트랙터 진입 시점·조건).

### 8af. 희소 보상 스크리닝 1차 — 채널이 살아났다 (09-11 19:10, Claude 집계, 진행 중)
사용자 제보(논문이 희소 보상 SMAC을 사용)로 주 세션이 09-11 17:06 투입.
`env_args.reward_sparse=True`, `reward_scale=False`(dense 피해·킬 보상 제거, 승패만),
ε 담금 300k, 지평 300k, F50·마스킹 on·temp0.2·캐시 off(논문 원형). 맵2 × 팔2 × 시드2.

**2s3z, 50k 이하 동일 창 비교**

| 팔 | 승률 평균 | 승률 최대 | 적 처치 | 아군 사망 | ep 길이 |
|---|---|---|---|---|---|
| **LEHCA full s0** | **0.0563** | **0.188** | 1.44 | 3.91 | 88.6 |
| **LEHCA full s1** | **0.0547** | **0.125** | 1.41 | 4.30 | 83.3 |
| qmix s0 | 0.0000 | 0.000 | 0.37 | 4.14 | 92.6 |
| qmix s1 | 0.0000 | 0.000 | 0.97 | 4.49 | 78.8 |

- **완주한 qmix는 291k까지 승률 0.000**(두 시드), 적 처치 0.07·0.28. 반면 LEHCA는
  **3만~5만 스텝에서 이미 12.5~18.8% 승리**한다. 두 시드 모두 부호 일치.
- 적 처치가 4배(1.4 대 0.4~1.0). 희소 보상에서 셰이핑이 **유일한 중간 신호**가 되어
  실제 전투 학습을 만든다는 가설과 정확히 일치.

**5m6m, 80k 이하 동일 창** — 아직 전 팔 승률 0.000. 적 처치는 full 0.87·0.30 대
qmix 0.41·0.00으로 방향만 우세. 어려운 맵이라 판정 이르다.

**의미.** §8aa~§8ae에서 "가이던스 채널이 QMIX를 못 이긴다"고 했던 결론은 **dense 보상
SMAC에 한정된 것**이었다. 보상 중복 가설(§8y 해석, 9/9 중간보고 §2-②)이 직접 확인됐다.
2s3z 희소에서는 QMIX가 291k에 아무것도 못 배우는데 LEHCA는 5만에 이긴다 — 지금까지
캠페인에서 가장 큰 효과 크기다.

**한정.** (a) full 런이 3만~9만으로 초기 구간뿐, 완주 후 재판정 필요. (b) n=2.
(c) ε 담금 300k가 지평 300k와 같아 전 구간이 탐색 구간이다 — qmix의 0.000이 희소 보상
탓인지 ε 탓인지 분리되지 않는다. **ε50k 판 대조가 필요**(§8ae에서 ε300k가 dense
5m6m을 전 팔 붕괴시킨 전례).

**후속 우선순위 개정.** (1) 희소 2s3z 완주 + 시드 확충(n≥4), (2) 희소 조건 ε50k 대조,
(3) 희소에서 셰이핑/마스킹 분해(§8ae는 dense 결론), (4) 그 다음에야 RSVP 대 고정 F.
채널이 사는 레짐을 찾았으므로 스케줄러 연구의 전제가 처음으로 성립한다.

### 8ag. 희소 2s3z 1M 캠페인 — 버티기 함정과 LEHCA의 예외성 (09-12 04:00, Claude, 진행 중)
사용자 승인으로 09-11 19:00 투입. 4팔 × 2시드, `reward_sparse=True`·`reward_scale=False`,
ε 담금 300k, 1M. 스크립트 `scripts/archive/run_sparse_1m_arm.sh`(신규). 집계 scratchpad/agg_s1m.py.
`aligned` = 손으로 만든 damage/kill 셰이핑(LLM 없음) — "LLM이 필요한가" 분리용.

**(1) 결함 발견: SMAC `reward_sparse`에는 버티기 최적해가 있다.**
SMAC 소스 확인 결과 `reward_sparse=True`는 승리 +1 / **패배 −1** / 시간초과 0을 준다.
승리가 어려우면 최적 정책은 교전 회피 후 시간초과(0 > −1)다. 실제로 전 팔이 그리로 간다.

| aligned s0 구간 | 승률 | 적 처치 | ep 길이 | 리턴 |
|---|---|---|---|---|
| 0–110k | 0.034 | 1.68 | 72.1 | −0.75 |
| 110–300k | 0.000 | 0.28 | 94.1 | −0.59 |
| 300–600k | 0.000 | 0.06 | 116.4 | −0.14 |
| 600k–1M | 0.000 | 0.03 | 119.1 | **−0.03** |

ep 길이가 상한(120)으로, 리턴이 0으로 단조 수렴한다. qmix도 동일(94.4→118.5, −0.54→−0.04).
**문헌의 희소 SMAC은 승리 +1 / 그 외 0**이며(2509.10656 등) 패배 페널티가 없어 이 퇴화
최적해가 없다. 우리 설정은 문헌 설정이 아니다.
→ 대응: `env/__init__.py`에 opt-in `sparse_win_only`(종단 음수 보상 0으로 절단) 추가,
기본 off, 실환경 스모크 통과. **미투입**(사용자 결정 대기).

**(2) 붕괴 전 창(≤110k) 판정, n=2**

| 팔 | 승률 | 승률 최대 | 적 처치 |
|---|---|---|---|
| **lehca** (셰이핑+마스킹, F50) | **0.0781±0.0181** | 0.219 | 1.88 |
| aligned (손 셰이핑) | 0.0227±0.0161 | 0.109 | 1.59 |
| rsvp (셰이핑만, vf F_max200) | 0.0028±0.0040 | 0.016 | 1.22 |
| qmix | 0.0000±0.0000 | 0.000 | 0.36 |

- qmix 전 구간 0 — 문헌과 일치(희소 2s3z에서 IPPO·MAPPO 0%, MASER 0.01).
- **LLM 가이던스가 손 셰이핑의 3.4배.** "아무 dense 셰이핑이나 되는 것 아니냐"에 대한
  첫 실증 반박. 단 n=2.

**(3) LEHCA만 버티기 함정을 벗어난다 (핵심).** 110–300k에서 타 팔은 승률 0·처치 0.01~0.40인데
lehca는 0.024/0.087, 처치 1.67/1.95를 유지. 50k 구간별로 보면 seed1은 300–350k에서
**0.164/2.45**로 캠페인 최고치까지 상승, seed0은 0.013~0.031로 감속하나 붕괴는 없음
(ep 92, 상한 120에서 멂). 즉 LLM 셰이핑은 −1 회피 유인을 이겨 교전을 유지시키며,
효과는 평균 이동이 아니라 **시드별 성장 속도 차이**로 나타난다(§8ae 분산 해석과 동형).

**(4) RSVP는 희소에서 실패.** s0 87만·s1 99만 완주, 전 구간 승률 ≈0(최대 0.031),
최근 20만 처치 0.15~0.23 — qmix와 구분 불가. lehca와의 차이는 **마스킹(off/on)**과
**주기(vf F_max200 / 고정 F50)** 둘. dense에서 마스킹은 순해악이었으나(§8ae) 희소에서는
반대로 **필수 성분일 가능성**. RSVP는 설계상 셰이핑 전용이라 이 점이 방법론에 직접 위협.
→ 필요한 분해 실험: 마스킹 켠 RSVP, 또는 F50 고정 셰이핑 전용 팔. 현재 자원 없어 미투입.

**보류 중 결정 사항.** (a) `sparse_win_only` 재실행 여부 — 현재 설정의 버티기 함정을
"LEHCA만 빠져나온다"는 결과로 쓸지, 문헌 정합 설정으로 갈아탈지. (b) 희소 마스킹 분해.
(c) lehca 시드 확충(현재 n=2, seed 간 격차 큼).

### 8ap. Win-only sparse 2s3z 재현 게이트 제출 (09-13 20:26, Codex)

사용자 지시로 §8ag의 보류를 해제했다. 승리 +1/패배·시간초과 0인 opt-in
`sparse_win_only=True`만 새로 적용하고 `paper_v2`는 적용하지 않았다. 300k,
seed0/1의 QMIX(979453/4)와 corrected full LEHCA(979455/6)를 제출했다. LEHCA는
β=.1, dedup=True, F50, shaping+train/test masking, collection-time shaping을 쓴다.
소스 snapshot은 `results/source_snapshots/winonly_sparse_gate_20260913_v1`, 제출 원장은
`results/diagnostics/winonly_sparse_gate_20260913_v1/queue-manifest.json`이다. 보상 래퍼
단위 확인과 LEHCA 회귀 unittest 11개를 통과했다. RSVP는 채널 재현 게이트 판정 뒤에만
추가한다.
20:29 KST에 네 job 모두 RUNNING이며 Sacred362/363(QMIX seed1/0),
364/365(LEHCA seed1/0)의 실제 config가 제출 원장과 일치한다.

### 8ad. 유한 shaping 구간 + LLM 조기 종료 검증 (09-10 18:25, Codex, 진행 중)

**질문.** 마스킹을 사용하지 않는 shaping-only 구성에서 `lambda`를 학습 끝까지 0.05로
남길 이유가 있는가? 탐색 annealing이 끝나는 300k에 shaping 계수를 정확히 0으로 만들고,
그 뒤 Commander·RSVP 내부 계산까지 끄면 성능을 보존하면서 LLM 비용을 줄일 수 있는지
검증한다. 논문 PDF가 공개한 300k는 epsilon-greedy annealing 값이지 lambda 스케줄이
아니므로, 이 설정은 "논문 재현값"이 아니라 별도의 검증 가설이다.

**코드 변경.** `lambda_zero_t > 0`이면
`lambda(t)=0.5*lambda_start*(1+cos(pi*t/lambda_zero_t))`를 사용하고, 300k 이후 정확히
0으로 고정한다. mask off이고 lambda가 0이면 train Commander refresh, predicate feature,
critic update, phase trace를 건너뛴다. test에서는 shaping이 적용되지 않고 mask도 없으므로
처음부터 Commander를 호출하지 않는다. 기본값은 0이라 기존 실험 의미는 바뀌지 않는다.

**통제된 3-arm 설계** (`5m_vs_6m`, 1.2M, seeds 0/1, cache off, temp 0.2,
paper prompt, learner-side shaping, mask off):

| arm | refresh | lambda | 분리하는 효과 |
|---|---|---|---|
| `fixed_floor` | F=50 | 기존 0.5→0.05 floor | 정확한 비용·성능 대조군 |
| `fixed_zero` | F=50 | cosine 0.5→0 @300k | lambda cutoff 자체의 효과 |
| `rsvp_zero` | RSVP, Fmax=200 | cosine 0.5→0 @300k | 활성 구간 내 동적 refresh 효과 |

주의: `fixed_floor`와 `fixed_zero`도 모두 RSVP runner를 써서 runner 차이를 제거했다.
따라서 `fixed_floor ↔ fixed_zero`만 cutoff의 인과 비교이며, `fixed_zero ↔ rsvp_zero`는
refresh 정책 비교다. 5m6m은 §8ac 이유로 성능 우월성의 주 무대가 아니며, 여기서는
LLM 내용의 해악이 shaping 종료 후에도 남는지와 호출 절감의 정확한 측정에 사용한다.

**실행 큐.** Slurm job 961846(`fixed_zero_s0` + `rsvp_zero_s0`)이 n034에서 실행 중이다.
Sacred 298/299, W&B run id `dyzy1yhz`/`7gdchfle`. 이후 961859(seed1 두 팔),
961901(`fixed_floor` 두 시드)이 `afterany` dependency로 연속 실행된다. zero 묶음은 한 A10에서 vLLM
(memory utilization 0.72)과 작은 QMIX trainer 두 개를 함께 실행하며 실측 GPU 메모리는
17.5/23.0 GiB다. 이는 벽시계 비교에는 쓰지 않고 동일 env-step 성능과 호출 수만 비교한다.
클러스터 QOS는 GPU 최대 12개지만 동시 job 최대 10개이며 현재 10/10 job이 실행 중이다.
floor 묶음은 48시간 초과를 피하려고 GPU 2개·독립 vLLM 2대로 두 시드를 병렬화했다.
5m6m 체인 뒤에는 Pursuit 동일 3팔을 한 서버에서 seed별로 실행하는 961902(seed0)과
961903(seed1)를 연결했다. Pursuit은 1M, epsilon/cutoff 300k이며 긴 에피소드에서 결과가
일반화되는지 보는 복제다. 두 환경 결과는 환경별로 판정하고 합쳐 평균내지 않는다.

**사전 판정.** `fixed_zero`가 `fixed_floor` 대비 final/AUC를 악화시키지 않으면서 300k 이후
LLM call 증가가 정확히 0이면 조기 종료를 채택한다. F50 기준 train call 상한은
24,000→6,000(75% 절감)이고, 불필요했던 test 호출 제거까지 포함한 실측 절감률을 별도로
보고한다. 성능이 악화되면 cutoff 시점을 300k로 고정하지 않고 lambda dose-response와
제로 도달 시점을 분리해 재검증한다.

### 8ae. finite-lambda 1차 결과와 실행 감사 (09-11 16:45, Codex)

**완주:** 5m6m `fixed_zero`와 `rsvp_zero`, seeds 0/1, 각 1.2M. AUC는 120회 평가의
단순 평균, final은 마지막 10%. `±`는 n=2 모집단 표준편차다.

| 방법 | win AUC / final | return AUC / final | 적 처치 AUC / final | 성공 guidance |
|---|---:|---:|---:|---:|
| paper-epsilon QMIX | 0.0004 / 0.000 | 3.694±0.261 / 4.093±2.761 | 0.923 / 1.135 | 0 |
| fixed F50, cosine→0 | 0 / 0 | 0.798±0.257 / 1.363±0.951 | 0.118 / 0.068 | 6,001.5/seed |
| RSVP Fmax200, cosine→0 | 0 / 0 | 0.799±0.188 / 1.298±0.088 | 0.136 / 0.139 | 1,552/seed |

- **성능:** 두 zero 팔은 사실상 동일하고 QMIX보다 return AUC가 약 78% 낮다. 승률은 세
  방법 모두 거의 0이라 이 배치만으로 승률 우위를 말할 수 없다. 다만 적 처치가 QMIX의
  13~15%이고 300k 뒤에도 회복되지 않아, 초반 shaping이 만든 정책/리플레이 편향이
  lambda=0 이후 자동으로 사라지지 않는다는 강한 경고다.
- **scheduler:** 동일 lambda에서 RSVP는 fixed 대비 호출 74.1% 절감(3,104 vs 12,003)했지만
  AUC/final 개선은 없다. 조기발화는 138회(전체 RSVP 갱신의 4.4%)뿐이다. 이 데이터는
  "동적 F가 더 좋은 성능을 만든다"를 지지하지 않고, 비용 절감만 지지한다.
- **dose:** 0–300k의 `|lambda F|/|r_env|` 평균이 fixed 약 0.684, RSVP 약 0.690이다.
  기존 분석의 약 1/3보다 커서 cosine 0.5→0@300k 자체가 너무 강한 개입일 가능성이 크다.
- **종료 검증:** W&B lambda는 두 시드 모두 300k 이후 정확히 0. RSVP는 300k 이후 호출
  0회. fixed는 seed1에서 300,050에 1회 발생(episode/learner 갱신 경계 지연). 성능에는
  무시 가능한 1/12,003회지만 "정확히 0" invariant는 실패하여 commit `2ce3ec8`에서
  runner가 global step 경계를 직접 보도록 수정했다. 이후 런부터 적용된다.
- **해석 정정:** `fixed_floor`는 legacy 지수감쇠 후 0.05 floor, `fixed_zero`는 cosine이므로
  두 팔은 0–300k dose도 다르다. 따라서 이 비교는 cutoff 단독 인과효과가 아니라 제안한
  **스케줄 전체 패키지** 비교다. cutoff 단독효과에는 동일 pre-cutoff 궤적의 hard-zero
  대조가 추가로 필요하다. 반면 fixed-zero↔RSVP-zero는 lambda가 같아 scheduler 비교다.

**동시에 끝난 paper-alignment 중간 판정:** 5m6m QMIX seeds0/1은 final win 0/0,
LEHCA full(mask+shape)도 seeds0/1 모두 0/0이다. mask-off는 seed0만 AUC/final
0.031/0.086이고 seed1은 현재 1.15M까지 0이다. 즉 PDF의 epsilon 300k를 적용한 1.2M
설정에서는 QMIX 자체가 붕괴해 LEHCA 재현이나 우위 검정이 불가능하다. 2s3z 300k seed0은
QMIX AUC/final 0.088/0.021, LEHCA full 0.129/0.135로 방향은 양수지만 n=1·짧은 지평이라
결론으로 쓰지 않는다.

**실행 결함과 복구:** 961901은 동일 노드 port 충돌로 23초 실패. Pursuit bundle은 한
A10에 vLLM+500-step learner 3개를 넣어 zero 두 팔이 CUDA OOM(유효 결과 제외), floor만
정상 진행했다. 단일 GPU/단일 learner와 job-id 고유 포트 wrapper(commit `ce2e997`,
`edf6905`)로 재구성했다. Pursuit용 vLLM 제한 0.64도 모델 적재 전에 실패하여 폐기했고,
기존에 2시간 이상 안정성이 확인된 0.70으로 고정했다. 현재 5m6m floor 967092/967093은
실행 중이고 Pursuit floor seed0은 961902에서 실행 중이다. 나머지 Pursuit 단일-run은
967133~967137로 재큐잉했으며 job 10개 상한을 계속 채운다.

### 8af. SMAC sparse-reward 재현 스크린 (09-11 17:10, Codex, 진행 중)

**근거.** 본문 Eq.2는 `R_env`를 "typically sparse in SMAC"이라 하고, shaping 제거
ablation을 sparse environment reward만으로 학습한다고 설명한다. 공식 supplement도
"Sparse win/loss and optional death/kill rewards"라고 쓰지만 실제 `reward_sparse`,
`reward_scale` 값은 공개하지 않았다. 표준 SMAC/PyMARL의 기본은 현재 설정처럼
`reward_sparse=False`인 damage+kill+win dense reward라 문서와 구현 사이에 미확정
불일치가 있다.

**설정.** 가장 자연스러운 terminal sparse 변형인 `reward_sparse=True`,
`reward_scale=False`를 사용한다. 중간 보상 0, 승리 +1, 패배 −1이다. scale을 켜면
5m6m terminal reward가 약 ±0.0377로 줄어 LLM shaping과 단위가 심하게 어긋나므로 첫
스크린에서 제외한다. 그 외에는 paper-alignment와 동일하게 epsilon 1→0.05@300k,
QMIX/LEHCA backbone, full LEHCA(F50, mask+shape, paper prompt, cache off, temp0.2)를 쓴다.
모든 런은 300k다.

| map | arm | seeds | Slurm | Sacred | 상태 |
|---|---|---:|---|---|---|
| 5m_vs_6m | sparse QMIX | 0,1 | 967231, 967232 | 314,312 | RUNNING |
| 5m_vs_6m | sparse LEHCA full | 0,1 | 967233, 967235 | 316,315 | RUNNING |
| 2s3z | sparse QMIX | 0 | 967236 | 313 | RUNNING |
| 2s3z | sparse LEHCA full | 0 | 967237 | 317 | RUNNING |
| 2s3z | sparse QMIX/full | 1 | 967248, 967249 | 대기 | AssocMaxJobsLimit |

기존 dense 비교군은 같은 seed·epsilon의 paper-alignment 런을 300k에서 절단해 재사용한다.
보상 단위가 다르므로 dense와 sparse의 raw return 크기를 직접 비교하지 않고 win AUC,
final win, first-win time, dead-enemies와 학습 발생 여부를 본다.

**사전 판정.** (1) sparse QMIX는 못 배우고 sparse LEHCA만 300k 내 승률이 뜨면 reward
mode가 논문 패턴과 재현 격차를 설명한다. 이후 n=4와 장기지평으로 확장한다. (2) 둘 다
못 배우면 sparse 설정만으로는 논문 결과가 재현되지 않는다. (3) 둘 다 비슷하게 배우면
LEHCA의 보고 이득 원인은 reward sparsity가 아니다. 어떤 경우에도 sparse에서 LEHCA가
이기는 사실만으로 표준 dense SMAC 우위를 주장하지 않는다.

**우선순위 정리.** cutoff/Pursuit 전체와 5m6m shaping-off seed0(t≈773k, 이미 붕괴), 그
전용 서버를 중단했다. 거의 끝난 mask-off seed1(t≈1.18M)과 shaping-off seed1(t≈1.08M),
필요한 서버 2개는 완주 보존했다. 중단 런은 sparse 결과에 합산하지 않는다.

### 8ag. 원문 reward 문맥 감사와 dense 2s3z 1M 재현 게이트 (09-12 00:15, Codex)

로컬 `paper/LEHCA.pdf`와 Nature 공식 Supplement를 직접 추출해 확인했다. 본문은 2s3z를
heterogeneous symmetric coordination으로 분류하고, sparse-reward 시나리오라고 명시한 맵은
`2m_vs_1z`와 `3s_vs_5z`다. Supplement의 Rewards 문장은 sparse win/loss와 optional
death/kill 및 SMAC config 인자를 열거하지만 실제 flag/value를 한 개도 제공하지 않는다.
따라서 09-11 sparse 2s3z 캠페인은 reward-regime 진단이지 논문 재현으로 간주하지 않는다.
Supplement는 PyMARL 기반이라고 명시하므로 `target_update_interval`의 "steps"는 PyMARL의
episode 기준 갱신을 느슨하게 부른 것으로 우선 해석한다. 이를 근거 없이 step 기준으로
변경하지 않는다.

새 게이트는 표준 SMAC dense reward(`reward_sparse=False`, `reward_scale=True`), 2s3z,
1M, epsilon 1→.05@300k, seeds0/1이다. 네 팔은 (a) `qmix_paper`, (b) full LEHCA F50,
(c) RSVP runner fixed F50 shaping-only, (d) RSVP Fmax200 shaping-only다. full은 논문 재현,
fixed↔RSVP는 동일 runner·mask off·learner shaping·lambda floor@40%에서 scheduler 직접
대조다. cache off, temp .2, paper prompt, 관측 가능한 d_t를 공통 사용한다.

실행 스크립트는 `scripts/archive/run_dense_2s3z_1m_arm.sh`. Slurm 968967/968968=qmix,
968969/968970=full LEHCA, 968971/968972=fixed shaping-only이며, qmix 각 seed 종료 후
968973/968974=RSVP가 자동 시작한다. 기존 sparse LEHCA/RSVP 4런과 합쳐 최대 10개 유효
학습 슬롯을 사용한다. 유휴 LLM 960650은 회수했다. 판정은 먼저 dense full LEHCA가 같은
seed QMIX 대비 AUCearly를 일관되게 개선하는지, 그 다음 fixed↔RSVP의 성능·호출을 본다.

### 8ah. sparse 2s3z 중간 분기와 fixed shaping-only 예약 (09-12 00:20, Codex)

1M sparse 진행 중 full LEHCA는 seed0/1 각각 180k/190k까지 승률 AUC
0.0576/0.0828, 마지막 평가 0.0625/0.0938로 초기 비영 신호를 유지한다. 반면 RSVP는
502k/573k까지 AUC 0/0.0011, 최근 승률 0/0이다. full과 RSVP는 masking과 scheduler,
runner-side/learner-side shaping이 함께 달라 이 차이를 scheduler 효과로 읽을 수 없다.

가장 필요한 누락 대조인 RSVP runner의 fixed F50 shaping-only를 `run_sparse_1m_arm.sh`에
추가했다. reward/lambda/prompt/mask/learner shaping은 RSVP와 같고 scheduler만 fixed F50이다.
Slurm 968993/968994가 기존 sparse RSVP job 967778/967780의 `afterany`로 예약되어 각 슬롯을
이어받는다. fixed가 학습하면 RSVP의 긴 refresh/검출 실패가 후보이고, fixed도 실패하면 full의
action masking 또는 runner-side shaping 차이가 후보로 남는다.

### 8ai. Commander subgoal 중복 감사와 corrected screen (09-12 00:36, Codex)

새 dense 2s3z full/fixed의 JSONL을 직접 감사했다. subgoal이 있는 guidance 1,068개 중
633개(59.3%)에 동일 `(predicate, unit_type)`가 반복됐고, 5,624 entry 중 1,360개(24.2%)가
중복이었다. 특히 `enemy_kill` 631건, `enemy_damage` 414건, `ally_survive` 309건이다.
단순 weight 합 4,237.3은 각 semantic feature의 최대 weight만 남긴 3,239.0보다 1.308배다.
기존 sparse full은 중복 guidance 55.3%/weight 1.281배, sparse RSVP는 54.7%/1.282배로
동일 현상이 재현됐다. LLM 출력 예시에는 같은 `enemy_kill`이 두 번 들어가며 기존
`sanitize_guidance`는 이를 통과시키고 `compute_shaping`은 두 번 합산했다.

수정은 semantic key별 하나만 남기되 최대 weight를 보존한다. typed predicate의 unit type은
공백 제거와 casefold 후 비교하고, 고유 subgoal 6개 제한은 dedup 뒤 적용한다. 실험 재현을
위해 `deduplicate_subgoals=False` legacy 경로도 명시적으로 유지했다. 기본은 corrected=True.
`analysis/test_lehca_paper_alignment.py` 11 tests와 py_compile이 통과했다. 원자료 재감사는
`analysis/audit_guidance_duplicates.py <glob...>`로 가능하다.

현재 1M legacy campaign은 실행 의미를 보존한다. 아직 시작하지 않은 dense RSVP와 sparse
fixed도 False로 잠가 대응하는 실행 중 legacy arm과 동일하게 했다. corrected screen은 표준
dense 2s3z, 300k, seeds0/1에서 LEHCA β=0.1+dedup(969095/6)과 fixed F50+dedup
(969062/3)이다. LEHCA의 중복 단독 효과는 soft action bias와 얽히므로 mask 없는 fixed가
dedup 인과 대조를 담당한다. RSVP corrected 969064/5는 이후 우선순위 재평가로 시작 전
취소됐다.

### 8aj. guidance 상태 설명력·soft action intervention 감사 (09-12 00:41, Codex)

중복 제거 canonical subgoal vector를 대상으로 같은 cache key가 10회 이상 나온 표본만
분산분해했다. sparse full은 132개 key/4,303호출에서 state R²=0.136, sparse RSVP는
117개 key/3,058호출에서 R²=0.175였다. coarse key가 설명하지 못한 동일-state 변동이
82~86%라는 뜻이며, key 자체가 불완전하므로 상태 무관성의 증명은 아니다. 그러나 현재
guidance의 상태 적응 신호보다 sampling/grounding 변동이 더 크다는 정량 경고다.
`analysis/audit_guidance_state_dependence.py`로 재현한다.

Sacred320/321의 sparse full에서 hard-mask override 평균은 0.004%/0.002%, forbid 가능한
행동 비율도 0.002%/0.004%에 불과했다. 반면 soft tilt argmax override는 평가 구간 평균
30.8%/31.1%이고 200~210k 최근값도 30.9%/31.2%다. β=0.5인 데 비해 최근 Q-gap은
0.031/0.037이므로 soft preference가 행동 정책에 강하게 개입한다. 따라서 full만 보인 sparse
승률은 reward shaping보다 action bias로도 설명된다.

직접 ablation은 sparse 2s3z 300k mask-off seeds0/1(969086/7)이며 현 sparse full 슬롯 뒤에
연결했다. full과 동일 LEHCA runner, F50, paper prompt, collection-time shaping, legacy duplicate
semantics를 쓰고 hard/soft mask만 끈다. mask-off가 full의 양의 신호를 잃으면 action channel,
유지하면 shaping channel을 우선 후보로 둔다. 제출 상한 때문에 gate 통과 후에만 필요한
corrected RSVP 969064/5는 시작 전 취소했으며 재제출 가능하다.

현재 dense full은 공개되지 않은 β에 config 기본 0.5를 사용한다. 기존 진단에서 β=0.5 tilt가
Q-gap보다 크고 β=0.1이 안정적이었으므로, dense full β=0.5는 그대로 민감도 상단으로
완주시키고 corrected LEHCA는 β=0.1+dedup(969095/6)으로 교체했다. 이 조합은 재현 후보
게이트이지 단일요인 ablation이 아니다. dedup 단독효과는 mask-off corrected fixed(969062/3),
β/action 효과는 legacy β=0.5 full 및 sparse mask-off와 함께 해석한다.

### 8ak. typed grounding 잔여 오류와 β 기본값 결정 (09-12 00:47, Codex)

실제 2s3z 병종 집합에 없는 typed 문자열 비율은 subgoal 기준 dense full 7/172(4.1%),
dense fixed 5/238(2.1%), sparse full 73/2,505(2.9%), sparse RSVP 44/2,640(1.7%)다.
`attack_type` token의 무효율은 0.28~0.47%다. 주로 `all`, `ranged`, `enemy_stalker`,
`EnemyStalker`처럼 grounding이 매칭하지 못하는 별칭이다. 이는 실제 무효 feature지만 수 %
수준이므로 중복 reward와 soft policy override보다 후순위 병목으로 둔다. 재현 감사 명령은
`analysis/audit_guidance_state_dependence.py --valid-unit-types Stalker,Zealot <globs>`다.

논문 본문·Supplement 어디에도 β 숫자는 없다. 따라서 β=0.5는 논문값이 아니라 초기 구현
선택이다. 실제 sparse full에서 약 31% argmax override가 관측됐고 과거 β=0.1 진단이 더
안정적이므로 `config/algs/lehca.yaml`의 신규 기본을 0.1로 변경했다. 실행 중 프로세스는 이미
기존 값을 읽어 영향받지 않는다. `run_dense_2s3z_1m_arm.sh`는 이미 시작한 legacy campaign
재현을 위해 기본 0.5를 명시하고, corrected gate 969095/6은 0.1을 command line으로 고정한다.

### 8al. dense/sparse 2s3z 1차 결과와 게이트 판정 (09-12 14:02, Codex)

**Corrected LEHCA 300k.** Slurm969095/6은 exit 0으로 완주했다. AUC는 30회 전체 평가,
final은 마지막 3회다. QMIX는 같은 seed의 1M run을 t≤300k로 절단했다.

| seed | QMIX win AUC/final | LEHCA β.1+dedup | QMIX/LEHCA return AUC | 판정 |
|---:|---:|---:|---:|---|
| 0 | .0875 / .0208 | .1115 / .0521 | 11.50 / 11.34 | win만 소폭 양수 |
| 1 | .1927 / 0 | .0927 / .0208 | 12.48 / 12.07 | AUC 큰 열세 |

두 seed 방향이 엇갈리고 LEHCA 절대 final이 .052/.021이며 return AUC도 둘 다 낮다.
따라서 “paper-style 2s3z에서 LEHCA가 QMIX를 안정적으로 개선”하는 재현 게이트는 실패다.
이 결과 뒤 RSVP corrected sweep은 수행하지 않는다. 완료 run의 LLM 호출은 각 5,818회다.

**Dense 1M 완주.** QMIX(968967/8)와 legacy RSVP(968973/4)는 모두 exit 0이다.

| 방법 | seed0 AUC/final | seed1 AUC/final | 호출 s0/s1 |
|---|---:|---:|---:|
| QMIX | .4225 / .9500 | .0638 / .0187 | 0 / 0 |
| RSVP Fmax200 | .1263 / .5437 | .0259 / .0031 | 7,636 / 6,796 |

QMIX seed0은 후반 회복했지만 seed1은 실패해 백본 자체가 극단적 고분산이다. 두 seed 모두
50~100k peak 뒤 150k부터 퇴행했고 loss, TD error, Q mean은 발산하지 않았다. seed0만 후반
회복했으므로 평가 노이즈보다 정책 퇴행·국소해와 seed 민감성 문제다. n=4 확인용 QMIX
seed2/3을 972100/1로 즉시 시작했다.

RSVP는 같은 seed QMIX보다 AUC와 final이 모두 낮아 현 상태에서 우위나 강건성을 지지하지
않는다. 진행 중 legacy fixed의 현재 지평에 맞추면 seed0 H=662k에서 fixed/RSVP final
.182/.037, seed1 H=642k에서 .490/0이다. 호출은 fixed 13,246/12,855 대 RSVP
4,961/4,669로 약 63% 적지만, RSVP seed1은 shaping nonzero .103, early refresh .115/ep로
함께 붕괴했다. 검출 입력이 풍부해지는 성공 궤적에서만 갱신이 늘어나는 endogenous feedback
가능성이 있으므로 fixed 완주 뒤 전체 곡선과 phase-conditioned refresh를 재검사한다.

**Sparse 1M 및 진행군.** QMIX final은 0/0, aligned도 0/0, RSVP도 0/0이다(RSVP AUC
0/.0006). terminal sparse reward만으로 shaping-only 학습은 재현되지 않았다. full LEHCA는
693~753k 현재 AUC .0246/.0897, final .0089/.1071로 seed1에만 신호가 남았다. fixed F50은
442~502k 현재 AUC .0014/.0006, final 0/0이다. full과 shaping-only 차이의 우선 후보는
약 17~24% 현재 soft argmax override이며, full 종료 후 동일 구성 mask-off 969086/7로
직접 판정한다.

**자원 재배치.** corrected LEHCA 완료로 빈 두 슬롯에 QMIX seed2/3을 넣고, legacy fixed
종료 dependency였던 corrected fixed dedup 969062/3을 즉시 해제해 시작했다. Sacred340/341은
QMIX seed2/3, 342/343은 corrected fixed seed1/0이며 설정을 재검증했다. 현재 10 RUNNING,
mask-off 969086/7만 dependency PENDING이다. 모니터는 현재 metric명 `llm_calls` fallback과
신규 ID를 포함한다. QMIX seed2/3 뒤에는 동일 seed corrected LEHCA β=.1+dedup 300k
972127/8을 추가 연결했다. n=2 방향 불일치와 QMIX 고분산 때문에 paired n=4까지 확인하되,
절대 성능 게이트를 완화하지는 않는다.

### 8am. 완료 결과 재검산과 LLM 출력 직접 검토 (09-12 22:58 KST, Codex)

사용자 요청에 따라 Slurm squeue/sacct와 Sacred info/config를 직접 확인했다. README의
14:02 스냅샷보다 진행됐다. 아래 평가평균은 전체 평가의 산술평균이며 논문 AUCearly가
아니다. final은 예산의 마지막 10% 구간에 있는 실제 평가 평균이다. 1M 완료 런은 마지막
평가가 약 993~996k에 있다. Sacred RUNNING 문자열로 완료 여부를 판정하지 않았다.

**Dense 2s3z 1M 완료:** QMIX seed0–3, fixed F50 seed0/1, RSVP seed0/1의 Slurm
종료 상태는 모두 COMPLETED/exit 0이다. fixed와 RSVP는 legacy 중복 의미, mask-off,
learner-time shaping, lambda floor@400k, paper prompt, epsilon300k를 공유한다.

| 방법 / Sacred | seed | 전 평가 승률 평균 | final 승률 | 마지막 train llm_calls |
|---|---:|---:|---:|---:|
| QMIX / 329 | 0 | .4225 | .9500 | 0 |
| QMIX / 328 | 1 | .0638 | .0187 | 0 |
| QMIX / 340 | 2 | .4556 | .9344 | 0 |
| QMIX / 341 | 3 | .6103 | .9469 | 0 |
| fixed F50 / 331 | 0 | .2241 | .6875 | 19,875 |
| fixed F50 / 332 | 1 | .3247 | .8719 | 19,884 |
| RSVP Fmax200 / 337 | 0 | .1263 | .5437 | 7,636 |
| RSVP Fmax200 / 336 | 1 | .0259 | .0031 | 6,796 |

QMIX는 네 seed 중 세 seed가 final 93% 이상이다. fixed는 QMIX 실패 seed1을 회복시키나
seed0을 훼손한다. RSVP는 fixed보다 두 seed 모두 평가평균과 final이 낮다. 마지막
train counter 합 기준 호출은 63.7% 적지만 성능 보존에 실패했다. 로그 마지막 지점의
누적 호출이며 종료 직전의 미기록 호출과 test 호출, HTTP 재시도 비용을 포함하는 총계는 아니다.

**Sparse 2s3z:** full LEHCA 321/320의 마지막 10% 승률은 seed0=0, seed1=.1281이며
전 평가평균은 .0187/.0947이다. QMIX 326/327, aligned 322/323, RSVP 325/324는
모두 1M final=0이다. fixed 339/338은 약 693k/754k에서 Slurm CANCELLED 상태다.
따라서 sparse fixed를 1M 완료 실패로 보고하지 않는다. full의 seed1 양의 신호도 안정적인
재현 성공이나 scheduler 효과를 입증하지 않는다.

**진행 중 8 jobs, 대기 0 (약 22:58):** corrected LEHCA seeds2/3 344/345는 약271k/271k
of300k; train-only soft RSVP 346/347은 약411k/421k of1M; 대응 soft fixed 349/350은
약100k/70k of1M; sparse mask-off 351/348은 약30k/120k of300k다. soft 팔은 β=.1,
dedup=True, training soft-only, test mask-off, paper prompt를 공유한다. 최근 1회 승률로
방법을 비교하지 않으며 soft fixed가 짧아 현재 RSVP와 전체 평균을 직접 비교하지 않는다.

**Corrected LEHCA n=4 공통 구간:** 아직 seed2/3이 미완주라 공통 0–260k를 보간
사다리꼴 적분(y(0)=0)했다. corrected LEHCA−QMIX win AUC는 seed0 +.0235,
seed1 −.1182, seed2 +.0184, seed3 −.0141이다. 신규 seed에서도 방향이 엇갈린다.
이 값은 전체 평가 산술평균 및 논문 첫20% AUC와 구분한다.

**비교 설계 정정:** corrected fixed 342/343은 완료됐으며 300k 내 평가평균이
seed0=.1260 / seed1=.1604로 legacy fixed 절단치 .0688/.0563보다 높다. 그러나
`t_max=300000, lambda_floor_frac=.4`여서 floor가 120k이고, legacy 1M 팔은 400k다.
실제 120k 근처 lambda 로그도 corrected=.05 / legacy≈.250이다. §8ai/8aj의
“dedup 단독효과/인과 대조”라는 설명은 성립하지 않는다. **중복 제거와 빠른 감쇠의
결합 변화**다. 기존 1M 시간축을 맞춘 300k 후속 대조라면 floor_frac=4/3이 필요하며,
이번 점검에서는 실행이나 설정을 변경하지 않았다.

LLM 문서 16개 사례와 실제 pipeline smoke 8개 응답을 직접 읽은 분석은
[llm-guidance-review-20260912.md](../llm-output/llm-guidance-review-20260912.md)에 기록했다.
기존 성능 실험은 모두 paper이며 paper_v2의 학습 성능 검증 결과가 아니다.

### 8an. RSVP 메커니즘 즉시 감사 (09-12 23:46 KST, Codex)

사용자 요청으로 현재 dense/sparse/soft RSVP의 trace·phase·guidance를 직접 분석했다.
범위·정량표·크리틱 검증의 기록 한계는
[rsvp-mechanism-audit-20260912.md](rsvp-mechanism-audit-20260912.md)에 둔다.
발급 시 저가치 게이트가 닫히면 중간에 감시를 재개하지 않고 상한까지 유지하는 구조,
새 지침에도 단독 생존 focus_fire가 반복되는 현상을 확인했다. 크리틱의 학습 loss를
별도 데이터 예측력으로 해석하지 않는다. 이번 작업은 오프라인 분석과 문서 추가이며
실행 중 실험·학습 소스·기존 결과를 변경하지 않았다.

### 8ao. 현황 재점검 및 RSVP 검증 계측 (09-13 04:05 KST, Codex)

Slurm 실행 5개, 대기 0개. corrected LEHCA seed2/3 jobs972127/972128 및 sparse
mask-off seed1 job969087은 sacct COMPLETED다. Sacred run.json의 잔류 RUNNING 대신
Slurm 완료 상태를 적용한다. 아래 진행량은 마지막 **평가** step으로 실제 학습량보다 작다.

| 실행 / Sacred | seed | 마지막 평가 step | 상태 | 마지막 1회 평가 승률 |
|---|---:|---:|---|---:|
| soft RSVP / 346 | 0 | 943,089 / 1M | RUNNING 973173 | .15625 |
| soft RSVP / 347 | 1 | 913,710 / 1M | RUNNING 973174 | .62500 |
| soft fixed / 349 | 0 | 330,848 / 1M | RUNNING 973175 | 0 |
| soft fixed / 350 | 1 | 290,712 / 1M | RUNNING 973176 | .03125 |
| sparse mask-off / 351 | 0 | 281,059 / 300k | RUNNING 969086 | 0 |
| sparse mask-off / 348 | 1 | 290,949 / 300k | COMPLETED | 0 |

soft RSVP seed1의 현재 900k 이후 평균은 .7031, seed0은 .1125다. 아직 미완주이고
평가점도 적어 final로 부르지 않는다. 공통 0–280k 내 **평가 승률 산술평균**은
RSVP seed0/1=.0703/.0826, fixed=.0223/.0569다. 초기 구간은 RSVP가 두 seed 모두
높지만 fixed 후반이 아직 없어 전체 학습 성능 우위를 확정하지 않는다.

corrected LEHCA 네 seed 모두 300k 완료. 동일 0–300k 평가 승률 산술평균은 다음과 같다.
이 표는 보간 AUC가 아니며 평가 간격은 약10k다.

| seed | corrected LEHCA | QMIX 첫300k | 차이 |
|---:|---:|---:|---:|
| 0 | .1115 | .0875 | +.0240 |
| 1 | .0927 | .1927 | −.1000 |
| 2 | .0885 | .0948 | −.0063 |
| 3 | .1792 | .2188 | −.0396 |

네 seed 중 하나만 개선한다. 따라서 현재 수정으로 LEHCA 재현이 해결됐다고 볼 수 없다.
sparse mask-off seed1은 모든 평가 승률 0이며 seed0도 현재 평균 .0022로 약하다.
full seed1의 1M 양의 결과와 300k mask-off를 지평 정렬 없이 인과 비교하지 않는다.

**향후 검증 준비:** 검증 계획（당시 파일 `validation/rsvp-validation-plan-20260913.md`; 원문은 서버 정리 백업）에 단계별 질문·판정·
명령을 기록했다. 학습 전 prediction과 episode 종료 target의 순차 검증 로그,
head별 baseline 대비 예측 점수 CLI, issuance gate를 공유하는 timer 대조군,
soft-only 무작위 early-refresh/hold trial, lambda floor 절대400k 실행 스크립트를 추가했다.
현재 실행에는 새 계측이 없으므로 기존 자료만으로 critic 예측력이나 갱신 인과효과를
산출할 수 없다. 관련 unittest 23개 중22개 통과/1개 optional archive 생략, shell 문법 및
`git diff --check` 통과. 실제 SC2/LLM 신규 통합 smoke는 미실행이며 새 job 제출·기존 job
재시작은 하지 않았다. 계측 구현의 합성 runner 테스트와 실제 학습 검증을 구분한다.

### 8ap. 연구 결과 종합 및 AAMAS 26일 계획 (09-13 04:19:55 KST snapshot, Codex)

사용자 요청으로 결과 종합 보고서（당시 파일 `archive/research-results-20260913.md`; 원문은 서버 정리 백업）와
26일 연구계획（당시 파일 `research-plan-aamas2027-20260913.md`; 원문은 서버 정리 백업）을 작성했다. 최근43개 실행의
config/info 집계와 원본 SHA256, Slurm 응답은
[JSON snapshot](results-snapshot-20260913.json)에 보존했다. 과거 연구는 기존 감사·
원장을 교차해 종합했으며 버전별 결과를 합산하지 않았다.

sparse mask-off seed0 job969086도 COMPLETED로 바뀌어 두 seed 모두300k 완료,
final 승률0/0이다. full LEHCA 첫300k 대비 mask-off의 손실은 train+test action
channel 제거의 결합 결과이며 훈련 효과만 분리하지 않는다. 현재 실행은 soft 네 job,
대기0이다. 마지막 평가 step은 RSVP seed0/1=973283/943773,
fixed seed0/1=340876/310780이며 후반 비교는 미완주로 남긴다.

새 계획은 무기한 재현 탐색을 중단하고 예측(H1)→갱신 이득(H2)→전체 성능·비용(H3)을
분리한다. 9/18 설정 동결, 9/20 방법 유지·측정 연구 전환·제출 보류 판단,
9/29 데이터 동결, 10/8 KST 내부 제출을 기준으로 한다. shaping-only가 기본이며
soft trial을 원래 방법의 추가학습 효과로 해석하지 않는다. 같은 ceiling F200 및
issuance gate timer 대조와 개발에 쓰지 않은 확인 seed를 우선한다.

이번 작업은 분석·MD/JSON 기록이며 신규 job 제출·기존 job 중단·학습 코드 변경은 없다.
공식 일정 및 제출 요건의 출처는 새 계획에 연결했다.

### 8aq. 저비용 refresh replacement gain 후보 측정 (09-13 18시 KST, Codex)

shaping-only에서 갱신 직후 같은 episode의 정책 행동이 guidance의 영향을 받지 않는 점을
이용해, 동일한 실제5-step suffix에 old/new guidance shaping을 모두 계산했다. 정의·해석·
판정 조건은 저비용 검증 문서（당시 파일 `validation/rsvp-efficient-refresh-metric-20260913.md`; 원문은 서버 정리 백업）, 원시 집계는
`results/diagnostics/rsvp_e1_20260913_v2/refresh-replacement-gain.json`에 보존했다.

완료 seed1의240개 early refresh에서 new−old gain은−.0182, episode-cluster bootstrap
95% CI [−.0334,−.0034], 양의 비율32.1%였다. 진행 중 seed0의190개 사건에서는
+.0060 [−.0119,.0237], 양의 비율35.8%였다. 기존 guidance의 trigger 전후 value 감소는
평균적으로 양수였고 selected-head prediction skill도 seed0/1=.441/.520이지만,
새 guidance의 즉시 value 회복은 두 seed에서 확인되지 않았다. 이 후보만 보면 예측기보다
replacement 단계가 병목일 가능성이 있지만, corrective guidance의 학습 효과까지 판정할
수는 없다. 이는 장기 승률 인과효과가 아니라 즉시 실행 가능성·상태 적합성 진단이며,
주 metric으로 확정하지 않는다.

계산기와 합성 테스트 `analysis/summarize_refresh_replacement.py`,
`analysis/test_refresh_replacement.py`를 추가했다. 관련 validation 테스트와 함께10개가
통과했고 `git diff --check`도 통과했다. 추가 환경 rollout·LLM 호출·GPU job은 없었다.

### 8ar. RSVP 무작위 지연 갱신 실험 구현·예약 (09-13 22:40 KST, Codex)

사용자 지시에 따라 단순 seed 확장이 아닌 RSVP 갱신 실효성 개입 실험을 구체화했다.
shaping-only에서 같은 episode return은 처치 결과가 아니므로 기존 `RefreshTrial`을 그대로
사용하지 않았다. 실제 trigger와 사전 표집한 armed non-trigger에서 candidate guidance를
두 팔 모두 생성한 뒤, 즉시 적용과 기존 지침 유지에 1:1 배정한다. trigger episode를
제외한 다음5개 training episode의 raw environment return이 주 outcome이며, block 안에서는
early와 Fmax refresh를 모두 잠근다. 종료 뒤 두 팔 모두 공통 refresh를 거쳐 scheduler를
재개한다. 정의·판정은
무작위 지연 실험 설계（당시 파일 `validation/rsvp-refresh-mrt-design-20260913.md`; 원문은 서버 정리 백업）에 기록했다.

`algorithm/rsvp/validation.py`, RSVP runner, 요약기, 설정과 실행 wrapper를 수정했다.
trigger/control cap은 각각50/seed, control episode probe 확률은.1, 개입 시작은200k다.
passive 예측 계측과 기존 비개입 실행은 변경하지 않으며 현재 Slurm run은 모두 기존 동결
소스를 사용한다. 관련 단위·통합·paper alignment 테스트26개가 통과했고 shell/Python 문법과
diff whitespace 검사도 통과했다.

실행 소스157개 파일을 `results/source_snapshots/rsvp_refresh_mrt_20260913_v1/`에 동결했다.
50k 실제 SMAC/LLM smoke980094는 E1 fixed976095 종료 뒤 시작한다. checker가 후속 episode
연결·trigger episode 제외·candidate 적용 일치와 block 완결을 검사하며, 실패하면 pilot
dependency가 풀리지 않는다. pilot seed0 job980098은 smoke 성공 뒤, seed1 job980099는
smoke와 win-only LEHCA seed0 성공 종료 뒤 시작한다. 두 pilot은 일반 RSVP 성능 확인군에
합산하지 않는 mechanism intervention run이다.

### 8as. 야간 실험 회수와 호출 예산 대조 추가 (09-14 11:19 KST, Codex)

Slurm 종료 상태와 Sacred 원자료를 다시 대조했다. E1 여섯 실행, win-only QMIX/LEHCA
각 네 seed, win-only soft RSVP/fixed F200 각 두 seed, MRT smoke와 seed0은 모두
exit `0:0`으로 완료됐다. 현재 traceback, OOM, LLM 요청 실패는 발견되지 않았다.

**Dense E1 1M 결과.** AUC는 `y(0)=0`을 포함한 0--1M 평가 승률의 정규화 사다리꼴
적분이며, 마지막 episode 경계와 1M 사이에는 마지막 평가값을 유지했다.

| 방법 | seed0 AUC / final100k | seed1 AUC / final100k | 평균 호출 수 |
|---|---:|---:|---:|
| RSVP | .2038 / .6375 | .1492 / .4500 | 7,362.5 |
| fixed F200 | .0646 / .0250 | .2804 / .7063 | 4,967.5 |
| gate-timer | .1618 / .5063 | .2299 / .7188 | 9,790.0 |

RSVP는 fixed F200보다 seed0에서 좋고 seed1에서 나쁘며, gate-timer도 seed별 방향이
엇갈린다. 두 seed 평균 AUC는 RSVP=.1765, fixed=.1725, gate-timer=.1958이다.
RSVP가 fixed F200보다 약48% 더 호출하므로 이 표만으로 갱신 시점 선택의 이득을 주장할
수 없다. gate-timer는 호출을 더 쓰면서 평균 AUC 차이도 작다.

**Win-only sparse 재현 게이트.** QMIX 네 seed는 모든 평가에서 승률0이다. full
LEHCA의 seed0--3 AUC는 각각 .0136/.0094/.0345/.0056이고 peak는
.125/.1563/.2188/.0625였지만, final100k는 .0031/0/.0125/.0031이다. 즉 네 seed
모두 한 번 이상 작은 탐색 신호를 만들었으나 학습이 유지되지 않았다. `LEHCA > QMIX`라는
안정적 baseline 재현으로 판정하지 않는다.

**Win-only soft 채널.** soft RSVP Fmax200 seed0/1 AUC는 .0240/.0010이고 둘 다
final100k=0이다. soft fixed F200은 0/.0042이며 역시 둘 다 final100k=0이다. soft
fixed F50은 seed0/1이 약261k/271k까지 진행했고 현재 AUC .0021/.0031, peak .03125,
최근100k=0이다. 완료 전 값이지만 soft masking이 희소 보상 학습을 안정화한다는 신호는 없다.

**Refresh MRT.** 실제 SMAC/LLM smoke는 여섯 block을 모두 통과했다. 두 pilot은 이미
각100 block(trigger50/control50)의 사전 cap을 채웠다. trigger에서 즉시 candidate를
적용한 뒤 다음5 episode의 raw return과 기존 지침을 유지한 팔의 차이는 seed0
`-.3189` (bootstrap 95% CI `[-.7329,.0888]`), seed1 `-.0355`
(`[-.3639,.2960]`)다. trigger 효과에서 control 효과를 뺀 timing selectivity gain도
seed0 `-.6255` (`[-1.8200,.5622]`), seed1 `-.0898`
(`[-.9558,.7828]`)로 두 seed 모두 음수다. 모든 candidate 생성은 성공했다. 현재 자료는
“갱신 trigger 순간에 새 지침을 즉시 적용하면 가까운 미래 return이 좋아진다”는 가설을
지지하지 않는다. CI가0을 포함하므로 손해를 확정한 것도 아니다. seed1 학습은 약642k에서
계속되지만 mechanism block 수는 더 늘지 않는다. 재현 가능한 집계는
`results/diagnostics/rsvp_refresh_mrt_20260913_v1/mrt-effect-summary-20260914.json`에 보존했다.

**호출 예산 일치 대조.** RSVP 두 seed의 평균 기록 호출7,362.5회에 맞춘 주기는
`1,000,000 / 7,362.5 = 135.82`이므로 fixed F136을 선택했다. seed0/1 jobs
982780/982781을 11:14 KST에 제출했고 둘 다 GPU를 잡아 private vLLM의 실제 요청과
Sacred379/380 생성을 확인했다. dense 2s3z, shaping-only, 1M, lambda floor 절대400k와
평가 설정은 E1과 같다. 동결 소스는
`results/source_snapshots/rsvp_e1_fmatch136_20260914_v1`, 제출 원장은
`results/diagnostics/rsvp_e1_fmatch136_20260914_v1/queue-manifest.json`이며, 동결 소스에서
관련 unittest26개가 통과했다. 이 비교가 끝나야 같은 호출 예산에서 RSVP timing의 H3를
판정할 수 있다. 현재 음성 결과 때문에 win-only seed 확장이나 threshold sweep은 추가하지 않았다.

### 8at. 09-14 13:27 KST 재점검

Win-only soft fixed F50 seed1 job980263은 11:43, seed0 job980260은 12:00 KST에
각각 exit `0:0`으로 완료됐다. 최종 seed0/1 AUC는 .0021/.0031, peak는 둘 다
.03125, final100k는 둘 다0이다. 호출은 5,820/5,825회다. 먼저 완료된 soft RSVP와
fixed F200까지 포함한 세 scheduler 모두 두 seed의 final100k가0이므로, soft masking과
갱신 빈도 증가가 win-only sparse의 학습 유지 문제를 해결하지 못했다.

현재 Slurm 실행은 세 개다. MRT seed1 job980099/Sacred376은 약843k, fixed F136
seed0 job982780/Sacred379는 약211k, seed1 job982781/Sacred380은 약211k다. MRT는
여전히 LLM failure0이며 전체 평가곡선은 최근 상승했지만, 개입 run이므로 일반 RSVP
성능 팔에 합산하지 않는다. F136의 공통0--200k AUC는 seed0/1=.0502/.1425이고,
같은 구간 RSVP는 .1724/.0789다. 시작부터 방향이 엇갈리므로 완료 전에 방법 차이를
판정하지 않는다. 세 job의 로그에서 traceback과 OOM은 발견되지 않았다.
