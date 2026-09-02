# Research track — VIGIL: LLM 커맨더 가이던스의 적응 갱신 스케줄링

세션 역할 B(HANDOFF.md). 베이스라인(`algorithm/lehca`)은 건드리지 않고 `algorithm/vigil/`에 격리.
문서는 역할별 하나씩. 논문 서사는 draft-paper, 결정은 여기 로그에, 수치는 experiments-log에만 둔다.
(problem.md·idea-draft.md는 2026-09-01 draft-paper.md로 흡수·삭제.)

| 문서 | 역할 | 언제 읽나 |
|---|---|---|
| [draft-paper.md](draft-paper.md) | **논문 초안 v1**(9/2 개정): 문제·배경·정식화·방법(VIGIL — 왜 V인가 §4.1, V 계산 §4.2, 결정 규칙·h 컨트롤러 §4.3)·예비 근거·실험 설계 | 먼저 읽을 것 |
| [related-work.md](related-work.md) | 관련연구 지도(직접 경쟁·HRL·MARL 시간추상화·변화 감지·CUSUM-in-RL·이론 선례) | 포지셔닝·인용 |
| [theory-cusum.md](theory-cusum.md) | CUSUM/QCD 설명, 가정 5개와 우리 설정의 대응, 조건부 판정 | 이론 근거 확인 |
| [grf-testbed.md](grf-testbed.md) | GRF 5v5 설계: d_t, 접지 어휘, predicate, 프롬프트, ν 정의, 빌드 레시피 | 구현 |
| [experiments-log.md](experiments-log.md) | 모든 프로브 결과·재생표·계산 방법 (단일 출처) | 수치 인용 |
| [code-guide.md](code-guide.md) | 코드 리딩 가이드: LEHCA·vigil 파일 지도, 읽는 순서, 러너 diff, 실행 명령 | 코드 읽기 전 |
| [retired.md](retired.md) | 폐기·보류 아이디어와 이유(regret, LLM 선언 유효조건, 값 CUSUM, φ-CUSUM 단독) | 같은 길 다시 가지 않기 |

## 현재 상태 (2026-09-02 저녁)
- 검증됨: temp 0에서 가이던스는 국면(소유권)에 체계적으로 의존(낡음 = 노이즈 12배); 고정 F는
  국면 지속시간 불규칙성(CV 0.99) 때문에 낭비·지연을 동시에 못 피함; 같은 호출 예산에서
  국면 시점 갱신이 고정 F를 지배(낡음 −21~−40%).
- 방법의 축(확정): **잔여가치 크리틱** — predicate별 가치 헤드 V̂_j(s)(리플레이 f-벡터의
  할인 접미합 타깃, γ_F=0.8), V_F=Σw_jV̂_j, 발급 대비 비율 v_t의 단측 CUSUM, h는 온라인 예산
  컨트롤러, 판단 불가 시 F_max 퇴화(LEHCA 지배 논거). 오프라인: 고정 F는 이김, 이벤트 상한엔 미달.
- 본 비교 1라운드 진행 중: GRF 5v5(500k) seed0 완료·seed1 진행, MMM(1M) seed0 진행
  (experiments-log §8~8b).
- 미검증(결정적): E2 — 학습 정책 위에서 낡음이 성능을 깎는가. 체크포인트 필요.
- 열린 설계 이슈: GRF 어휘가 저수준(선수 행동)이라 낡음 해악 일부를 제조함 → 전술 수준 어휘(L1)
  또는 SMAC(이미 L1)에서 재측정 필요. 접지 모듈을 명시적으로 정의·평가할 것.

## 결정 로그
- 08-29 A축(전술 갱신 시점) 메인, 마스킹은 논문 충실(β≤0.1).
- 08-31 regret 프레임 트리거로서 폐기. 가설 명명 S-시리즈. SMAC→GRF 전환(에피소드 내 급변 필요).
- 08-31 temp 0을 방법 전제로(접지 노이즈). 하이브리드 봇 정책(교착 해소).
- 09-01 LLM 선언 유효조건은 후순위(이론 부재). 전제조건 적용 가능 비율 → 학습 발화 확률로.
- 09-02 16:23 본 비교 1차 라운드 6런 투입(experiments-log §8). 코드 리뷰 6건 반영(tick 분리·keep_possession 이벤트화·shaping_in_learner=True·실패 재시도·버퍼 축소).
- 09-02 본 비교 테스트베드 확정: **GRF 5v5 + SMAC MMM2** (MMM 후퇴 조건 사전 등록). 실행 스크립트 scripts/run_vigil_{sc2,grf}.sh.
- 09-02 방법 가칭 **VIGIL** 확정 (wandb 그룹 접두 `VIGIL_`).
- 09-01 초안 모델 고정: **Guidance-Progress CUSUM** — 통계량 = 보유 서브골의 셰이핑 신호를 교차 스펠 기대값으로 정규화한 진척률 v_t(종료는 지지확률로 확정), 단측 CUSUM(k,h), β=0(셰이핑 채널), 훈련 시 주장. 손 진척 함수·손 전제조건 금지. 재생 결과 experiments-log §6.
- 09-01 다음: SMAC 이질 맵(MMM2/3s5z) temp 0 프로브 → GRF 셰이핑을 학습 러너에 연결 → 고정 F vs gp 학습 런.
- 09-02 통계량 교체: 컨텍스트 테이블 정규화 폐기(설계 자유도 과다, retired §5) → **잔여가치
  크리틱** 정식화 채택(γ 스윕으로 0.8 확정, experiments-log §7).
- 09-02 17:01 wandb 규칙 개편(프로젝트 AAMAS_<ENV>_<MAP>, run=알고리즘_디테일_seed, 그룹=시드 묶음,
  train/·test/ 분리) 후 1라운드 6런 재기동. LEHCA-원형 대조군 = MMM 캐시 off + temp 0.2 + β=0.1 확정.
- 09-02 17:12 SMAC 맵 **MMM2 → MMM**(사용자 결정, 사유 = 계산 예산 — 사전 등록 후퇴 조건과 구분 기록).
- 09-02 저녁 draft-paper **v1 전면 개정**(문제→아이디어 흐름, 왜 V인가/V 계산 명시, h 컨트롤러 반영).

## 코드
`algorithm/vigil/`(lehca와 같은 배치): runner.py·critic.py·predlib.py·commander/grf.py·shaping/grf.py
+ analysis/(probe_phase·probe_grf·effect·analyze_probe·analyze_guidance_quality·vf_replay).
환경: `env/gfootball.py`, `env/semantic/grf.py`. conda env `grf`(레시피 grf-testbed §10a → experiments-log §2a).
