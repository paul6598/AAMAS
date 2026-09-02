# 코드 리딩 가이드 — LEHCA 재현과 VIGIL

2026-09-02 (구조 개편 반영). 두 코드베이스를 처음 읽는 순서와, 각 파일이 무엇을 하고
서로 어떻게 연결되는지 정리. 세션 규칙: `algorithm/lehca/`는 베이스라인 세션 소유
(우리는 import만), `algorithm/vigil/`가 연구 코드.

---

## 0. 한눈에

```
                    ┌── LEHCA (베이스라인, 무수정) ──────────────────────────┐
env ─ snapshot ──→  │ semantic iface → d_t 요약 → LLM Commander → guidance   │
                    │ guidance ─┬→ shaping F_t (predicates) → r + λF → QMIX │
                    │           └→ masks (compiler) → Q̃ = Q + β·logW        │
                    │ 갱신: 매 f_update 스텝 (고정)                          │
                    └────────────────────────────────────────────────────────┘
VIGIL이 바꾸는 것: **갱신 시점 하나** (+ GRF 환경 지원). 나머지는 LEHCA 부품을 그대로
소비한다. 갱신 결정 = 잔여가치 크리틱 V_F(s;G)의 발급 대비 비율에 단측 CUSUM.
```

---

## 1. 저장소 지도

```
config/algs/lehca.yaml        LEHCA 하이퍼 (논문 Table 2)
config/algs/vigil.yaml        우리 설정 (셰이핑만, 스케줄러 키 포함)
config/envs/sc2.yaml          SMAC 환경 인자
config/envs/gfootball.yaml    GRF 환경 인자 (우리 추가)

env/__init__.py               env REGISTRY: sc2, gfootball (SMAC은 try/except로 선택적)
env/gfootball.py              GRF pymarl 래퍼 (좌팀 4명, 19행동, 팀 보상)   [우리]
env/semantic/sc2.py           SMAC snapshot/d_t/cache_key/토큰 접지        [LEHCA]
env/semantic/grf.py           GRF   〃  + tick(스텝 시계; snapshot은 무부수효과) [우리]

algorithm/lehca/
  runner.py                   에피소드 루프 + 고정주기 갱신 + 셰이핑 합성
  commander/base.py           guidance 스키마·sanitize·어휘 상수
  commander/llm_commander.py  vLLM 호출·프롬프트 3종·캐시·통계
  commander/rule_commander.py 고정 규칙 커맨더 (ablation용)
  shaping/predicates.py       SMAC predicate 8종 + compute_shaping (Eq.1)
  masking/compiler.py         규칙 → hard/soft 마스크 (Eq.7, 매 스텝 재접지)
  controller.py               LehcaMAC: Q̃ = Q + β·log W (Eq.5-6) + 마스크 통계
  learner.py                  QMIX 학습 + λ 감쇠 (+ shaping_in_learner 경로)
  state.py                    guidance/λ 전역 상태

algorithm/vigil/                (lehca와 같은 배치; 런타임 5개 + analysis/)
  runner.py                   ★ 본체 SchedRunner: LehcaRunner 적응 복사 + vf 스케줄러
  critic.py                   ★ 멀티헤드 잔여가치 V_j(s) + 온라인 학습
  predlib.py                  ★ 환경 디스패치(SMAC/GRF): 헤드·f-벡터·특징·셰이핑
  commander/grf.py            GRF 프롬프트·sanitize (LLMCommander 전송 상속)
  shaping/grf.py              GRF predicate 9종 (+ applicable: 재생 분석용)
  analysis/                   런타임 아님 — 오프라인 도구 격리 (REPO 경로 4단계)
    probe_phase.py, probe_grf.py   봇 궤적 + LLM shadow 프로브 → jsonl
    effect.py                      가이던스 쌍의 효과 공간 거리
    analyze_probe.py               낡음·이벤트 정렬·재생표
    analyze_guidance_quality.py    신선 가이던스의 국면 적합성
    vf_replay.py                   크리틱 LOO 검증 + vf 트리거 재생
```

runner가 서브클래스가 아니라 복사인 이유: lehca `run()`이 셰이핑을 self 경유가 아닌
모듈 전역 호출로 하기 때문(동결 규칙상 seam 불가). 베이스라인 세션에 동작 불변 seam
2개(self 경유 셰이핑 호출, 스텝 훅) 패치를 제안하면 ~80줄 서브클래스로 축소 가능 —
전달 예정. 등록: `algorithm/vigil/__init__.py`가 RUNNER_REGISTRY["vigil"]에 SchedRunner를
올리고, `algorithm/__init__.py`가 vigil을 import한다.

---

## 2. LEHCA 읽는 순서 (베이스라인 이해)

1. **config/algs/lehca.yaml** — 키 이름이 곧 기능 목록. f_update/beta/lambda_*/prompt_style.
2. **algorithm/lehca/runner.py** — 전부 여기서 시작.
   - `run()`: 스텝 루프. `snapshot → _maybe_refresh_commander → build_masks →
     select_actions → env.step → compute_shaping → batch에 (r+λF, shaping_f) 저장`.
   - `_maybe_refresh_commander()`: `t_global − last_refresh ≥ f_update`면 d_t 요약을
     만들어 커맨더 호출. **우리 연구가 바꾸는 지점이 이 함수 하나.**
   - 테스트 에피소드: 갱신 없음·셰이핑 없음(주석 참고) — "훈련 시 주장"의 근거.
3. **commander/base.py** — guidance JSON 스키마, sanitize(어휘 밖 토큰 제거), 어휘 상수.
4. **commander/llm_commander.py** — 프롬프트 3종(default/paper/twostage), 캐시(cache_key),
   reasoning_effort 처리, 실패 시 이전 guidance 유지.
5. **shaping/predicates.py** — f_j 정의 8종. `compute_shaping` = Σ w_j f_j (클립).
6. **masking/compiler.py** + **controller.py** — 규칙→마스크(매 스텝 재접지), Q 틸트.
   vigil은 β=0이라 소비하지 않지만 효과 거리(analysis/effect.py)가 이 컴파일러를 씀.
7. **learner.py** — λ 감쇠 위치(업데이트마다), shaping_in_learner(리플레이 시점 재합성;
   vigil 기본 True — HANDOFF 발견 2).
8. **env/semantic/sc2.py** — snapshot(유닛 dict), summary(d_t 문장), cache_key(거친 키),
   resolve_action_token(토큰→행동 인덱스). LLM이 보고 접지되는 모든 것.

---

## 3. vigil 읽는 순서 (우리 코드)

1. **config/algs/vigil.yaml** — lehca.yaml과의 diff만 보면 됨:
   `use_action_masking False, beta 0, llm_temperature 0, shaping_in_learner True`
   + 스케줄러 블록.

   | 키 | 의미 |
   |---|---|
   | scheduler | fixed(=LEHCA 재현) / vf(잔여가치 CUSUM) |
   | f_update | fixed의 주기이자 vf의 **상한 주기**(이보다 늦게 갱신하지 않음) |
   | sched_k, sched_h | CUSUM 손잡이: 발급 가치의 k 미만이 저하, 누적 h에서 발화 |
   | sched_min_interval | 연속 발화 방지 최소 간격 |
   | sched_gamma | 크리틱 지평 (0.8 ≈ 5스텝; 0.97은 실패 — experiments-log §7) |
   | sched_eps_frac | 발급 가치 < frac × 버퍼 평균 가이던스 가치 → 판단 불가(상한 주기만; 스케일 무관) |
   | sched_warmup_episodes | 크리틱 워밍업 동안 fixed로 동작 |
   | sched_trusted_weight_min | 신뢰 헤드의 가중 비중이 이보다 작으면 판단 불가 |
   | sched_target_early_per_ep | >0이면 h를 피드백 조절해 조기 갱신/ep를 이 값에 맞춤(예산 보정 자동화) |
   | sched_fail_retry | LLM 호출 실패 시 이 스텝 뒤 재시도(주기·CUSUM 상태는 보존) |

2. **predlib.py** — 환경 추상화가 전부 여기.
   - `build_library`: 헤드 목록. GRF=9 고정, SMAC=단순 5 + (kill/damage×적 타입) + (protect×아군 타입).
   - `f_vector`: 한 스텝의 predicate 값 벡터(크리틱 타깃 재료).
   - `FeatureExtractor`: 크리틱 입력 x(s). GRF 14차원(필드 인원으로 정규화), SMAC은
     타입별 (생존비, hp비)+거리+교전.
   - `shaping`: 환경별 compute_shaping 디스패치 (러너의 보상 합성이 이걸 씀).
3. **critic.py** — `add_episode`(할인 접미합 타깃 생성 + 링버퍼), `train`(표준화 MSE),
   `predict`, `trusted`(타깃 분산이 0에 가까운 헤드 = 신호 없음).
4. **runner.py** — LehcaRunner의 **적응 복사**. diff 포인트만 읽으면 됨:

   | 위치 | LEHCA | vigil |
   |---|---|---|
   | 커맨더 생성 | make_commander | env=gfootball이면 commander/grf의 GRFLLMCommander |
   | 갱신 판정 | 경과 ≥ f_update | `_maybe_refresh`: 상한 주기 ∨ (vf: S≥h ∧ 최소간격) |
   | 실패 처리 | 다음 f_update까지 대기 | last_refresh·CUSUM 보존, sched_fail_retry 뒤 재시도 |
   | vf 상태 | — | `_v_ref/_ref_heads`(발급 시 고정), `_S`, `_guidance_heads`(신뢰 필터) |
   | 스텝 부가 | — | x_pre·f_vector 에피소드 누적 → 끝나면 critic.add+train; `iface.tick(snap_post)` 1회 |
   | 셰이핑 | lehca compute_shaping | predlib.shaping (환경 디스패치) |
   | 로깅 | 마스크 통계 | sched_refresh_per_ep/early_per_ep/v_mean, sched_ref_*(사유), critic_loss, sched_h_current |

   결정 의사코드:
   ```
   due = (첫 호출) or (경과 ≥ f_update)                # 상한: LEHCA로의 지배
   if not due and scheduler==vf and 워밍업 지남 and v_ref 유효:
       v = clip( V_F(x_t; 발급시 헤드집합) / v_ref, 0, 1 )
       S = max(0, S + k − v)
       due |= (S ≥ h and 경과 ≥ min_interval)          # 조기 갱신
   if due and 재시도 대기 아님:
       g = 커맨더 호출
       if g is None: retry_after = now + sched_fail_retry; return   # 상태 보존
       guidance = g; S = 0
       v_ref = V_F(x_t; 새 guidance의 신뢰 헤드)                    # 발급 가치 고정
       (신뢰 비중 < wmin, 또는 v_ref < eps_frac×버퍼평균 → v_ref=None = 판단 불가,
        상한 주기로만 동작; 사유는 sched_ref_{warmup,no_heads,low_vref,ok}에 집계)
   ```
5. **GRF 경로**: env/gfootball.py(래퍼) → env/semantic/grf.py(d_t·접지·tick) →
   commander/grf.py(프롬프트·sanitize) → shaping/grf.py(predicate 9종; keep_possession은
   이벤트 스케일 — 턴오버 −1, 유지 +0.1, 패스 비행 0). 각 파일 머리 주석에 좌표계·
   행동 인덱스 등 규약 명시.

---

## 4. 프로브·분석 스크립트 (학습 없는 실험층)

(모두 `algorithm/vigil/analysis/`; repo 루트 기준 실행, 학습 코드와 의존 격리)

| 스크립트 | 만든다 | 읽는다 |
|---|---|---|
| probe_phase.py / probe_grf.py | 봇 궤적 + 10스텝마다 LLM shadow 2회 → results/vigil/*.jsonl | — |
| analyze_probe.py | 노이즈/낡음/이벤트 정렬/**재생표**(고정F·이벤트·적용가능성·gp·keych) | 프로브 jsonl |
| analyze_guidance_quality.py | 신선 가이던스의 국면 적합성 표 | 〃 |
| vf_replay.py | 크리틱 LOO R² + vf 트리거 재생 | 〃 |

재생표의 정의(열 계산·트리거 정의)는 experiments-log.md §3이 단일 출처.
**주의**: keep_possession 의미 변경(9/2) 이전에 뽑은 프로브 jsonl 기반의 (k,h) 캘리브레이션·
critic R²는 구 의미론 — 새 결정에 쓰려면 vf_replay를 재실행할 것.

---

## 5. 실행 명령

```bash
# SMAC (aamas env)
conda activate aamas; export SC2PATH=~/StarCraftII
python main.py --config=vigil --env-config=sc2 with env_args.map_name=2s3z \
    scheduler=vf llm_api_base=http://<node>:8356/v1 use_wandb=True wandb_group=<GN> seed=0
# scheduler=fixed 로 두면 셰이핑-only LEHCA 재현(비교군)

# GRF (aamas 가능 — GPU 학습용; grf env는 CPU 프로브용)
conda activate aamas; export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
python main.py --config=vigil --env-config=gfootball with scheduler=vf ...

# 배치: scripts/run_vigil_sc2.sh <MAP> <SCHED>, scripts/run_vigil_grf.sh <SCHED>

# 프로브/재생 (grf env)
python algorithm/vigil/analysis/probe_grf.py --episodes 10 --api http://<node>:8356/v1
python algorithm/vigil/analysis/analyze_probe.py results/vigil/probe_grf_*.jsonl
python algorithm/vigil/analysis/vf_replay.py results/vigil/probe_grf_*.jsonl
```

## 6. 알려진 미완·주의 (2026-09-02 갱신)
- **gfootball은 aamas에도 설치됨**(9/2 15:58, --freeze-installed로 기존 패키지 무변경,
  numpy 2.2.6 유지, torch cu121·smac 정상, GRF env 생성·스텝 검증). GRF 실행 시
  `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib` 필수(엔진 .so의 GLIBCXX). 설치 로그
  results/vigil/aamas_gfootball_install.log, 사전 백업 aamas_pkgs_backup_20260902.txt.
- grf env(CPU)는 프로브·재생 전용으로 유지.
- **GRF 리플레이 버퍼 선할당**: buffer_size 5000 × (episode_limit+1) ≈ 16GB RAM.
  t_max 500k면 에피소드 ~500개라 실사용 ~10% — GRF 런은 buffer_size 500–1000 권장
  (오버라이드로; yaml 기본은 SMAC 겸용이라 미변경).
- SMAC 스모크는 2s3z로 플러밍만 검증 — 방법 효과 검증은 이질 맵(MMM2)·GRF에서.
- vf의 오프라인 성적: 고정 F는 이김, 이벤트류엔 미달 (experiments-log §7). 즉시 임계
  변형·분모 처리 개선이 열린 항목. 오프라인(vf_replay: γ=0.97, 절대 게이트 EPS_DEN)과
  온라인(runner: γ=0.8, 상대 게이트 eps_frac)의 파라미터가 다름 — 오프라인 (k,h)를
  온라인에 그대로 이식하지 말 것.
- 크리틱 신뢰(trusted)는 타깃 분산 기준의 소극적 판정 — 헤드별 검증 R²로 강화 여지.
- 리뷰(9/2) 보류 항목: controller n_base_actions=6은 lehca 동결 코드(=mask_consistency_w
  0 유지 조건), _sc2_features는 visible-only 고정(dt_observable=True 운용 전제), GRF
  `shot` 접지는 x<0.5에서 forbid도 무력화됨, GRF 커맨더의 전송 로직 ~40줄 중복(베이스라인에
  base 헬퍼 추가 제안 전달 예정), vigil guidance jsonl에 phase/plan_text 미기록, yaml 기본
  포트 8355(스크립트가 덮어씀).
