# LEHCA 재현 노트

논문: Bai et al., *A hierarchical multi-agent reinforcement learning framework
with high-level guidance from large language models*, Scientific Reports 2026.
(`paper/LEHCA.pdf`) — 공식 코드 미공개("available upon publication")라 논문
기술(Algorithm 1, Eq. 1–7, Table 2)만으로 pymarl 위에 직접 구현.

## 논문 → 코드 매핑

| 논문 요소 | 구현 위치 |
|---|---|
| 시맨틱 요약 d_t (F_update 스케줄) | `env/semantic/sc2.py: summary()`, `algorithm/lehca/runner.py` |
| LLM Commander (gpt-oss:20b, T=0.2, 3072tok) | `algorithm/lehca/commander/llm_commander.py` |
| 서브골 → F_t = Σ w_j f_j (Eq.1) | `algorithm/lehca/shaping/predicates.py` (8종 predicate) |
| R_t = R_env + λ_t F_t, λ 감쇠 (Eq.2, Alg.1 l.22) | runner(수집 시 합성) + `algorithm/lehca/learner.py`(업데이트마다 감쇠) |
| 하드/소프트 마스크, Q̃=Q+β·log W_soft (Eq.5-6) | `algorithm/lehca/masking/compiler.py`, `algorithm/lehca/controller.py` |
| 마스크 매 스텝 재해석 (Eq.7) | 심볼 룰을 현재 상태에 매 스텝 그라운딩 |
| Commander 실패 시 마지막 유효 가이던스 유지 | commander가 None 반환 시 이전 가이던스 유지 |
| Rule-Commander + QMIX (Table 6 통제군) | `commander=rule` |
| Ablation (Table 5) | `use_reward_shaping` / `use_action_masking` 플래그 |
| QMIX 공통 설정 (Table 2) | lr 1e-3, Adam, batch 128, buffer 5000, γ 0.99 |

## 논문에 없는 값 (supplementary 페이월) — 우리 기본값

- `f_update=200` (Commander 갱신 주기)
- `beta=0.5` (소프트 마스크 계수)
- `lambda_start=0.5`, `lambda_min=0.05`, `lambda_decay=0.9995` (per update)
- `shaping_clip=3.0`, ε-anneal 50k (pymarl 기본), target update 200 에피소드
- predicate 스케일: 적 전멸 damage ≈ 10, kill 1.0/기, ally 사망 −1.0 등
  (SMAC 스케일 보상 max 20과 비슷한 자릿수로 설계)

## 의도적 편차 / 실용적 결정

1. **응답 캐싱**: 유사 상태(타입별 생존수+체력 버킷+phase)에서 Commander 응답
   재사용 — 논문도 "reused between refreshes" 언급. `llm_cache=False`로 끌 수 있음.
2. **reasoning_effort=low**: gpt-oss 추론 토큰을 제한해 호출당 ~5초 유지.
   미지원 모델이면 자동 제거.
3. **Medivac**: 공격 슬롯이 힐이므로 attack_* 마스크 토큰 미적용.
4. **학습 return 로깅**: `return_mean`은 환경 보상만(비교 가능성),
   셰이핑 크기는 `shaped_return_mean`으로 별도 기록.
5. **테스트 에피소드**: 최신 가이던스로 masked greedy 실행하되 LLM 호출과
   보상 셰이핑은 없음.

## 인프라

- conda `aamas` (py3.10, torch 2.5.1+cu121, smac 1.0, sacred, wandb; setuptools<81)
- SC2 4.10 + SMAC 맵: `/gpfs/home1/paul6598/StarCraftII`
- LLM 서빙: conda `vllm` (0.25.1), 세션 37 (n064, RTX 6000 Ada 48GB), 포트 8355.
  컴퓨트 노드에서 `CUDA_HOME=/usr/local/cuda-12.7`, `VLLM_USE_FLASHINFER_SAMPLER=0` 필요.
- gpt-oss-20b 호출 실측: ~5.3초/호출 (reasoning low, 3072 max_tokens)
- wandb: project `AAMAS-LEHCA`, entity `joonhuk6598-university-of-seoul`

## 재현 상태 (2026-08-25)

- [x] 전체 파이프라인 검증 (CPU smoke + LLM 실호출)
- [ ] 2s3z: LEHCA(gpt-oss) vs QMIX, 각 5시드 — 진행 중 (tmux 2:0 / 34)
- [ ] 나머지 7개 맵 (노드 38/40 할당 대기)
- [ ] AUC_early 비교표 작성 (`analysis/auc_early.py`)
