# GRF(Google Research Football) 이식 설계 — d_t · 접지 · 셰이핑 · 프롬프트

작성 2026-08-31. 목적: LEHCA 파이프라인(QMIX + LLM Commander → 서브골 셰이핑 +
규칙 마스크)을 GRF로 옮겨 **에피소드 내 급변(공수 전환)에 대한 가이던스 갱신
스케줄링**을 검증한다. SMAC은 접지 노이즈 진단(2s3z, `results/vigil/`)만 남긴다.

관련: docs/research/draft-paper.md §3–4, theory-cusum.md, related-work §4c.

---

## 0. 왜 GRF인가 — 문제 조건과의 대응

| 조건 | GRF |
|---|---|
| 내생적·빈번·라벨 가능한 급변 ν | `ball_owned_team` 전환(0→−1→1 등), `game_mode` 변화(세트피스). 관측에서 자동 라벨 |
| 급변 후 올바른 팀 규칙이 *반전* | 소유 시 "전진·전방 패스·슛" ↔ 상실 시 "복귀·압박·전진 금지". 낡은 규칙은 능동적으로 해로움(실점) |
| 급변이 컴팩트 d_t에 보임 | 소유 팀·공 구역·양 팀의 구역별 인원·스코어·잔여 시간 |
| 이산 행동·협력·CTDE | 19 이산 행동, 좌팀 필드 플레이어 N명 제어, QMIX/MAPPO 표준 |
| 팀 수준 기호 규칙 어휘 | 이동 방향(진영 상대)·패스 종류·슛·스프린트·드리블이 그대로 토큰 |
| 에피소드 ≫ 갱신 단위 | 5_vs_5 기본 3000스텝(단축본 500–1000 사용) |

**주의**: academy_* 시나리오는 대개 `end_episode_on_possession_change=True`라
에피소드 내 ν가 없다. 우리 목적에는 **5_vs_5**(기본값: 스코어·아웃·소유권 전환에
에피소드를 끊지 않음, GK는 봇, 필드 4명 제어) 또는 그 단축본을 쓴다.

---

## 1. 시나리오

- **주**: `5_vs_5` 복제본 `vigil_5v5_short` — `game_duration` 1000(≈ 5분 경기의
  1/3), 나머지 플래그 원본과 동일, `deterministic=False`. 좌팀 필드 4명 제어
  (`number_of_left_players_agent_controls=4`), 우팀 내장 AI(난이도 0.05 원본 →
  필요 시 0.6/0.95 스윕). 좌팀은 항상 +x 방향 공격이 되도록 래퍼에서 진영 고정.
- **보조(학습 난이도 확인용)**: `academy_counterattack_hard`, `academy_3_vs_1_with_keeper`
  — ν는 없지만 QMIX 학습 곡선의 정상 동작 확인과 셰이핑 predicate 디버깅용.
- 보상: 기본 `scoring` + `checkpoints`(전진 시 소액) — LEHCA의 "환경은 이미 조밀
  보상"과 같은 상황을 만들어 서브골이 *조정 수준*에 머물게 한다.

---

## 2. pymarl 래퍼 (`env/gfootball.py`, `REGISTRY["gfootball"]`)

- `get_obs()`: 에이전트별 `simple115_v2` + 자기 인덱스 one-hot(+ 역할 one-hot 10).
- `get_state()`: raw에서 공 (x,y,z,dx,dy), 소유(팀 one-hot 3 + 소유자 역할 one-hot 10),
  양 팀 위치·방향(5×4 각각), 스코어 차, 잔여 스텝 비율, game_mode one-hot 7.
- `get_avail_actions()`: 19개 전부 1. GRF는 모든 행동이 항상 legal이므로 **hard
  forbid는 곧바로 행동을 바꾼다** — SMAC보다 마스크 채널이 강하다. β는 0.1 유지.
- 에피소드 종료: `game_duration` 소진(단축본). `battle_won` 대응 지표 = 득실차 > 0.
- 시드/결정성: `deterministic=False`, 시드는 래퍼에서 전달.

---

## 3. snapshot (구조화 상태; SMAC의 allies/enemies에 대응)

```
{
 "possession": "ours" | "theirs" | "loose",
 "carrier": {"role": "CF", "idx": 2, "x":.., "y":.., "pressure": 0.08}   # ours일 때
 "ball": {"x","y","z","dx","dy","zone": "def|mid|att" × "L|C|R"},
 "game_mode": "normal|kickoff|goalkick|freekick|corner|throwin|penalty",
 "score_diff": +1, "time_frac": 0.42,
 "allies":  [{"idx","role","x","y","dx","dy","tired","has_ball","zone"} ×5],
 "enemies": [{"idx","role","x","y","dx","dy","zone"} ×5],
 "shape": {"ours_behind_ball": 3, "theirs_behind_ball": 4,
           "ours_in_att_third": 1, "theirs_in_our_third": 2,
           "nearest_enemy_to_ball": 0.05, "nearest_ally_to_ball": 0.02},
 "n_actions": 19
}
```
zone: x를 [-1,-1/3),[-1/3,1/3),[1/3,1]로 def/mid/att, y를 3등분 L/C/R(좌팀 기준).

---

## 4. summary d_t (LLM 입력; 결정적 함수 φ(snapshot))

예문(소유 상실 직후):
```
Scenario: 5v5 football, we are the LEFT team attacking to the right. 1000-step match, 58% remaining. Score 1-1.
Possession: THEIRS (their CF, mid-centre zone, under light pressure: nearest defender 0.09 away).
Ball: mid-centre zone, moving toward our goal.
Our shape: 2 of 4 field players behind the ball; 1 in their attacking third (CF), 1 in midfield.
Their shape: 3 players ahead of the ball in our half; 1 behind.
Set piece: none (normal play).
Recent: possession changed 4 steps ago (we lost it in their half).
```
- 방향·좌표는 항상 **좌팀 기준**("toward their goal"/"toward our goal").
- "Recent" 줄은 최근 ν 이후 경과 스텝을 담아 LLM이 전환을 명시적으로 인지하게 한다.
  (CUSUM 입력 φ에는 넣지 않는다 — 트리거가 자기 자신을 참조하면 안 됨.)
- 학습 컨텍스트(rolling 득실, 스텝)는 SMAC과 동일하게 훈련 시에만 부가.

---

## 5. cache_key (거친 이산화)

`possession | ball zone(3×3) | game_mode | sign(score_diff) | time tercile |
ours_behind_ball bucket(0-1,2,3-4)` — SMAC과 같은 정신(재호출 비용 상각)이되,
**소유권과 공 구역이 키에 들어가므로 ν에서 키가 반드시 바뀐다**. E2의 A1(F=ep)이
캐시 때문에 정적 계획으로 퇴화하는 SMAC 결함은 재현되지 않는다. 그래도 방법
비교에서는 캐시 off 또는 "캐시 적중은 호출로 세지 않음"을 명시.

---

## 6. 규칙 어휘와 접지 (마스크)

**applies_to 선택자**: `all` | `role:<GK|CB|LB|RB|DM|CM|LM|RM|AM|CF>` |
`carrier`(공 소유자) | `off_ball`(비소유 아군) | `nearest_to_ball` | `deepest`(최후방).

**토큰 → 행동 인덱스** (`resolve_action_token(token, agent, snap)`):

| 토큰 | 접지 | 비고 |
|---|---|---|
| `move_forward` / `move_back` | {5,4,6} / {1,2,8} (좌팀 기준 +x/−x) | 3개 행동 집합 |
| `move_up` / `move_down` | {3,2,4} / {7,6,8} | |
| `move_toward_ball` | 공 방향 벡터를 8방향으로 양자화 → 1개 | 상태 의존(매 스텝 재접지) |
| `move_toward_goal` | 상대 골(1,0) 방향 양자화 → 1개 | |
| `move_toward_own_goal` | (−1,0) 방향 양자화 → 1개 | 수비 복귀 |
| `short_pass` / `long_pass` / `high_pass` | 11 / 9 / 10 | carrier에게만 의미, 그 외 [] |
| `shot` | 12 | carrier & 공 x>0.5일 때만 접지, 아니면 [] |
| `sprint` / `release_sprint` | 13 / 15 | |
| `dribble` / `release_dribble` | 17 / 18 | carrier에게만 |
| `slide` | 16 | off_ball에게만 (파울 위험 → forbid 후보) |
| `idle` | 0 | |

- 자기근거화: `shot`/패스는 carrier 외에는 빈 집합 → 낡아도 무해. **해로운 낡음**은
  `move_forward`/`sprint`/`move_back` 같은 방향·강도 토큰에서 나온다(소유 상실 후
  전진 선호 잔존). 이것이 C2를 검증 가능하게 만드는 어휘 설계 포인트.
- forbid 상한: 이동 8방향을 전부 금지할 수 없게 sanitize(방향 토큰은 한 축만 forbid).
- 컴파일러(`algorithm/lehca/masking/compiler.py`)는 그대로 재사용, 접지만 iface에서.

---

## 7. 셰이핑 predicate (`compute_shaping(subgoals, snap_pre, snap_post, actions)`)

| predicate | 정의(스텝 단위) | 국면 |
|---|---|---|
| `ball_progress` | 소유 중 공 x 증가량(정규화, 클립) | 공격 |
| `keep_possession` | pre·post 모두 ours: +1 / ours→theirs: −1 | 공격 |
| `shot_in_box` | carrier가 x>0.7에서 shot | 공격 |
| `pass_forward` | 패스 행동 후 공 x 증가 & 소유 유지 | 공격 |
| `regain_possession` | theirs/loose → ours | 수비 |
| `defensive_shape` | 수비 시 공 뒤 아군 수 증가(또는 ≥3 유지) | 수비 |
| `press_carrier` | 수비 시 nearest_ally_to_ball 감소 | 수비 |
| `compactness` | 아군 위치 분산 감소(수비) / 폭 증가(공격) | 양쪽 |
| `no_slide_foul` | slide 사용 시 −1 | 수비 |

typed 예: `protect_type`류 대응은 `role:` 선택자로 대체(`defensive_shape` with
role:CB). 가중치 ∈[0,1], λ 감쇠·클립은 LEHCA 설정 그대로.

**요점**: 공격/수비 predicate가 서로 반대라서 낡은 서브골(공격형)은 수비 국면에서
잘못된 보상을 준다 — 셰이핑 채널에서도 낡음이 정의된다(SMAC에선 아니었음).

---

## 8. 프롬프트

- `prompt_context`: "우리는 좌팀, 오른쪽으로 공격. 필드 플레이어 4명(역할 나열)을
  당신이 안내하며 GK는 자동. 행동: 8방향 이동, 3종 패스, 슛, 스프린트, 드리블,
  슬라이드. 보상은 득점과 전진에 이미 주어짐."
- 시스템 프롬프트는 LEHCA `paper` 스타일(3단계 추론 → JSON)을 기본으로 하되:
  - "국면(공격/수비/전환/세트피스)을 먼저 판정하고 그 국면에 맞는 팀 규칙을 써라."
  - forbid는 위험·비합리 행동에만(예: 수비 시 `slide`, 소유 시 `move_back` 과다).
  - prefer_weight 1.5–2.5, 규칙 ≤4, 서브골 ≤4.
- **temperature 0** 고정(2s3z 진단: 0.2에서 접지 노이즈 hard 불일치 0.31). 논문
  설정과의 차이는 명시하고, 0.2 대비 노이즈를 부록에 제시.

---

## 9. ν(변화점)의 정의와 라벨

- 소유 상태 기계: ours / loose / theirs. **ν = ours↔theirs 전이**(loose 경유 포함,
  loose 체류 ≤ 10스텝이면 하나의 전이로 병합). 라벨은 관측에서 자동.
- 부차 ν: `game_mode` 0→비0(세트피스 진입)과 복귀, 득점(리셋).
- 검출 지연 = ν 이후 첫 갱신까지 스텝. 오경보 = ν 없는 구간의 갱신.
- φ(CUSUM 입력) = [possession one-hot, ball x, ball y, ours_behind_ball/4,
  theirs_in_our_third/4, nearest_enemy_to_ball, game_mode≠0]. "Recent" 줄은 제외.

---

## 10. 학습 전 검증 프로토콜 (E1-lite GRF)

1. 좌팀 4명을 **내장 AI로 두고**(제어 0명 또는 `action_set` idle에 봇 위임) 봇 대
   봇 경기 20판 기록 → snapshot/φ/이벤트/ν 자동 라벨.
2. 5스텝마다 shadow LLM 2회(temp 0; 노이즈 바닥선은 0.2로 별도 10판).
3. 지표: D_noise, D_stale(Δ), **ν 정렬 점프**(ν 전후 창의 규칙 효과 거리), 낡은
   규칙이 "잘못된 방향"(수비 국면에 move_forward/sprint 선호, 공격 국면에
   move_back 선호)을 내는 비율 = C2 프록시.
4. 재생표: F=ep / 고정 F∈{50,100,200} / 이벤트(ν) / φ-CUSUM / 단발 임계값 →
   (호출/ep, 낡음, 지연, 오경보).
5. 판정: D_stale(ν 직후) ≥ 2·D_noise이고 잘못된 방향 비율이 ν 후 유의하게 크면
   C1·C2 프록시 통과 → 래퍼 완성·QMIX 베이스라인(마스크 off) 학습으로 진행.

---

### 10a–c. 실행 기록과 결과 → docs/research/experiments-log.md §2

## 11. 리스크

- 봇 궤적 ≠ 학습 정책 궤적(특히 초기 QMIX는 공을 거의 못 잡음 → 소유 국면 희소).
  → 초기 학습 구간에서는 ν가 적어 스케줄링 이득이 늦게 나타날 수 있음; 5v5
  단축본 + checkpoints 보상으로 완화.
- QMIX의 GRF 5v5 성능이 낮다는 보고 다수 → 비교는 "같은 QMIX 위에서 고정 F vs
  적응 갱신"이므로 절대 성능보다 상대 차이가 중요하지만, 학습이 아예 안 되면
  무의미 → academy로 파이프라인 검증 후 5v5.
- 19 행동 항상 legal → hard forbid가 강함. β와 forbid 절제 프롬프트로 조절, 마스크
  off 베이스라인 필수.
- gfootball 빌드(SDL/boost/mesa) — conda env `grf`에서 레시피 확정 후 aamas에 적용.
