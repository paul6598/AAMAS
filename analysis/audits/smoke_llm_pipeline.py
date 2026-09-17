"""Capture fresh SMAC observations and audit the real Commander pipeline.

No training, checkpoints, or W&B writes. A bounded scripted rollout supplies
observations; both prompt versions receive the same frozen snapshots.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from env import sc2_env_fn
from env.semantic.sc2 import SC2SemanticInterface
from algorithm.lehca.commander.llm_commander import LLMCommander, extract_json
from algorithm.lehca.masking.compiler import build_masks


class RecordingSession:
    def __init__(self, session):
        self.session = session
        self.calls = []

    def post(self, url, **kwargs):
        record = {"url": url, "request": copy.deepcopy(kwargs["json"])}
        self.calls.append(record)
        try:
            response = self.session.post(url, **kwargs)
            record.update(status=response.status_code, response_text=response.text)
            try:
                record["response"] = response.json()
            except ValueError:
                pass
            return response
        except Exception as exc:
            record["error"] = str(exc)
            raise


def block(obj, lang="json"):
    value = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, indent=2)
    return "\n```" + lang + "\n" + value + "\n```\n\n"


def render(data, destination):
    lines = ["# 실제 SMAC → LLM → JSON → 마스크 스모크 테스트\n\n",
             "생성 시각(UTC): " + data["created_utc"] + "\n\n",
             "별도의 실제 2s3z 에피소드에서 수집한 관측입니다. 과거 로그의 재현이나 학습 정책 평가가 아닙니다. "
             "아군은 공격 가능한 적이 있으면 첫 대상을 공격하고, 없으면 동쪽으로 이동합니다. "
             "한 에피소드 최대 120스텝에서 상황별 첫 관측과 마지막 비종료 관측을 선택합니다.\n\n",
             "같은 관측을 paper와 paper_v2에 각 1회 입력합니다(실패 시 Commander의 기존 재시도 적용). "
             "모델 openai/gpt-oss-20b, temperature=0.2, reasoning_effort=low, max_tokens=3072, "
             "응답 캐시 off, f_update=200. 반복 통계나 성능 비교가 아닙니다.\n\n",
             "전체 원자료: [JSON](" + destination.with_suffix('.json').name + "). "
             "여기에 per-agent get_obs, availability, 요청·응답 전체, 파싱 및 마스크를 보존했습니다.\n\n",
             "주의: snapshot에는 training-time 전체 상태도 포함됩니다. 이것을 LLM에 직접 보내지 않습니다. "
             "실제 전송 내용은 아래 system/user message이며, 적 정보는 기존 summary의 visibility 필터를 통과합니다. "
             "JSON 파싱과 sanitizer는 조건 검증기가 아니므로 정제 성공이 의미적 타당성을 보장하지 않습니다.\n\n"]
    for case in data["cases"]:
        lines += ["## " + case["label"] + " — env step " + str(case["step"]) + "\n\n",
                  "### 1. 실제 환경에서 추출한 숫자 snapshot\n", block(case["snapshot"]),
                  "원래 per-agent 관측 벡터와 availability는 첨부 JSON의 obs/avail_actions에 있습니다.\n\n",
                  "### 2. 기존 summary 함수가 만든 자연어\n", block(case["summary"], "text")]
        for result in case["outputs"]:
            lines += ["### 3–6. " + result["style"] + "\n\n"]
            for index, call in enumerate(result["calls"], 1):
                lines += ["#### 요청 " + str(index) + ": 실제 messages\n", block(call["request"]["messages"])]
                response = call.get("response", {})
                choice = (response.get("choices") or [{}])[0]
                content = choice.get("message", {}).get("content")
                lines += ["#### 실제 응답 content 원문\n", block(content, "text"),
                          "응답 메타데이터:\n", block({"status": call.get("status"),
                          "finish_reason": choice.get("finish_reason"), "usage": response.get("usage"),
                          "error": call.get("error")}),
                          "#### JSON 추출 결과（sanitizer 이전）\n", block(extract_json(content))]
            lines += ["#### sanitizer 이후 실제 반환 지침\n", block(result["guidance"]),
                      "#### 컴파일된 마스크\n", block(result.get("masks")),
                      "행: ally index, 열: [noop, stop, north, south, east, west, enemy0…enemy4]. "
                      "soft>1은 선호이며 확률이 아닙니다. availability가 최종 실행 가능성을 제한합니다. "
                      "RSVP soft-only 실행에서는 아래 hard를 모두 1로 덮어씁니다.\n\n"]
    destination.write_text("".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError("Use a fresh output path; previous artifacts are preserved")
    opts = SimpleNamespace(dt_observable=True, llm_api_base=args.api,
                           llm_model="openai/gpt-oss-20b", llm_temperature=0.2,
                           llm_max_tokens=3072, llm_timeout=90, llm_cache=False,
                           llm_reasoning_effort="low", deduplicate_subgoals=True,
                           f_update=200, prompt_style="paper")
    env = sc2_env_fn(map_name="2s3z", seed=42, reward_sparse=False,
                     obs_all_health=True, obs_own_health=True)
    data = {"created_utc": datetime.now(timezone.utc).isoformat(), "cases": []}
    try:
        env.reset()
        iface = SC2SemanticInterface(env, opts)
        context = iface.prompt_context()
        selected = {}
        last = None
        for step in range(120):
            snap = iface.snapshot()
            avail = env.get_avail_actions()
            case = {"step": step, "snapshot": snap,
                    "obs": [x.tolist() for x in env.get_obs()],
                    "avail_actions": [list(map(int, x)) for x in avail],
                    "summary": iface.summary(snap), "cache_key": iface.cache_key(snap),
                    "outputs": []}
            last = case
            visible = any(e["visible"] for e in snap["enemies"])
            alive = sum(u["alive"] for u in snap["allies"])
            if not visible:
                selected.setdefault("적 미관측", case)
            if visible and alive >= 3:
                selected.setdefault("다수 아군 교전", case)
            if alive <= 2:
                selected.setdefault("소수 아군 생존", case)
            actions = []
            for allowed in avail:
                attacks = [a for a in range(6, len(allowed)) if allowed[a]]
                actions.append(attacks[0] if attacks else 4 if allowed[4]
                               else 1 if allowed[1] else 0)
            _, terminated, _ = env.step(actions)
            if terminated:
                break
        selected.setdefault("마지막 비종료 관측", last)
        seen = set()
        for label, case in selected.items():
            if case["step"] in seen:
                continue
            seen.add(case["step"])
            case["label"] = label
            data["cases"].append(case)
        # No SC2 processes remain while inference requests run.
    finally:
        env.close()
    # prompt_context must not consult a closed/mutated environment.
    iface.prompt_context = lambda: context
    for case in data["cases"]:
        for style in ("paper", "paper_v2"):
            opts.prompt_style = style
            commander = LLMCommander(opts, iface)
            recording = RecordingSession(commander._session)
            commander._session = recording
            guidance = commander(case["summary"], case["cache_key"], iface)
            result = {"style": style, "guidance": guidance, "calls": recording.calls}
            if guidance is not None:
                hard, soft = build_masks(guidance["action_rules"], case["snapshot"], iface,
                                         len(case["snapshot"]["allies"]), case["snapshot"]["n_actions"])
                result["masks"] = {"hard": hard.tolist(), "soft": soft.tolist(),
                                   "avail_actions": case["avail_actions"]}
            case["outputs"].append(result)
            args.output.with_suffix('.json').write_text(json.dumps(data, ensure_ascii=False, indent=2))
            render(data, args.output)
            print(case["label"], style, "parsed=", guidance is not None, flush=True)
            recording.session.close()


if __name__ == "__main__":
    main()
