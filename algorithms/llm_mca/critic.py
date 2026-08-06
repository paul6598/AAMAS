"""LLM critic: a HuggingFace causal LM (default Gemma-7B-it) acting as the centralized
credit-assignment critic. Model-agnostic -- pass any causal LM id.
"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from time import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompts import build_base_prompt, build_critic_prompt, build_batch_prompt
from .parser import parse_credits, parse_batch_credits, valid_credit_arrays


class LLMCritic:
    def __init__(self, model_name="google/gemma-7b-it", device="cuda",
                 dtype=torch.bfloat16, max_new_tokens=800, group_size=0,
                 parallel_requests=1, chunk_steps=0,
                 backend="hf", api_base="http://localhost:8000/v1", max_retries=1,
                 compact_output=False, single_prompt=False, structured_output=False,
                 grounded_filter=False, fallback="env", trace_path=None,
                 lenient_arrays=False, array_length_policy="right", paper_faithful=False):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        # score `group_size` episodes per LLM call (0 = whole batch at once). Small groups keep the
        # prompt short enough for the LLM to reason per-episode (credit-vs-oracle corr degrades as
        # more episodes are packed into one prompt).
        self.group_size = group_size
        self.parallel_requests = max(1, parallel_requests)
        self.chunk_steps = max(0, chunk_steps)
        self.max_retries = max(0, max_retries)
        self.compact_output = compact_output
        self.single_prompt = single_prompt
        self.structured_output = structured_output
        self.grounded_filter = grounded_filter
        self.fallback = fallback
        self.lenient_arrays = lenient_arrays
        self.array_length_policy = array_length_policy
        self.paper_faithful = paper_faithful
        self.trace_path = Path(trace_path) if trace_path else None
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._trace_lock = Lock()
        # Retries happen after the parallel first-pass calls have joined, so this flag is only
        # toggled by the main training thread.  It lets a retry differ from a deterministic
        # first pass; otherwise temperature-0 vLLM simply repeats the same all-zero answer.
        self._retry_active = False
        self.last_stats = {
            "critic_parse_rate": 1.0,
            "critic_retry_calls": 0,
            "critic_fallback_episodes": 0,
        }
        self.backend = backend
        self.api_base = api_base.rstrip("/")
        if backend == "vllm":
            return  # no local model: generate() calls a vLLM OpenAI-compatible server
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, device_map=device,
        )
        self.model.eval()
        self._chat = self.tokenizer.chat_template is not None

    def generate(self, prompt, json_schema=None):
        if self.backend == "vllm":
            import requests
            payload = {"model": self.model_name,
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": self.max_new_tokens,
                       "temperature": 0.2 if self._retry_active else 0.0}
            if json_schema is not None:
                payload["structured_outputs"] = {"json": json_schema}
            r = requests.post(
                f"{self.api_base}/chat/completions",
                json=payload,
                timeout=600,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        return self._generate_hf(prompt)

    @torch.no_grad()
    def _generate_hf(self, prompt):
        if self._chat:
            messages = [{"role": "user", "content": prompt}]
            enc = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
            )
        else:
            enc = self.tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]
        out = self.model.generate(
            **enc, max_new_tokens=self.max_new_tokens, do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        gen = out[0, prompt_len:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)

    def assign_credits(self, traj, env, trajectory_text):
        """Build the base+trajectory prompt, generate, and parse per-agent credit.

        `env` must provide describe(traj), get_example(kind), agent_names and n_agents.
        Returns (credits [n_agents, T], raw_text).
        """
        T = len(traj)
        base = build_base_prompt(env, traj, T, paper_faithful=self.paper_faithful)
        prompt = build_critic_prompt(base, trajectory_text)
        text = self.generate(prompt)
        credits = parse_credits(
            text, env.n_agents, T,
            agent_names=[f"{n.lower()}_credit" for n in env.agent_names],
        )
        return credits, text

    def _score_group(self, trajs, env):
        if self.single_prompt and len(trajs) == 1:
            traj = trajs[0]
            prompt = build_critic_prompt(
                build_base_prompt(env, traj, len(traj), paper_faithful=self.paper_faithful),
                env.serialize(traj, include_progress=not self.paper_faithful),
            )
            names = [f"{n.lower()}_credit" for n in env.agent_names]
            schema = self._json_schema(names, len(traj)) if self.structured_output else None
            if schema is not None:
                prompt += self._json_instruction(names)
            if self._retry_active:
                prompt += self._retry_instruction()
            text = self.generate(prompt, schema)
            if schema is not None:
                try:
                    credit = self._parse_json_group(
                        text, [len(traj)], env.agent_names, single=True,
                    )[0]
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    # Guided decoding can still end mid-JSON when max_tokens is exhausted.
                    # Treat it as a malformed answer so the selective retry/fallback path runs.
                    credit = np.zeros((env.n_agents, len(traj)), dtype=np.float32)
            else:
                credit = parse_credits(
                    text, env.n_agents, len(traj), agent_names=names,
                    length_policy=self.array_length_policy,
                )
            valid = [
                (schema is not None or valid_credit_arrays(
                    text, names, len(traj), exact_length=not self.lenient_arrays,
                ))
                and bool(torch.as_tensor(credit).abs().sum().item() > 1e-8)
            ]
            self._write_trace(prompt, text, trajs, valid)
            return [credit], text, valid

        prompt = build_batch_prompt(
            env, trajs, compact=self.compact_output, paper_faithful=self.paper_faithful,
        )
        batch_names = [
            f"{n.lower()}_credit_{e}"
            for e in range(1, len(trajs) + 1) for n in env.agent_names
        ]
        schema = self._json_schema(
            batch_names,
            {f"{n.lower()}_credit_{e}": len(traj)
             for e, traj in enumerate(trajs, 1) for n in env.agent_names},
        ) if self.structured_output else None
        if schema is not None:
            prompt += self._json_instruction(batch_names)
        if self._retry_active:
            prompt += self._retry_instruction()
        text = self.generate(prompt, schema)
        if schema is not None:
            try:
                credits = self._parse_json_group(
                    text, [len(t) for t in trajs], env.agent_names, single=False,
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                credits = [
                    np.zeros((env.n_agents, len(traj)), dtype=np.float32)
                    for traj in trajs
                ]
        else:
            credits = parse_batch_credits(
                text, env.n_agents, [len(t) for t in trajs], env.agent_names,
                length_policy=self.array_length_policy,
            )
        valid = [
            (schema is not None or valid_credit_arrays(
                text,
                [f"{n.lower()}_credit_{e}" for n in env.agent_names],
                len(traj), exact_length=not self.lenient_arrays,
            )) and bool(torch.as_tensor(credits[e - 1]).abs().sum().item() > 1e-8)
            for e, traj in enumerate(trajs, 1)
        ]
        self._write_trace(prompt, text, trajs, valid)
        return credits, text, valid

    @staticmethod
    def _json_schema(names, lengths):
        if isinstance(lengths, int):
            lengths = {name: lengths for name in names}
        return {
            "type": "object",
            "properties": {
                name: {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": int(lengths[name]),
                    "maxItems": int(lengths[name]),
                }
                for name in names
            },
            "required": names,
            "additionalProperties": False,
        }

    @staticmethod
    def _json_instruction(names):
        return (
            "\n\nReturn a JSON object (not Python and no explanation) with exactly these keys: "
            + ", ".join(names) + ". The response schema enforces every array length."
        )

    @staticmethod
    def _retry_instruction():
        return (
            "\n\nCORRECTION: Your previous answer for this exact episode was rejected because it "
            "was invalid or assigned zero total credit. Recompute the coordinate changes one "
            "transition at a time. You MUST put at least one nonzero number in the response: use "
            "positive credit for concrete progress, or negative credit for a concrete backward, "
            "boundary-hitting, or failed-load action. Preserve the exact JSON schema."
        )

    @staticmethod
    def _parse_json_group(text, episode_lengths, agent_names, single=False, cap=10.0):
        data = json.loads(text)
        out = []
        for e, T in enumerate(episode_lengths, 1):
            rows = []
            for name in agent_names:
                key = f"{name.lower()}_credit" if single else f"{name.lower()}_credit_{e}"
                rows.append(np.asarray(data[key], dtype=np.float32))
            out.append(np.stack(rows))
        peak = max((float(np.abs(c[i]).sum())
                    for c in out for i in range(len(agent_names))), default=0.0)
        if cap is not None and peak > cap:
            for c in out:
                c *= cap / peak
        return out

    def _write_trace(self, prompt, text, trajs, valid):
        if self.trace_path is None:
            return
        record = {
            "time": time(),
            "model": self.model_name,
            "backend": self.backend,
            "compact": self.compact_output,
            "single_prompt": self.single_prompt and len(trajs) == 1,
            "structured_output": self.structured_output,
            "episode_lengths": [len(t) for t in trajs],
            # Keep the sparse environment signal beside the LLM output.  This makes it
            # possible to distinguish useful anticipatory credit from unconditional
            # positive shaping without storing the (much larger) full prompts.
            "global_rewards": [
                [float(r) for r in traj.global_reward] for traj in trajs
            ],
            "episode_returns": [
                float(sum(traj.global_reward)) for traj in trajs
            ],
            "prompt_chars": len(prompt),
            "output_chars": len(text),
            "valid": valid,
            "output": text,
        }
        # vLLM calls can complete concurrently; serialize each JSONL append.
        with self._trace_lock:
            with self.trace_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _retry_one(self, traj, env):
        """Retry one malformed episode so valid episodes in the original batch are preserved."""
        texts = []
        for _ in range(self.max_retries):
            previous = getattr(self, "_retry_active", False)
            self._retry_active = True
            try:
                credits, text, valid = self._score_group([traj], env)
            finally:
                self._retry_active = previous
            texts.append(text)
            if valid[0]:
                return credits[0], "\n--- RETRY ---\n".join(texts), True
        return None, "\n--- RETRY ---\n".join(texts), False

    def _fallback_credits(self, traj, env):
        credits = torch.zeros((env.n_agents, len(traj)), dtype=torch.float32).numpy()
        if self.fallback == "env":
            rewards = torch.as_tensor(traj.global_reward, dtype=torch.float32).numpy()
            credits[:] = rewards[None, :] / env.n_agents
        return credits

    def assign_credits_batch(self, trajs, env):
        """Optionally score a long trajectory as short temporal windows.

        Long LBF episodes make small LMs emit generic or nearly constant arrays instead of
        evaluating each transition.  Windowing keeps the centralized full-state critic but
        reduces the sequence-reasoning burden.  Credits are concatenated and capped again over
        the original episode so windowing cannot inflate the reward scale.
        """
        chunk_steps = getattr(self, "chunk_steps", 0)
        if chunk_steps <= 0 or all(len(traj) <= chunk_steps for traj in trajs):
            credits, text = self._assign_credits_batch_flat(trajs, env)
            self._apply_grounded_filter(credits, trajs)
            return credits, text

        from envs.base import Trajectory

        sequence_fields = (
            "obs", "actions", "global_reward", "next_obs", "done", "state", "next_state",
        )
        chunks = []
        mapping = []
        for episode_idx, traj in enumerate(trajs):
            for start in range(0, len(traj), chunk_steps):
                end = min(start + chunk_steps, len(traj))
                chunks.append(Trajectory(**{
                    name: getattr(traj, name)[start:end] for name in sequence_fields
                }))
                mapping.append((episode_idx, start, end))

        chunk_credits, text = self._assign_credits_batch_flat(chunks, env)
        combined = [
            np.zeros((env.n_agents, len(traj)), dtype=np.float32) for traj in trajs
        ]
        for credit, (episode_idx, start, end) in zip(chunk_credits, mapping):
            combined[episode_idx][:, start:end] = credit

        for credit in combined:
            peak = max(
                (float(np.abs(credit[i]).sum()) for i in range(env.n_agents)),
                default=0.0,
            )
            if peak > 10.0:
                credit *= 10.0 / peak
        self._apply_grounded_filter(combined, trajs)
        return combined, text

    def _apply_grounded_filter(self, credits, trajs):
        """Remove only credits that contradict an unambiguous LBF transition.

        This is a semantic parser guard, not a reward generator: it never adds or changes the
        sign of an LLM value.  Mixed progress (closer to one apple, farther from another) is left
        to the LLM's structural target assignment.
        """
        if not getattr(self, "grounded_filter", False):
            return
        removed = nonzero = 0
        for credit, traj in zip(credits, trajs):
            for k in range(len(traj)):
                state = traj.state[k]
                next_state = traj.next_state[k]
                if not isinstance(state, dict) or "agents" not in state or "foods" not in state:
                    continue
                reward = float(traj.global_reward[k])
                for i in range(credit.shape[0]):
                    value = float(credit[i, k])
                    if abs(value) <= 1e-8:
                        continue
                    nonzero += 1
                    action = int(traj.actions[k][i])
                    pos = state["agents"][i][:2]
                    next_pos = next_state["agents"][i][:2]
                    progress = []
                    for r, c, _ in state["foods"]:
                        if r < 0:
                            continue
                        before = max(0, abs(pos[0] - r) + abs(pos[1] - c) - 1)
                        after = max(0, abs(next_pos[0] - r) + abs(next_pos[1] - c) - 1)
                        progress.append(before - after)

                    contradiction = value > 0.0 and (
                        action == 0
                        or (action == 5 and reward <= 0.0)
                        or (
                            action in (1, 2, 3, 4)
                            and progress
                            and max(progress) <= 0
                        )
                    )
                    contradiction = contradiction or (
                        value < 0.0
                        and action in (1, 2, 3, 4)
                        and progress
                        and min(progress) >= 0
                        and max(progress) > 0
                    )
                    if contradiction:
                        credit[i, k] = 0.0
                        removed += 1
        self.last_stats["critic_grounded_removed_frac"] = (
            float(removed / nonzero) if nonzero else 0.0
        )

    def _assign_credits_batch_flat(self, trajs, env):
        """Score the batch of episodes and return (list of credit matrices, raw_text). With
        group_size>0, split the batch into groups of that many episodes, one LLM call each, so the
        prompt stays short enough for the LLM to reason per-episode."""
        g = self.group_size
        group_size = len(trajs) if not g or g >= len(trajs) else g
        all_credits, all_valid, texts = [], [], []
        groups = [trajs[s:s + group_size] for s in range(0, len(trajs), group_size)]
        parallel = getattr(self, "parallel_requests", 1)
        if getattr(self, "backend", "hf") == "vllm" and parallel > 1 and len(groups) > 1:
            with ThreadPoolExecutor(max_workers=min(parallel, len(groups))) as pool:
                scored = list(pool.map(lambda group: self._score_group(group, env), groups))
        else:
            scored = [self._score_group(group, env) for group in groups]
        for credits, text, valid in scored:
            all_credits.extend(credits)
            all_valid.extend(valid)
            texts.append(text)

        retry_calls = 0
        fallback_episodes = 0
        for i, valid in enumerate(all_valid):
            if valid or self.max_retries == 0:
                continue
            replacement, retry_text, retry_valid = self._retry_one(trajs[i], env)
            retry_calls += self.max_retries if not retry_valid else retry_text.count("--- RETRY ---") + 1
            texts.append(retry_text)
            if retry_valid:
                all_credits[i] = replacement
                all_valid[i] = True
            else:
                all_credits[i] = self._fallback_credits(trajs[i], env)
                fallback_episodes += 1

        self.last_stats = {
            "critic_parse_rate": float(sum(all_valid) / len(all_valid)) if all_valid else 1.0,
            "critic_retry_calls": retry_calls,
            "critic_fallback_episodes": fallback_episodes,
        }
        return all_credits, "\n---\n".join(texts)
