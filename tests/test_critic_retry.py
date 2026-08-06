"""Regression test for selective retry of malformed batch credit outputs (no LLM required)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from algorithms.llm_mca.critic import LLMCritic
from envs.base import Trajectory


class FakeCritic(LLMCritic):
    def __init__(self):
        self.group_size = 0
        self.max_retries = 1
        self.fallback = "env"
        self.last_stats = {}
        self.calls = []

    def _score_group(self, trajs, env):
        self.calls.append(len(trajs))
        if len(trajs) == 2:
            return [
                np.full((2, 3), 1.0, dtype=np.float32),
                np.zeros((2, 3), dtype=np.float32),
            ], "batch", [True, False]
        return [np.full((2, 3), 2.0, dtype=np.float32)], "retry", [True]


def test_selective_retry():
    critic = FakeCritic()
    credits, text = critic.assign_credits_batch([[0, 1, 2], [0, 1, 2]], object())

    assert critic.calls == [2, 1]
    assert np.all(credits[0] == 1.0)
    assert np.all(credits[1] == 2.0)
    assert critic.last_stats == {
        "critic_parse_rate": 1.0,
        "critic_retry_calls": 1,
        "critic_fallback_episodes": 0,
    }
    assert "batch" in text and "retry" in text


def test_env_reward_fallback():
    critic = FakeCritic()
    critic._score_group = lambda trajs, env: (
        [np.full((2, len(t)), 99.0, dtype=np.float32) for t in trajs],
        "malformed",
        [False] * len(trajs),
    )
    traj = Trajectory(
        actions=[None, None],
        global_reward=[0.0, 2.0],
    )
    env = type("Env", (), {"n_agents": 2})()

    credits, _ = critic.assign_credits_batch([traj], env)

    assert np.allclose(credits[0], [[0.0, 1.0], [0.0, 1.0]])
    assert critic.last_stats["critic_fallback_episodes"] == 1


def test_chunk_reassembly_and_episode_cap():
    class FakeChunkCritic(LLMCritic):
        def __init__(self):
            self.chunk_steps = 2
            self.seen_lengths = None

        def _assign_credits_batch_flat(self, trajs, env):
            self.seen_lengths = [len(t) for t in trajs]
            return [
                np.full((env.n_agents, len(t)), 5.0, dtype=np.float32) for t in trajs
            ], "chunked"

    critic = FakeChunkCritic()
    trajs = [
        Trajectory(actions=[None] * 5),
        Trajectory(actions=[None] * 3),
    ]
    env = type("Env", (), {"n_agents": 2})()

    credits, text = critic.assign_credits_batch(trajs, env)

    assert critic.seen_lengths == [2, 2, 1, 2, 1]
    assert [x.shape for x in credits] == [(2, 5), (2, 3)]
    # Each original episode, not each window, is capped to |credit_i| sum <= 10.
    assert np.allclose(credits[0], 2.0)
    assert np.allclose(credits[1], 10.0 / 3.0)
    assert text == "chunked"


def test_grounded_filter_only_removes_contradictions():
    critic = object.__new__(LLMCritic)
    critic.grounded_filter = True
    critic.last_stats = {}
    traj = Trajectory(
        actions=[
            np.array([4, 0]),  # Alice gets closer; Bob does nothing.
            np.array([3, 5]),  # Alice gets farther; Bob attempts a failed load.
        ],
        global_reward=[0.0, 0.0],
        state=[
            {"agents": [(0, 0, 1), (3, 3, 1)], "foods": [(0, 2, 2)]},
            {"agents": [(0, 1, 1), (3, 3, 1)], "foods": [(0, 2, 2)]},
        ],
        next_state=[
            {"agents": [(0, 1, 1), (3, 3, 1)], "foods": [(0, 2, 2)]},
            {"agents": [(0, 0, 1), (3, 3, 1)], "foods": [(0, 2, 2)]},
        ],
    )
    credits = [np.array([[-1.0, -1.0], [1.0, -1.0]], dtype=np.float32)]

    critic._apply_grounded_filter(credits, [traj])

    # Contradictory closer-negative and idle-positive values are removed. Consistent
    # farther-negative and failed-load-negative values are preserved.
    assert np.allclose(credits[0], [[0.0, -1.0], [0.0, -1.0]])
    assert critic.last_stats["critic_grounded_removed_frac"] == 0.5


if __name__ == "__main__":
    test_selective_retry()
    test_env_reward_fallback()
    test_chunk_reassembly_and_episode_cap()
    test_grounded_filter_only_removes_contradictions()
    print("CRITIC_RETRY_TEST_PASSED")
