"""Smoke tests for env wrapper, DDQN, and parser -- no LLM required."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from envs import make_env
from algorithms.common.ddqn import DDQNAgent
from algorithms.llm_mca.parser import parse_credits, valid_credit_arrays
from algorithms.llm_mca.prompts import build_batch_task_prompt


def test_env_and_ddqn():
    env = make_env("Foraging-8x8-2p-2f-coop-v3", seed=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agents = [DDQNAgent(env.obs_dim, env.n_actions, device, batch_size=8) for _ in range(env.n_agents)]

    batch = env.collect_batch(agents, epsilon=1.0, n_episodes=2)
    assert len(batch) == 2
    traj = batch[0]
    assert len(traj) > 0
    assert traj.done[-1], "the final transition must stop bootstrap at the episode horizon"
    print(f"[env] episode length={len(traj)}, obs_dim={env.obs_dim}, n_actions={env.n_actions}")
    txt = env.serialize(traj)
    print("[env] first 2 serialized lines:")
    print("\n".join(txt.splitlines()[:2]))

    # push transitions with dummy per-agent credit and update
    for i, ag in enumerate(agents):
        for k in range(len(traj)):
            ag.buffer.push(traj.obs[k][i], int(traj.actions[k][i]),
                           float(traj.global_reward[k]), traj.next_obs[k][i], traj.done[k])
        # add more transitions so buffer >= batch_size
        for _ in range(20):
            ag.buffer.push(traj.obs[0][i], 0, 0.0, traj.next_obs[0][i], False)
        loss = ag.update()
        assert loss is not None
        print(f"[ddqn] agent{i} update loss={loss:.4f}, act={ag.act(traj.obs[0][i], 0.0)}")


def test_parser_paper_sample():
    # The actual LLM-TACA output text from Figure 4 of the paper (alice/bob, length 9).
    sample = """
    Here's a structured credit assignment ...
    ```python
    import numpy as np
    alice_credit = np.array([3, 1, 1, 1, 1, 1, 1, 1, 5]) # Alice contributed ...
    bob_credit = np.array([2, 1, 1, 1, 1, 1, 1, 1, 5]) # Bob also contributed ...
    task_alice = np.array([[0], [4]])
    task_bob = np.array([[1], [4]])
    ```
    """
    T = 9
    credits = parse_credits(sample, n_agents=2, num_timesteps=T,
                            agent_names=["alice_credit", "bob_credit"], cap=10.0)
    assert credits.shape == (2, T)
    # raw totals: alice=15 (>10 -> scaled to 10), bob=13 (>10 -> scaled to 10)
    assert abs(np.abs(credits[0]).sum() - 10.0) < 1e-3, credits[0].sum()
    assert abs(np.abs(credits[1]).sum() - 10.0) < 1e-3, credits[1].sum()
    print(f"[parser] alice total={np.abs(credits[0]).sum():.3f} -> {np.round(credits[0],3).tolist()}")
    print(f"[parser] bob   total={np.abs(credits[1]).sum():.3f} -> {np.round(credits[1],3).tolist()}")

    # values already under cap are left untouched (densification preserved)
    c_dense = parse_credits("alice_credit = np.array([0,0,0.5,0]) bob_credit = np.array([0,0,0.5,0])",
                            n_agents=2, num_timesteps=4, agent_names=["alice_credit", "bob_credit"])
    assert abs(c_dense.sum() - 1.0) < 1e-3, c_dense.sum()
    print(f"[parser] under-cap untouched: total={c_dense.sum():.3f}")

    # missing agent -> zeros; no crash
    c2 = parse_credits("no arrays here", 2, 5)
    assert c2.shape == (2, 5) and c2.sum() == 0.0
    print("[parser] missing-array fallback OK")

    assert valid_credit_arrays(
        "alice_credit = np.array([0, 1])\nbob_credit = np.array([1, 0])",
        ["alice_credit", "bob_credit"], 2,
    )
    assert not valid_credit_arrays(
        "alice_credit = np.array([0, 1])\nbob_credit = np.array([1])",
        ["alice_credit", "bob_credit"], 2,
    )
    print("[parser] strict training validation OK")

    compact = build_batch_task_prompt(["Alice", "Bob"], [3], compact=True)
    assert "nothing else" in compact
    assert "exactly 3 numbers" in compact
    assert "Work through your reasoning" not in compact
    print("[prompt] compact array-only mode OK")


if __name__ == "__main__":
    test_env_and_ddqn()
    test_parser_paper_sample()
    print("\nALL_CORE_TESTS_PASSED")
