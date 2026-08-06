"""TACA tests: F_TACA parsing on the paper's Figure 4 output, and trainer plumbing (no LLM)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from algorithms.llm_taca.parser import parse_credits_and_tasks
from algorithms.llm_taca.algorithm import build_task_vec, task_keep_prob, TaskPolicy, TASK_DIM
from algorithms.common.ddqn import DDQNAgent
from envs import make_env


def test_ftaca_paper_sample():
    # Figure 4 output: credit arrays (length 9) + task arrays (targets).
    sample = """
    ```python
    import numpy as np
    alice_credit = np.array([3, 1, 1, 1, 1, 1, 1, 1, 5])
    bob_credit = np.array([2, 1, 1, 1, 1, 1, 1, 1, 5])
    task_alice = np.array([[0], [4]])   # Alice should go to (0,4)
    task_bob = np.array([[1], [4]])
    ```
    """
    credits, targets = parse_credits_and_tasks(sample, 2, 9, ["Alice", "Bob"])
    assert credits.shape == (2, 9)
    assert targets == [(0, 4), (1, 4)], targets
    print(f"[ftaca] targets={targets}, credit totals={[round(float(np.abs(credits[i]).sum()),2) for i in range(2)]}")

    # "no task" sentinel and missing task -> None
    c2, t2 = parse_credits_and_tasks(
        "alice_credit=np.array([1]) task_alice=np.array([-1,-1]) bob_credit=np.array([1])",
        2, 1, ["Alice", "Bob"])
    assert t2 == [None, None], t2
    print(f"[ftaca] no-task sentinel + missing -> {t2}")


def test_task_vec_and_schedule():
    assert np.allclose(build_task_vec(None, 8), [0, 0, 0])
    assert np.allclose(build_task_vec((0, 4), 8), [1.0, 0.0, 0.5])
    assert task_keep_prob(0, 100, 0.8) == 1.0
    assert task_keep_prob(80, 100, 0.8) == 0.0
    assert task_keep_prob(40, 100, 0.8) == 0.5
    print("[taca] task_vec + weaning schedule OK")


def test_task_policy_plumbing():
    env = make_env("Foraging-8x8-2p-2f-coop-v3", seed=0)
    agents = [DDQNAgent(env.obs_dim + TASK_DIM, env.n_actions, "cpu", batch_size=8)
              for _ in range(env.n_agents)]
    policies = [TaskPolicy(a) for a in agents]
    traj = env.rollout(policies, epsilon=1.0)   # acts on [obs, task=0]
    assert len(traj) > 0
    # push a task-augmented transition and update
    for i, ag in enumerate(agents):
        tvec = build_task_vec((3, 5), env.grid_size)
        for k in range(len(traj)):
            ag.buffer.push(np.concatenate([traj.obs[k][i], tvec]), int(traj.actions[k][i]),
                           0.5, np.concatenate([traj.next_obs[k][i], tvec]), traj.done[k])
        for _ in range(10):
            ag.buffer.push(np.concatenate([traj.obs[0][i], np.zeros(TASK_DIM, np.float32)]),
                           0, 0.0, np.concatenate([traj.next_obs[0][i], np.zeros(TASK_DIM, np.float32)]), False)
        loss = ag.update()
        assert loss is not None
    print(f"[taca] rollout len={len(traj)}, task-conditioned DDQN update OK (input_dim={env.obs_dim + TASK_DIM})")


if __name__ == "__main__":
    test_ftaca_paper_sample()
    test_task_vec_and_schedule()
    test_task_policy_plumbing()
    print("\nALL_TACA_TESTS_PASSED")
