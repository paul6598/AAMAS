import numpy as np

from algorithms.common.permutation_data import inverse_agent_axis, permuted_lbf_view
from envs.base import Trajectory


class Env:
    n_agents = 3
    agent_names = ["A", "B", "C"]


def test_permuted_view_moves_tags_with_agent_data_and_inverse_scatter_restores_axis():
    trajectory = Trajectory(
        actions=[np.asarray([0, 1, 2])],
        state=[{"agents": [(0,), (1,), (2,)], "foods": [(3,), (4,)]}],
        next_state=[{"agents": [(0,), (1,), (2,)], "foods": [(3,), (4,)]}],
    )
    view, env = permuted_lbf_view(trajectory, Env(), [2, 0, 1], [1, 0])
    assert env.agent_names == ["C", "A", "B"]
    assert view.state[0]["agents"] == [(2,), (0,), (1,)]
    assert view.state[0]["foods"] == [(4,), (3,)]
    original = np.asarray([[10.], [20.], [30.]])
    serialized = original[[2, 0, 1]]
    np.testing.assert_allclose(inverse_agent_axis(serialized, [2, 0, 1]), original)
