import numpy as np
import torch

from algorithms.peel.model import PEELEventEncoder
from algorithms.peel.permutation import audit_event_equivariance
from algorithms.peel.transition import TransitionRecord, permute_record, rware_transition
from envs.base import Trajectory


def _record():
    return TransitionRecord(
        agent_features=np.asarray([[0, 0, 1, 3], [4, 2, 1, 1], [1, 3, 2, 0]], np.float32),
        object_features=np.asarray([[2, 2, 2, 1], [3, 4, 1, 1]], np.float32),
        agent_delta=np.zeros((3, 4), np.float32), object_delta=np.zeros((2, 4), np.float32),
        joint_actions=np.asarray([3, 1, 0]), reward=0.0, done=False,
        agent_ids=np.arange(3), object_ids=np.arange(2), domain="test",
    )


def test_record_permutation_only_moves_entity_axes():
    record = _record()
    permuted = permute_record(record, [2, 0, 1], [1, 0])
    np.testing.assert_allclose(permuted.agent_features[0], record.agent_features[2])
    np.testing.assert_allclose(permuted.object_features[0], record.object_features[1])


def test_event_encoder_is_agent_and_object_equivariant():
    torch.manual_seed(0)
    model = PEELEventEncoder(dim=32, event_slots=3, layers=2, heads=4)
    drift = audit_event_equivariance(model, _record(), [2, 0, 1], [1, 0])
    assert max(drift.values()) < 1e-5


def test_rware_transition_uses_entity_union_across_requested_shelf_change():
    trajectory = Trajectory(
        actions=[np.asarray([1, 4])], global_reward=[1.0], done=[False],
        state=[{"agents": [(0, 0, "UP", None), (1, 0, "RIGHT", 3)],
                "goals": [(2, 2)], "requested": [(1, 1, 3)]}],
        next_state=[{"agents": [(0, 1, "UP", None), (1, 0, "RIGHT", None)],
                     "goals": [(2, 2)], "requested": [(3, 3, 7)]}],
    )
    record = rware_transition(trajectory, 0)
    assert record.agent_features.shape == (2, 5)
    assert record.object_features.shape == (3, 4)  # one goal + removed shelf 3 + new shelf 7
    assert set(record.object_ids.tolist()) == {-1, 3, 7}
