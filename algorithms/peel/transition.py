"""Environment-neutral transition records for the relational event model.

An adapter may know how to expose state fields, but it must not encode a particular agent's
slot/index as a feature.  Entity order is deliberately kept outside the representation so that
the same record can be permuted for the adversarial consistency evaluation.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class TransitionRecord:
    agent_features: np.ndarray      # [N, D_a], excluding agent index
    object_features: np.ndarray     # [M, D_o], excluding object index
    agent_delta: np.ndarray         # [N, D_a]
    object_delta: np.ndarray        # [M, D_o]
    joint_actions: np.ndarray       # [N]
    reward: float
    done: bool
    agent_ids: np.ndarray            # external scatter/evaluation bookkeeping only
    object_ids: np.ndarray
    domain: str


def _lbf_agent(agent, action):
    row, col, level = agent
    return np.asarray([row, col, level, action], dtype=np.float32)


def _lbf_object(food):
    row, col, level = food
    # active flag makes a disappeared object observable without assigning it a semantic slot.
    return np.asarray([row, col, level, float(row >= 0)], dtype=np.float32)


def lbf_transition(traj, timestep):
    """Create a one-transition, entity-set record from the existing LBF global snapshots."""
    state = traj.state[timestep]
    next_state = traj.next_state[timestep]
    actions = np.asarray(traj.actions[timestep], dtype=np.float32)
    agents = np.stack([_lbf_agent(agent, actions[i]) for i, agent in enumerate(state["agents"])])
    next_agents = np.stack([
        _lbf_agent(agent, actions[i]) for i, agent in enumerate(next_state["agents"])
    ])
    objects = np.stack([_lbf_object(food) for food in state["foods"]])
    next_objects = np.stack([_lbf_object(food) for food in next_state["foods"]])
    return TransitionRecord(
        agent_features=agents,
        object_features=objects,
        agent_delta=next_agents - agents,
        object_delta=next_objects - objects,
        joint_actions=actions.astype(np.int64),
        reward=float(traj.global_reward[timestep]),
        done=bool(traj.done[timestep]),
        agent_ids=np.arange(len(agents), dtype=np.int64),
        object_ids=np.arange(len(objects), dtype=np.int64),
        domain="lbf",
    )


def permute_record(record, agent_permutation=None, object_permutation=None):
    """Reorder only homogeneous entity axes; IDs move as external bookkeeping metadata."""
    n_agents, n_objects = len(record.agent_features), len(record.object_features)
    ap = np.arange(n_agents) if agent_permutation is None else np.asarray(agent_permutation)
    op = np.arange(n_objects) if object_permutation is None else np.asarray(object_permutation)
    if sorted(ap.tolist()) != list(range(n_agents)) or sorted(op.tolist()) != list(range(n_objects)):
        raise ValueError("permutations must contain every entity index exactly once")
    return TransitionRecord(
        agent_features=record.agent_features[ap],
        object_features=record.object_features[op],
        agent_delta=record.agent_delta[ap],
        object_delta=record.object_delta[op],
        joint_actions=record.joint_actions[ap],
        reward=record.reward,
        done=record.done,
        agent_ids=record.agent_ids[ap],
        object_ids=record.object_ids[op],
        domain=record.domain,
    )


_DIRECTION_CODE = {"UP": 0.0, "RIGHT": 1.0, "DOWN": 2.0, "LEFT": 3.0}


def _rware_agent(agent, action):
    x, y, direction, carrying = agent
    return np.asarray([x, y, _DIRECTION_CODE[direction], float(carrying is not None), action], np.float32)


def _rware_object(x, y, object_id, kind, active):
    # kind 0=goal, 1=requested shelf; ID is external identity metadata, not a model feature.
    return np.asarray([x, y, kind, float(active)], np.float32)


def rware_transition(traj, timestep):
    """RWARE adapter: union requested-shelf identities across a transition plus stable goals."""
    state, next_state = traj.state[timestep], traj.next_state[timestep]
    actions = np.asarray(traj.actions[timestep], dtype=np.float32)
    agents = np.stack([_rware_agent(agent, actions[i]) for i, agent in enumerate(state["agents"])])
    next_agents = np.stack([
        _rware_agent(agent, actions[i]) for i, agent in enumerate(next_state["agents"])
    ])
    current_requested = {shelf_id: (x, y) for x, y, shelf_id in state["requested"]}
    next_requested = {shelf_id: (x, y) for x, y, shelf_id in next_state["requested"]}
    entries = []
    for goal_index, (x, y) in enumerate(state["goals"]):
        entries.append(("goal", goal_index, x, y, x, y, 0.0, 1.0))
    for shelf_id in sorted(set(current_requested) | set(next_requested)):
        x, y = current_requested.get(shelf_id, (-1, -1))
        nx, ny = next_requested.get(shelf_id, (-1, -1))
        entries.append(("shelf", shelf_id, x, y, nx, ny,
                        float(shelf_id in current_requested), float(shelf_id in next_requested)))
    objects = np.stack([
        _rware_object(x, y, object_id, 0.0 if kind == "goal" else 1.0, active)
        for kind, object_id, x, y, _, _, active, _ in entries
    ])
    next_objects = np.stack([
        _rware_object(nx, ny, object_id, 0.0 if kind == "goal" else 1.0, next_active)
        for kind, object_id, _, _, nx, ny, _, next_active in entries
    ])
    object_ids = np.asarray([
        object_id if kind == "shelf" else -(object_id + 1) for kind, object_id, *_ in entries
    ])
    return TransitionRecord(
        agent_features=agents, object_features=objects,
        agent_delta=next_agents - agents, object_delta=next_objects - objects,
        joint_actions=actions.astype(np.int64), reward=float(traj.global_reward[timestep]),
        done=bool(traj.done[timestep]), agent_ids=np.arange(len(agents), dtype=np.int64),
        object_ids=object_ids, domain="rware",
    )
