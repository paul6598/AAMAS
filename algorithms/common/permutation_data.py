"""Environment-side reserialization without changing the underlying MARL transition."""
import copy

import numpy as np

from envs.base import Trajectory


def _permuted_lbf_state(state, agent_permutation, object_permutation):
    return {
        "agents": [state["agents"][i] for i in agent_permutation],
        "foods": [state["foods"][i] for i in object_permutation],
    }


def permuted_lbf_view(traj, env, agent_permutation, object_permutation):
    """Return a view with blocks reordered but opaque agent tags moved with their data.

    ``env.agent_names`` is reordered to preserve the semantic tag-to-physical-agent mapping in
    raw text baselines.  The real environment/policy ordering is never mutated.
    """
    ap, op = np.asarray(agent_permutation), np.asarray(object_permutation)
    if sorted(ap.tolist()) != list(range(env.n_agents)):
        raise ValueError("invalid agent permutation")
    copied = Trajectory(
        obs=[tuple(step[i] for i in ap) for step in traj.obs],
        actions=[np.asarray(step)[ap] for step in traj.actions],
        global_reward=list(traj.global_reward),
        next_obs=[tuple(step[i] for i in ap) for step in traj.next_obs],
        done=list(traj.done),
        state=[_permuted_lbf_state(state, ap, op) for state in traj.state],
        next_state=[_permuted_lbf_state(state, ap, op) for state in traj.next_state],
    )
    view = copy.copy(env)
    view.agent_names = [env.agent_names[i] for i in ap]
    return copied, view


def inverse_agent_axis(values, agent_permutation):
    """Map rows generated in serialized order back to original environment agent order."""
    return np.asarray(values)[np.argsort(np.asarray(agent_permutation))]
