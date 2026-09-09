"""Effect-space distance between two guidances compiled on the same state."""
import numpy as np

from algorithm.lehca.masking import build_masks


def mask_distance(g1, g2, snap, avail, iface, n_agents):
    """Effect-space distance between two guidances compiled on the same state.
    Restricted to alive agents x env-available actions."""
    n_actions = snap["n_actions"]
    h1, s1 = build_masks((g1 or {}).get("action_rules"), snap, iface, n_agents, n_actions)
    h2, s2 = build_masks((g2 or {}).get("action_rules"), snap, iface, n_agents, n_actions)
    av = np.array(avail, dtype=bool)
    ctrl = snap.get("controlled")  # GRF: agent i -> allies[controlled[i]]; SMAC: agent i -> allies[i]
    units = [snap["allies"][c] for c in ctrl] if ctrl else snap["allies"][:n_agents]
    alive = np.array([u["alive"] for u in units], dtype=bool)
    m = av & alive[:, None]
    if m.sum() == 0:
        return None
    hard_diff = float(((h1 > 0) != (h2 > 0))[m].mean())
    prefer_diff = float(((s1 > 1) != (s2 > 1))[m].mean())
    soft_l1 = float(np.abs(np.log(s1) - np.log(s2))[m].mean())
    # g1 (stale) forbids what g2 (fresh) prefers, per agent
    conflict = ((h1 == 0) & (s2 > 1) & m).any(-1)[alive]
    forbid_conflict = float(conflict.mean()) if alive.any() else 0.0
    return {"hard_diff": hard_diff, "prefer_diff": prefer_diff,
            "soft_l1": soft_l1, "forbid_conflict": forbid_conflict}


