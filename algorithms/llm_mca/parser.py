"""Parser F_MCA: extract per-agent credit arrays from the LLM critic's text output.

Per the paper, the critic is asked to emit numpy arrays (one per agent). This module does a
regex search for those arrays plus a normalization step, mapping the text to c_k in R^N.
"""
import re

import numpy as np

_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def _extract_array(text, name):
    """Find `<name> = np.array([ ... ])` (or `= [ ... ]`) and return a float list, or None."""
    pat = re.compile(rf"{re.escape(name)}\s*=\s*(?:np\.array\(\s*)?\[([^\]]*)\]", re.IGNORECASE)
    m = pat.search(text)
    if m is None:
        return None
    nums = re.findall(_NUM, m.group(1))
    if not nums:
        return None
    return [float(x) for x in nums]


def valid_credit_arrays(text, agent_names, num_timesteps, exact_length=True):
    """Return True only when every requested array exists with exactly the requested length.

    Parsing remains permissive for diagnostics, but training uses this strict check to avoid
    silently turning malformed generations into zero-padded reward signals.
    """
    for name in agent_names:
        arr = _extract_array(text, name)
        if arr is None or not np.isfinite(arr).all():
            return False
        if exact_length and len(arr) != num_timesteps:
            return False
    return True


def parse_credits(text, n_agents, num_timesteps, agent_names=None, cap=10.0,
                  length_policy="right"):
    """Map critic text -> credit matrix of shape (n_agents, num_timesteps).

    - Missing/short arrays are zero-padded; long ones are truncated.
    - If an agent's array is entirely missing, that agent gets zeros.
    - Normalization (the paper's "normalization step"): the base prompt asks the critic to keep
      each robot's reward values adding up to less than ten. We enforce that only as a soft
      upper bound -- if an agent's total magnitude exceeds `cap`, scale that agent's row down to
      `cap`; otherwise leave the raw values untouched. This bounds outliers while preserving the
      LLM's intermediate ("densified") sub-goal rewards even when the episode's global reward is
      zero -- the key mechanism of the paper. Pass cap=None to disable.
    """
    if length_policy not in ("right", "left"):
        raise ValueError(f"unknown credit length policy: {length_policy}")
    names = agent_names or [f"agent{i}_credit" for i in range(n_agents)]
    credits = np.zeros((n_agents, num_timesteps), dtype=np.float32)
    for i, name in enumerate(names):
        arr = _extract_array(text, name)
        if arr is None:
            continue
        # The legacy/default policy right-aligns malformed arrays because LBF collection rewards
        # are often late.  The paper-style free-form prompt instead commonly emits a correct
        # t=0 prefix followed by a few explanatory/trailing zeros; its permissive regex parser
        # path must preserve that prefix (``length_policy=left``).
        arr = np.asarray(arr, dtype=np.float32)
        if len(arr) >= num_timesteps and length_policy == "right":
            credits[i] = arr[-num_timesteps:]
        elif len(arr) >= num_timesteps:
            credits[i] = arr[:num_timesteps]
        elif length_policy == "right":
            credits[i, num_timesteps - len(arr):] = arr
        else:
            credits[i, :len(arr)] = arr

    if cap is not None:
        for i in range(n_agents):
            total = float(np.sum(np.abs(credits[i])))
            if total > cap:
                credits[i] *= cap / total
    return credits


def parse_batch_credits(text, n_agents, episode_lengths, agent_names, cap=10.0,
                        length_policy="right"):
    """Parse a batch-mode critic output into one credit matrix per episode.

    Arrays are named `{name}_credit_{e}` (1-indexed episode e). Returns a list of arrays, each of
    shape (n_agents, T_e). Missing episodes/agents get zeros.

    Normalization is applied ACROSS THE WHOLE BATCH with a single factor, not per episode. This
    is essential: batching lets the critic assign relative credit (a successful episode gets more
    than a failed one), and a per-episode cap would flatten that -- if two episodes both exceed
    the cap they would both be scaled to it and become indistinguishable. Here we find the largest
    per-agent episode total across the batch and, only if it exceeds `cap`, scale every episode by
    the same factor, preserving the cross-episode ordering the comparison produced.
    """
    out = []
    for e, T in enumerate(episode_lengths, 1):
        names = [f"{n.lower()}_credit_{e}" for n in agent_names]
        out.append(parse_credits(
            text, n_agents, T, agent_names=names, cap=None,
            length_policy=length_policy,
        ))

    if cap is not None and out:
        peak = max((float(np.abs(c[i]).sum()) for c in out for i in range(n_agents)), default=0.0)
        if peak > cap:
            for c in out:
                c *= cap / peak
    return out
