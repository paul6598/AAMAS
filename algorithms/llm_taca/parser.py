"""Parser F_TACA: extract per-agent credit arrays (as in F_MCA) AND a per-agent recommended
action. Returns (credits [n_agents, T], actions) where actions[i] is an int in 0..n_actions-1
or None ("no recommendation").
"""
import re

from ..llm_mca.parser import parse_credits

_INT = r"-?\d+"


def _extract_action(text, name, n_actions):
    """Find `task_<name> = <int>` and return the action index, or None (for -1 / out of range)."""
    m = re.search(rf"task_{re.escape(name)}\s*=\s*(?:np\.array\(\s*)?\[?\s*({_INT})",
                  text, re.IGNORECASE)
    if m is None:
        return None
    a = int(m.group(1))
    return a if 0 <= a < n_actions else None


def parse_credits_and_tasks(text, n_agents, num_timesteps, agent_names, n_actions, cap=10.0):
    """agent_names: display names (e.g. ["Alice", "Bob"]). Returns (credits, actions)."""
    credit_names = [f"{n.lower()}_credit" for n in agent_names]
    credits = parse_credits(text, n_agents, num_timesteps, agent_names=credit_names, cap=cap)
    actions = [_extract_action(text, n.lower(), n_actions) for n in agent_names]
    return credits, actions


def parse_batch(text, n_agents, episode_lengths, agent_names, n_actions, cap=10.0):
    """Batch F_TACA: per-episode credit matrices (batch-normalized) + per-episode recommended
    actions. Returns (list of credit matrices, list of action lists)."""
    from ..llm_mca.parser import parse_batch_credits
    credit_names = agent_names
    credits = parse_batch_credits(text, n_agents, episode_lengths, credit_names, cap=cap)
    # one recommended action per agent for the whole batch (forward guidance)
    actions = [_extract_action(text, n.lower(), n_actions) for n in agent_names]
    return credits, actions
