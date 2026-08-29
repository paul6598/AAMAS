"""Abstract semantic interface between an environment and the LEHCA layer.

A SemanticInterface exposes exactly three things to the rest of LEHCA:
  * snapshot()   : structured, observable unit/entity information (numeric)
  * summary()    : structured natural-language bundle d_t for the Commander
  * vocabulary   : action-token / predicate grounding used by masking & shaping

To support a new environment (e.g. MPE), subclass SemanticInterface and
register it in lehca/envs/__init__.py under the pymarl env name.
"""


class SemanticInterface:
    def __init__(self, env, args):
        self.env = env
        self.args = args

    def snapshot(self):
        """Return a dict describing the current observable situation."""
        raise NotImplementedError

    def summary(self, snap, extra_stats=None):
        """Return the natural-language bundle d_t built from a snapshot."""
        raise NotImplementedError

    def cache_key(self, snap):
        """Coarse discretisation of the snapshot used to cache Commander calls."""
        raise NotImplementedError

    def prompt_context(self):
        """Static, environment-specific text for the Commander system prompt
        (action tokens, unit types present, rules of the scenario)."""
        raise NotImplementedError

    # --- grounding helpers used by masking / shaping ---
    def resolve_action_token(self, token, agent_idx, snap):
        """Map a symbolic action token (e.g. 'attack_type:Marine') to a list of
        concrete action indices for the given agent under snapshot `snap`."""
        raise NotImplementedError

    def agent_matches(self, selector, agent_idx, snap):
        """Whether agent `agent_idx` matches an 'applies_to' selector."""
        raise NotImplementedError
