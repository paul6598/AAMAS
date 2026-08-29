"""LEHCA multi-agent controller.

Applies Commander guidance at action-selection time only (Eq. 5-6):
    Q~_i(tau, u) = Q_i(tau, u) + beta * log W_soft,i(u)   if M_hard,i(u) = 1
                   -inf                                    otherwise
Greedy/epsilon-greedy selection then operates on the allowed set
(env-available AND hard-allowed; falls back to env-available if empty).
The learner's forward passes are untouched, so TD backups use the raw
utilities Q_i as in standard QMIX.
"""
import torch as th

from algorithm.src.controllers.basic_controller import BasicMAC


class LehcaMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(LehcaMAC, self).__init__(scheme, groups, args)
        self.beta = getattr(args, "beta", 0.5)
        # True: bias epsilon-greedy RANDOM draws by W_soft as well (the paper
        # describes masking as "guiding exploration"; default only tilts greedy)
        self.explore_soft_bias = getattr(args, "explore_soft_bias", False)
        self._hard = None
        self._soft = None
        self._st = {"q_gap_mean": [], "mask_forbid_frac": [],
                    "mask_override_rate": [], "mask_fallback_rate": []}

    def pop_mask_stats(self):
        out = {k: float(sum(v) / len(v)) for k, v in self._st.items() if v}
        for v in self._st.values():
            v.clear()
        return out

    def _record_stats(self, q, avail, hard=None, empty=None, allowed=None, tilted=None):
        with th.no_grad():
            q0, av = q[0], avail[0] > 0
            qm = q0.masked_fill(~av, -1e9)
            top2 = qm.topk(2, dim=-1).values
            valid = av.sum(-1) >= 2
            if valid.any():
                self._st["q_gap_mean"].append(float((top2[:, 0] - top2[:, 1])[valid].mean()))
            if hard is None:
                return
            h = hard[0] > 0
            n_av = av.sum(-1).clamp(min=1).float()
            self._st["mask_forbid_frac"].append(float(((av & ~h).sum(-1).float() / n_av).mean()))
            self._st["mask_fallback_rate"].append(float(empty[0].float().mean()))
            al = allowed[0] > 0
            raw_arg = q0.masked_fill(~al, -1e9).argmax(-1)
            tilt_arg = tilted[0].masked_fill(~al, -1e9).argmax(-1)
            self._st["mask_override_rate"].append(float((raw_arg != tilt_arg).float().mean()))

    def set_guidance(self, hard, soft):
        """hard/soft: (n_agents, n_actions) numpy arrays, or None to disable."""
        self._hard = hard
        self._soft = soft

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)

        if self._hard is None:
            if not test_mode:
                self._record_stats(agent_outputs, avail_actions)
            return self.action_selector.select_action(
                agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)

        hard = th.as_tensor(self._hard, dtype=avail_actions.dtype,
                            device=avail_actions.device).unsqueeze(0)
        soft = th.as_tensor(self._soft, dtype=agent_outputs.dtype,
                            device=agent_outputs.device).unsqueeze(0)

        allowed = avail_actions * hard
        # Non-emptiness fallback: ignore hard constraints for agents whose
        # allowed set would be empty.
        empty = (allowed.sum(dim=-1, keepdim=True) == 0)
        allowed = th.where(empty, avail_actions, allowed)

        tilted_q = agent_outputs + self.beta * th.log(soft)
        if not test_mode:
            self._record_stats(agent_outputs, avail_actions, hard, empty, allowed, tilted_q)
        if self.explore_soft_bias:
            # Categorical over avail weights in the selector -> W_soft-biased
            # random exploration; -inf mask and greedy argmax are unaffected.
            allowed = allowed.float() * soft
        return self.action_selector.select_action(
            tilted_q[bs], allowed[bs], t_env, test_mode=test_mode)
