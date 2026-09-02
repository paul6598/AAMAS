"""Episode runner with adaptive Commander refresh (Guidance-Progress CUSUM).

Adapted from algorithm/lehca/runner.py (kept untouched per session rules).
Differences:
  * env dispatch: works for SMAC ("sc2") and GRF ("gfootball") — commander,
    predicate library, shaping and state features chosen per env.
  * refresh schedule: scheduler="fixed" reproduces LEHCA (period f_update);
    scheduler="vf" adds value-based early refreshes under a hard ceiling —
    refresh at latest every f_update steps, earlier when the one-sided CUSUM
    on v_t = V_F(s_t;G)/V_F(s_r;G) exceeds sched_h.
  * a multi-head shaping-value critic V_j(s) is trained online from the
    per-step predicate vectors of finished episodes.
"""
import json
import os
from collections import deque
from functools import partial

import numpy as np

from env import REGISTRY as env_REGISTRY
from algorithm.src.components.episode_buffer import EpisodeBatch
from env.semantic import make_interface
from algorithm.lehca.commander import make_commander
from algorithm.lehca.masking import build_masks
from algorithm.lehca.state import get_state
from algorithm.vigil import predlib
from algorithm.vigil.critic import ValueCritic


class SchedRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0
        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}
        self.log_train_stats_t = -1000000

        self.state = get_state()
        self.state.configure(args)
        self.iface = make_interface(args.env, self.env, args)
        if args.env == "gfootball":
            from algorithm.vigil.commander.grf import GRFLLMCommander
            self.commander = GRFLLMCommander(args, self.iface, logger) \
                if getattr(args, "commander", "llm") == "llm" else None
        else:
            self.commander = make_commander(args, self.iface, logger)

        self.use_shaping = getattr(args, "use_reward_shaping", True)
        self.use_masking = getattr(args, "use_action_masking", False)
        self.mask_at_test = getattr(args, "use_masking_at_test", False)
        self.shaping_in_learner = getattr(args, "shaping_in_learner", False)

        # ---- scheduler ----
        self.scheduler = getattr(args, "scheduler", "fixed")
        self.f_max = getattr(args, "f_update", 100)          # fixed period / vf ceiling
        self.k = getattr(args, "sched_k", 0.6)
        self.h = getattr(args, "sched_h", 3.0)
        self.min_interval = getattr(args, "sched_min_interval", 10)
        self.eps_frac = getattr(args, "sched_eps_frac", 0.25)  # issuance value vs buffer-mean value
        self.warmup_eps = getattr(args, "sched_warmup_episodes", 20)
        self.trusted_wmin = getattr(args, "sched_trusted_weight_min", 0.5)
        # budget controller: adapt h so realized EARLY refreshes/ep track the target
        self.target_early = getattr(args, "sched_target_early_per_ep", 0.0)  # 0 = off
        self.h_adapt_every = getattr(args, "sched_h_adapt_episodes", 20)
        self._early_window = deque(maxlen=self.h_adapt_every)
        self.critic = None      # built lazily (needs a snapshot)
        self.lib = None
        self.fx = None
        self._episodes_seen = 0
        self._S = 0.0
        self._v_ref = None
        self._ref_heads = None
        self._x_ref = None
        self.last_refresh_t = None
        self._ep_X, self._ep_F = [], []
        self._log_v, self._log_refresh, self._log_early = [], 0, 0
        self._ep_early = 0
        self._ep_armed = False
        self._why = {"warmup": 0, "no_heads": 0, "low_vref": 0, "ok": 0}
        self._episodes_since_log = 0

        self.recent_wins = deque(maxlen=32)
        self._shaping_sums = []
        self._guidance_log = None
        if self.commander is not None:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            gdir = os.path.join(repo_root, "results", "guidance")
            os.makedirs(gdir, exist_ok=True)
            fname = "%s_%s_s%s_p%d.jsonl" % (
                getattr(args, "unique_token", "run"),
                getattr(args, "wandb_group", "nogroup"),
                getattr(args, "seed", "x"), os.getpid())
            self._guidance_log = open(os.path.join(gdir, fname), "a")

    # ------------------------------------------------------------- plumbing
    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size,
                                 self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        if hasattr(self.iface, "reset_episode"):
            self.iface.reset_episode()
        self.t = 0

    # ------------------------------------------------------------ scheduler
    def _lazy_init(self, snap):
        if self.lib is not None:
            return
        self.lib = predlib.build_library(self.args.env, snap)
        self.fx = predlib.FeatureExtractor(self.args.env, snap)
        self.critic = ValueCritic(
            in_dim=len(self.fx(snap)), n_heads=len(self.lib),
            gamma=getattr(self.args, "sched_gamma", 0.8),
            lr=getattr(self.args, "critic_lr", 1e-3),
            buffer_steps=getattr(self.args, "critic_buffer", 60000))

    def _guidance_heads(self, guidance):
        """[(head_idx, weight)] for trusted, library-resolvable sub-goals."""
        out, w_all = [], 0.0
        for sg in (guidance or {}).get("subgoals") or []:
            w = float(sg.get("weight", 0.5))
            w_all += w
            hi = predlib.head_index(self.lib, sg)
            if hi is not None and self.critic.trusted(hi):
                out.append((hi, w))
        w_tr = sum(w for _, w in out)
        if w_all <= 0 or w_tr / w_all < self.trusted_wmin:
            return None
        return out

    def _vF(self, x, heads):
        pred = self.critic.predict(x)
        return float(sum(w * pred[hi] for hi, w in heads))

    def _maybe_refresh(self, snap, x, test_mode):
        if self.commander is None or test_mode:
            return
        t_global = self.t_env + self.t
        since = None if self.last_refresh_t is None else t_global - self.last_refresh_t
        due, early = self.last_refresh_t is None or since >= self.f_max, False
        ready = (self.scheduler == "vf" and self._episodes_seen >= self.warmup_eps
                 and self._v_ref is not None)
        if not due and ready:
            # re-evaluate the issuance state with the CURRENT critic so that
            # estimator drift over the spell cancels out of the ratio
            v_den = max(self._vF(self._x_ref, self._ref_heads), 1e-3)
            v = min(1.0, max(0.0, self._vF(x, self._ref_heads)) / v_den)
            self._log_v.append(v)
            self._S = max(0.0, self._S + self.k - v)
            if self._S >= self.h and since >= self.min_interval:
                due, early = True, True
        if not due:
            return
        if t_global < getattr(self, "_retry_after", -1):
            return
        stats = {"t_env": self.t_env}
        if self.recent_wins:
            stats["rolling_win_rate"] = float(np.mean(self.recent_wins))
        summary = self.iface.summary(snap, stats)
        key = self.iface.cache_key(snap)
        hits_before = getattr(self.commander, "n_cache_hits", 0)
        guidance = self.commander(summary, key, self.iface)
        if guidance is None:
            # failed call: keep the old guidance AND the schedule state; retry soon
            self._retry_after = t_global + getattr(self.args, "sched_fail_retry", 5)
            return
        self.state.guidance = guidance
        self.last_refresh_t = t_global
        self._S = 0.0
        self._log_refresh += 1
        self._log_early += int(early)
        self._ep_early += int(early)
        self._v_ref, self._ref_heads, self._x_ref = None, None, None
        if self.scheduler == "vf" and self.critic is not None:
            if self._episodes_seen < self.warmup_eps:
                self._why["warmup"] += 1
            else:
                heads = self._guidance_heads(self.state.guidance)
                if not heads:
                    self._why["no_heads"] += 1
                else:
                    vr = self._vF(x, heads)
                    mu = self.critic.mu.numpy()
                    base = sum(w * max(float(mu[hi]), 0.0) for hi, w in heads)
                    if base < 1e-3 or vr < self.eps_frac * base:
                        self._why["low_vref"] += 1
                    else:
                        self._ref_heads, self._v_ref = heads, vr
                        self._x_ref = np.array(x, dtype=np.float32, copy=True)
                        self._why["ok"] += 1
                        self._ep_armed = True
        if self._guidance_log is not None:
            self._guidance_log.write(json.dumps({
                "t_env": self.t_env, "t_global": t_global, "cache_key": key,
                "early": early,
                "cache_hit": getattr(self.commander, "n_cache_hits", 0) > hits_before,
                "guidance": guidance}) + "\n")
            self._guidance_log.flush()

    # ------------------------------------------------------------------ run
    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        shaped_return = 0.0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            snap_pre = self.iface.snapshot()
            self._lazy_init(snap_pre)
            x_pre = self.fx(snap_pre)
            self._maybe_refresh(snap_pre, x_pre, test_mode)

            guidance = self.state.guidance
            mask_active = self.use_masking and guidance is not None \
                and (not test_mode or self.mask_at_test)
            if mask_active:
                hard, soft = build_masks(guidance.get("action_rules"), snap_pre,
                                         self.iface, self.args.n_agents,
                                         snap_pre["n_actions"])
                self.mac.set_guidance(hard, soft)
            else:
                self.mac.set_guidance(None, None)

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }
            self.batch.update(pre_transition_data, ts=self.t)

            actions = self.mac.select_actions(self.batch, t_ep=self.t,
                                              t_env=self.t_env, test_mode=test_mode)

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            snap_post = self.iface.snapshot()
            if hasattr(self.iface, "tick"):
                self.iface.tick(snap_post)
            acts = actions[0].tolist()
            stored_reward = reward
            f_t = 0.0
            if not test_mode:
                self._ep_X.append(x_pre)
                self._ep_F.append(predlib.f_vector(self.args.env, self.lib,
                                                   snap_pre, snap_post, acts))
                if self.use_shaping and guidance is not None:
                    f_t = predlib.shaping(self.args.env, guidance.get("subgoals"),
                                          snap_pre, snap_post, acts,
                                          clip=getattr(self.args, "shaping_clip", 3.0))
                    if not self.shaping_in_learner:
                        stored_reward = reward + self.state.lambda_val * f_t
                    shaped_return += self.state.lambda_val * f_t

            post_transition_data = {
                "actions": actions,
                "reward": [(stored_reward,)],
                "shaping_f": [(f_t,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            self.batch.update(post_transition_data, ts=self.t)
            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)
        actions = self.mac.select_actions(self.batch, t_ep=self.t,
                                          t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0)
                          for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t
            self.recent_wins.append(1.0 if env_info.get("battle_won", False) else 0.0)
            self._shaping_sums.append(shaped_return)
            self._episodes_seen += 1
            self._episodes_since_log += 1
            # h adapts only on episodes where the CUSUM was actually armed —
            # gate-closed (warmup/low_vref) episodes are no evidence that h is
            # too high, and previously drove h to the floor (hair-trigger).
            if self._ep_armed:
                self._early_window.append(self._ep_early)
            self._ep_early = 0
            self._ep_armed = False
            if (self.target_early > 0 and len(self._early_window) == self.h_adapt_every):
                rate = float(np.mean(self._early_window))
                ratio = (rate + 0.05) / (self.target_early + 0.05)
                self.h = float(np.clip(self.h * np.sqrt(ratio), 2.0, 50.0))
                self._early_window.clear()
            if self._ep_X:
                self.critic.add_episode(self._ep_X, self._ep_F)
                self.critic.train(iters=getattr(self.args, "critic_iters", 50))
            self._ep_X, self._ep_F = [], []

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon,
                                     self.t_env)
            self.logger.log_stat("lehca_lambda", self.state.lambda_val, self.t_env)
            if self._shaping_sums:
                self.logger.log_stat("shaped_return_mean",
                                     float(np.mean(self._shaping_sums)), self.t_env)
                self._shaping_sums = []
            n_ep = max(1, self._episodes_since_log)
            self.logger.log_stat("sched_refresh_per_ep", self._log_refresh / n_ep, self.t_env)
            self.logger.log_stat("sched_early_per_ep", self._log_early / n_ep, self.t_env)
            # F_max-due (+first/retry) refreshes vs CUSUM-fired ones
            self.logger.log_stat("sched_fallback_per_ep",
                                 (self._log_refresh - self._log_early) / n_ep, self.t_env)
            if self._log_v:
                self.logger.log_stat("sched_v_mean", float(np.mean(self._log_v)), self.t_env)
            if self.critic is not None and self.critic.loss_ema is not None:
                self.logger.log_stat("critic_loss", self.critic.loss_ema, self.t_env)
            for kk, vv in self._why.items():
                self.logger.log_stat("sched_ref_%s" % kk, vv, self.t_env)
            self.logger.log_stat("sched_h_current", self.h, self.t_env)
            self._why = {"warmup": 0, "no_heads": 0, "low_vref": 0, "ok": 0}
            self._log_refresh, self._log_early, self._log_v = 0, 0, []
            self._episodes_since_log = 0
            if self.commander is not None:
                for k, v in self.commander.stats().items():
                    self.logger.log_stat(k, v, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()
        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"],
                                     self.t_env)
        stats.clear()
