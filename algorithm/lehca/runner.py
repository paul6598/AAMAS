"""LEHCA episode runner (Algorithm 1).

Extends pymarl's EpisodeRunner with:
  * scheduled Commander refresh every f_update env steps (semantic summary
    d_t rebuilt on the same schedule; last valid guidance kept on failure)
  * per-step action-mask compilation from the current symbolic rules
  * semantic reward shaping r_total = r_env + lambda * F_t at collection time

Test episodes use the latest guidance for masked greedy execution but never
trigger LLM calls and are never reward-shaped.
"""
import json
import os
from collections import deque
from functools import partial

import numpy as np

from env import REGISTRY as env_REGISTRY
from algorithm.src.components.episode_buffer import EpisodeBatch
from env.semantic import make_interface
from .commander import make_commander
from .masking import build_masks
from .shaping import compute_shaping
from .state import get_state


class LehcaRunner:

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

        # --- LEHCA additions ---
        self.state = get_state()
        self.state.configure(args)
        self.iface = make_interface(args.env, self.env, args)
        self.commander = make_commander(args, self.iface, logger)
        self.f_update = getattr(args, "f_update", 200)
        self.use_shaping = getattr(args, "use_reward_shaping", True)
        self.use_masking = getattr(args, "use_action_masking", True)
        self.mask_at_test = getattr(args, "use_masking_at_test", True)
        # >0: disable masking entirely once t_env passes this step count
        self.mask_anneal_t = getattr(args, "mask_anneal_t", 0)
        # True: store raw env reward + F_t separately; learner composes
        # r + lambda_now * F_t at train time (avoids stale-lambda rewards
        # lingering in the replay buffer)
        self.shaping_in_learner = getattr(args, "shaping_in_learner", False)
        self.last_refresh_t = None
        self.recent_wins = deque(maxlen=32)
        self._shaping_sums = []
        # Every Commander call is dumped so guidance content can be audited.
        self._guidance_log = None
        if self.commander is not None:
            gdir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))), "results", "guidance")
            os.makedirs(gdir, exist_ok=True)
            self._guidance_log = open(os.path.join(
                gdir, "%s.jsonl" % getattr(args, "unique_token", "run")), "a")

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
        self.t = 0

    def _maybe_refresh_commander(self, snap, test_mode):
        if self.commander is None or test_mode:
            return
        t_global = self.t_env + self.t  # t_env only advances at episode end
        due = (self.last_refresh_t is None
               or t_global - self.last_refresh_t >= self.f_update)
        if not due:
            return
        self.last_refresh_t = t_global
        stats = {"t_env": self.t_env}
        if self.recent_wins:
            stats["rolling_win_rate"] = float(np.mean(self.recent_wins))
        summary = self.iface.summary(snap, stats)
        key = self.iface.cache_key(snap)
        hits_before = getattr(self.commander, "n_cache_hits", 0)
        guidance = self.commander(summary, key, self.iface)
        if guidance is not None:
            self.state.guidance = guidance
        if self._guidance_log is not None:
            self._guidance_log.write(json.dumps({
                "t_env": self.t_env, "t_global": t_global, "cache_key": key,
                "cache_hit": getattr(self.commander, "n_cache_hits", 0) > hits_before,
                "phase": snap.get("phase"), "guidance": guidance}) + "\n")
            self._guidance_log.flush()

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0        # environment reward only (comparable)
        shaped_return = 0.0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            snap_pre = self.iface.snapshot()
            self._maybe_refresh_commander(snap_pre, test_mode)

            guidance = self.state.guidance
            mask_active = self.use_masking and guidance is not None \
                and (not test_mode or self.mask_at_test) \
                and (self.mask_anneal_t <= 0 or self.t_env < self.mask_anneal_t)
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

            stored_reward = reward
            f_t = 0.0
            if self.use_shaping and guidance is not None and not test_mode:
                snap_post = self.iface.snapshot()
                f_t = compute_shaping(guidance.get("subgoals"), snap_pre,
                                      snap_post, actions[0].tolist(),
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
            if self.commander is not None:
                for k, v in self.commander.stats().items():
                    self.logger.log_stat(k, v, self.t_env)
            if hasattr(self.mac, "pop_mask_stats"):
                for k, v in self.mac.pop_mask_stats().items():
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
