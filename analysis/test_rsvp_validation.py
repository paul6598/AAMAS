import gzip
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algorithm.rsvp.validation import ValidationLog, RefreshTrial, discounted_targets
from algorithm.rsvp.critic import ValueCritic
from algorithm.rsvp.runner import RSVPRunner
from analysis.summarize_rsvp_validation import summarize, Scores


class ValidationTests(unittest.TestCase):
    def test_prequential_targets_baseline_rng_and_roundtrip(self):
        critic = ValueCritic(1, 1)
        critic.mu = torch.tensor([7.])
        state = torch.get_rng_state().clone()
        npstate = np.random.get_state()
        pyst = random.getstate()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'audit.gz'
            args = SimpleNamespace(seed=1, rsvp_validation_every=1, rsvp_refresh_trial=False)
            log = ValidationLog(path, args)
            log.begin(0)
            for t, f in enumerate([1., 2.]):
                r = log.capture(critic, [t], [('enemy_kill', None)],
                    {'subgoals': [{'predicate': 'enemy_kill', 'weight': .5}]},
                    [(0, .5)], {}, t, t, 'key')
                log.observe(r, [f], 3., .5*f)
            log.finish(0, [('enemy_kill', None)], .8, {'battle_won': True})
            log.close()
            with gzip.open(path, 'rt') as f:
                meta, ep = [json.loads(line) for line in f]
            self.assertAlmostEqual(ep['records'][0]['target'][0], 2.6)
            self.assertEqual(ep['records'][1]['target'], [2.])
            self.assertEqual(ep['records'][0]['baseline'], [7.])
            self.assertEqual(critic.X, [])  # logger did not train or insert episode
            self.assertEqual(summarize(path)['heads'][0]['all']['n'], 2)
            self.assertEqual(summarize(path, start=1)['heads'][0]['all']['n'], 1)
            p = np.array([r['prediction'][0] for r in ep['records']])
            y = np.array([r['target'][0] for r in ep['records']])
            b = np.array([r['baseline'][0] for r in ep['records']])
            expected = Scores()
            expected.add(p, y, b)
            self.assertEqual(summarize(path)['heads'][0]['all'], expected.result())
            self.assertEqual(summarize(path, trials_only=True)['heads'][0]['all']['n'], 0)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertTrue(np.array_equal(npstate[1], np.random.get_state()[1]))
        self.assertEqual(pyst, random.getstate())

    def test_constant_target_is_not_perfect_r2(self):
        score = Scores()
        score.add([0, 0], [0, 0], [0, 0])
        self.assertIsNone(score.result()['r2'])
        self.assertIsNone(score.result()['baseline_skill'])
        score = Scores()
        score.add([1, 3], [1, 3], [2, 2])
        self.assertEqual(score.result()['r2'], 1.)
        self.assertEqual(score.result()['baseline_skill'], 1.)

    def test_signed_targets_and_episode_boundary(self):
        np.testing.assert_allclose(discounted_targets([[1, -2], [2, -1]], .5), [[2, -2.5], [2, -1]])
        np.testing.assert_allclose(discounted_targets([[3]], .5), [[3]])

    def runner(self, trial=None):
        r = object.__new__(RSVPRunner)
        class Commander:
            calls = 0
            def __call__(self, *args):
                self.calls += 1
                return {'subgoals': [{'predicate': 'enemy_kill', 'weight': 1}], 'action_rules': []}
        r.commander = Commander()
        r.args = SimpleNamespace(sched_fail_retry=5, sched_gate_interval=20)
        r._guidance_needed = lambda test: not test
        r._validation = SimpleNamespace(trial=trial) if trial else None
        r.t_env, r.t, r.last_refresh_t = 0, 10, 0
        r.f_max, r.scheduler, r._episodes_seen, r.warmup_eps = 200, 'vf', 30, 20
        r._v_ref, r._ref_heads, r._x_ref = 1., [(0, 1.)], [1.]
        r._vF = lambda x, heads: float(x[0])
        r._S, r.k, r.h, r.min_interval = 3., .6, 3., 10
        r._log_v, r._trace, r._guidance_log = [], None, None
        r.include_training_stats = False
        r.iface = SimpleNamespace(summary=lambda *a: '', cache_key=lambda *a: 'key')
        r.state = SimpleNamespace(guidance={'old': True})
        r._log_refresh = r._log_early = r._ep_early = 0
        r.critic = SimpleNamespace(mu=torch.ones(1))
        r._guidance_heads = lambda g: [(0, 1.)]
        r.eps_frac = .25
        r._why = dict(warmup=0, no_heads=0, low_vref=0, ok=0)
        return r

    def test_trial_hold_locks_ceiling_and_forces_common_end_refresh(self):
        trial = RefreshTrial(0, window=2, control_probability=0, start_t=0)
        trial.rng = SimpleNamespace(random=lambda: .9)
        trial.begin_episode(0)
        r = self.runner(trial)
        r._maybe_refresh({}, [0.], False)
        self.assertEqual(r.commander.calls, 1)  # candidate generated in both arms
        self.assertTrue(r._audit_decision['candidate_early'])
        self.assertFalse(r._audit_decision['early'])
        self.assertTrue(trial.locked)
        self.assertEqual(r.state.guidance, {'old': True})
        r.t = 20
        r._maybe_refresh({}, [0.], False)
        self.assertEqual(r.commander.calls, 1)
        r.t = 200
        r._maybe_refresh({}, [0.], False)
        self.assertEqual(r.commander.calls, 1)
        self.assertFalse(r._audit_decision['refresh_success'])
        self.assertIsNone(trial.finish_episode(9., {'battle_won': True}))
        trial.begin_episode(1)
        self.assertIsNone(trial.finish_episode(1., {'battle_won': False}))
        trial.begin_episode(2)
        completed = trial.finish_episode(3., {'battle_won': True})
        self.assertEqual(completed['outcome_return_mean'], 2.)
        self.assertEqual(completed['outcome_win_rate'], .5)
        self.assertTrue(trial.force_refresh)
        r.t = 220
        r._maybe_refresh({}, [0.], False)
        self.assertEqual(r.commander.calls, 2)
        self.assertFalse(trial.force_refresh)

    def test_trial_can_probe_nontrigger_and_apply_candidate(self):
        trial = RefreshTrial(0, control_probability=1, start_t=0)
        trial.rng = SimpleNamespace(random=lambda: .1)
        trial.begin_episode(0)
        r = self.runner(trial)
        r._S = 0.
        r._maybe_refresh({}, [1.], False)
        self.assertEqual(r.commander.calls, 1)
        self.assertEqual(trial.block['kind'], 'control')
        self.assertTrue(trial.block['applied'])
        self.assertEqual(r.state.guidance['subgoals'][0]['predicate'], 'enemy_kill')

    def test_validation_writes_post_trigger_block_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'trial.gz'
            args = SimpleNamespace(seed=2, rsvp_validation_every=1,
                rsvp_refresh_trial=True, rsvp_trial_probability=.5,
                rsvp_trial_window_episodes=2, rsvp_trial_control_probability=0,
                rsvp_trial_max_per_kind=2, rsvp_trial_start_t=0,
                use_action_masking=False, scheduler='vf')
            log = ValidationLog(path, args)
            critic = SimpleNamespace(predict=lambda x: np.array([1.]),
                mu=np.array([0.]), trusted=lambda i: True)
            guidance = {'subgoals': [{'predicate': 'enemy_kill', 'weight': 1}],
                        'action_rules': []}

            def episode(ep, ret):
                log.begin(ep)
                record = log.capture(critic, [0.], [('enemy_kill', None)],
                    guidance, [(0, 1.)], {}, ep*10, 0, 'key')
                log.observe(record, [1.], ret, 1.)
                return log.finish(ep, [('enemy_kill', None)], .8,
                                  {'battle_won': ret > 0}, episode_return=ret)

            log.begin(0)
            block = log.trial.propose(True, True, 10, 10, 0, 0, {}, guidance)
            log.trial.candidate_result(guidance)
            record = log.capture(critic, [0.], [('enemy_kill', None)],
                guidance, [(0, 1.)], {}, 0, 0, 'key')
            log.observe(record, [1.], 9., 1.)
            log.finish(0, [('enemy_kill', None)], .8,
                       {'battle_won': True}, episode_return=9.)
            episode(1, 1.)
            metrics = episode(2, 3.)
            self.assertEqual(metrics['validation_trial_return'], 2.)
            log.close()
            with gzip.open(path, 'rt') as f:
                rows = [json.loads(line) for line in f]
            completed = [r['trial_completed'] for r in rows
                         if r.get('trial_completed')]
            self.assertEqual(len(completed), 1)
            self.assertEqual([x['episode'] for x in completed[0]['outcomes']], [1, 2])
            self.assertNotIn(0, [x['episode'] for x in completed[0]['outcomes']])
            trial_summary = summarize(path, trials_only=True)['trial']
            self.assertEqual(trial_summary['completed_blocks'], 1)
            arm_name = 'refresh' if completed[0]['refresh'] else 'hold'
            self.assertEqual(trial_summary['trigger'][arm_name]['return_mean'], 2.)
            self.assertIsNone(trial_summary['trigger']['apply_minus_hold'])

    def test_passive_mode_does_not_change_scheduler(self):
        a, b = self.runner(), self.runner()
        b._validation = SimpleNamespace(trial=None)
        a._maybe_refresh({}, [0.], False)
        b._maybe_refresh({}, [0.], False)
        for key in ['_S', 'last_refresh_t', '_v_ref', '_log_refresh', '_log_early']:
            self.assertEqual(getattr(a, key), getattr(b, key))
        self.assertEqual(a.commander.calls, b.commander.calls)

    def test_gate_timer_uses_delay_not_cusum(self):
        r = self.runner()
        r.scheduler = 'gate_timer'
        r.args = SimpleNamespace(sched_gate_interval=20)
        r._maybe_refresh({}, [0.], False)
        self.assertEqual(r.commander.calls, 0)  # S already exceeds h at t=10
        r.t = 20
        r._maybe_refresh({}, [0.], False)
        self.assertEqual(r.commander.calls, 1)
        self.assertIsNone(r._v_ref)  # newly issued at low value, gate closes
        r.t = 40
        r._maybe_refresh({}, [0.], False)
        self.assertEqual(r.commander.calls, 1)
        r.t = 220
        r._maybe_refresh({}, [0.], False)
        self.assertEqual(r.commander.calls, 2)

    def test_trial_requires_shaping_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(seed=1, rsvp_validation_every=1, rsvp_refresh_trial=True,
                use_action_masking=True, scheduler='vf', soft_guidance_only=True)
            with self.assertRaises(ValueError):
                ValidationLog(Path(tmp)/'x.gz', args)

    def test_real_runner_logging_before_update_and_passive_equivalence(self):
        def rollout(path=None):
            torch.manual_seed(99)
            r = self.runner()
            r.t = 0
            r.args = SimpleNamespace(env='sc2', n_agents=1, critic_iters=2,
                seed=0, rsvp_validation_every=1, rsvp_refresh_trial=False,
                runner_log_interval=100000, test_nepisode=32)
            r.batch_size = 1
            r.critic = ValueCritic(1, 1)
            r._validation = ValidationLog(path, r.args) if path else None
            r.lib = [('enemy_kill', None)]
            r.fx = lambda snap: np.array([r.t], dtype=np.float32)
            r._lazy_init = lambda snap: None
            r._maybe_refresh = lambda *args: None
            r._shaping_active = lambda *args: True
            r.state = SimpleNamespace(guidance={'subgoals': [{'predicate': 'enemy_kill', 'weight': .5}]}, lambda_val=.5)
            r.iface = SimpleNamespace(snapshot=lambda: {}, cache_key=lambda s: 'key')
            step_count = [0]
            def step(actions):
                step_count[0] += 1
                return float(step_count[0]), step_count[0]==2, {'battle_won': True}
            r.env = SimpleNamespace(get_state=lambda: [0], get_obs=lambda: [[0]],
                get_avail_actions=lambda: [[1]], step=step)
            r.mac = SimpleNamespace(init_hidden=lambda **kw: None,
                set_guidance=lambda *a: None, select_actions=lambda *a,**kw: torch.tensor([[0]]))
            class Batch:
                def __init__(self): self.data = []
                def update(self, data, ts): self.data.append((ts, data))
            r.reset = lambda: setattr(r, 'batch', Batch())
            r.use_masking = False
            r.shaping_in_learner = True
            r._ep_X, r._ep_F = [], []
            r.train_stats, r.test_stats = {}, {}
            r.train_returns, r.test_returns, r.recent_wins, r._shaping_sums = [], [], [], []
            r._episodes_since_log = 0
            r._ep_armed, r.target_early = False, 0
            r._reward_diag = dict.fromkeys(['n','env_abs','env_sq','env_nonzero','f_abs','f_sq',
                'f_nonzero','f_at_clip','lambda_f_abs'], 0.)
            r.log_train_stats_t = 0
            r.logger = SimpleNamespace(log_stat=lambda *a: None)
            order = []
            if r._validation:
                original_finish = r._validation.finish
                def finish(*args, **kwargs):
                    self.assertEqual(r.critic.X, [])
                    order.append('validate')
                    return original_finish(*args, **kwargs)
                r._validation.finish = finish
            original_add = r.critic.add_episode
            def add(*args):
                order.append('insert')
                return original_add(*args)
            r.critic.add_episode = add
            with patch('algorithm.rsvp.runner.predlib.f_vector', side_effect=lambda *a: np.array([step_count[0]])), \
                 patch('algorithm.rsvp.runner.predlib.shaping', return_value=.5):
                batch = r.run(False)
            if r._validation:
                r._validation.close()
            return r, order, torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as tmp:
            a, _, rng_a = rollout()
            b, order, rng_b = rollout(Path(tmp)/'integration.gz')
            self.assertEqual(order, ['validate', 'insert'])
            self.assertEqual(a.train_returns, b.train_returns)
            self.assertTrue(torch.equal(rng_a, rng_b))
            for p,q in zip(a.critic.net.parameters(), b.critic.net.parameters()):
                self.assertTrue(torch.equal(p,q))


if __name__ == '__main__':
    unittest.main()
