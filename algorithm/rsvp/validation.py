"""Passive prequential logging and an opt-in randomized refresh trial.

Neither sampling nor trial assignment consumes the policy/critic RNG streams.
Episode targets are measured before the episode enters the critic's buffer.
"""
import gzip
import copy
import json
import random
from pathlib import Path

import numpy as np


# 에피소드 끝에서 역순으로 할인 접미합 정답을 계산한다.
def discounted_targets(values, gamma):
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(values)
    future = np.zeros(values.shape[1])
    for t in range(len(values)-1, -1, -1):
        future = values[t] + gamma * future
        out[t] = future
    return out


class RefreshTrial:
    """Non-overlapping immediate-vs-delay blocks for shaping-only RSVP.

    A candidate guidance is generated in both arms.  The refresh arm applies it
    immediately; the hold arm keeps the old guidance.  All other refreshes are
    locked until ``window`` *following* training episodes have supplied an
    outcome.  Separate RNG streams keep assignment/probing out of policy and
    critic randomness.
    """
    def __init__(self, seed, probability=.5, window=5,
                 control_probability=.1, max_per_kind=50, start_t=200000):
        if not 0 < probability < 1:
            raise ValueError('trial probability must be strictly between zero and one')
        if window < 1 or max_per_kind < 1:
            raise ValueError('trial window and cap must be positive')
        if not 0 <= control_probability <= 1:
            raise ValueError('control probability must be in [0, 1]')
        self.rng = random.Random(int(seed) + 735191)
        self.probe_rng = random.Random(int(seed) + 915731)
        self.probability = probability
        self.window = int(window)
        self.control_probability = float(control_probability)
        self.max_per_kind = int(max_per_kind)
        self.start_t = int(start_t)
        self.counts = {'trigger': 0, 'control': 0}
        self.block = None
        self.force_refresh = False
        self.episode = None
        self.control_selected = False
        self.control_used = False
        self.started_this_episode = None

    def begin_episode(self, episode):
        self.episode = int(episode)
        self.control_selected = self.probe_rng.random() < self.control_probability
        self.control_used = False
        self.started_this_episode = None

    @property
    def locked(self):
        return self.block is not None

    # 실제 trigger 또는 사전 추출한 non-trigger에서 한 번만 무작위 block을 연다.
    def propose(self, trigger, ready, since, min_interval, t, t_ep, context,
                old_guidance):
        if self.block is not None or self.force_refresh or t < self.start_t:
            return None
        kind = None
        if trigger and self.counts['trigger'] < self.max_per_kind:
            kind = 'trigger'
        elif (not trigger and ready and since is not None and since >= min_interval
              and self.control_selected and not self.control_used
              and self.counts['control'] < self.max_per_kind):
            kind = 'control'
            self.control_used = True
        if kind is None:
            return None
        assignment = dict(kind=kind, t=int(t), t_ep=int(t_ep),
                          trigger_episode=self.episode,
                          probability=self.probability,
                          refresh=self.rng.random() < self.probability,
                          context=copy.deepcopy(context),
                          old_guidance=copy.deepcopy(old_guidance),
                          candidate_success=False, applied=False,
                          outcomes=[])
        self.counts[kind] += 1
        self.block = assignment
        self.started_this_episode = assignment
        return assignment

    def candidate_result(self, guidance):
        if self.block is None:
            return
        self.block['candidate_success'] = guidance is not None
        self.block['new_guidance'] = copy.deepcopy(guidance)
        self.block['applied'] = bool(guidance is not None and self.block['refresh'])

    def finish_episode(self, episode_return, env_info):
        """Add only post-trigger episodes and close a block after ``window``."""
        if self.block is None or self.episode == self.block['trigger_episode']:
            return None
        self.block['outcomes'].append(dict(
            episode=int(self.episode), env_return=float(episode_return),
            battle_won=bool(env_info.get('battle_won', False)),
            episode_limit=bool(env_info.get('episode_limit', False))))
        if len(self.block['outcomes']) < self.window:
            return None
        completed = self.block
        completed['outcome_return_mean'] = float(np.mean(
            [x['env_return'] for x in completed['outcomes']]))
        completed['outcome_win_rate'] = float(np.mean(
            [x['battle_won'] for x in completed['outcomes']]))
        self.block = None
        self.force_refresh = True
        return completed

    def end_refresh_result(self, success):
        if success:
            self.force_refresh = False


class ValidationLog:
    def __init__(self, path, args):
        self.every = int(getattr(args, 'rsvp_validation_every', 0))
        if self.every < 0:
            raise ValueError('rsvp_validation_every must be nonnegative')
        self.trial = RefreshTrial(
            args.seed, getattr(args, 'rsvp_trial_probability', .5),
            getattr(args, 'rsvp_trial_window_episodes', 5),
            getattr(args, 'rsvp_trial_control_probability', .1),
            getattr(args, 'rsvp_trial_max_per_kind', 50),
            getattr(args, 'rsvp_trial_start_t', 200000)) \
            if getattr(args, 'rsvp_refresh_trial', False) else None
        if self.trial and (self.every != 1 or args.use_action_masking
                           or args.scheduler != 'vf'):
            raise ValueError('refresh trial requires validation_every=1, vf and shaping-only guidance')
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.file = gzip.open(path, 'xt', encoding='utf-8')
        self.args = args
        self.records = []
        self.active = False
        self.header_written = False

    def begin(self, episode):
        self.active = self.every > 0 and episode % self.every == 0
        self.records = []
        if self.trial:
            self.trial.begin_episode(episode)

    # 현재 에피소드 학습 전 예측·과거 평균·지침·판단 상태를 저장한다.
    def capture(self, critic, x, lib, guidance, ref_heads, decision, t, t_ep, key):
        if not self.active:
            return None
        from algorithm.rsvp.predlib import head_index
        weights = np.zeros(len(lib))
        for sg in (guidance or {}).get('subgoals', []):
            hi = head_index(lib, sg)
            if hi is not None:
                weights[hi] += min(1., max(0., float(sg.get('weight', .5))))
        selected = np.zeros(len(lib))
        for hi, w in ref_heads or []:
            selected[hi] += w
        return dict(t=t, t_ep=t_ep, state_key=key,
                    prediction=critic.predict(x).tolist(),
                    baseline=critic.mu.tolist(), trusted=[bool(critic.trusted(i)) for i in range(len(lib))],
                    weights=weights.tolist(), selected_weights=selected.tolist(),
                    decision=decision, guidance=guidance, x=np.asarray(x).tolist())

    # 저장한 예측에 실제 전이의 predicate와 외부·보조 보상을 연결한다.
    def observe(self, record, predicates, reward, shaping):
        if record is not None:
            record.update(predicates=np.asarray(predicates).tolist(), reward=float(reward),
                          actual_shaping=float(shaping))
            self.records.append(record)

    # 에피소드 정답과 선택적 개입 결과를 기록하고 검증 요약 통계를 반환한다.
    def finish(self, episode, lib, gamma, env_info, episode_return=None):
        if not self.active or not self.records:
            return
        if not self.header_written:
            keys = ['seed', 'env', 'scheduler', 'f_update', 'sched_gamma', 'sched_h', 'sched_k',
                    't_max', 'lambda_floor_frac', 'beta', 'prompt_style', 'use_action_masking',
                    'use_masking_at_test', 'soft_guidance_only', 'rsvp_validation_every',
                    'rsvp_refresh_trial', 'rsvp_trial_probability',
                    'rsvp_trial_window_episodes', 'rsvp_trial_control_probability',
                    'rsvp_trial_max_per_kind', 'rsvp_trial_start_t', 'wandb_run', 'wandb_group',
                    'env_args', 'deduplicate_subgoals', 'sched_gate_interval', 'shaping_in_learner']
            self.write(dict(kind='metadata', schema=1, library=lib, gamma=gamma,
                            config={k: getattr(self.args, k, None) for k in keys}))
            self.header_written = True
        targets = discounted_targets([r['predicates'] for r in self.records], gamma)
        for r, target in zip(self.records, targets):
            r['target'] = target.tolist()
        started = copy.deepcopy(self.trial.started_this_episode) if self.trial else None
        if started:
            started.pop('outcomes', None)
        completed = self.trial.finish_episode(episode_return, env_info) \
            if self.trial and episode_return is not None else None
        self.write(dict(kind='episode', episode=episode, records=self.records,
                        trial_started=started, trial_completed=completed,
                        episode_limit=bool(env_info.get('episode_limit', False)),
                        battle_won=env_info.get('battle_won')))
        self.file.flush()
        predictions = np.array([r['prediction'] for r in self.records])
        baselines = np.array([r['baseline'] for r in self.records])
        metrics = dict(validation_mae=float(np.abs(predictions-targets).mean()),
                       validation_baseline_mae=float(np.abs(baselines-targets).mean()),
                       validation_target_nonzero_frac=float((np.abs(targets)>1e-8).mean()))
        if completed:
            metrics.update(validation_trial_refresh=int(completed['refresh']),
                           validation_trial_return=completed['outcome_return_mean'],
                           validation_trial_win_rate=completed['outcome_win_rate'])
        self.records = []
        return metrics

    def write(self, value):
        self.file.write(json.dumps(value, allow_nan=False) + '\n')

    def close(self):
        self.file.close()
