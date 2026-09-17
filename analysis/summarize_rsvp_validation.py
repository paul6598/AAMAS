"""Summarize prequential head accuracy and randomized trial ITT by run.

Usage: python analysis/summarize_rsvp_validation.py results/validation/*.gz
Never pool seeds as independent timesteps for uncertainty statements.
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np


class Scores:
    def __init__(self):
        self.n = 0
        self.sum_y = self.sum_y2 = self.sse = self.sae = self.baseline_sse = 0.

    def add(self, prediction, target, baseline):
        p, y, b = map(lambda a: np.asarray(a, dtype=float), (prediction, target, baseline))
        self.n += y.size
        self.sum_y += y.sum()
        self.sum_y2 += (y*y).sum()
        self.sse += ((p-y)**2).sum()
        self.sae += np.abs(p-y).sum()
        self.baseline_sse += ((b-y)**2).sum()

    def result(self):
        if not self.n:
            return {'n': 0, 'mae': None, 'r2': None, 'baseline_skill': None}
        variance = self.sum_y2-self.sum_y**2/self.n
        return dict(n=self.n, mae=self.sae/self.n, rmse=float(np.sqrt(self.sse/self.n)),
                    r2=1-self.sse/variance if variance > 1e-10 else None,
                    baseline_skill=1-self.sse/self.baseline_sse if self.baseline_sse > 1e-10 else None)


def summarize(path, start=0, stop=float('inf'), trials_only=False):
    all_heads = trusted_heads = None
    weighted, selected = Scores(), Scores()
    episodes, trials, metadata = 0, [], None
    partial = False
    with gzip.open(path, 'rt') as f:
        try:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    partial = True
                    break
                if row['kind'] == 'metadata':
                    metadata = row
                    all_heads = [Scores() for _ in row['library']]
                    trusted_heads = [Scores() for _ in row['library']]
                    continue
                if metadata is None:
                    raise ValueError('missing metadata')
                episodes += 1
                records = [] if trials_only else [
                    r for r in row['records'] if start <= r['t'] < stop]
                # episode 단위 배열로 합산해 긴 검증 로그도 동일한 점수를 빠르게 계산한다.
                if records:
                    p, y, b = (np.asarray([r[k] for r in records], dtype=float)
                               for k in ['prediction', 'target', 'baseline'])
                    trusted = np.asarray([r['trusted'] for r in records], dtype=bool)
                    for i, score in enumerate(all_heads):
                        score.add(p[:, i], y[:, i], b[:, i])
                        mask = trusted[:, i]
                        trusted_heads[i].add(p[mask, i], y[mask, i], b[mask, i])
                    for key, score in [('weights', weighted), ('selected_weights', selected)]:
                        w = np.asarray([r[key] for r in records], dtype=float)
                        mask = np.any(w != 0, axis=1)
                        score.add(*(np.einsum('ij,ij->i', w[mask], a[mask])
                                    for a in [p, y, b]))
                trial = row.get('trial_completed') or row.get('trial')
                if trial and start <= trial['t'] < stop:
                    trials.append(trial)
        except EOFError:
            partial = True
    result = dict(file=str(path), metadata=metadata, episodes_read=episodes,
                  start=start, stop=stop if np.isfinite(stop) else None, partial_file=partial,
                  trials_only=trials_only,
                  heads=[dict(head=h, all=a.result(), trusted=t.result())
                         for h,a,t in zip(metadata['library'], all_heads, trusted_heads)] if metadata else [],
                  guidance_weighted=weighted.result(), selected_weighted=selected.result())
    result['trial'] = {'completed_blocks': len(trials), 'note':
        'block-level proximal ITT; trigger episode excluded and all refreshes locked during the following outcome window'}
    if trials:
        def outcome(r):
            return r.get('outcome_return_mean', r.get('return_after'))
        contributions = [outcome(r)*(1/r['probability'] if r['refresh']
                         else -1/(1-r['probability'])) for r in trials]
        result['trial']['return_itt_ipw'] = float(np.mean(contributions))
        for kind in ['all', 'trigger', 'control']:
            subset = trials if kind == 'all' else [r for r in trials if r.get('kind') == kind]
            summary = {'n': len(subset)}
            if subset:
                cs = [outcome(r)*(1/r['probability'] if r['refresh']
                      else -1/(1-r['probability'])) for r in subset]
                summary['return_itt_ipw'] = float(np.mean(cs))
            for arm in [True, False]:
                rs = [r for r in subset if r['refresh'] == arm]
                wins = [r.get('outcome_win_rate', r.get('battle_won')) for r in rs]
                wins = [x for x in wins if x is not None]
                summary['refresh' if arm else 'hold'] = dict(
                    n=len(rs), return_mean=float(np.mean([outcome(r) for r in rs])) if rs else None,
                    win_mean=float(np.mean(wins)) if wins else None,
                    candidate_successes=sum(bool(r.get('candidate_success', True)) for r in rs))
            refresh_mean, hold_mean = (summary[k]['return_mean'] for k in ['refresh', 'hold'])
            summary['apply_minus_hold'] = refresh_mean - hold_mean \
                if refresh_mean is not None and hold_mean is not None else None
            result['trial'][kind] = summary
        trigger, control = (result['trial'][k]['apply_minus_hold'] for k in ['trigger', 'control'])
        result['trial']['timing_selectivity_gain'] = trigger - control \
            if trigger is not None and control is not None else None
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+', type=Path)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int, default=None)
    parser.add_argument('--trials-only', action='store_true',
                        help='Skip prediction scores when only completed MRT blocks are needed')
    args = parser.parse_args()
    print(json.dumps([summarize(p, args.start, args.stop or float('inf'), args.trials_only)
                      for p in args.paths], indent=2, allow_nan=False))
