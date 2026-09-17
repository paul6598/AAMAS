"""Offline trace audit. No training/API calls; phase keys are coarse proxies.

Example: python analysis/audit_rsvp_mechanism.py > /tmp/rsvp_audit.json
Critic targets and counterfactual returns are not present in these traces.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
RUNS = {337: ('*02-31-19*', 1000000), 336: ('*02-20-19*', 1000000),
        346: ('*18-52-21*', 400000), 347: ('*18-52-22*', 400000),
        325: ('*19-12-30*', 1000000), 324: ('*19-12-20*', 1000000)}


def rows(path):
    with path.open() as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A concurrently written final partial line is excluded.
                continue


def state(key):
    parts = (key or '').split('|')
    a = sum(int(x.split(':')[2]) for x in parts if x.startswith('A:'))
    e = sum(int(x.split(':')[2]) for x in parts if x.startswith('E:'))
    return a, e


def vector(g):
    # Aggregate exact predicate/type duplicates as compute_shaping does.
    out = Counter()
    for sg in (g or {}).get('subgoals', []):
        key = (sg['predicate'], str(sg.get('unit_type', '')).strip().casefold())
        out[key] += min(1., max(0., float(sg['weight'])))
    return out


def quantile(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[int((len(xs)-1)*q)]


def audit(run_id, pattern, limit):
    path = next((ROOT/'results/trace').glob(pattern))
    phase_path = ROOT/'results/phase'/path.name
    guidance = list(rows(ROOT/'results/guidance'/path.name))
    guidance = [g for g in guidance if g['t_global'] < limit]
    by_t = {g['t_global']: g for g in guidance}
    phases = iter(rows(phase_path))
    pending = next(phases, None)
    key = None
    stats = defaultdict(Counter)
    ages, matched = [], defaultdict(lambda: [0, 0, 0, 0])
    next_guidance = {a['t_global']: b for a,b in zip(guidance, guidance[1:])}
    spell_gaps = defaultdict(list)
    spell_end = defaultdict(Counter)
    episode = []

    def finish_episode():
        if not episode:
            return
        end = episode[-1][0]['t_ep'] + 1
        for r, a, e in episode:
            # Non-trigger comparison restricted to actually eligible decisions.
            eligible = r['armed'] and r['v'] is not None and (r['since'] or 0) >= 10
            if not eligible or r['t_ep'] == 0:
                continue
            event = r['early']
            if r['due'] and not event:
                continue
            terminal = int(end - r['t_ep'] <= 5)
            label = 'early_context' if event else 'eligible_control'
            for b in ['all', str(r['t']//200000)]:
                s = stats[b]
                s[label+'_n'] += 1
                s[label+'_within5end'] += terminal
                s[label+'_alone'] += int(a <= 1)
                s[label+'_no_visible'] += int(e == 0)
            stratum = (r['t']//200000, a, bool(e), r['since']//25, r['t_ep']//10)
            m = matched[stratum]
            idx = 0 if event else 2
            m[idx] += 1
            m[idx+1] += terminal

    last_t = None
    for r in rows(path):
        t = r['t']
        if t >= limit:
            # Don't call a censored episode complete.
            if r['t_ep'] == 0:
                finish_episode()
                episode = []
            break
        if episode and r['ep'] != episode[-1][0]['ep']:
            finish_episode()
            episode = []
        while pending and pending['t_global'] < t:
            key = pending['key']
            pending = next(phases, None)
        current_key = by_t[t]['cache_key'] if t in by_t else key
        a, e = state(current_key)
        episode.append((r, a, e))
        last_t = t
        if t-1 in by_t and t-1 in next_guidance:
            following = next_guidance[t-1]
            label = 'armed_after_issuance' if r['armed'] else 'unarmed_after_issuance'
            spell_gaps[label].append(following['t_global']-(t-1))
            spell_end[label]['n'] += 1
            spell_end[label]['next_early'] += int(following['early'])
        for b in ['all', str(t//200000)]:
            s = stats[b]
            s['steps'] += 1
            s['armed_steps'] += int(r['armed'])
            s['episodes_started'] += int(r['t_ep'] == 0)
            if r['v'] is not None:
                s['ratio_steps'] += 1
                s['ratio_zero'] += int(r['v'] == 0)
                s['negative_den'] += int(r['den_raw'] <= 0)
                s['v_sum'] += r['v']
            if t in by_t:
                s['refresh'] += 1
                s['early'] += int(r['early'])
            if r['early']:
                s['early_cross_episode_reference'] += int(r['since'] > r['t_ep'])
                s['early_t_ep_zero'] += int(r['t_ep'] == 0)
                s['early_v_sum'] += r['v']
                s['early_v_not_low'] += int(r['v'] >= .6)
        if t in by_t and r['since'] is not None:
            ages.append(r['since'])
    # EOF can end in a buffered partial episode: do not count it as terminal.
    change = defaultdict(Counter)
    for old, new in zip(guidance, guidance[1:]):
        x, y = vector(old['guidance']), vector(new['guidance'])
        s = change['early' if new['early'] else 'ceiling']
        s['n'] += 1
        s['coefficient_changed'] += int(x != y)
        s['coefficient_l1_sum'] += sum(abs(x[k]-y[k]) for k in x.keys() | y.keys())
        s['action_rules_changed'] += int(old['guidance']['action_rules'] != new['guidance']['action_rules'])
        s['alone'] += int(state(new['cache_key'])[0] <= 1)
        s['alone_with_focus'] += int(state(new['cache_key'])[0] <= 1 and
            any(g['predicate']=='focus_fire' for g in new['guidance']['subgoals']))
    overlap = [m for m in matched.values() if m[0] and m[2]]
    n = sum(m[0] for m in overlap)
    match_summary = {'matched_early_n': n,
        'early_within5end': sum(m[1] for m in overlap)/n if n else None,
        'control_within5end_reweighted': sum(m[0]*m[3]/m[2] for m in overlap)/n if n else None,
        'note': 'coarse training-bin/alive/visibility/age/episode-step matching; descriptive, not causal'}
    info = json.loads((ROOT/f'results/sacred/{run_id}/info.json').read_text())
    why = {k: sum(v for t,v in zip(info[k+'_T'], vals) if t < limit)
           for k,vals in info.items() if k.startswith('sched_ref_') and not k.endswith('_T')}
    result = {'run': run_id, 'trace': str(path.relative_to(ROOT)), 'limit': limit,
        'last_t': last_t, 'stats': dict(stats), 'matched_terminal_proxy': match_summary,
        'guidance_change': dict(change), 'issuance_reasons_logged': why,
        'spell_outcomes': {k: {**dict(v), 'age_median': quantile(spell_gaps[k], .5),
            'age_p90': quantile(spell_gaps[k], .9),
            'age_mean': statistics.mean(spell_gaps[k])} for k,v in spell_end.items()},
        'age_median': quantile(ages, .5), 'age_p90': quantile(ages, .9)}
    return result


if __name__ == '__main__':
    results = []
    for n, (pattern, limit) in RUNS.items():
        results.append(audit(n, pattern, limit))
    print(json.dumps({'notes': ['No held-out targets, per-step rewards or critic weights in trace.',
        'Phase rows are post-step; use strictly earlier rows, exclude t_ep=0 from matched contexts.',
        'Matching cannot identify benefits of refresh versus keeping guidance.',
        'Coefficient changes are not measured reward or policy changes.'], 'runs': results}, indent=2))
