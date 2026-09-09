"""Read-only, offline audit of the completed lam40 round and matched QMIX runs.

Run from repository root with the aamas Python environment. Uses local W&B
history (not console's five-point moving averages), cross-checks Sacred, and
prints reproducible JSON. Does not contact W&B or modify experiment artifacts.
Final follows the existing convention t >= 0.9 * last evaluation t, not t_max.
Std is population std across seeds; neither std nor evaluation points are a CI.
"""
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yaml
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


IDS = [140, 142, 153, *range(157, 166), *range(170, 199)]


def scalar(value):
    return float(value['value'] if isinstance(value, dict) else value)


def history(path):
    ds = DataStore()
    ds.open_for_scan(str(next(path.glob('*.wandb'))))
    values = defaultdict(list)
    while True:
        data = ds.scan_data()
        if data is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if not record.HasField('history'):
            continue
        row = {('.'.join(v.nested_key) if v.nested_key else v.key):
               json.loads(v.value_json) for v in record.history.item}
        if 't_env' not in row:
            continue
        for key, value in row.items():
            if key.startswith(('test/', 'train/')):
                values[key].append((row['t_env'], value))
    return values


def metric(values, key):
    points = np.array(values[key], dtype=float)
    ts, ys = points.T
    return dict(mean=float(ys.mean()),
                final=float(ys[ts >= .9 * ts[-1]].mean()),
                n=len(ts), last_t=int(ts[-1]), last=float(ys[-1]))


def main():
    runs = []
    for sid in IDS:
        root = Path('results/sacred') / str(sid)
        config = json.loads((root / 'config.json').read_text())
        info = json.loads((root / 'info.json').read_text())
        meta = json.loads((root / 'run.json').read_text())
        local_start = datetime.fromisoformat(meta['start_time']) + timedelta(hours=9)
        candidates = []
        for wp in Path('wandb').glob('run-*'):
            try:
                stamp = datetime.strptime(wp.name[4:19], '%Y%m%d_%H%M%S')
            except ValueError:
                continue
            if abs((stamp - local_start).total_seconds()) > 120:
                continue
            wc = {k: v['value'] for k, v in
                  yaml.safe_load((wp / 'files/config.yaml').read_text()).items()}
            expected_env = dict(config['env_args'], seed=config['seed'])
            if (wc['env_args'] == expected_env and
                all(wc.get(k) == config.get(k) for k in
                    ['seed', 'wandb_run', 'env', 't_max'])):
                candidates.append(wp)
        assert len(candidates) == 1, (sid, candidates)
        wp = candidates[0]
        values = history(wp)
        key = 'test/battle_won_mean'
        sacred_pairs = list(zip(info['test_battle_won_mean_T'],
                                info['test_battle_won_mean']))
        assert sacred_pairs == values[key], (sid, 'Sacred / W&B mismatch')
        win = metric(values, key)
        assert win['last_t'] >= .98 * config['t_max'], sid
        row = dict(sacred=sid, run=wp.name.rsplit('-', 1)[-1],
                   map=config['env_args'].get('map_name', config['env']),
                   group=config['wandb_run'], seed=config['seed'],
                   t_max=config['t_max'], win=win,
                   test_return=metric(values, 'test/return_mean'),
                   test_length=metric(values, 'test/ep_length_mean'))
        if config.get('scheduler'):
            assert config['lambda_floor_frac'] == .4
            assert not config['use_action_masking']
            assert not config['use_masking_at_test']
            guidance_files = []
            for gp in Path('results/guidance').glob(
                    f"vigil__*_default_s{config['seed']}_*.jsonl"):
                stamp = datetime.strptime(gp.name[7:26], '%Y-%m-%d_%H-%M-%S')
                if abs((stamp - local_start).total_seconds()) <= 3:
                    ev = [json.loads(line) for line in gp.read_text().splitlines()]
                    log_t, hits = values['train/llm_cache_hits'][-1]
                    seen = [e for e in ev if e['t_global'] < log_t]
                    if (ev[0]['cache_key'].split('|')[0] == row['map'] and
                        sum(e['cache_hit'] for e in seen) == hits):
                        guidance_files.append(gp)
            assert len(guidance_files) == 1, (sid, guidance_files)
            events = [json.loads(line) for line in
                      guidance_files[0].read_text().splitlines()]
            gates = {k: sum(scalar(v) for _, v in values['train/sched_ref_' + k])
                     for k in ['ok', 'no_heads', 'low_vref', 'warmup']}
            total_gate = sum(gates.values())
            row['schedule'] = dict(
                refresh=len(events), early=sum(e['early'] for e in events),
                cache_hits=sum(e['cache_hit'] for e in events),
                api_calls_last_logged=values['train/llm_calls'][-1][1],
                api_failures_last_logged=values['train/llm_failures'][-1][1],
                refresh_per_M=len(events) / (config['t_max'] / 1e6),
                gate_counts=gates,
                armed_spell_fraction=gates['ok'] / total_gate if total_gate else None,
                last_h=values['train/sched_h_current'][-1][1])
        runs.append(row)
    groups = defaultdict(list)
    for row in runs:
        groups[(row['map'], row['group'])].append(row)
    aggregate = []
    for (map_name, group), rows in sorted(groups.items()):
        def stats(key, field):
            vals = [r[key][field] for r in rows]
            return [float(np.mean(vals)), float(np.std(vals))]
        aggregate.append(dict(map=map_name, group=group,
                              seeds=sorted(r['seed'] for r in rows),
                              auc=stats('win', 'mean'), final=stats('win', 'final'),
                              return_final=stats('test_return', 'final')))
    print(json.dumps(dict(runs=runs, aggregate=aggregate), indent=2))


if __name__ == '__main__':
    main()
