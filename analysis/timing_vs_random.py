"""Q-001 sparse-transition redesign: are CUSUM firing times better aligned with
real tactical events than matched-budget random / periodic firings?

Ground truth is deliberately SPARSE (the earlier protocol used raw cache_key
changes, which fire on ~60% of steps and make any schedule look aligned):
  L1 = own-force state change (ally alive count OR ally HP bucket), reset steps removed
  L2 = ally alive count change only (a marine died), reset steps removed
Both are sight-independent, so enemy-visibility flicker cannot inflate them.

Baselines are drawn from the SAME eligible pool (armed AND since >= min_interval),
so this isolates the statistic's contribution from the gating.
"""
import json, sys, numpy as np, glob, os

MIN_INTERVAL = 10
WINDOWS = [2, 5, 10]
NBOOT = 200

def ally_fields(key):
    for part in key.split('|'):
        if part.startswith('A:'):
            f = part.split(':')
            return (f[2], f[3])          # alive, hp bucket
    return None

def load(stem):
    tr = [json.loads(l) for l in open(f'results/trace/{stem}.jsonl')]
    ph = [json.loads(l) for l in open(f'results/phase/{stem}.jsonl')]
    return tr, ph

def events(ph, tep):
    """Return (L1, L2) sparse event step lists."""
    l1, l2, prev = [], [], None
    for r in ph:
        cur = ally_fields(r['key'])
        t = r['t_global']
        if cur is None:
            continue
        if prev is not None and cur != prev:
            if tep.get(t, 99) > 1:            # drop episode-reset transitions
                l1.append(t)
                if cur[0] != prev[0]:
                    l2.append(t)
        prev = cur
    return np.array(l1), np.array(l2)

def hit_rate(fire, ev, w):
    if len(fire) == 0 or len(ev) == 0:
        return float('nan')
    idx = np.searchsorted(ev, fire)
    d = np.full(len(fire), np.inf)
    left = np.clip(idx - 1, 0, len(ev) - 1)
    right = np.clip(idx, 0, len(ev) - 1)
    d = np.minimum(np.abs(fire - ev[left]), np.abs(fire - ev[right]))
    return float((d <= w).mean())

def report(stem):
    tr, ph = load(stem)
    tep = {r['t']: r['t_ep'] for r in tr}
    l1, l2 = events(ph, tep)
    cusum   = np.array([r['t'] for r in tr if r['due'] and r['early']])
    expiry  = np.array([r['t'] for r in tr if r['due'] and not r['early']])
    elig    = np.array([r['t'] for r in tr
                        if r['armed'] and (r['since'] or 0) >= MIN_INTERVAL])
    print(f"\n=== {stem}")
    print(f"    steps={len(tr)}  eligible={len(elig)}  CUSUM fires={len(cusum)} "
          f"expiry fires={len(expiry)}  L1 events={len(l1)} ({100*len(l1)/len(tr):.1f}% of steps) "
          f"L2 events={len(l2)} ({100*len(l2)/len(tr):.1f}%)")
    if len(cusum) == 0 or len(elig) < len(cusum):
        print("    (insufficient data)"); return
    rng = np.random.default_rng(0)
    for name, ev in (("L1 own-force change", l1), ("L2 ally death", l2)):
        if len(ev) == 0:
            continue
        print(f"    --- {name}")
        for w in WINDOWS:
            c = hit_rate(cusum, ev, w)
            e = hit_rate(expiry, ev, w) if len(expiry) else float('nan')
            boot = [hit_rate(rng.choice(elig, size=len(cusum), replace=False), ev, w)
                    for _ in range(NBOOT)]
            per = np.linspace(elig[0], elig[-1], len(cusum))
            per = elig[np.searchsorted(elig, per).clip(0, len(elig)-1)]
            p = hit_rate(per, ev, w)
            mu, sd = float(np.mean(boot)), float(np.std(boot))
            z = (c - mu) / sd if sd > 0 else float('nan')
            print(f"      w=+-{w:<3d} CUSUM={c:.3f}  random(matched)={mu:.3f}+-{sd:.3f}  "
                  f"periodic={p:.3f}  expiry={e:.3f}   z={z:+.2f}")

for f in sorted(glob.glob('results/trace/*.jsonl')):
    report(os.path.basename(f)[:-6])
