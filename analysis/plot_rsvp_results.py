"""Regenerate RSVP-labelled curves from immutable Sacred metrics, not image edits.

Produces an auditable snapshot beside the plots. Restrict IDs to <=235 to
exclude the new cache-off round, short smokes, and the later 2s3z seed expansion.
Completion is a >=98% evaluation-budget filter, NOT Sacred's stale RUNNING flag.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
GROUPS={'qmix':('QMIX','gray'),
 'lehca-shape_F100_lam40':('F100','red'),
 'lehca-shape_F200_lam40':('F200','orange'),
 'vigil_Fmax200_te0.15_lam40':('RSVP Fmax200','tab:blue'),
 'vigil_Fmax600_te0.15_lam40':('RSVP Fmax600','tab:green')}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    selected={}
    for p in sorted((ROOT/'results/sacred').glob('[0-9]*'),key=lambda p:int(p.name)):
        if int(p.name)>235:continue
        try:
            c=json.loads((p/'config.json').read_text());i=json.loads((p/'info.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):continue
        group=c.get('wandb_run','');m=c.get('env_args',{}).get('map_name',c.get('env'))
        if m not in ['5m_vs_6m','3s_vs_5z','2s3z','pursuit'] or c.get('seed') not in range(4):continue
        base=group.removesuffix('_v2')
        if base not in GROUPS:continue
        if m=='pursuit' and base!='qmix' and not group.endswith('_v2'):continue
        if m!='pursuit' and group!=base:continue
        horizon={'5m_vs_6m':1200000,'3s_vs_5z':2000000,'2s3z':1000000,'pursuit':1000000}[m]
        key='test_return_mean' if m=='pursuit' else 'test_battle_won_mean'
        ts=np.asarray(i.get(key+'_T',[]));ys=np.asarray([
            v['value'] if isinstance(v,dict) else v for v in i.get(key,[])],dtype=float)
        if c['t_max']!=horizon or len(ts)<10 or ts[-1]<.98*horizon:continue
        selected[(m,group,c['seed'])]=dict(sacred=int(p.name),group=group,map=m,seed=c['seed'],
            t=ts.tolist(),y=ys.tolist(),horizon=horizon,
            final=float(ys[ts>=.9*horizon].mean()),evaluation_mean=float(ys.mean()),
            time_auc_observed=float(np.trapezoid(ys,ts)/(ts[-1]-ts[0])))
    plots={}
    for m,file in [('5m_vs_6m','fig_5m6m_rsvp.png'),('3s_vs_5z','fig_3s5z_rsvp.png'),
                   ('2s3z','fig_2s3z_rsvp.png'),('pursuit','fig_pursuit_rsvp.png')]:
        fig,ax=plt.subplots(figsize=(8,4.6));metadata=[]
        for group,(label,color) in GROUPS.items():
            target=group+'_v2' if m=='pursuit' and group!='qmix' else group
            rr=[r for (mm,g,s),r in selected.items() if mm==m and g==target]
            if not rr:continue
            start=max(r['t'][0] for r in rr);end=min(r['t'][-1] for r in rr)
            grid=np.linspace(start,end,201)
            yy=np.array([np.interp(grid,r['t'],r['y']) for r in rr])
            mu,sd=yy.mean(0),yy.std(0)
            ax.plot(grid/1e6,mu,color=color,label=f'{label} (n={len(rr)})')
            ax.fill_between(grid/1e6,mu-sd,mu+sd,color=color,alpha=.12)
            metadata.append(dict(group=target,sacred=[r['sacred'] for r in rr],common_range=[start,end]))
        ax.set(title=m+(' (corrected Pursuit v2)' if m=='pursuit' else ' (lam40)'),
               xlabel='Environment steps (M)',ylabel='Test return' if m=='pursuit' else 'Test win rate')
        ax.legend(fontsize=8);ax.grid(alpha=.25)
        fig.text(.5,.01,'Mean ± seed SD; interpolated within common observed range; snapshot 2026-09-07',
                 ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.035,1,1));fig.savefig(a.out/file,dpi=140);plt.close(fig)
        plots[file]=metadata
    with (a.out/'rsvp_plot_snapshot.json').open('x') as f:
        json.dump(dict(utc=datetime.now(timezone.utc).isoformat(),plots=plots,
                       runs=list(selected.values())),f,indent=2)
    print(json.dumps(plots,indent=2))


if __name__=='__main__':main()
