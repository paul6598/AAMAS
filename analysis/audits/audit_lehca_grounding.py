"""Bounded, additive LEHCA grounding diagnostics; never changes training config.

Run from the repository root with aamas Python:
  --mode collect : eight heuristic SMAC episodes, no learner or LLM
  --mode analyze : structural tests and frozen-trajectory reward replay
  --mode llm     : 6 states x 3 existing prompt styles x 2 repeats; cache off

LLM probe: 48 nominal HTTP requests (twostage uses two); hard cap 64 with
retries, serial requests. Outputs are exclusive-create JSON/JSONL artifacts.
The heuristic/synthetic measurements are diagnostics, NOT policy evaluation.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import requests
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from algorithm.lehca.commander.base import sanitize_guidance
from algorithm.lehca.commander.llm_commander import LLMCommander
from algorithm.lehca.shaping.predicates import compute_shaping, evaluate_predicate
from algorithm.src.components.epsilon_schedules import DecayThenFlatSchedule
from algorithm.rsvp.predlib import build_library, f_vector, head_index
from env.semantic.sc2 import SC2SemanticInterface


def save(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def provenance():
    files = ['env/semantic/sc2.py', 'algorithm/lehca/commander/llm_commander.py',
             'algorithm/lehca/commander/base.py', 'algorithm/lehca/shaping/predicates.py',
             'algorithm/rsvp/runner.py', 'analysis/audits/audit_lehca_grounding.py']
    return dict(utc=datetime.now(timezone.utc).isoformat(),
                sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files})


def fake_interface():
    env = SimpleNamespace(map_name='5m_vs_6m', n_agents=5, n_enemies=6)
    return SC2SemanticInterface(env, SimpleNamespace(dt_observable=True))


def sample_state():
    def unit(x, y):
        return dict(type='Marine', hp=45., hp_max=45., x=x, y=y,
                    alive=True, visible=True)
    return dict(allies=[unit(10., 8.+i) for i in range(5)],
                enemies=[unit(16., 7.5+i) for i in range(6)], n_actions=12)


def collect(out):
    from smac.env import StarCraft2Env
    args = yaml.safe_load((ROOT / 'config/envs/sc2.yaml').read_text())['env_args']
    args.update(map_name='5m_vs_6m', heuristic_ai=True, seed=0)
    env = StarCraft2Env(**args)
    iface = SC2SemanticInterface(env, SimpleNamespace(dt_observable=True))
    ends = []
    try:
        with (out / 'transitions.jsonl').open('x') as f:
            for ep in range(8):
                env.reset()
                terminated, t, ret = False, 0, 0.
                while not terminated:
                    pre = iface.snapshot()
                    actions = [0] * 5  # SMAC heuristic overwrites actions in place
                    reward, terminated, info = env.step(actions)
                    post = iface.snapshot()
                    row = dict(ep=ep, t=t, pre=pre, post=post,
                               actions=[int(x) for x in actions], reward=float(reward),
                               summary=iface.summary(pre), cache_key=iface.cache_key(pre))
                    f.write(json.dumps(row) + '\n')
                    t += 1
                    ret += reward
                    if t > 100:
                        raise RuntimeError('Unexpected episode length; stopping diagnostic')
                ends.append(dict(ep=ep, length=t, external_return=float(ret),
                                 won=bool(info.get('battle_won', False))))
                print(ends[-1], flush=True)
        save(out / 'collection.json', dict(provenance=provenance(), episodes=ends,
                                          prompt_context=iface.prompt_context()))
    finally:
        env.close()


def read_transitions(out):
    return [json.loads(s) for s in (out / 'transitions.jsonl').read_text().splitlines()]


def synthetic_tests():
    iface = fake_interface()
    pre = sample_state()
    spread = deepcopy(pre)
    # Same centroid, counts and HP; one tightly clustered vs one spread enemy formation.
    for j, u in enumerate(spread['enemies']):
        u['y'] = 10. + (j-2.5)*2.
    east_west = deepcopy(pre)
    for u in east_west['enemies']:
        u['x'] = 4.
    healthy = dict(predicate='retreat_low_health', weight=1.,
                   condition='only when retreat increases immediate survival', target_id=0,
                   time_window=3)
    cleaned = sanitize_guidance(dict(strategy='Retreat only under the stated condition',
                                    subgoals=[healthy], action_rules=[]))
    attack = dict(strategy='Focus fire', subgoals=[dict(predicate='focus_fire', weight=1.)],
                  action_rules=[])
    other = deepcopy(attack)
    other['strategy'] = 'Spread out and kite instead'
    no_damage = deepcopy(pre)
    acts = [6]*5
    result = dict(
        formation_same_summary=iface.summary(pre) == iface.summary(spread),
        formation_same_key=iface.cache_key(pre) == iface.cache_key(spread),
        direction_same_key=iface.cache_key(pre) == iface.cache_key(east_west),
        direction_same_summary=iface.summary(pre) == iface.summary(east_west),
        removed_subgoal_fields=sorted(set(healthy)-set(cleaned['subgoals'][0])),
        strategy_only_change_same_shaping=(
            compute_shaping(attack['subgoals'],pre,no_damage,acts) ==
            compute_shaping(other['subgoals'],pre,no_damage,acts)),
        focus_fire_without_damage=evaluate_predicate('focus_fire',None,pre,no_damage,acts))
    assert result['formation_same_summary'] and result['formation_same_key']
    assert result['direction_same_key'] and not result['direction_same_summary']
    assert result['strategy_only_change_same_shaping']
    assert result['removed_subgoal_fields'] == ['condition', 'target_id', 'time_window']
    return result


def effective_vector(goals):
    # Bases: enemy deaths, enemy damage / total max HP, ally deaths,
    # ally damage / total max HP, focus_fire predicate, retreat predicate.
    v = np.zeros(6)
    for g in goals:
        p, typ = g.get('predicate'), g.get('unit_type')
        w = min(1., max(0., float(g.get('weight',0))))
        if p in ['kill_type','damage_type','protect_type'] and str(typ).lower() != 'marine':
            continue
        if p == 'enemy_kill': v[0] += w
        elif p == 'kill_type': v[0] += 1.5*w
        elif p == 'enemy_damage': v[1] += 10*w
        elif p == 'damage_type': v[1] += 5*w
        elif p == 'ally_survive': v[2] -= w
        elif p == 'protect_type': v[2] -= 1.5*w; v[3] -= 3*w
        elif p == 'focus_fire': v[4] += w
        elif p == 'retreat_low_health': v[5] += w
    return v


def analyze(out):
    rows = read_transitions(out)
    lib = build_library('sc2', rows[0]['pre'])
    X = np.array([f_vector('sc2',lib,r['pre'],r['post'],r['actions']) for r in rows])
    idx = {p:j for j,(p,_) in enumerate(lib)}
    errs = dict(kill=float(np.max(np.abs(X[:,idx['kill_type']]-1.5*X[:,idx['enemy_kill']]))),
                damage=float(np.max(np.abs(X[:,idx['damage_type']]-.5*X[:,idx['enemy_damage']]))))
    assert max(errs.values()) < 1e-6
    rewards = np.array([r['reward'] for r in rows])
    heads = []
    for j,(p,typ) in enumerate(lib):
        y=X[:,j]
        heads.append(dict(predicate=p,type=typ,mean=float(y.mean()),
                          nonzero=float((np.abs(y)>1e-8).mean()),
                          correlation_external=float(np.corrcoef(y,rewards)[0,1])
                          if y.std()>1e-8 and rewards.std()>1e-8 else None))
    # Audit all recorded completed lam40 F200/V200 runs for this map.
    history = []
    filecache = {}
    for root in sorted((ROOT/'results/sacred').glob('[0-9]*'),key=lambda p:int(p.name)):
        if not (root/'config.json').exists():
            continue
        c=json.loads((root/'config.json').read_text())
        if (c.get('env_args',{}).get('map_name') != '5m_vs_6m' or
            c.get('wandb_run') not in ['lehca-shape_F200_lam40','vigil_Fmax200_te0.15_lam40']):
            continue
        info=json.loads((root/'info.json').read_text())
        if not info.get('test_return_mean_T') or info['test_return_mean_T'][-1]<.98*c['t_max']:
            continue
        meta=json.loads((root/'run.json').read_text())
        start=datetime.fromisoformat(meta['start_time'])+timedelta(hours=9)
        selected=[]
        for p in (ROOT/'results/guidance').glob(f"vigil__*_default_s{c['seed']}_*.jsonl"):
            stamp=datetime.strptime(p.name[7:26],'%Y-%m-%d_%H-%M-%S')
            if abs((stamp-start).total_seconds())>3: continue
            if p not in filecache:
                filecache[p]=[json.loads(x) for x in p.read_text().splitlines()]
            ev=filecache[p]
            hits=sum(e['cache_hit'] for e in ev if e['t_global']<info['llm_cache_hits_T'][-1])
            if ev[0]['cache_key'].startswith('5m_vs_6m|') and hits==info['llm_cache_hits'][-1]:
                selected.append((p,ev))
        assert len(selected)==1,(root,selected)
        p,ev=selected[0]
        vectors=np.array([effective_vector(e['guidance'].get('subgoals',[])) for e in ev])
        texts=[e['guidance'].get('strategy','') for e in ev]
        textchanges=np.array([a!=b for a,b in zip(texts,texts[1:])])
        samevector=np.max(np.abs(np.diff(vectors,axis=0)),axis=1)<1e-9
        history.append(dict(sacred=int(root.name),seed=c['seed'],group=c['wandb_run'],
          guidance_file=p.name,n=len(ev),unique_text=len(set(texts)),
          unique_effective_vectors=len(set(map(tuple,np.round(vectors,8)))),
          same_vector_refresh_fraction=float(samevector.mean()),
          text_changed_but_same_vector=int(np.sum(textchanges & samevector)),
          vector_mean=vectors.mean(0).tolist(),vector_sd=vectors.std(0).tolist()))
    schedules={}
    for length in [50000,300000]:
        sched=DecayThenFlatSchedule(1.,.05,length,decay='linear')
        ts=np.linspace(0,300000,3001)
        schedules[str(length)]=dict(epsilon={str(t):sched.eval(t) for t in [0,50000,100000,200000,300000]},
          integrated_epsilon_first300k=float(np.trapezoid([sched.eval(t) for t in ts],ts)))
    result=dict(provenance=provenance(),synthetic=synthetic_tests(),trajectory_steps=len(rows),
                library=lib,empirical_rank=int(np.linalg.matrix_rank(X,tol=1e-6)),
                exact_alias_errors=errs,heads=heads,history=history,schedules=schedules)
    save(out/'analysis.json',result)
    print(json.dumps(result,indent=2),flush=True)


class RecordingSession:
    def __init__(self, path):
        self.session=requests.Session()
        self.f=path.open('x')
        self.count=0
        self.label={}
    def post(self,url,**kwargs):
        if self.count>=64:
            raise RuntimeError('Hard limit of 64 HTTP calls reached')
        self.count+=1
        t0=time.monotonic()
        response=self.session.post(url,**kwargs)
        try: body=response.json()
        except ValueError: body={'error_text':response.text[:1000]}
        self.f.write(json.dumps(dict(label=self.label,request=kwargs.get('json'),
             status=response.status_code,seconds=time.monotonic()-t0,response=body))+ '\n')
        self.f.flush()
        return response


def llm_probe(out,api):
    rows=read_transitions(out)
    ep0=[r for r in rows if r['ep']==0]
    chosen=[ep0[int(j)] for j in np.linspace(0,len(ep0)-1,5,dtype=int)] + [rows[-1]]
    context=json.loads((out/'collection.json').read_text())['prompt_context']
    iface=SimpleNamespace(prompt_context=lambda:context)
    lib=build_library('sc2',rows[0]['pre'])
    transport=RecordingSession(out/'llm_http.jsonl')
    replies=[]
    try:
        with (out/'llm_probe.jsonl').open('x') as f:
            for n,r in enumerate(chosen):
                for repeat in range(2):
                    # Rotate order to reduce confounding prompt style with server time.
                    styles=['default','paper','twostage']
                    offset=(n+repeat)%3
                    for style in styles[offset:]+styles[:offset]:
                        args=SimpleNamespace(llm_api_base=api,llm_model='openai/gpt-oss-20b',
                            llm_temperature=.2,llm_max_tokens=3072,llm_timeout=90,
                            llm_cache=False,llm_reasoning_effort='low',prompt_style=style)
                        commander=LLMCommander(args,iface)
                        commander._session=transport
                        label=dict(case=n,ep=r['ep'],t=r['t'],style=style,repeat=repeat)
                        transport.label=label
                        g=commander(r['summary'],r['cache_key'],iface)
                        row=dict(**label,summary=r['summary'],guidance=g,
                            plan=commander.last_plan_text,stats=commander.stats(),
                            cumulative_http=transport.count)
                        if g:
                            goals=g['subgoals']
                            row['unresolved_heads']=[s for s in goals if head_index(lib,s) is None]
                            row['effective_vector']=effective_vector(goals).tolist()
                            # Replay on a common fixed trajectory; does not estimate policy gain.
                            yy=[compute_shaping(goals,z['pre'],z['post'],z['actions']) for z in rows]
                            row['replay_mean']=float(np.mean(yy))
                            row['replay_nonzero']=float(np.mean(np.abs(yy)>1e-8))
                        replies.append(row)
                        f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
                        print(label,'valid',g is not None,'HTTP',transport.count,flush=True)
    finally:
        transport.f.close()
        transport.session.close()
    save(out/'llm_summary.json',dict(provenance=provenance(),http=transport.count,
         replies=len(replies),valid=sum(r['guidance'] is not None for r in replies),
         note='Six frozen heuristic states, two repeats; not win-rate evidence.'))


def summarize(out):
    rows = read_transitions(out)
    lib = build_library('sc2', rows[0]['pre'])
    xx = np.array([f_vector('sc2',lib,r['pre'],r['post'],r['actions']) for r in rows])
    ix = {p:j for j,(p,_) in enumerate(lib)}
    basis = np.column_stack([xx[:,ix['enemy_kill']],xx[:,ix['enemy_damage']]/10,
       -xx[:,ix['ally_survive']],(1.5*xx[:,ix['ally_survive']]-xx[:,ix['protect_type']])/3,
       xx[:,ix['focus_fire']],xx[:,ix['retreat_low_health']]])
    replies = [json.loads(s) for s in (out/'llm_probe.jsonl').read_text().splitlines()]
    http = [json.loads(s) for s in (out/'llm_http.jsonl').read_text().splitlines()]
    by_style = {}
    for style in ['default','paper','twostage']:
        rr = [r for r in replies if r['style']==style]
        good = [r for r in rr if r['guidance'] is not None]
        hh = [h for h in http if h['label']['style']==style]
        within, centroids, equivalence_errors = [], [], []
        for case in sorted({r['case'] for r in good}):
            cc = [r for r in good if r['case']==case]
            ys = np.array([[compute_shaping(r['guidance']['subgoals'],z['pre'],z['post'],z['actions'])
                            for z in rows] for r in cc])
            for r,y in zip(cc,ys):
                reconstructed = np.clip(basis @ effective_vector(r['guidance']['subgoals']),-3,3)
                equivalence_errors.append(float(np.max(np.abs(y-reconstructed))))
            if len(ys)==2:
                within.append(float(np.sqrt(np.mean((ys[0]-ys[1])**2))))
                centroids.append(ys.mean(0))
        between = [float(np.sqrt(np.mean((a-b)**2))) for i,a in enumerate(centroids)
                   for b in centroids[i+1:]]
        within_mean = float(np.mean(within)) if within else None
        between_mean = float(np.mean(between)) if between else None
        by_style[style] = dict(requested=len(rr),valid=len(good),http=len(hh),
          vector_replay_max_error=max(equivalence_errors,default=0.),
          empty_shaping=sum(not r['guidance']['subgoals'] for r in good),
          unresolved_heads=sum(len(r.get('unresolved_heads',[])) for r in good),
          reported_calls=sum(r['stats']['llm_calls'] for r in rr),
          unique_vectors=len({tuple(r['effective_vector']) for r in good}),
          within_state_reward_rms_mean=within_mean,within_state_reward_rms=within,
          between_state_centroid_reward_rms_mean=between_mean,
          within_over_between=(within_mean/between_mean if between_mean else None),
          http_seconds_sum=sum(h['seconds'] for h in hh),
          completion_tokens=sum(h['response'].get('usage',{}).get('completion_tokens',0) for h in hh),
          finish_reasons=dict(Counter(c.get('finish_reason') for h in hh for c in h['response'].get('choices',[]))))
        assert max(equivalence_errors,default=0.) < 1e-5
    key_to_summary = defaultdict(set)
    for r in rows:
        key_to_summary[r['cache_key']].add(r['summary'])
    result = dict(provenance=provenance(),styles=by_style,
      observed_cache_keys=len(key_to_summary),
      keys_with_multiple_summaries=sum(len(s)>1 for s in key_to_summary.values()),
      notes=['Replay includes shaping clipping but not lambda, masking, policy change or learning.',
             'Six states/two repetitions is a diagnostic, not a significance test.',
             'All eight heuristic episodes lost; trajectories are not representative of trained QMIX.'])
    save(out/'probe_analysis.json',result)
    print(json.dumps(result,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['collect','analyze','llm','summarize'],required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--api')
    a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    if a.mode=='collect': collect(a.out)
    elif a.mode=='analyze': analyze(a.out)
    elif a.mode=='summarize': summarize(a.out)
    else:
        if not a.api: ap.error('--api is required for llm')
        llm_probe(a.out,a.api)


if __name__=='__main__':
    main()
