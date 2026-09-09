"""Bounded, no-LLM backbone audit; real SC2 batches and paired updates.

Uses a fresh SC2 instance for each path. Writes new diagnostic artifacts only.
No production code patches, no long-run performance claims.
"""
import argparse
import copy
import hashlib
import json
import logging
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from algorithm.src.components.episode_buffer import EpisodeBatch, ReplayBuffer
from algorithm.src.components.transforms import OneHot
from algorithm.src.controllers.basic_controller import BasicMAC
from algorithm.lehca.controller import LehcaMAC
from algorithm.lehca.learner import LehcaQLearner
from algorithm.lehca.state import get_state
from algorithm.src.runners.episode_runner import EpisodeRunner
from algorithm.lehca.runner import LehcaRunner
from algorithm.rsvp.runner import RSVPRunner
from algorithm.rsvp.critic import ValueCritic


class Logger:
    console_logger = logging.getLogger('backbone-audit')

    def __init__(self):
        self.values = {}

    def log_stat(self, key, value, step):
        self.values[key] = float(value)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def scheme_for(info):
    return {
        'state': {'vshape': info['state_shape']},
        'obs': {'vshape': info['obs_shape'], 'group': 'agents'},
        'actions': {'vshape': (1,), 'group': 'agents', 'dtype': torch.long},
        'avail_actions': {'vshape': (info['n_actions'],), 'group': 'agents', 'dtype': torch.int},
        'reward': {'vshape': (1,)}, 'shaping_f': {'vshape': (1,)},
        'terminated': {'vshape': (1,), 'dtype': torch.uint8},
    }


def tensor_delta(a, b):
    if a.shape != b.shape:
        return {'shape_equal': False, 'a': list(a.shape), 'b': list(b.shape)}
    return {'shape_equal': True, 'max_abs': float((a.float()-b.float()).abs().max()),
            'equal': bool(torch.equal(a, b))}


def collect(config, path, seed, episodes, device):
    args = SimpleNamespace(**copy.deepcopy(config))
    args.seed = seed
    args.env_args['seed'] = seed
    args.device = device
    args.use_cuda = device == 'cuda'
    args.commander = 'none'
    args.use_reward_shaping = args.use_action_masking = args.use_masking_at_test = False
    args.shaping_in_learner = True
    args.scheduler = 'fixed'
    args.sched_trace = False
    get_state().guidance = None
    seed_all(seed)
    cls = {'qmix': EpisodeRunner, 'lehca_off': LehcaRunner, 'rsvp_off': RSVPRunner}[path]
    runner = cls(args, Logger())
    try:
        info = runner.get_env_info()
        args.n_agents, args.n_actions, args.state_shape = (
            info['n_agents'], info['n_actions'], info['state_shape'])
        scheme = scheme_for(info)
        groups = {'agents': args.n_agents}
        preprocess = {'actions': ('actions_onehot', [OneHot(args.n_actions)])}
        template = EpisodeBatch(copy.deepcopy(scheme), groups, 1, info['episode_limit']+1,
                                preprocess=preprocess)
        mac = (BasicMAC if path == 'qmix' else LehcaMAC)(template.scheme, groups, args)
        # Production initializes learner/mixer before the first rollout.
        learner = LehcaQLearner(mac, template.scheme, Logger(), args)
        if args.use_cuda:
            learner.cuda()
        runner.setup(copy.deepcopy(scheme), groups, preprocess, mac)
        batches = []
        for _ in range(episodes):
            batches.append(copy.deepcopy(runner.run(False)))
        return args, batches
    finally:
        runner.close_env()


def paired_updates(args, source, updates=8):
    args = copy.deepcopy(args)
    args.learner_log_interval = 1
    args.target_update_interval = 2
    left_args, right_args = copy.deepcopy(args), copy.deepcopy(args)
    left_args.shaping_in_learner = False
    left_args.lambda_floor_frac = 0
    right_args.shaping_in_learner = True
    right_args.lambda_floor_frac = .4
    get_state().configure(right_args)
    seed_all(731)
    left_log = Logger()
    left = LehcaQLearner(BasicMAC(source.scheme, source.groups, left_args),
                        source.scheme, left_log, left_args)
    if args.use_cuda:
        left.cuda()
    seed_all(731)
    right_log = Logger()
    right = LehcaQLearner(LehcaMAC(source.scheme, source.groups, right_args),
                         source.scheme, right_log, right_args)
    if args.use_cuda:
        right.cuda()
    records = []
    original = source['reward'].clone()
    for i in range(updates):
        # Independent sampled copies: the learner replaces reward in its batch.
        left.train(copy.deepcopy(source), 1000*(i+1), i+1)
        right.train(copy.deepcopy(source), 1000*(i+1), i+1)
        pdelta = max(float((a-b).abs().max()) for a,b in zip(left.params,right.params))
        target_delta = max(float((a-b).abs().max()) for a,b in zip(
            left.target_mac.parameters(), right.target_mac.parameters()))
        mixer_delta = max(float((a-b).abs().max()) for a,b in zip(
            left.target_mixer.parameters(), right.target_mixer.parameters()))
        records.append({'update': i+1, 'parameter_max_abs': pdelta,
                        'target_max_abs': max(target_delta,mixer_delta),
                        'loss_abs': abs(left_log.values['loss']-right_log.values['loss'])})
    assert torch.equal(original, source['reward'])
    assert all(r['parameter_max_abs']==r['target_max_abs']==r['loss_abs']==0 for r in records)
    # Nonzero shaping: compare learner composition to explicit reward composition.
    shaped = copy.deepcopy(source)
    shaped.data.transition_data['shaping_f'].fill_(.25)
    get_state().set_lambda_progress(20000,.4*args.t_max)
    expected = copy.deepcopy(shaped)
    expected.data.transition_data['reward'] += get_state().lambda_val*expected['shaping_f']
    left.train(expected,20000,20)
    right.train(shaped,20000,20)
    nonzero_delta=max(float((a-b).abs().max()) for a,b in zip(left.params,right.params))
    assert nonzero_delta==0
    # Actual replay sampling must not contaminate stored rewards.
    scheme=copy.deepcopy(source.scheme)
    scheme.pop('filled')
    replay=ReplayBuffer(scheme,source.groups,2,source.max_seq_length,device=args.device)
    replay.insert_episode_batch(source)
    before=replay['reward'].clone()
    sample=replay.sample(1)
    sample.data.transition_data['shaping_f'].fill_(.25)
    right.train(sample,21000,21)
    assert torch.equal(before,replay['reward'])
    return {'zero_shaping_updates':records,'nonzero_shaping_parameter_max_abs':nonzero_delta,
            'replay_reward_unchanged':True}


def rng_probe():
    seed_all(81)
    before=torch.get_rng_state().clone()
    critic=ValueCritic(4,3)
    init_changed=not torch.equal(before,torch.get_rng_state())
    critic.add_episode(np.zeros((4,4)),np.ones((4,3)))
    before=torch.get_rng_state().clone()
    critic.train(iters=2)
    train_changed=not torch.equal(before,torch.get_rng_state())
    return {'critic_init_changes_global_torch_rng':init_changed,
            'critic_train_changes_global_torch_rng':train_changed}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--episodes',type=int,default=2)
    ap.add_argument('--seeds',type=int,nargs='+',default=[0,1])
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    config=json.loads((ROOT/'results/sacred/242/config.json').read_text())
    result={'reference':242,'seeds':a.seeds,'episodes_per_path':a.episodes,'device':a.device,
            'rng_probe':rng_probe(),'rollouts':{},'updates':{}}
    for seed in a.seeds:
        runs={}
        for path in ['qmix','lehca_off','rsvp_off']:
            args,batches=collect(config,path,seed,a.episodes,a.device)
            runs[path]=batches
            torch.save([b.data.transition_data for b in batches],a.out/f'{path}_s{seed}.pt')
        comparisons={}
        for path in ['lehca_off','rsvp_off']:
            comparisons[path]=[{k:tensor_delta(q[k],l[k]) for k in
                               ['state','obs','avail_actions','actions','reward','terminated','filled']}
                              for q,l in zip(runs['qmix'],runs[path])]
        result['rollouts'][seed]=comparisons
        assert all(v.get('equal',False) for ep in comparisons['lehca_off'] for v in ep.values())
        result['updates'][seed]=paired_updates(args,runs['qmix'][0])
        print('SEED_DONE',seed,flush=True)
    result['code_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for folder in ['algorithm/lehca','algorithm/rsvp','algorithm/src']
                           for p in (ROOT/folder).rglob('*.py')}
    (a.out/'report.json').write_text(json.dumps(result,indent=2))
    print('REPORT',a.out/'report.json',flush=True)


if __name__=='__main__':
    main()
