# AAMAS — LLM × MARL

Research codebase for our AAMAS submission on applying LLMs to multi-agent
reinforcement learning. Built on [oxwhirl/pymarl](https://github.com/oxwhirl/pymarl)
(QMIX backbone). First milestone: reproducing the **LEHCA** baseline
(Bai et al., Scientific Reports 2026, DOI 10.1038/s41598-026-54971-6) —
no official code exists, so it is implemented from scratch here
(see `docs/lehca-reproduction.md` for method mapping and decisions).

## Layout

```
main.py, run.py       pymarl entrypoint (sacred) + training loop
algorithm/            one folder per algorithm + shared infra; registration in
                      algorithm/__init__.py (import your package there to add one)
  src/                shared MARL infrastructure
    learners/         loss/optimization (q_learner, coma, qtran)
    controllers/      action selection (basic_mac)
    runners/          env interaction loops (episode, parallel)
    modules/          networks (DRQN agents, QMIX mixer, critics)
    components/       replay buffer, action selectors, epsilon schedules
  lehca/              LEHCA: runner.py (Algorithm 1), controller.py (masked Q~),
                      learner.py (QMIX+Adam+lambda decay), commander/ (LLM+rule),
                      shaping/, masking/, state.py
  qmix/               QMIX baseline (native src components; see its __init__.py)
env/                  environments: pymarl env registry (SMAC) +
  semantic/           LEHCA semantic interfaces (obs -> d_t text + grounding)
config/               default.yaml, algs/ (lehca, qmix_paper), envs/ (sc2)
scripts/              experiment scripts (positional args + EXTRA + seed loop)
analysis/             metrics (auc_early.py: paper Eq. 9 from wandb groups)
paper/                reference papers (LEHCA.pdf, ...)
docs/                 notes & decisions
results/              logs, sacred outputs, saved models
wandb/                wandb run dirs (project AAMAS-LEHCA)
```

## Quick start

```bash
conda activate aamas          # torch 2.5.1+cu121, smac, sacred, wandb
# 1) serve the Commander LLM on a GPU node (e.g. tmux 37, RTX 6000 Ada):
cd scripts && bash serve_llm.sh openai/gpt-oss-20b 8355
# 2) train (any GPU node; API base points at the serving node):
bash run_lehca.sh 2s3z true LEHCA_2s3z llm http://n064:8355/v1 openai/gpt-oss-20b
bash run_qmix.sh  2s3z true QMIX_2s3z
bash run_ablation.sh 2s3z false true true ABL_noreward http://n064:8355/v1 openai/gpt-oss-20b
# quick CPU smoke test (no LLM):
python main.py --config=lehca --env-config=sc2 with env_args.map_name=3m \
    t_max=400 use_cuda=False commander=rule
# metrics:
python analysis/auc_early.py LEHCA_2s3z
```

SC2 4.10 + SMAC maps: `/gpfs/home1/paul6598/StarCraftII` (SC2PATH).
Paper protocol: 3m/8m/MMM/2s3z @1M, 2m_vs_1z @500k,
3s_vs_5z/5m_vs_6m/27m_vs_30m @5M steps, 5 seeds.
