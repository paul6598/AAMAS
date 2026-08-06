"""Algorithm registry. New algorithms: add a package with a train(env, eval_env, args) entry
and register it here (e.g. mappo, llm_taca).

Backbone (MLP vs RNN) is a swappable module inside llm_mca/llm_taca, selected with --backbone
(default rnn), not a separate algorithm name.
"""


def get_algorithm(name):
    if name == "ddqn":
        from .ddqn.algorithm import train
    elif name == "llm_mca":                    # LLM-MCA (--backbone rnn|mlp, default rnn)
        from .llm_mca.algorithm import train
    elif name == "llm_taca":                   # LLM-TACA (--backbone rnn|mlp, default rnn)
        from .llm_taca.algorithm import train
    elif name == "mappo":                      # MAPPO (--backbone rnn|mlp, default rnn)
        from .mappo.algorithm import train
    elif name == "rnn_iql":                    # true-reward (equal-split) baseline
        from .rnn_iql.algorithm import train
    elif name == "oracle":                     # diagnostic: dense-credit upper bound (rnn backbone)
        from .rnn_iql.algorithm import train_oracle as train
    else:
        raise ValueError(f"Unknown algorithm: {name}")
    return train
