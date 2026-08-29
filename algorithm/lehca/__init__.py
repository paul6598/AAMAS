"""LEHCA: QMIX backbone + coarse-timescale LLM Commander guidance.

Registers its runner/controller/learner into the shared registries under the
names used by config/algs/lehca.yaml.
"""
from algorithm.src.runners import REGISTRY as RUNNER_REGISTRY
from algorithm.src.controllers import REGISTRY as MAC_REGISTRY
from algorithm.src.learners import REGISTRY as LEARNER_REGISTRY

from .runner import LehcaRunner
from .controller import LehcaMAC
from .learner import LehcaQLearner

RUNNER_REGISTRY["lehca"] = LehcaRunner
MAC_REGISTRY["lehca_mac"] = LehcaMAC
LEARNER_REGISTRY["lehca_q_learner"] = LehcaQLearner
