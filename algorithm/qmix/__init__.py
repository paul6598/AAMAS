"""QMIX baseline (Rashid et al., 2018).

QMIX is pymarl's native value-decomposition algorithm: it is composed
entirely of shared components in algorithm/src — q_learner ("q_learner"),
BasicMAC ("basic_mac"), EpisodeRunner ("episode") and the QMixer in
src/modules/mixers/qmix.py — so this package registers nothing extra.
Config: config/algs/qmix_paper.yaml (paper Table 2 settings; uses
"lehca_q_learner" only for its Adam option, with all LEHCA guidance off).
"""
