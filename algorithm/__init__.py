"""Algorithm registry root.

algorithm/src holds the shared MARL infrastructure (runners, controllers,
learners, modules, components) with base registry entries; each algorithm
package below registers its own classes on import. To add an algorithm,
create algorithm/<name>/ and import it here.
"""
import algorithm.lehca  # noqa: F401  (registers lehca runner/mac/learner)
import algorithm.qmix   # noqa: F401  (native pymarl components; no-op)
