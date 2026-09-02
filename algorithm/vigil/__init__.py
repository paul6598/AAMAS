"""Adaptive guidance scheduling (research track; imports LEHCA, never edits it)."""

from algorithm.src.runners import REGISTRY as RUNNER_REGISTRY  # noqa: E402
from .runner import SchedRunner  # noqa: E402

RUNNER_REGISTRY["vigil"] = SchedRunner
