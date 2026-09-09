"""RSVP — Residual Shaping-Value Prediction for On-Demand LLM Guidance Refresh."""

from algorithm.src.runners import REGISTRY as RUNNER_REGISTRY  # noqa: E402
from .runner import RSVPRunner  # noqa: E402

SchedRunner = RSVPRunner  # compatibility for historical imports
RUNNER_REGISTRY["rsvp"] = RSVPRunner
RUNNER_REGISTRY["vigil"] = RSVPRunner  # historical configs / queued commands
