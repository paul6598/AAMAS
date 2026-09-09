"""Legacy import bridge. New code must import algorithm.rsvp.

Retained for queued jobs and old analysis/checkpoints; implementation lives
only in algorithm/rsvp. Core modules share identity with their RSVP versions.
"""
from importlib import import_module
import sys

from algorithm import rsvp as _rsvp

RSVPRunner = _rsvp.RSVPRunner
SchedRunner = RSVPRunner
__path__ = _rsvp.__path__
for _name in ("runner", "critic", "predlib", "shaping", "shaping.grf",
              "shaping.pursuit", "commander", "commander.grf", "commander.pursuit"):
    _module = import_module("algorithm.rsvp." + _name)
    sys.modules[__name__ + "." + _name] = _module
    if "." not in _name:
        globals()[_name] = _module
