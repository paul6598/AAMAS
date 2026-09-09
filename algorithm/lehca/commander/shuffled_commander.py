"""Deterministic state-independent guidance control for content ablations."""
import copy
import json
import random

from .base import Commander, sanitize_guidance


class ShuffledCommander(Commander):
    """Replay valid LLM guidance from another seed in shuffled order.

    The current summary/cache key are intentionally ignored.  This preserves
    the marginal form of LLM outputs while breaking their alignment with the
    current state.  It performs no LLM requests.
    """
    def __init__(self, args):
        path = getattr(args, "guidance_replay_path", "")
        if not path:
            raise ValueError("commander=shuffle requires guidance_replay_path")
        items = []
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                guidance = sanitize_guidance(row.get("guidance"))
                if guidance is not None:
                    items.append(guidance)
        if not items:
            raise ValueError("no valid guidance in %s" % path)
        random.Random(104729 + int(getattr(args, "seed", 0))).shuffle(items)
        self.items = items
        self.i = int(getattr(args, "guidance_replay_offset", 0)) % len(items)

    def __call__(self, summary, cache_key, iface):
        guidance = copy.deepcopy(self.items[self.i])
        self.i = (self.i + 1) % len(self.items)
        return guidance
