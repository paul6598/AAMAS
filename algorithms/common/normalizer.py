"""Running mean/std normalizer for value targets (Welford).

Networks predict NORMALIZED values; we denormalize for bootstrapping and normalize the TD target
for the loss. This is the standard MAPPO "value normalization" trick, and it matters even more for
LLM credit, whose magnitude swings wildly between iterations.
"""
import torch


class ValueNormalizer:
    def __init__(self, device):
        self.mean = torch.zeros((), device=device)
        self.var = torch.ones((), device=device)
        self.count = 1e-4

    @torch.no_grad()
    def update(self, x):
        if x.numel() == 0:
            return
        bm, bv, bc = x.mean(), x.var(unbiased=False), x.numel()
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        M2 = self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot
        self.var = M2 / tot
        self.count = tot

    def normalize(self, x):
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)

    def denormalize(self, x):
        return x * torch.sqrt(self.var + 1e-8) + self.mean
