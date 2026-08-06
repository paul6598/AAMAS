"""Unified experiment logger.

Always prints a compact line per iteration to stdout (so training is visible live in a terminal
/ tmux); additionally mirrors every metric to Weights & Biases when --use_wandb is set. wandb is
flag-gated and default-off, matching the terminal workflow.
"""


class Logger:
    def __init__(self, use_wandb=False, project="AAMAS", name=None, group=None,
                 entity=None, config=None):
        self.use_wandb = use_wandb
        self.wandb = None
        if use_wandb:
            import wandb
            self.wandb = wandb
            wandb.init(project=project, name=name, group=group, entity=entity, config=config or {})

    def log(self, metrics, step):
        """Log a dict of scalar metrics at a given iteration step."""
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)
        parts = []
        for k, v in metrics.items():
            parts.append(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}")
        print(f"[iter {step:3d}] " + " ".join(parts), flush=True)

    def finish(self):
        if self.wandb is not None:
            self.wandb.finish()
