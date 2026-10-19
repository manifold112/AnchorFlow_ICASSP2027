"""Method hyperparameters from Section 3.1 of the accompanying manuscript."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AnchorFlowConfig:
    weight: float = 0.3
    delta: float = 0.20
    targets_per_sample: int = 1
    ema_decay: float = 0.99
    min_samples: int = 32
    variance_floor: float = 1e-6
    epsilon: float = 1e-5
    warmup_epochs: int = 1

    def __post_init__(self):
        if not 0 < self.delta < 1:
            raise ValueError("delta must lie strictly between zero and one")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if self.targets_per_sample < 1 or self.min_samples < 1:
            raise ValueError("target and sample counts must be positive")
        if self.variance_floor < 0 or self.epsilon <= 0:
            raise ValueError("scales require variance_floor >= 0 and epsilon > 0")
        if self.weight < 0 or self.warmup_epochs < 0:
            raise ValueError("weight and warm-up must be nonnegative")
