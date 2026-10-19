"""Closed-form diagonal-Gaussian device paths, Equations (6)-(8)."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class DeviceFlow:
    source_mean: Tensor
    source_scale: Tensor
    target_mean: Tensor
    target_scale: Tensor
    residual: Tensor

    @classmethod
    @torch.no_grad()
    def from_features(cls, features, source_mean, source_scale, target_mean, target_scale):
        r = (features.detach().float() - source_mean) / source_scale
        return cls(source_mean.detach(), source_scale.detach(), target_mean.detach(),
                   target_scale.detach(), r.detach())

    def at(self, tau: Tensor | float) -> Tensor:
        if isinstance(tau, Tensor) and tau.ndim == self.residual.ndim - 1:
            tau = tau.unsqueeze(-1)
        mean = self.source_mean + tau * (self.target_mean - self.source_mean)
        scale = self.source_scale + tau * (self.target_scale - self.source_scale)
        return mean + scale * self.residual

    @property
    def velocity(self) -> Tensor:
        return ((self.target_mean - self.source_mean)
                + (self.target_scale - self.source_scale) * self.residual)
