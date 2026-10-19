"""Transactional EMA moments indexed jointly by scene and observed device.

Stages are accumulated over microbatches. Only a successful optimizer update
commits them, so the current auxiliary branch sees the preceding anchor state.
"""

import torch
import torch.distributed as dist
from torch import Tensor, nn

from .config import AnchorFlowConfig


class AnchorBank(nn.Module):
    def __init__(self, num_scenes: int, num_devices: int, feature_dim: int,
                 config: AnchorFlowConfig):
        super().__init__()
        if min(num_scenes, num_devices, feature_dim) <= 0:
            raise ValueError("bank dimensions must be positive")
        self.config = config
        self.num_scenes = num_scenes
        self.num_devices = num_devices
        self.feature_dim = feature_dim
        shape = (num_scenes, num_devices, feature_dim)
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("second", torch.zeros(shape))
        self.register_buffer("counts", torch.zeros(shape[:2], dtype=torch.long))
        self.register_buffer("pending_sum", torch.zeros(shape), persistent=False)
        self.register_buffer("pending_square", torch.zeros(shape), persistent=False)
        self.register_buffer("pending_count", torch.zeros(shape[:2], dtype=torch.long),
                             persistent=False)

    @property
    def scale(self) -> Tensor:
        variance = (self.second - self.mean.square()).clamp_min(self.config.variance_floor)
        return (variance + self.config.epsilon).sqrt()

    @property
    def eligible(self) -> Tensor:
        return self.counts >= self.config.min_samples

    @torch.no_grad()
    def stage(self, features: Tensor, scenes: Tensor, devices: Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(f"features must have shape [B, {self.feature_dim}]")
        if scenes.shape != (len(features),) or devices.shape != scenes.shape:
            raise ValueError("scene and device labels must have shape [B]")
        if scenes.dtype != torch.long or devices.dtype != torch.long:
            raise ValueError("scene and device labels must be int64")
        if ((scenes < 0) | (scenes >= self.num_scenes)).any():
            raise ValueError("scene ID outside bank vocabulary")
        if ((devices < 0) | (devices >= self.num_devices)).any():
            raise ValueError("device ID outside observed training vocabulary")
        h = features.detach().float()
        if not torch.isfinite(h).all():
            raise ValueError("anchor features must be finite")
        cells = scenes * self.num_devices + devices
        self.pending_sum.view(-1, self.feature_dim).index_add_(0, cells, h)
        self.pending_square.view(-1, self.feature_dim).index_add_(0, cells, h.square())
        self.pending_count.view(-1).index_add_(0, cells, torch.ones_like(cells))

    @torch.no_grad()
    def commit(self, synchronize: bool = False) -> None:
        """Commit sufficient statistics; all ranks must call when synchronize=True."""
        if synchronize:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("initialize torch.distributed before synchronizing anchors")
            for buffer in (self.pending_sum, self.pending_square, self.pending_count):
                dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
        active = self.pending_count > 0
        fresh = active & (self.counts == 0)
        old = active & ~fresh
        denominator = self.pending_count.clamp_min(1).unsqueeze(-1)
        first = self.pending_sum / denominator
        second = self.pending_square / denominator
        beta = self.config.ema_decay
        self.mean[fresh] = first[fresh]
        self.second[fresh] = second[fresh]
        self.mean[old] = beta * self.mean[old] + (1 - beta) * first[old]
        self.second[old] = beta * self.second[old] + (1 - beta) * second[old]
        self.counts.add_(self.pending_count)
        self.discard()

    @torch.no_grad()
    def discard(self) -> None:
        """Drop an update, for example after an AMP overflow or an invalid loss."""
        self.pending_sum.zero_()
        self.pending_square.zero_()
        self.pending_count.zero_()
