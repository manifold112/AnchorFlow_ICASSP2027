"""First-order finite-difference regularization with a shared auxiliary head."""

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .anchors import AnchorBank
from .config import AnchorFlowConfig
from .flows import DeviceFlow


@contextmanager
def deterministic_head(head: nn.Module):
    """Disable train-mode randomness and freeze BN statistics, retaining gradients.

    Custom masking layers must respect Module.training. There is deliberately no
    no_grad context: the same head parameters must receive the auxiliary gradient.
    """
    state = [(module, module.training) for module in head.modules()]
    head.eval()
    try:
        yield
    finally:
        for module, training in state:
            module.training = training


@dataclass
class FlowLoss:
    energy: Tensor
    eligible_samples: int
    paths: int


class AnchorFlow(nn.Module):
    def __init__(self, num_scenes: int, num_devices: int, feature_dim: int,
                 config: AnchorFlowConfig | None = None):
        super().__init__()
        self.config = config or AnchorFlowConfig()
        self.bank = AnchorBank(num_scenes, num_devices, feature_dim, self.config)

    def forward(self, features: Tensor, scenes: Tensor, devices: Tensor,
                head: nn.Module, epoch: int) -> FlowLoss:
        # This zero is connected only to head parameters, never to the encoder.
        zero = sum((p.sum() * 0 for p in head.parameters()), features.new_zeros(()))
        if epoch < self.config.warmup_epochs:
            return FlowLoss(zero, 0, 0)
        valid = self.bank.eligible
        # A source cell must be valid and have another valid device in the scene.
        eligible = valid[scenes, devices] & (valid[scenes].sum(-1) >= 2)
        rows = eligible.nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            return FlowLoss(zero, 0, 0)
        y, source = scenes[rows], devices[rows]
        candidates = valid[y].clone()
        candidates.scatter_(1, source[:, None], False)
        k = self.config.targets_per_sample
        target = torch.multinomial(candidates.float(), k, replacement=True).reshape(-1)
        y = y.repeat_interleave(k)
        source = source.repeat_interleave(k)
        h = features[rows].detach().float().repeat_interleave(k, dim=0)
        scale = self.bank.scale
        flow = DeviceFlow.from_features(h, self.bank.mean[y, source], scale[y, source],
                                       self.bank.mean[y, target], scale[y, target])
        tau = torch.rand(len(y), device=h.device) * (1 - self.config.delta)
        # Both points use the same target, residual, and deterministic head state.
        z0 = flow.at(tau).to(features.dtype)
        z1 = flow.at(tau + self.config.delta).to(features.dtype)
        with deterministic_head(head):
            logits = head(torch.cat((z0, z1), dim=0))
            loss0, loss1 = F.cross_entropy(logits.float(), y.repeat(2),
                                           reduction="none").chunk(2)
        energy = ((loss1 - loss0) / self.config.delta).square().mean()
        return FlowLoss(energy, len(rows), len(y))
