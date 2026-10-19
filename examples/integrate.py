"""Attach AnchorFlow at an existing model's representation/head boundary."""

from anchorflow import AnchorFlow, AnchorFlowConfig


def training_step(encoder, head, optimizer, anchorflow, batch, epoch):
    import torch
    from torch.nn import functional as F

    x, y, d = batch
    optimizer.zero_grad(set_to_none=True)
    features = encoder(x)
    base_loss = F.cross_entropy(head(features), y)
    flow = anchorflow(features, y, d, head, epoch=epoch)
    objective = base_loss + anchorflow.config.weight * flow.energy
    anchorflow.bank.stage(features, y, d)
    if not torch.isfinite(objective):
        anchorflow.bank.discard()
        raise FloatingPointError("non-finite objective")
    objective.backward()
    optimizer.step()
    anchorflow.bank.commit()
    return {"base_loss": base_loss.detach(), "flow_energy": flow.energy.detach()}


def create_regularizer(feature_dim, num_scenes=10, num_devices=6, device="cpu"):
    return AnchorFlow(num_scenes, num_devices, feature_dim, AnchorFlowConfig()).to(device)
