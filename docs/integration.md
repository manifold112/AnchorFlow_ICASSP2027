# Integrate AnchorFlow into an ASC system

[← Project](../README.md) · [Method](method.md) · [Data](data.md)

## Choose a representation boundary

Expose an encoder `f(x) -> [B, P]` and the remaining predictor
`g(h) -> [B, num_scenes]`. AnchorFlow operates at this boundary. A larger head
lets the auxiliary branch update every trainable layer after the chosen
representation; encoder layers receive the base objective.

```python
from anchorflow.models import SplitASC

model = SplitASC(encoder=encoder, head=scene_predictor, feature_dim=feature_dim)
```

For a backbone factory, return an object with `.encoder`, `.head`, `.feature_dim`,
and `forward(x)`. Point the configuration at its import path:

```yaml
model:
  factory: my_asc.backbones:build_model
  kwargs:
    feature_dim: 256
```

The trainer passes `num_scenes` plus these keyword arguments. Published CP-Mobile,
BC-ResNet, TF-SepNet and PaSST recipes can use this interface at an appropriate
representation boundary. Preserve their backbone definitions and preprocessing
when setting up matched comparisons.

## Update order

1. Compute original features and base scene loss.
2. Evaluate the auxiliary path loss using the current committed bank.
3. Stage detached feature sums, squared sums and counts.
4. Backpropagate the combined objective and update model parameters.
5. Commit staged statistics after the update succeeds.

The bank never updates itself inside `AnchorFlow.forward`. This makes gradient
accumulation and skipped optimizer updates explicit. On a skipped or invalid
update, call `af.bank.discard()`.

```python
optimizer.zero_grad(set_to_none=True)
h = encoder(x)
base = criterion(head(h), y)
flow = af(h, y, d, head, epoch)
af.bank.stage(h, y, d)
loss = base + af.config.weight * flow.energy
loss.backward()
optimizer.step()
af.bank.commit()
```

During the first epoch, the default warm-up accumulates anchors but returns zero
auxiliary energy. Later, each source cell still needs at least 32 committed
observations and another valid device in its scene.

## Deterministic auxiliary head

The context switches the head to evaluation mode without disabling gradients.
Ordinary dropout and BatchNorm then behave deterministically; BatchNorm affine
parameters still receive gradients. Custom random masking, stochastic depth or
patchout must respect the module's `training` flag.

Both positions of a finite-difference pair are evaluated in one concatenated
forward. Individual module modes are restored even if the call raises an error.
Use a head that evaluates samples independently in evaluation mode; avoid operations
that mix different samples across the batch dimension.

PyTorch's [`Module.eval()` documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval)
describes the distinction between evaluation mode and disabling autograd.

## Effective updates and gradient accumulation

Stage every microbatch, but commit once per successful effective update. The bank
combines raw sums and counts before forming the update's cell moments.

For the auxiliary objective, the denominator is the total number of eligible paths
in the effective update. If microbatches have different eligible counts, combine
`flow.energy * flow.paths` across them and divide by the total path count. Averaging
microbatch means would reweight samples. Base-loss reduction should independently
match the original recipe.

## Distributed anchor statistics

Keep one AnchorFlow bank per process, outside a DDP module's automatic buffer
broadcast. All workers must call `bank.commit(synchronize=True)` in the same order.
This all-reduces pending feature sums, squared sums and counts so every bank commits
the same global cell moments. See PyTorch's
[`all_reduce` documentation](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce).

When using DDP's averaged gradients, weight a rank's auxiliary energy by
`world_size * local_paths / global_paths` to preserve the global eligible-path mean.
An empty rank contributes a head-connected zero. A custom DDP integration must route
both normal and auxiliary predictor calls through the wrapped training forward so
the reducer observes the complete parameter graph. The command-line trainer is
the single-device entry point.

## Checkpoints and deployment

Training checkpoints contain model weights, anchor moments/counts, optimizer,
scheduler, effective configuration, epoch, best source score, and Torch/DataLoader
random states. Save at committed boundaries, when no pending anchor statistics remain.
The CLI resumes only when the configuration and protocol match.

```bash
anchorflow train --config configs/default.yaml \
  --protocol configs/protocols/dcase2024.yaml \
  --train-csv data/train.csv --val-csv data/val_source.csv \
  --feature-root data --output runs/tau22_af_seed0 --device cuda \
  --resume runs/tau22_af_seed0/last.pt
```

Export packages only the original scene model and its construction metadata:

```bash
anchorflow export --checkpoint runs/tau22_af_seed0/best.pt \
  --output checkpoints/scene_model.pt
```

```python
import torch
from anchorflow.engine import build_model

payload = torch.load("checkpoints/scene_model.pt", weights_only=True)
model = build_model(payload["model_config"], num_scenes=len(payload["scenes"]))
model.load_state_dict(payload["model"])
model.eval()
# Scene inference: logits = model(log_mel)
```
