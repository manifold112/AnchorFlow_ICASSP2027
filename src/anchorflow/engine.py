"""Training, source-only selection, checkpointing and deployment model export."""

import importlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .config import AnchorFlowConfig
from .data import CachedLogMel, Protocol, read_manifest, validate_rows, validate_split_pair
from .metrics import DeviceMetrics
from .regularizer import AnchorFlow


def build_model(settings, num_scenes):
    module, name = settings["factory"].split(":", 1)
    factory = getattr(importlib.import_module(module), name)
    return factory(num_scenes=num_scenes, **settings.get("kwargs", {}))


@torch.no_grad()
def evaluate(model, loader, protocol, device):
    model.eval()
    metrics = DeviceMetrics(protocol.seen_devices, protocol.segments_per_recording)
    for batch in loader:
        logits = model(batch["features"].to(device))
        metrics.update(logits.softmax(-1), batch["scene"], batch["device_name"],
                       batch["recording_id"])
    return metrics.compute()


def save_atomic(payload, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def train(settings, protocol: Protocol, train_csv, val_csv, feature_root, output,
          device="cpu", seed=0, resume=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    training = settings["training"]
    train_rows, val_rows = read_manifest(train_csv), read_manifest(val_csv)
    validate_rows(train_rows, protocol, source_only=True)
    # Held-out devices never participate in model or hyperparameter selection.
    validate_rows(val_rows, protocol, source_only=True)
    validate_split_pair(train_rows, val_rows, protocol)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(CachedLogMel(train_rows, protocol, feature_root),
                              batch_size=training["batch_size"], shuffle=True,
                              num_workers=training["num_workers"], generator=generator)
    val_loader = DataLoader(CachedLogMel(val_rows, protocol, feature_root),
                            batch_size=training["batch_size"], shuffle=False,
                            num_workers=training["num_workers"])
    model = build_model(settings["model"], len(protocol.scenes)).to(device)
    config = AnchorFlowConfig(**settings["anchorflow"])
    regularizer = AnchorFlow(len(protocol.scenes), len(protocol.seen_devices),
                             model.feature_dim, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"],
                                  weight_decay=training["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, training["epochs"])
    best, start = -1.0, 0
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if checkpoint["settings"] != settings or checkpoint["protocol"] != asdict(protocol):
            raise ValueError("resume configuration or protocol differs from checkpoint")
        model.load_state_dict(checkpoint["model"])
        regularizer.load_state_dict(checkpoint["anchorflow"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start, best = checkpoint["epoch"] + 1, checkpoint["best"]
        generator.set_state(checkpoint["loader_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    (output / "config.json").write_text(json.dumps(
        {"settings": settings, "protocol": asdict(protocol), "seed": seed}, indent=2),
        encoding="utf-8")
    for epoch in range(start, training["epochs"]):
        model.train()
        running, energy_total, eligible_total, steps = 0.0, 0.0, 0, 0
        for batch in train_loader:
            inputs = batch["features"].to(device)
            scenes, devices = batch["scene"].to(device), batch["device"].to(device)
            optimizer.zero_grad(set_to_none=True)
            features = model.encoder(inputs)
            logits = model.head(features)
            base = F.cross_entropy(logits, scenes)
            flow_loss = regularizer(features, scenes, devices, model.head, epoch)
            loss = base + config.weight * flow_loss.energy
            regularizer.bank.stage(features, scenes, devices)
            if not torch.isfinite(loss):
                regularizer.bank.discard()
                raise FloatingPointError("non-finite objective; staged anchors discarded")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip"])
            if not torch.isfinite(norm):
                regularizer.bank.discard()
                raise FloatingPointError("non-finite gradient; staged anchors discarded")
            optimizer.step()
            regularizer.bank.commit()
            running += base.item()
            energy_total += flow_loss.energy.item()
            eligible_total += flow_loss.eligible_samples
            steps += 1
        scheduler.step()
        metrics = evaluate(model, val_loader, protocol, device)
        improved = metrics["overall"] > best
        best = max(best, metrics["overall"])
        record = {"epoch": epoch, "base_loss": running / steps,
                  "path_energy": energy_total / steps, "eligible_samples": eligible_total,
                  "source_validation": metrics}
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        payload = {"model": model.state_dict(), "anchorflow": regularizer.state_dict(),
                   "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                   "epoch": epoch, "best": best, "settings": settings,
                   "protocol": asdict(protocol), "loader_rng": generator.get_state(),
                   "torch_rng": torch.get_rng_state(),
                   "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        save_atomic(payload, output / "last.pt")
        if improved:
            save_atomic(payload, output / "best.pt")


def load_scene_model(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    protocol = Protocol.from_dict(checkpoint["protocol"])
    model = build_model(checkpoint["settings"]["model"], len(protocol.scenes))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), protocol, checkpoint
