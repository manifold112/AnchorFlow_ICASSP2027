"""Command-line entry points for training and data management."""

import argparse
import json
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

from .data import CachedLogMel, Protocol, read_manifest, validate_rows
from .engine import evaluate, load_scene_model, save_atomic, train


def load_yaml(path):
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def main(argv=None):
    parser = argparse.ArgumentParser(description="AnchorFlow | Scene-conditioned device flows")
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train", help="train with source-only model selection")
    fit.add_argument("--config", required=True)
    fit.add_argument("--protocol", required=True)
    fit.add_argument("--train-csv", required=True)
    fit.add_argument("--val-csv", required=True)
    fit.add_argument("--feature-root", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--device", default="cpu")
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--resume")
    validation = sub.add_parser("validate-data", help="validate a device-aware manifest")
    validation.add_argument("--manifest", required=True)
    validation.add_argument("--protocol", required=True)
    validation.add_argument("--source-only", action="store_true")
    evaluation = sub.add_parser("evaluate", help="evaluate a scene checkpoint")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--manifest", required=True)
    evaluation.add_argument("--feature-root", required=True)
    evaluation.add_argument("--device", default="cpu")
    evaluation.add_argument("--batch-size", type=int, default=64)
    export = sub.add_parser("export", help="export encoder/head weights without training state")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "train":
        train(load_yaml(args.config), Protocol.from_dict(load_yaml(args.protocol)),
              args.train_csv, args.val_csv, args.feature_root, args.output,
              args.device, args.seed, args.resume)
    elif args.command == "validate-data":
        rows = read_manifest(args.manifest)
        validate_rows(rows, Protocol.from_dict(load_yaml(args.protocol)), args.source_only)
        print(json.dumps({"segments": len(rows), "valid": True}))
    elif args.command == "evaluate":
        model, protocol, _ = load_scene_model(args.checkpoint, args.device)
        rows = read_manifest(args.manifest)
        validate_rows(rows, protocol)
        loader = DataLoader(CachedLogMel(rows, protocol, args.feature_root),
                            batch_size=args.batch_size, shuffle=False)
        print(json.dumps(evaluate(model, loader, protocol, args.device), indent=2))
    elif args.command == "export":
        model, protocol, checkpoint = load_scene_model(args.checkpoint)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_atomic({"model": model.state_dict(), "model_config": checkpoint["settings"]["model"],
                     "scenes": list(protocol.scenes)}, path)
        print(f"Scene model exported to {path}")


if __name__ == "__main__":
    main()
