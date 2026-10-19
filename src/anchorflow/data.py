"""Cached log-mel datasets and device-aware manifest validation."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Protocol:
    name: str
    scenes: tuple[str, ...]
    seen_devices: tuple[str, ...]
    unseen_devices: tuple[str, ...]
    segments_per_recording: int
    user_disjoint: bool = False

    @classmethod
    def from_dict(cls, values: dict):
        return cls(values["name"], tuple(values["scenes"]), tuple(values["seen_devices"]),
                   tuple(values["unseen_devices"]), values["segments_per_recording"],
                   values.get("user_disjoint", False))

    def __post_init__(self):
        if set(self.seen_devices) & set(self.unseen_devices):
            raise ValueError("seen and unseen device sets must be disjoint")
        if self.segments_per_recording < 1:
            raise ValueError("segments_per_recording must be positive")
        for values in (self.scenes, self.seen_devices, self.unseen_devices):
            if len(values) != len(set(values)) or not values:
                raise ValueError("protocol vocabularies must be nonempty and unique")


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"feature_path", "scene", "device", "recording_id"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"manifest requires columns {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError("manifest must contain at least one segment")
    for row in rows:
        if any(not row.get(key, "").strip() for key in required):
            raise ValueError("manifest contains an empty required field")
    return rows


def validate_rows(rows, protocol: Protocol, source_only: bool = False,
                  check_groups: bool = True) -> None:
    allowed = set(protocol.seen_devices)
    if not source_only:
        allowed.update(protocol.unseen_devices)
    paths, groups = set(), {}
    for row in rows:
        if row["scene"] not in protocol.scenes or row["device"] not in allowed:
            raise ValueError(f"invalid scene/device for protocol: {row['scene']}/{row['device']}")
        if row["feature_path"] in paths:
            raise ValueError(f"duplicate feature path: {row['feature_path']}")
        paths.add(row["feature_path"])
        if protocol.user_disjoint and not row.get("user_id"):
            raise ValueError("this protocol requires a nonempty user_id for every row")
        key = (row["recording_id"], row["device"])
        groups.setdefault(key, []).append(row)
    for key, segments in groups.items():
        if len({row["scene"] for row in segments}) != 1:
            raise ValueError(f"inconsistent scene labels within recording {key}")
        if check_groups and len(segments) != protocol.segments_per_recording:
            raise ValueError(f"recording {key} requires {protocol.segments_per_recording} segments")


def validate_split_pair(train_rows, other_rows, protocol: Protocol) -> None:
    # Recording identities are deliberately checked without device: paired captures
    # of the same underlying recording must not cross training/selection partitions.
    overlap = {r["recording_id"] for r in train_rows} & {r["recording_id"] for r in other_rows}
    if overlap:
        raise ValueError(f"recording leakage across partitions: {sorted(overlap)[:3]}")
    if protocol.user_disjoint:
        overlap = {r["user_id"] for r in train_rows} & {r["user_id"] for r in other_rows}
        if overlap:
            raise ValueError("user leakage across CochlScene partitions")


class CachedLogMel(Dataset):
    def __init__(self, rows, protocol: Protocol, root: str | Path):
        self.rows = rows
        self.protocol = protocol
        self.root = Path(root).resolve()
        self.scene_ids = {name: index for index, name in enumerate(protocol.scenes)}
        self.device_ids = {name: index for index, name in
                           enumerate(protocol.seen_devices + protocol.unseen_devices)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = (self.root / row["feature_path"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("feature_path must stay within feature-root")
        array = np.load(path, allow_pickle=False)
        if array.ndim == 2:
            array = array[None]
        if array.ndim != 3 or array.shape[0] != 1 or not np.isfinite(array).all():
            raise ValueError(f"invalid log-mel tensor: {path}")
        return {"features": torch.from_numpy(array.astype(np.float32)),
                "scene": self.scene_ids[row["scene"]],
                "device": self.device_ids[row["device"]],
                "device_name": row["device"], "recording_id": row["recording_id"]}
