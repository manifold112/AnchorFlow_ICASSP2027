"""Recording-level probability aggregation and seen/unseen accounting."""

from collections import defaultdict

import torch


class DeviceMetrics:
    def __init__(self, seen_devices, expected_segments: int = 1):
        self.seen = set(seen_devices)
        self.expected_segments = expected_segments
        self.groups = defaultdict(list)

    def update(self, probabilities, labels, devices, recording_ids):
        for probability, label, device, recording in zip(
                probabilities.detach().cpu(), labels, devices, recording_ids, strict=True):
            self.groups[(recording, device)].append((probability, int(label)))

    def compute(self):
        outcomes = defaultdict(list)
        for (_, device), segments in self.groups.items():
            if len(segments) != self.expected_segments:
                raise ValueError("incorrect number of segments in an evaluation recording")
            labels = {label for _, label in segments}
            if len(labels) != 1:
                raise ValueError("inconsistent labels in a recording group")
            prediction = torch.stack([p for p, _ in segments]).mean(0).argmax().item()
            outcomes[device].append(prediction == labels.pop())
        flattened = [correct for values in outcomes.values() for correct in values]
        seen = [x for d, values in outcomes.items() if d in self.seen for x in values]
        unseen = [x for d, values in outcomes.items() if d not in self.seen for x in values]

        def accuracy(values):
            return 100.0 * sum(values) / len(values) if values else None

        return {"overall": accuracy(flattened), "seen": accuracy(seen),
                "unseen": accuracy(unseen), "recordings": len(flattened),
                "per_device": {name: accuracy(values) for name, values in sorted(outcomes.items())}}
