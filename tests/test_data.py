import unittest

import torch

from anchorflow.data import Protocol, validate_rows, validate_split_pair
from anchorflow.metrics import DeviceMetrics


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol = Protocol("test", ("park", "metro"), ("A", "B"), ("S4",), 1)
        self.row = {"feature_path": "x.npy", "scene": "park", "device": "A",
                    "recording_id": "recording_1"}

    def test_held_out_devices_excluded_from_selection(self):
        with self.assertRaises(ValueError):
            validate_rows([{**self.row, "device": "S4"}], self.protocol, source_only=True)

    def test_recording_leakage_rejected(self):
        with self.assertRaises(ValueError):
            validate_split_pair([self.row], [{**self.row, "device": "B"}], self.protocol)

    def test_recording_probabilities_averaged_before_prediction(self):
        metric = DeviceMetrics(("A",), expected_segments=2)
        metric.update(torch.tensor([[0.51, 0.49], [0.01, 0.99]]),
                      torch.tensor([1, 1]), ["A", "A"], ["r1", "r1"])
        result = metric.compute()
        self.assertEqual(result["seen"], 100)
        self.assertEqual(result["recordings"], 1)
        self.assertIsNone(result["unseen"])


if __name__ == "__main__":
    unittest.main()
