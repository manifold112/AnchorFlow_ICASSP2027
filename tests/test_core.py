"""Mathematical and gradient-boundary invariants, on small CPU tensors."""

import unittest

import torch
from torch import nn

from anchorflow import AnchorBank, AnchorFlow, AnchorFlowConfig, DeviceFlow
from anchorflow.regularizer import deterministic_head


class AnchorTests(unittest.TestCase):
    def setUp(self):
        self.bank = AnchorBank(2, 2, 2, AnchorFlowConfig(min_samples=2, ema_decay=0.5))
        self.h = torch.tensor([[1., 3.], [3., 5.]], requires_grad=True)
        self.y = torch.tensor([0, 0])
        self.d = torch.tensor([0, 0])

    def test_first_commit_uses_raw_moments(self):
        self.bank.stage(self.h, self.y, self.d)
        self.bank.commit()
        torch.testing.assert_close(self.bank.mean[0, 0], torch.tensor([2., 4.]))
        torch.testing.assert_close(self.bank.second[0, 0], torch.tensor([5., 17.]))
        self.assertEqual(self.bank.counts[0, 0], 2)
        self.assertIsNone(self.bank.mean.grad_fn)

    def test_paths_use_preceding_update(self):
        self.bank.stage(self.h, self.y, self.d)
        self.assertEqual(int(self.bank.counts.sum()), 0)
        self.assertFalse(bool(self.bank.eligible.any()))
        self.bank.commit()
        self.assertTrue(bool(self.bank.eligible[0, 0]))

    def test_microbatches_form_one_effective_update(self):
        self.bank.stage(self.h[:1], self.y[:1], self.d[:1])
        self.bank.stage(self.h[1:], self.y[1:], self.d[1:])
        self.bank.commit()
        torch.testing.assert_close(self.bank.mean[0, 0], torch.tensor([2., 4.]))

    def test_discard_does_not_change_anchors(self):
        self.bank.stage(self.h, self.y, self.d)
        self.bank.discard()
        self.bank.commit()
        self.assertEqual(int(self.bank.counts.sum()), 0)

    def test_ema_updates_moments_not_variance(self):
        self.bank.stage(self.h, self.y, self.d)
        self.bank.commit()
        self.bank.stage(self.h + 2, self.y, self.d)
        self.bank.commit()
        torch.testing.assert_close(self.bank.mean[0, 0], torch.tensor([3., 5.]))
        torch.testing.assert_close(self.bank.second[0, 0], torch.tensor([11., 27.]))

    def test_training_vocabulary_rejects_held_out_devices(self):
        with self.assertRaises(ValueError):
            self.bank.stage(self.h, self.y, torch.tensor([0, 6]))


class FlowTests(unittest.TestCase):
    def test_endpoints_velocity_and_standardized_residual(self):
        h = torch.tensor([[2., 6.]])
        source_mean, source_scale = torch.tensor([[0., 2.]]), torch.tensor([[2., 4.]])
        target_mean, target_scale = torch.tensor([[5., 1.]]), torch.tensor([[3., 2.]])
        flow = DeviceFlow.from_features(h, source_mean, source_scale, target_mean, target_scale)
        torch.testing.assert_close(flow.at(0), h)
        torch.testing.assert_close(flow.at(1), target_mean + target_scale * flow.residual)
        torch.testing.assert_close(flow.at(0.7) - flow.at(0.2), 0.5 * flow.velocity)
        torch.testing.assert_close((flow.at(0.5) - (source_mean + target_mean) / 2)
                                   / ((source_scale + target_scale) / 2), flow.residual)


class GradientTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.af = AnchorFlow(1, 2, 3, AnchorFlowConfig(min_samples=2, warmup_epochs=0))
        h = torch.tensor([[1., 2., 1.], [2., 3., 2.],
                          [4., 1., 3.], [5., 2., 4.]])
        self.y = torch.zeros(4, dtype=torch.long)
        self.d = torch.tensor([0, 0, 1, 1])
        self.af.bank.stage(h, self.y, self.d)
        self.af.bank.commit()
        self.features = h.clone().requires_grad_(True)
        self.head = nn.Sequential(nn.BatchNorm1d(3), nn.Dropout(0.8), nn.Linear(3, 2))

    def test_auxiliary_updates_head_and_detaches_encoder(self):
        result = self.af(self.features, self.y, self.d, self.head, epoch=1)
        result.energy.backward()
        self.assertEqual(result.eligible_samples, 4)
        self.assertIsNone(self.features.grad)
        self.assertGreater(self.head[-1].weight.grad.abs().sum().item(), 0)

    def test_auxiliary_keeps_bn_statistics_and_restores_modes(self):
        self.head.train()
        self.head[-1].eval()
        modes = [m.training for m in self.head.modules()]
        running = self.head[0].running_mean.clone()
        self.af(self.features, self.y, self.d, self.head, epoch=1)
        torch.testing.assert_close(running, self.head[0].running_mean)
        self.assertEqual(modes, [m.training for m in self.head.modules()])

    def test_auxiliary_mode_restored_after_exception(self):
        with self.assertRaises(RuntimeError):
            with deterministic_head(self.head):
                raise RuntimeError("interrupt auxiliary forward")
        self.assertTrue(self.head.training)

    def test_no_cross_scene_pairing(self):
        af = AnchorFlow(2, 2, 3, AnchorFlowConfig(min_samples=1, warmup_epochs=0))
        af.bank.stage(self.features[:2], torch.tensor([0, 1]), torch.tensor([0, 1]))
        af.bank.commit()
        result = af(self.features[:2], torch.tensor([0, 1]), torch.tensor([0, 1]),
                    self.head, epoch=1)
        self.assertEqual(result.paths, 0)
        self.assertEqual(result.energy.item(), 0)


if __name__ == "__main__":
    unittest.main()
