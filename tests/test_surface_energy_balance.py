import math
import unittest

import numpy as np
import torch

from urbanheat.models import build_model
from urbanheat.seb import SIGMA_SB, SurfaceEnergyBalanceLoss


class _Scaler:
    mean_ = np.array([0.0])
    scale_ = np.array([1.0])


class SurfaceEnergyBalanceTests(unittest.TestCase):
    def test_m5_has_no_trainable_physical_coefficient(self):
        model, _ = build_model('m5')
        physical_tokens = ('sky', 'c_areal', 'ceff', 'h_conv',
                           'h_roof', 'h_wall')
        found = [name for name, _ in model.named_parameters()
                 if any(token in name.lower() for token in physical_tokens)]
        self.assertEqual(found, [])

    def test_roof_and_wall_convection_are_distinct_terms(self):
        loss_fn = SurfaceEnergyBalanceLoss(
            _Scaler(), h_roof=10.0, h_wall=8.0,
            c_areal=1.0, resid_scale=100.0)
        _, detail = loss_fn(
            torch.tensor([[30.0], [30.0]]), torch.empty((0, 1)),
            torch.tensor([[30.0], [30.0]]),
            torch.tensor([[30.0], [30.0]]),
            torch.tensor([[-1, -2], [-1, -2]]), torch.empty(0),
            torch.tensor([0.0, 0.0]), torch.tensor([1.0, 0.0]),
            torch.tensor(25.0), torch.tensor(3600.0),
            torch.tensor([1.0, 1.0]), return_details=True)
        torch.testing.assert_close(
            detail['q_conv'], torch.tensor([50.0, 40.0]))

    def test_flux_arithmetic_weighting_and_gradients(self):
        loss_fn = SurfaceEnergyBalanceLoss(
            _Scaler(), h_roof=10.0, h_wall=8.0,
            c_areal=100_000.0, resid_scale=100.0)
        self.assertEqual(list(loss_fn.named_parameters()), [])
        self.assertTrue(all(not value.requires_grad
                            for _, value in loss_fn.named_buffers()))

        receiver = torch.tensor([[30.0]], requires_grad=True)
        emitter = torch.tensor([[20.0]], requires_grad=True)
        previous = torch.tensor([[29.0]], requires_grad=True)
        following = torch.tensor([[31.0]], requires_grad=True)
        ray_index = torch.tensor([[0, -1, -2, 0]])
        emitter_eps = torch.tensor([0.95])
        qsw = torch.tensor([500.0])
        normal_z = torch.tensor([1.0])
        ambient = torch.tensor(25.0)
        dt_seconds = torch.tensor(7200.0)
        confidence = torch.tensor([0.4])

        loss, detail = loss_fn(
            receiver, emitter, previous, following, ray_index,
            emitter_eps, qsw, normal_z, ambient, dt_seconds, confidence,
            return_details=True)

        incoming = 0.5 * 0.95 * SIGMA_SB * (20.0 + 273.15) ** 4
        outgoing = SIGMA_SB * (30.0 + 273.15) ** 4
        q_lw = 0.5 * (incoming - outgoing)
        q_conv = 10.0 * (30.0 - 25.0)
        q_storage = 100_000.0 * (31.0 - 29.0) / 7200.0
        expected_residual = 500.0 + q_lw - q_conv - q_storage
        expected_loss = (expected_residual / 100.0) ** 2

        self.assertAlmostEqual(detail['residual'].item(), expected_residual, places=4)
        self.assertAlmostEqual(loss.item(), expected_loss, places=5)

        loss.backward()
        for value in (receiver, emitter, previous, following):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0.0)

    def test_original_ray_denominator_is_preserved(self):
        loss_fn = SurfaceEnergyBalanceLoss(
            _Scaler(), h_roof=10.0, h_wall=10.0,
            c_areal=1.0, resid_scale=1.0)
        base = dict(
            pred_receiver=torch.tensor([[25.0]]),
            pred_emitter=torch.tensor([[25.0]]),
            pred_prev=torch.tensor([[25.0]]),
            pred_next=torch.tensor([[25.0]]),
            emitter_emissivity=torch.tensor([1.0]),
            qsw=torch.tensor([0.0]),
            receiver_normal_z=torch.tensor([1.0]),
            ambient_temperature=torch.tensor(25.0),
            dt_seconds=torch.tensor(3600.0),
            confidence=torch.tensor([1.0]),
            return_details=True,
        )
        _, one_hit = loss_fn(ray_index=torch.tensor([[0, -1, -2, -2]]), **base)
        expected = SIGMA_SB * (25.0 + 273.15) ** 4 / 4.0
        self.assertAlmostEqual(one_hit['incoming_lw'].item(), expected, places=4)

    def test_invalid_zero_weight_is_rejected(self):
        loss_fn = SurfaceEnergyBalanceLoss(
            _Scaler(), h_roof=10.0, h_wall=10.0,
            c_areal=1.0, resid_scale=1.0)
        with self.assertRaises(ValueError):
            loss_fn(
                torch.tensor([[25.0]]), torch.empty((0, 1)),
                torch.tensor([[25.0]]), torch.tensor([[25.0]]),
                torch.tensor([[-1, -2]]), torch.empty(0),
                torch.tensor([0.0]), torch.tensor([1.0]),
                torch.tensor(25.0), torch.tensor(3600.0),
                torch.tensor([0.0]))

    def test_weight_normalization_is_composition_invariant(self):
        loss_fn = SurfaceEnergyBalanceLoss(
            _Scaler(), h_roof=10.0, h_wall=10.0,
            c_areal=1.0, resid_scale=100.0)
        common = dict(
            pred_emitter=torch.empty((0, 1)),
            emitter_emissivity=torch.empty(0),
            qsw=torch.tensor([100.0, 100.0]),
            receiver_normal_z=torch.tensor([1.0, 1.0]),
            ambient_temperature=torch.tensor(25.0),
            dt_seconds=torch.tensor(3600.0),
        )
        loss = loss_fn(
            torch.tensor([[25.0], [25.0]]), common['pred_emitter'],
            torch.tensor([[25.0], [25.0]]),
            torch.tensor([[25.0], [25.0]]),
            torch.tensor([[-1, -2], [-1, -2]]),
            common['emitter_emissivity'], common['qsw'],
            common['receiver_normal_z'], common['ambient_temperature'],
            common['dt_seconds'], torch.tensor([0.1, 0.9]))
        outgoing = 0.5 * SIGMA_SB * (25.0 + 273.15) ** 4
        expected = ((100.0 - outgoing) / 100.0) ** 2
        self.assertAlmostEqual(loss.item(), expected, places=5)


if __name__ == '__main__':
    unittest.main()
