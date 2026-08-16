import unittest

import numpy as np
import torch

from urbanheat.config import SPLIT_SEED
from urbanheat.data import split_cases, validate_physics_time_coverage
from urbanheat.models import MultiScaleFourierFeatures


class ReproducibilityTests(unittest.TestCase):
    def test_physics_trajectory_gate(self):
        cases = []
        for hour in range(7, 19):
            cases.append({
                'filename': f'Case_50_33_{hour}_x.csv',
                'params': {'humidity': 50.0, 'temperature': 33.0,
                           'hour': float(hour)},
            })
        validate_physics_time_coverage(cases)
        with self.assertRaises(ValueError):
            validate_physics_time_coverage(cases[:-1])

    def test_fourier_rng_contract_is_stable(self):
        torch.manual_seed(42)
        expected_parts = [torch.randn(4, 2) * scale
                          for scale in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)]
        expected_parts.append(torch.randn(4, 4) * 5.0)
        expected_basis = torch.cat(expected_parts, dim=1)
        expected_torch_state = torch.get_rng_state().clone()
        np.random.seed(42)
        expected_numpy_state = np.random.get_state()

        np.random.seed(123)
        torch.manual_seed(123)
        layer = MultiScaleFourierFeatures(
            input_dim=4, num_frequencies=16, seed=42)

        torch.testing.assert_close(layer.B, expected_basis)
        self.assertTrue(torch.equal(torch.get_rng_state(),
                                    expected_torch_state))
        actual_numpy_state = np.random.get_state()
        self.assertEqual(expected_numpy_state[0], actual_numpy_state[0])
        np.testing.assert_array_equal(
            expected_numpy_state[1], actual_numpy_state[1])
        self.assertEqual(expected_numpy_state[2:], actual_numpy_state[2:])

    def test_split_rng_progression_is_stable(self):
        cases = [
            {'params': {'humidity': float(i), 'temperature': float(30 + i)}}
            for i in range(10)]
        expected_keys = sorted(
            (c['params']['humidity'], c['params']['temperature'])
            for c in cases)
        np.random.seed(SPLIT_SEED)
        np.random.shuffle(expected_keys)
        expected_next = np.random.choice(1000, size=20, replace=False)

        np.random.seed(999)
        split_cases(cases, log=lambda *_: None)
        actual_next = np.random.choice(1000, size=20, replace=False)
        np.testing.assert_array_equal(actual_next, expected_next)


if __name__ == '__main__':
    unittest.main()
