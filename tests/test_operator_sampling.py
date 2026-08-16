import unittest

import numpy as np

from urbanheat.physics_operator import (RayPhysicsOperator, absorbed_shortwave,
                                        adjacent_hours)
from urbanheat.solar import (HOUR_TO_ROW, RHO_CAT, SUN_BHI, SUN_DIF, SUN_DNI,
                             SUN_VECS)


class OperatorSamplingTests(unittest.TestCase):
    @staticmethod
    def operator():
        op = RayPhysicsOperator.__new__(RayPhysicsOperator)
        op.receiver_xyz = np.arange(24, dtype=np.float32).reshape(8, 3)
        op.receiver_normal = np.tile([0.0, 0.0, 1.0], (8, 1)).astype(np.float32)
        op.receiver_shortwave_normal = op.receiver_normal.copy()
        op.receiver_dbp = np.linspace(-1, 1, 8, dtype=np.float32)
        op.receiver_confidence = np.linspace(0.2, 0.9, 8, dtype=np.float32)
        op.resolved_fraction = np.linspace(0.8, 1.0, 8, dtype=np.float32)
        op.coarse_fraction = np.linspace(0.1, 0.0, 8, dtype=np.float32)
        op.sampling_probability = np.full(8, 1 / 8, dtype=np.float64)
        op.ray_eid = np.array([
            [0, -1, -2, 1], [0, -1, 1, 2], [1, -1, 2, 3], [2, -1, 3, 0],
            [3, -1, 0, 1], [0, -1, 2, 3], [1, -1, 3, 2], [3, -1, -2, 0],
        ], dtype=np.int64)
        op.sky_code = -1
        op.unresolved_code = -2
        op.shadow_lit = np.ones((12, 8), dtype=bool)
        op.emitter_xyz = np.arange(12, dtype=np.float32).reshape(4, 3)
        op.emitter_dbp = np.zeros(4, dtype=np.float32)
        op.emitter_category = np.array([0, 1, 2, 4], dtype=np.int64)
        return op

    def test_weighting_does_not_change_receiver_sample(self):
        op = self.operator()
        uniform = op.sample(5, np.random.default_rng(91), 12, 'uniform')
        confidence = op.sample(5, np.random.default_rng(91), 12, 'confidence')
        np.testing.assert_array_equal(
            uniform['receiver_id'], confidence['receiver_id'])
        np.testing.assert_array_equal(
            uniform['ray_index'], confidence['ray_index'])
        np.testing.assert_allclose(uniform['confidence'], 1.0)
        np.testing.assert_allclose(
            confidence['confidence'],
            op.receiver_confidence[confidence['receiver_id']])

    def test_unresolved_is_not_relabelled_as_sky(self):
        op = self.operator()
        batch = op.sample(8, np.random.default_rng(2), 12, 'confidence')
        self.assertTrue((batch['ray_index'] == -2).any())
        self.assertTrue((batch['ray_index'] == -1).any())
        self.assertFalse(np.array_equal(
            batch['ray_index'] == -2, batch['ray_index'] == -1))

    def test_emitter_deduplication_preserves_ray_ids(self):
        op = self.operator()
        batch = op.sample(8, np.random.default_rng(7), 12, 'uniform')
        original = op.ray_eid[batch['receiver_id']]
        resolved = original >= 0
        reconstructed = np.full(original.shape, -99, dtype=np.int64)
        reconstructed[resolved] = batch['emitter_id'][
            batch['ray_index'][resolved]]
        np.testing.assert_array_equal(
            reconstructed[resolved], original[resolved])

    def test_shortwave_horizontal_vertical_back_and_shade(self):
        hour = 12
        row = HOUR_TO_ROW[hour]
        sun = SUN_VECS[row]
        horizontal = np.array([[0.0, 0.0, 1.0]])
        q_horizontal = absorbed_shortwave(
            horizontal, [1.0], [1.0], hour)[0]
        expected_horizontal = (1.0 - RHO_CAT[0]) * (
            SUN_BHI[row] + SUN_DIF[row])
        self.assertAlmostEqual(q_horizontal, expected_horizontal, places=4)

        vertical = np.array([[sun[0], sun[1], 0.0]])
        vertical /= np.linalg.norm(vertical, axis=1, keepdims=True)
        q_vertical = absorbed_shortwave(vertical, [1.0], [0.5], hour)[0]
        expected_vertical = (1.0 - RHO_CAT[0]) * (
            SUN_DNI[row] * max(0.0, float(vertical[0] @ sun)) +
            0.5 * SUN_DIF[row])
        self.assertAlmostEqual(q_vertical, expected_vertical, places=4)

        q_back = absorbed_shortwave(-sun.reshape(1, 3), [1.0], [0.5], hour)[0]
        q_shade = absorbed_shortwave(horizontal, [0.0], [0.5], hour)[0]
        diffuse_only = (1.0 - RHO_CAT[0]) * 0.5 * SUN_DIF[row]
        self.assertAlmostEqual(q_back, diffuse_only, places=4)
        self.assertAlmostEqual(q_shade, diffuse_only, places=4)
        with self.assertRaises(ValueError):
            absorbed_shortwave(horizontal, [1.0], [1.0], 6)

    def test_adjacent_hour_stencil(self):
        self.assertEqual(adjacent_hours(7), (7.0, 8.0, 3600.0))
        self.assertEqual(adjacent_hours(12), (11.0, 13.0, 7200.0))
        self.assertEqual(adjacent_hours(18), (17.0, 18.0, 3600.0))
        with self.assertRaises(ValueError):
            adjacent_hours(12.5)


if __name__ == '__main__':
    unittest.main()
