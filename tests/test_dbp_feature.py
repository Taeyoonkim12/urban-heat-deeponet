"""Signed building-proximity feature: pin every branch of the definition.

Section 2.3.3 assigns a negative value inside the 1 m horizontal buffer, the
horizontal distance for query points at or below the height of the nearest
building-surface point, and the three-dimensional Euclidean distance above it.
"""

import os
import sys
import unittest

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from urbanheat.geometry import compute_dbp  # noqa: E402
from utils.dbp_feature import dbp_feature  # noqa: E402


class SignedBuildingProximityTests(unittest.TestCase):
    def setUp(self):
        self.building_xy = np.array([[0.0, 0.0]])
        self.building_z = np.array([5.0])
        self.tree = cKDTree(self.building_xy)

    def _both(self, query):
        return (compute_dbp(query, self.tree, self.building_z),
                dbp_feature(query, self.tree, self.building_z))

    def test_at_or_below_building_height_uses_horizontal_distance(self):
        # (3, 4) is 5 m away horizontally; z = 3 lies below the building top,
        # so d_BP is the horizontal distance and NOT sqrt(5^2 + 2^2).
        query = np.array([[3.0, 4.0, 3.0]])
        for values in self._both(query):
            self.assertAlmostEqual(float(values[0]), 5.0, places=5)

    def test_exactly_at_building_height_uses_horizontal_distance(self):
        query = np.array([[3.0, 4.0, 5.0]])
        for values in self._both(query):
            self.assertAlmostEqual(float(values[0]), 5.0, places=5)

    def test_above_building_height_uses_three_dimensional_distance(self):
        query = np.array([[3.0, 4.0, 17.0]])
        expected = float(np.sqrt(5.0 ** 2 + 12.0 ** 2))
        for values in self._both(query):
            self.assertAlmostEqual(float(values[0]), expected, places=5)

    def test_inside_one_metre_buffer_is_negative(self):
        query = np.array([[0.2, 0.0, 9.0]])
        for values in self._both(query):
            self.assertAlmostEqual(float(values[0]), -(4.0 + 0.1), places=5)
            self.assertLess(float(values[0]), 0.0)

    def test_pipeline_and_reference_implementations_agree(self):
        rng = np.random.default_rng(0)
        query = np.column_stack([rng.uniform(-50, 50, 500),
                                 rng.uniform(-50, 50, 500),
                                 rng.uniform(-5, 40, 500)])
        pipeline, reference = self._both(query)
        self.assertTrue(np.allclose(pipeline, reference, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
