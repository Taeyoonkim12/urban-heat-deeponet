import inspect
import unittest

from urbanheat.physics_operator import RayPhysicsOperator, RayPhysicsRegularizer
from urbanheat.models import MODEL_SPECS


class PhysicsContractTests(unittest.TestCase):
    def test_public_model_registry_and_matched_solar_control(self):
        self.assertEqual(
            set(MODEL_SPECS),
            {'m0', 'm1', 'm2', 'm3', 'm4', 'm5', 'd1', 'solar_control'})
        m5 = MODEL_SPECS['m5']
        control = MODEL_SPECS['solar_control']
        self.assertTrue(m5['use_seb'])
        self.assertFalse(control['use_seb'])
        architecture_keys = (
            'branch', 'use_dbp', 'use_fourier', 'use_category', 'coupling')
        self.assertEqual(
            {key: m5[key] for key in architecture_keys},
            {key: control[key] for key in architecture_keys})

    def test_training_interface_has_no_target_argument(self):
        regularizer_args = set(inspect.signature(
            RayPhysicsRegularizer.__call__).parameters)
        sample_args = set(inspect.signature(RayPhysicsOperator.sample).parameters)
        forbidden = {'target', 'targets', 'temperature_target', 'residual', 'error'}
        self.assertFalse(regularizer_args & forbidden)
        self.assertFalse(sample_args & forbidden)


if __name__ == '__main__':
    unittest.main()
