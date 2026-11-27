"""Original 12 unit tests from pilot v0.1 (EHPC-QCP-2026Q04-092 Feasibility Proof)."""
import math
import unittest

import numpy as np

import pilot


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.weights = np.array([2, 5, 3])
        self.edges = [(0, 1), (1, 2)]

    def test_all_energies_match_independent_definition(self):
        actual = pilot.energy_table(self.weights, self.edges)
        penalty = sum(self.weights) + 1
        expected = []
        for index in range(8):
            bits = [(index >> i) & 1 for i in range(3)]
            expected.append(-sum(w * x for w, x in zip(self.weights, bits))
                            + penalty * sum(bits[i] * bits[j] for i, j in self.edges))
        np.testing.assert_array_equal(actual, expected)

    def test_optimum_is_feasible_and_has_known_weight(self):
        energy = pilot.energy_table(self.weights, self.edges)
        self.assertEqual(float(np.min(energy)), -5.0)
        self.assertEqual(pilot.valid_mask(3, self.edges)[np.argmin(energy)], True)

    def test_validity_uses_original_conflicts(self):
        mask = pilot.valid_mask(3, self.edges)
        self.assertEqual(mask.tolist(), [True, True, True, False, True, True, False, False])

    def test_seeded_instances_repeat(self):
        wa, ea = pilot.make_instance(8, 4201)
        wb, eb = pilot.make_instance(8, 4201)
        np.testing.assert_array_equal(wa, wb)
        self.assertEqual(ea, eb)
        self.assertTrue(np.all(wa > 0))
        self.assertTrue(all(0 <= a < b < 8 for a, b in ea))

    def test_invalid_samples_are_not_repaired(self):
        energies = pilot.energy_table(self.weights, self.edges)
        row = pilot.sample_metrics(np.array([7, 7]), energies,
                                   pilot.valid_mask(3, self.edges))
        self.assertFalse(row['selected_valid'])
        self.assertIsNone(row['feasible_weight_ratio'])
        self.assertEqual(row['valid_fraction'], 0.0)

    def test_metrics_and_ties(self):
        energies = pilot.energy_table(self.weights, self.edges)
        row = pilot.sample_metrics(np.array([5, 2, 7, 0]), energies,
                                   pilot.valid_mask(3, self.edges))
        self.assertEqual(row['selected_index'], 2)
        self.assertEqual(row['objective_gap'], 0.0)
        self.assertEqual(row['valid_fraction'], 0.75)
        self.assertEqual(row['optimum_fraction'], 0.5)
        self.assertEqual(row['feasible_weight_ratio'], 1.0)


class StatevectorTests(unittest.TestCase):
    def test_zero_angles_are_uniform(self):
        state = pilot.qaoa_state(np.array([0., -2., -1., 7.]), [0.], [0.])
        np.testing.assert_allclose(state, np.full(4, 0.5), atol=1e-14)

    def test_matches_independent_dense_operator(self):
        energies = np.array([0., -2., -1., 4., -3., 2., -4., 8.])
        gammas, betas = [0.37, -0.29], [0.23, 0.61]
        expected = np.full(8, 1 / math.sqrt(8), dtype=complex)
        for gamma, beta in zip(gammas, betas):
            expected = np.diag(np.exp(-1j * gamma * energies)) @ expected
            one = np.array([[math.cos(beta), -1j * math.sin(beta)],
                            [-1j * math.sin(beta), math.cos(beta)]])
            mixer = np.kron(np.kron(one, one), one)
            expected = mixer @ expected
        actual = pilot.qaoa_state(energies, gammas, betas)
        np.testing.assert_allclose(actual, expected, atol=2e-14)
        self.assertAlmostEqual(float(np.vdot(actual, actual).real), 1.0, places=13)

    def test_analytic_one_qubit_minimum(self):
        state = pilot.qaoa_state(np.array([0., 1.]), [math.pi / 2], [-math.pi / 4])
        np.testing.assert_allclose(abs(state) ** 2, [1., 0.], atol=1e-14)

    def test_depth_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            pilot.qaoa_state(np.zeros(4), [0.1], [0.2, 0.3])


class AnnealingTests(unittest.TestCase):
    def test_every_returned_energy_matches_enumeration(self):
        weights, edges = pilot.make_instance(8, 4202)
        indices, terminal_energies = pilot.anneal(weights, edges, seed=3, reads=32, sweeps=10)
        np.testing.assert_array_equal(terminal_energies, pilot.energy_table(weights, edges)[indices])
        self.assertEqual(len(indices), 32)

    def test_single_variable_finds_known_solution(self):
        indices, energies = pilot.anneal(np.array([5]), [], seed=4, reads=32, sweeps=20)
        self.assertEqual(float(np.min(energies)), -5.0)
        self.assertIn(1, indices)


if __name__ == '__main__':
    unittest.main()
