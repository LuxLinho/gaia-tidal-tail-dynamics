"""Project 05 scientific invariants, numerical checks, and serialized products."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
from astropy import units as u

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/project05'))
import common as c

a = c.module('test05a', 'src/project05/01_cluster_reference.py')
b = c.module('test05b', 'src/project05/02_galactic_orbit.py')
d = c.module('test05c', 'src/project05/03_orbital_diagnostics.py')


class Project05Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = a.build_reference()
        cls.orbit = b.integrate(cls.ref)
        cls.diag = d.diagnostics(cls.orbit)

    def test_canonical_population(self):
        self.assertEqual(c.population_invariants()['canonical_unique_sources'], 1456)

    def test_reference_provenance_and_counts(self):
        self.assertEqual(self.ref['n_position_contributors'][0], 940)
        self.assertEqual(self.ref['n_velocity_contributors'][0], 594)
        self.assertIn('homogenized_class == C', self.ref.meta['reference_definition'])
        old = c.read(c.REF04)
        for q in c.POSITION + c.VELOCITY:
            np.testing.assert_array_equal(self.ref[q], old[q])

    def test_reference_complete_6d_units_frame(self):
        a.validate(self.ref)
        for q in ['ra', 'dec', 'distance', 'pmra', 'pmdec', 'radial_velocity'] + c.POSITION + c.VELOCITY:
            self.assertTrue(np.isfinite(c.array(self.ref, q)).all())
            self.assertIsNotNone(self.ref[q].unit)
        self.assertEqual(self.ref.meta['adopted_parameters'], c.p04a.load_frame()[1])

    def test_reference_nonfinite_rejected(self):
        broken = self.ref.copy(); broken['radial_velocity'][0] = np.nan
        with self.assertRaises(ValueError): a.validate(broken)

    def test_time_zero_matches_reference(self):
        t = c.array(self.orbit, 'time_myr')
        self.assertEqual(np.count_nonzero(t == 0), 1)
        for q in c.POSITION + c.VELOCITY:
            np.testing.assert_allclose(c.array(self.orbit, q)[t == 0], c.array(self.ref, q), atol=1e-9, rtol=0)

    def test_time_range_and_finiteness(self):
        t = c.array(self.orbit, 'time_myr')
        np.testing.assert_array_equal(t[[0, -1]], [-500., 500.])
        self.assertEqual(len(t), 4001)
        for q in self.orbit.colnames:
            self.assertTrue(np.isfinite(c.array(self.orbit, q)).all())
            self.assertIsNotNone(self.orbit[q].unit)

    def test_continuity_conservation(self):
        self.assertEqual(b.validate(self.ref, self.orbit)['status'], 'PASS')

    def test_corrupt_trajectory_rejected(self):
        broken = self.orbit.copy(); broken['x_gc_kpc'][10] += 1
        with self.assertRaises(ValueError): b.validate(self.ref, broken)

    def test_integrator_uses_supplied_initial_state(self):
        other = self.ref.copy(); other['vz_gc_kms'][0] += 1
        orbit = b.integrate(other)
        self.assertAlmostEqual(float(orbit['vz_gc_kms'][2000]), float(other['vz_gc_kms'][0]), places=9)
        self.assertFalse(np.allclose(c.array(orbit, 'z_gc_kpc'), c.array(self.orbit, 'z_gc_kpc')))

    def test_diagnostic_bounds_and_definitions(self):
        diag = self.diag; d.validate(diag)
        R = c.array(self.orbit, 'R_gc_kpc')
        self.assertLess(diag['pericentre_kpc'], diag['apocentre_kpc'])
        self.assertAlmostEqual(diag['eccentricity'], (R.max()-R.min())/(R.max()+R.min()))
        self.assertGreaterEqual(diag['z_max_kpc'], abs(diag['z_now_kpc']))
        i = 2000
        self.assertAlmostEqual(diag['Lz_now_kpc_kms'], float(self.orbit['x_gc_kpc'][i]*self.orbit['vy_gc_kms'][i]-self.orbit['y_gc_kpc'][i]*self.orbit['vx_gc_kms'][i]))
        self.assertLess(diag['Lz_now_kpc_kms'], 0)

    def test_extrema_resolution(self):
        fine = d.diagnostics(b.integrate(self.ref, step=.125))
        for key in ('pericentre_kpc', 'apocentre_kpc', 'z_max_kpc', 'eccentricity'):
            self.assertLess(abs(self.diag[key] - fine[key]), 1e-5)

    def test_period_estimator_and_unavailable(self):
        t = np.linspace(0, 100, 10001)
        value, count = d.period(t, np.cos(2*np.pi*t/10))
        self.assertAlmostEqual(value, 10, places=2)
        self.assertIsNone(d.period(t, t)[0])

    def test_products_exist_and_roundtrip(self):
        paths = [c.REFERENCE, c.ORBIT, c.DATA/'stock2_orbital_diagnostics.ecsv']
        paths += [c.RESULTS/name for name in ['cluster_reference_metadata.json', 'cluster_reference_summary.txt',
                  'orbit_integration_metadata.json', 'orbital_diagnostics.json', 'project05_summary.txt', 'upstream_integrity.json']]
        paths += [d.FIGURES/f'{name}.{ext}' for name in d.FIGURE_NAMES for ext in ('png', 'pdf')]
        for path in paths: self.assertTrue(path.is_file(), str(path))
        a.validate(c.read(c.REFERENCE))
        b.validate(c.read(c.REFERENCE), c.read(c.ORBIT))


if __name__ == '__main__':
    unittest.main()
