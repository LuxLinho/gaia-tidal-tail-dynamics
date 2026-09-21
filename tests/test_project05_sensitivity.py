"""Tests for the focused sensitivity audit, without altering baseline products."""
from pathlib import Path
import json
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/project05'))
import common as c
s = c.module('test_reference_sensitivity', 'src/project05/reference_sensitivity.py')


class ReferenceSensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = c.read(s.INPUT)
        cls.ref, cls.stored, cls.mask = s.coherent_reference(cls.table)
        cls.orbit = s.B.integrate(cls.ref)

    def test_exact_matched_sample_and_medians(self):
        self.assertEqual(int(self.mask.sum()), 594)
        self.assertTrue(np.all(self.table['homogenized_class'][self.mask] == 'C'))
        fields = s.P.b.POSITION[:3] + s.P.b.VELOCITY[:3]
        raw = np.column_stack([c.array(self.table, q) for q in fields])
        expected = (self.table['homogenized_class']=='C') & np.all(np.isfinite(raw), axis=1)
        np.testing.assert_array_equal(self.mask, expected)
        np.testing.assert_array_equal([self.stored[q][0] for q in s.P.REF_POSITION+s.P.REF_VELOCITY], np.median(raw[expected], axis=0))
        self.assertEqual(self.stored['n_position_contributors'][0], self.stored['n_velocity_contributors'][0])

    def test_excluded_sources_do_not_influence_reference(self):
        t = self.table.copy()
        # Complete non-C and C rows lacking a velocity must not influence either median.
        for q in s.P.b.POSITION[:3]:
            t[q][~self.mask] = 1e9
        ref, stored, mask = s.coherent_reference(t)
        np.testing.assert_array_equal(mask, self.mask)
        for q in c.POSITION+c.VELOCITY:
            np.testing.assert_array_equal(ref[q], self.ref[q])

    def test_velocity_matches_frozen_reference(self):
        baseline = c.read(c.REFERENCE)
        for q in c.VELOCITY:
            np.testing.assert_array_equal(baseline[q], self.ref[q])
        self.assertEqual(s.B.validate(self.ref, self.orbit)['status'], 'PASS')
        original = c.read(c.ORBIT)

        # Scientific configuration must remain identical between the frozen
        # baseline orbit and the sensitivity orbit.
        for key in ('integration', 'potential', 'adopted_parameters'):
            self.assertEqual(original.meta[key], self.orbit.meta[key])

        # Software metadata records the environment used to generate each
        # product. Historical products may legitimately have been generated
        # under different Python/Astropy/NumPy/SciPy versions, so exact
        # dictionary equality is not a scientific invariant.
        for meta in (original.meta, self.orbit.meta):
            self.assertIn('software', meta)
            self.assertIn('galpy', meta['software'])

        # The dynamics package itself is part of the frozen Project 05 setup.
        self.assertEqual(
            original.meta['software']['galpy'],
            self.orbit.meta['software']['galpy'],
        )

    def test_handedness_and_native_galpy_prograde(self):
        audit = s.convention_audit(self.ref, self.orbit)
        self.assertEqual(audit['reflection_determinant'], -1)
        self.assertTrue(audit['physically_prograde'])
        self.assertAlmostEqual(audit['Astropy_Lz_kpc_kms'], -audit['galpy_Lz_kpc_kms'])
        self.assertAlmostEqual(audit['Astropy_vphi_kms'], -audit['galpy_vT_kms'])

    def test_review_gate_stops_for_material_change(self):
        base = s.D.diagnostics(c.read(c.ORBIT))
        rows = {key: s.comparison(base[key], base[key], '') for key in s.KEYS}
        self.assertEqual(s.assess(rows, base)['decision'], 'RETAIN_FROZEN_REFERENCE')
        rows['apocentre_kpc'] = s.comparison(base['apocentre_kpc'], base['apocentre_kpc']*1.1, 'kpc')
        self.assertEqual(s.assess(rows, base)['decision'], 'STOP_FOR_SCIENTIFIC_REVIEW')

    def test_zero_and_unavailable_relative_differences(self):
        self.assertIsNone(s.comparison(0, .1, 'kpc')['absolute_relative_difference'])
        self.assertEqual(s.comparison(-2, -1, '')['absolute_relative_difference'], .5)
        base = s.D.diagnostics(c.read(c.ORBIT)); base['radial_period_myr'] = None
        rows = {key: s.comparison(base[key], base[key], '') for key in s.KEYS}
        self.assertEqual(s.assess(rows, base)['decision'], 'STOP_FOR_SCIENTIFIC_REVIEW')

    def test_serialized_results_and_baseline_integrity(self):
        result = json.loads((s.OUT/'reference_state_sensitivity.json').read_text())
        self.assertEqual(result['sample_n'], 594)
        self.assertEqual(len(set(result['source_ids'])), 594)
        self.assertEqual(result['assessment']['decision'], 'RETAIN_FROZEN_REFERENCE')
        self.assertTrue((s.OUT/'reference_state_sensitivity.txt').is_file())
        # The historical audit checked README at execution time; later projects
        # legitimately update its status. Keep scientific baseline hashes frozen.
        self.assertTrue(result['preservation']['README_unchanged'])
        for path in [c.REFERENCE, c.ORBIT, c.POPULATION]:
            self.assertEqual(c.sha(path), result['preservation']['protected_sha256'][str(path.relative_to(c.ROOT))])
        diag = s.D.diagnostics(self.orbit)
        for key in s.KEYS:
            self.assertAlmostEqual(result['orbital_diagnostics'][key]['comparison'], diag[key], places=10)


if __name__ == '__main__':
    unittest.main()
