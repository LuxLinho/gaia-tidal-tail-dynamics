import importlib.util
from pathlib import Path
import unittest
import numpy as np
from astropy.table import Table, MaskedColumn

PATH = Path(__file__).resolve().parents[1] / 'src/project03/01_observed_phase_space_diagnostics.py'
spec = importlib.util.spec_from_file_location('phase_space', PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class PhaseSpaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = Table.read(m.INPUT_PATH)
        cls.master = Table.read(m.MASTER_PATH)

    def test_missing_and_extreme_values(self):
        a = MaskedColumn([1., 2., 1e9, 0., np.nan, np.inf], mask=[0, 0, 0, 1, 0, 0])
        s = m.percentile_summary(a)
        self.assertEqual(s['n_valid'], 3)
        self.assertEqual(s['median'], 2)
        self.assertEqual(s['max'], 1e9)
        self.assertAlmostEqual(s['p16'], 1.32)
        self.assertTrue(np.isnan(m.percentile_summary([np.nan])['median']))

    def test_real_invariants(self):
        m.validate_invariants(self.table, self.master)

    def test_reject_duplicates_missing_rows_and_replacements(self):
        for variant in ('duplicate', 'missing', 'replacement'):
            t = self.table.copy()
            if variant == 'duplicate':
                t['source_id'][0] = t['source_id'][1]
            elif variant == 'missing':
                t = t[:-1]
            else:
                t['source_id'][0] = 1
            with self.assertRaises(ValueError):
                m.validate_invariants(t, self.master)

    def test_reject_label_swap_preserving_counts(self):
        t = self.table.copy()
        a = np.flatnonzero(t['homogenized_class'] == 'C')[0]
        b = np.flatnonzero(t['homogenized_class'] == 'L')[0]
        t['homogenized_class'][a], t['homogenized_class'][b] = 'L', 'C'
        with self.assertRaises(ValueError):
            m.validate_invariants(t, self.master)

    def test_summary_counts_and_ecsv_roundtrip(self):
        import tempfile
        full, grouped = m.summary_tables(self.table)
        self.assertEqual(len(full), 6)
        self.assertEqual(len(grouped), 36)
        self.assertEqual(full[full['quantity'] == 'radial_velocity']['n_valid'][0], 886)
        for field in m.GROUPS:
            rows = grouped[(grouped['group_column'] == field) & (grouped['quantity'] == 'radial_velocity')]
            self.assertEqual(sum(rows['n_total']), 1456)
            self.assertEqual(sum(rows['n_valid']), 886)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'summary.ecsv'
            full.write(p)
            loaded = Table.read(p)
            np.testing.assert_equal(loaded['median'], full['median'])
            self.assertEqual(list(loaded['unit']), list(full['unit']))

    def test_scatter_coordinates_and_full_limits(self):
        for x, y in [('ra', 'dec'), ('pmra', 'pmdec'), ('parallax', 'pmra'), ('parallax', 'pmdec')]:
            fig, ax = m.scatter(self.table, x, y, 'homogenized_class')
            self.assertEqual(sum(len(c.get_offsets()) for c in ax.collections), 1456)
            for group, collection in zip(m.GROUPS['homogenized_class'], ax.collections):
                mask = self.table['homogenized_class'] == group
                np.testing.assert_equal(collection.get_offsets()[:, 0], m.values(self.table[x])[mask])
                np.testing.assert_equal(collection.get_offsets()[:, 1], m.values(self.table[y])[mask])
            for quantity, limits in [(x, ax.get_xlim()), (y, ax.get_ylim())]:
                a = m.values(self.table[quantity])
                self.assertLessEqual(limits[0], np.nanmin(a))
                self.assertGreaterEqual(limits[1], np.nanmax(a))
            self.assertEqual(ax.get_xlabel(), m.LABELS[x])
            self.assertEqual(ax.get_ylabel(), m.LABELS[y])
            m.plt.close(fig)

    def test_histogram_contains_all_finite_measurements(self):
        for q in ('parallax', 'radial_velocity'):
            fig, ax = m.histogram(self.table, q)
            total = 0
            for patch in ax.patches:
                # Step histogram polygon has two vertices per bin plus endpoints.
                total += patch.get_xy()[1:-1:2, 1].sum()
            self.assertEqual(total, np.isfinite(m.values(self.table[q])).sum())
            a = m.values(self.table[q])
            self.assertLessEqual(ax.get_xlim()[0], np.nanmin(a))
            self.assertGreaterEqual(ax.get_xlim()[1], np.nanmax(a))
            m.plt.close(fig)


if __name__ == '__main__':
    unittest.main()
