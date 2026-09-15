import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import tempfile
import numpy as np
from astropy.table import Table, MaskedColumn
from astropy import units as u
from astropy.coordinates import SkyCoord

PATH = Path(__file__).resolve().parents[1]/'src/project03/02_heliocentric_phase_space.py'
spec = importlib.util.spec_from_file_location('helio', PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class HeliocentricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = Table.read(m.a.INPUT_PATH)
        cls.result = m.transform(cls.table)

    def test_rows_ids_observables_and_roundtrip(self):
        checks = m.validate_output(self.table, self.result)
        self.assertEqual(len(self.result), 1456)
        self.assertEqual(len(set(self.result['source_id'])), 1456)
        self.assertEqual(checks['n_position'], 1456)
        self.assertEqual(checks['n_velocity'], 886)

    def test_missing_rv_remains_missing(self):
        missing = ~np.isfinite(m.a.values(self.table['radial_velocity']))
        self.assertEqual(missing.sum(), 570)
        for q in m.VELOCITY:
            self.assertTrue(np.isnan(m.a.values(self.result[q])[missing]).all())
        self.assertTrue(np.isfinite(m.a.values(self.result['r_helio_pc'])[missing]).all())

    def test_invalid_parallax_and_astrometry_preserve_rows(self):
        t = self.table[:6].copy()
        t['parallax'][:4] = [0, -1, np.nan, np.inf]
        t['pmra'][4] = np.nan
        t['dec'][5] = np.nan
        result = m.transform(t)
        checks = m.validate_output(t, result)
        self.assertEqual(len(result), 6)
        self.assertEqual(checks['n_position'], 1)
        self.assertEqual(checks['n_velocity'], 0)

    def test_masked_positive_parallax_not_used(self):
        t = self.table[:2].copy()
        t.replace_column('parallax', MaskedColumn([2., 2.], mask=[True, False], unit='mas'))
        result = m.transform(t)
        self.assertTrue(np.isnan(result['x_helio_pc'][0]))
        self.assertTrue(np.isfinite(result['x_helio_pc'][1]))

    def test_units_norm_and_ecsv(self):
        for q, unit in m.UNITS.items():
            self.assertEqual(self.result[q].unit, u.Unit(unit))
        xyz = np.array([self.result[q] for q in m.POSITION[:3]])
        np.testing.assert_allclose(np.linalg.norm(xyz, axis=0), 1000/self.table['parallax'], rtol=1e-12)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'roundtrip.ecsv'
            self.result.write(p)
            m.validate_output(self.table, Table.read(p))

    def test_axis_and_velocity_signs_using_astropy(self):
        # Galactic cardinal directions converted by Astropy, no copied rotation constants.
        c = SkyCoord(l=[0,90,0]*u.deg, b=[0,0,90]*u.deg, distance=[100]*3*u.pc,
                     pm_l_cosb=[0]*3*u.mas/u.yr, pm_b=[0]*3*u.mas/u.yr,
                     radial_velocity=[10]*3*u.km/u.s, frame='galactic').icrs
        t = Table(dict(ra=c.ra, dec=c.dec, parallax=[10]*3*u.mas,
                       pmra=c.pm_ra_cosdec, pmdec=c.pm_dec, radial_velocity=c.radial_velocity))
        result = m.transform(t)
        np.testing.assert_allclose(np.array([result[q] for q in m.POSITION[:3]]), np.eye(3)*100, atol=1e-10)
        np.testing.assert_allclose(np.array([result[q] for q in m.VELOCITY[:3]]), np.eye(3)*10, atol=1e-10)

    def test_group_counts(self):
        full, groups = m.summaries(self.result)
        self.assertEqual(len(full), 8)
        self.assertEqual(len(groups), 48)
        for field in m.a.GROUPS:
            rows = groups[(groups['group_column'] == field) & (groups['quantity'] == 'x_helio_pc')]
            self.assertEqual(sum(rows['n_positional']), 1456)
            self.assertEqual(sum(rows['n_velocity']), 886)

    def test_plots_coordinates_counts_and_limits(self):
        for fields in (m.POSITION, m.VELOCITY):
            for i,j in [(0,1),(0,2),(1,2)]:
                x,y = fields[i],fields[j]
                fig, ax = m.scatter(self.result,x,y,'homogenized_class')
                total = 0
                for group, collection in zip(m.a.GROUPS['homogenized_class'], ax.collections):
                    mask = (self.result['homogenized_class'] == group) & np.isfinite(m.a.values(self.result[x]))
                    offsets = collection.get_offsets()
                    total += len(offsets)
                    np.testing.assert_allclose(offsets[:,0], self.result[x][mask])
                    np.testing.assert_allclose(offsets[:,1], self.result[y][mask])
                self.assertEqual(total, 1456 if fields is m.POSITION else 886)
                if fields is m.POSITION:
                    self.assertEqual(ax.get_aspect(), 1.)
                for q, limits in [(x,ax.get_xlim()),(y,ax.get_ylim())]:
                    self.assertLessEqual(limits[0],np.nanmin(self.result[q]))
                    self.assertGreaterEqual(limits[1],np.nanmax(self.result[q]))
                m.plt.close(fig)
        fig,ax = m.histogram(self.result)
        self.assertEqual(sum(p.get_xy()[1:-1:2,1].sum() for p in ax.patches),1456)
        m.plt.close(fig)

    def test_pipeline_does_not_overwrite_previous_products(self):
        before = m.protected_hashes()
        with tempfile.TemporaryDirectory(dir=m.ROOT) as d:
            with patch.object(m,'OUTPUT',Path(d)/'tables'), patch.object(m,'FIGURES',Path(d)/'figures'):
                with patch.object(m,'generate_plots',return_value=[]), patch('builtins.print'):
                    m.main()
        self.assertEqual(before,m.protected_hashes())


if __name__ == '__main__':
    unittest.main()
