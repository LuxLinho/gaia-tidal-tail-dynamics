import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord, Galactic, CartesianRepresentation

PATH = Path(__file__).resolve().parents[1]/'src/project03/03_cluster_centered_reference_frame.py'
spec = importlib.util.spec_from_file_location('cluster_reference', PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ClusterReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = Table.read(m.INPUT)
        cls.ref = m.reference_frame(cls.table)
        cls.result = m.relative_phase_space(cls.table, cls.ref)

    def test_canonical_preservation(self):
        m.validate_canonical(self.result)
        checks = m.validate_relative(self.table, self.result, self.ref)
        self.assertEqual(len(self.result), 1456)
        self.assertEqual(len(set(self.result['source_id'])), 1456)
        self.assertEqual(dict(zip(*np.unique(self.result['homogenized_class'], return_counts=True))),
                         {'C':940,'L':184,'T':332})
        self.assertEqual(checks['n_position'],1456)
        self.assertEqual(checks['n_velocity'],886)

    def test_reference_medians_from_existing_c_only(self):
        c = self.table['homogenized_class'] == 'C'
        self.assertEqual(self.ref['n_c_total'][0],940)
        self.assertEqual(self.ref['n_position_contributors'][0],940)
        self.assertEqual(self.ref['n_velocity_contributors'][0],594)
        for original, ref in [(m.b.POSITION,m.REF_POSITION),(m.b.VELOCITY,m.REF_VELOCITY)]:
            complete = np.all(np.isfinite(np.array([m.a.values(self.table[q]) for q in original[:3]])),axis=0)
            for q,r in zip(original,ref):
                self.assertEqual(self.ref[r][0],np.median(self.table[q][c & complete]))
        altered = self.table.copy()
        for q in m.b.POSITION[:3] + m.b.VELOCITY[:3]:
            altered[q][~c] = 1e10
        second = m.reference_frame(altered)
        for q in m.REF_POSITION + m.REF_VELOCITY:
            self.assertEqual(second[q][0], self.ref[q][0])

    def test_reference_extremes_and_incomplete_vectors(self):
        t = self.table[:4].copy()
        t['homogenized_class'] = ['C','C','C','L']
        for q in m.b.POSITION[:3]:
            t[q] = [1.,100.,1000.,-1e9]*u.pc
        for q in m.b.VELOCITY[:3]:
            t[q] = [0.,100.,200.,-1e9]*u.km/u.s
        t['vx_helio_kms'][0] = np.nan
        ref = m.reference_frame(t)
        self.assertEqual(ref['n_position_contributors'][0],3)
        self.assertEqual(ref['n_velocity_contributors'][0],2)
        for q in m.REF_POSITION:
            self.assertEqual(ref[q][0],100.)
        for q in m.REF_VELOCITY:
            self.assertEqual(ref[q][0],150.)
        t['homogenized_class'] = ['L']*4
        with self.assertRaises(ValueError):
            m.reference_frame(t)

    def test_arithmetic_norms_and_c_component_medians(self):
        for original, derived, refs in [(m.b.POSITION,m.POSITION,m.REF_POSITION),
                                       (m.b.VELOCITY,m.VELOCITY,m.REF_VELOCITY)]:
            array = []
            for q,d,r in zip(original,derived,refs):
                expected = m.a.values(self.table[q])-self.ref[r][0]
                np.testing.assert_allclose(m.a.values(self.result[d]),expected,atol=1e-12)
                array.append(expected)
                c_values = m.a.values(self.result[d])[self.result['homogenized_class']=='C']
                self.assertAlmostEqual(np.nanmedian(c_values),0.,places=12)
            np.testing.assert_allclose(m.a.values(self.result[derived[3]]),np.linalg.norm(array,axis=0),atol=1e-12)

    def test_missing_rv_and_units_roundtrip(self):
        missing = ~np.isfinite(m.a.values(self.table['radial_velocity']))
        self.assertEqual(missing.sum(),570)
        for q in m.VELOCITY + m.DECOMPOSITION:
            self.assertTrue(np.isnan(m.a.values(self.result[q])[missing]).all())
        for q,unit in m.UNITS.items():
            self.assertEqual(self.result[q].unit,u.Unit(unit))
        with tempfile.TemporaryDirectory() as d:
            p,r = Path(d)/'result.ecsv',Path(d)/'ref.ecsv'
            self.result.write(p)
            self.ref.write(r)
            m.validate_relative(self.table,Table.read(p),Table.read(r))

    def test_reference_sky_backtransform(self):
        position = np.array([self.ref[q][0] for q in m.REF_POSITION])
        sky = SkyCoord(Galactic(CartesianRepresentation(position*u.pc)))
        np.testing.assert_allclose(self.ref['r_ref_pc'][0],np.linalg.norm(position),rtol=1e-12)
        np.testing.assert_allclose(self.ref['speed_ref_kms'][0],np.linalg.norm([self.ref[q][0] for q in m.REF_VELOCITY]))
        for q,value in [('l_ref_deg',sky.l.deg),('b_ref_deg',sky.b.deg),
                        ('ra_ref_deg',sky.icrs.ra.deg),('dec_ref_deg',sky.icrs.dec.deg)]:
            self.assertAlmostEqual(self.ref[q][0],value,places=12)

    def test_radial_tangential_and_exact_origin(self):
        t = self.table[:3].copy()
        displacement = np.array([[0.,0.,0.],[1.,0.,0.],[1.,0.,0.]])
        velocity = np.array([[1.,2.,3.],[3.,4.,0.],[-3.,4.,0.]])
        for i,q in enumerate(m.b.POSITION[:3]):
            t[q] = (displacement[:,i]+self.ref[m.REF_POSITION[i]][0])*u.pc
        for i,q in enumerate(m.b.VELOCITY[:3]):
            t[q] = (velocity[:,i]+self.ref[m.REF_VELOCITY[i]][0])*u.km/u.s
        result = m.relative_phase_space(t,self.ref)
        for q in m.DECOMPOSITION:
            self.assertTrue(np.isnan(result[q][0]))
        np.testing.assert_allclose(result[m.DECOMPOSITION[0]][1:],[3.,-3.])
        np.testing.assert_allclose(result[m.DECOMPOSITION[1]][1:],[4.,4.])

    def test_summary_counts_and_independent_statistics(self):
        full,groups = m.summaries(self.result)
        self.assertEqual(len(full),10)
        self.assertEqual(len(groups),60)
        for field in m.a.GROUPS:
            subset = groups[(groups['group_column']==field)&(groups['quantity']=='r_cluster_pc')]
            self.assertEqual(sum(subset['n_total']),1456)
            self.assertEqual(sum(subset['n_position']),1456)
            self.assertEqual(sum(subset['n_velocity']),886)
        for row in groups:
            values = m.a.values(self.result[row['quantity']])[self.result[row['group_column']]==row['group']]
            values = values[np.isfinite(values)]
            self.assertEqual(row['n_valid'],len(values))
            np.testing.assert_allclose([row['p16'],row['median'],row['p84']],np.percentile(values,[16,50,84]))

    def test_scatter_axes_and_full_extrema(self):
        specs = [(m.POSITION[i],m.POSITION[j],'homogenized_class') for i,j in [(0,1),(0,2),(1,2)]]
        specs += [(m.POSITION[0],m.POSITION[1],'catalogue_status')]
        specs += [(m.VELOCITY[i],m.VELOCITY[j],'homogenized_class') for i,j in [(0,1),(0,2),(1,2)]]
        specs += [('r_cluster_pc','dv_cluster_kms','homogenized_class')]
        for x,y,field in specs:
            fig,ax = m.scatter(self.result,x,y,field)
            for group,collection in zip(m.a.GROUPS[field],ax.collections):
                mask = (self.result[field]==group)&np.isfinite(m.a.values(self.result[x]))&np.isfinite(m.a.values(self.result[y]))
                np.testing.assert_allclose(collection.get_offsets()[:,0],self.result[x][mask])
                np.testing.assert_allclose(collection.get_offsets()[:,1],self.result[y][mask])
            self.assertEqual(ax.get_xlabel(),m.LABELS[x])
            self.assertEqual(ax.get_ylabel(),m.LABELS[y])
            if x in m.POSITION[:3]:
                self.assertEqual(ax.get_aspect(),1.)
            if x != 'r_cluster_pc':
                np.testing.assert_array_equal(ax.lines[0].get_xydata(),[[0,0]])
            for q,limits in [(x,ax.get_xlim()),(y,ax.get_ylim())]:
                self.assertLessEqual(limits[0],np.nanmin(self.result[q]))
                self.assertGreaterEqual(limits[1],np.nanmax(self.result[q]))
            m.plt.close(fig)

    def test_ecdf_and_overview(self):
        for q in ('r_cluster_pc','dv_cluster_kms'):
            fig,ax = m.ecdf(self.result,q)
            for group,line in zip(m.a.GROUPS['homogenized_class'],ax.lines):
                values = m.a.values(self.result[q])[self.result['homogenized_class']==group]
                values = np.sort(values[np.isfinite(values)])
                np.testing.assert_array_equal(line.get_xdata()[1:],values)
                self.assertEqual(line.get_ydata()[0],0.)
                self.assertEqual(line.get_ydata()[-1],1.)
                self.assertTrue(np.all(np.diff(line.get_ydata())>=0))
            m.plt.close(fig)
        fig,axes = m.reference_overview(self.result)
        for ax in axes:
            self.assertEqual(ax.get_aspect(),1.)
            self.assertEqual(ax.get_xlim(),axes[0].get_xlim())
            self.assertEqual(ax.get_ylim(),axes[0].get_ylim())
        m.plt.close(fig)

    def test_prior_products_unchanged_by_pipeline(self):
        before = m.protected_hashes()
        with tempfile.TemporaryDirectory(dir=m.ROOT) as d:
            with patch.object(m,'OUTPUT',Path(d)/'tables'),patch.object(m,'FIGURES',Path(d)/'figures'):
                with patch.object(m,'generate_plots',return_value=[]),patch('builtins.print'):
                    m.main()
        self.assertEqual(before,m.protected_hashes())


if __name__ == '__main__':
    unittest.main()
