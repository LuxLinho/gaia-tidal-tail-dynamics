import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from astropy.table import Table
from astropy import units as u

PATH=Path(__file__).resolve().parents[1]/'src/project03/04_relative_morphology_coherence.py'
spec=importlib.util.spec_from_file_location('morphology',PATH)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class MorphologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table=Table.read(m.INPUT)
        cls.ref=Table.read(m.REFERENCE)
        cls.fits,cls.pca=m.pca_populations(cls.table)
        cls.basis=cls.fits['L+T']['axes']
        cls.result=m.project(cls.table,cls.basis)

    def test_canonical_preservation_and_missing_rv(self):
        m.validate_input(self.table,self.ref)
        m.validate_projection(self.table,self.result,self.basis)
        self.assertEqual(len(self.result),1456)
        self.assertEqual(len(set(self.result['source_id'])),1456)
        self.assertEqual(dict(zip(*np.unique(self.result['homogenized_class'],return_counts=True))),{'C':940,'L':184,'T':332})
        missing=~np.isfinite(m.data(self.table,'radial_velocity'))
        self.assertEqual(missing.sum(),570)
        for q in m.U+['u_perp_ext_kms']:
            self.assertTrue(np.isnan(m.data(self.result,q)[missing]).all())
            self.assertEqual(np.isfinite(m.data(self.result,q)).sum(),886)
        for q in m.S+['r_perp_ext_pc']:
            self.assertEqual(np.isfinite(m.data(self.result,q)).sum(),1456)
        for q,unit in m.UNITS.items(): self.assertEqual(self.result[q].unit,u.Unit(unit))

    def test_reference_metadata_mismatch_rejected(self):
        t=self.table.copy()
        t.meta['x_ref_pc']+=1e-8
        with self.assertRaises(ValueError): m.validate_input(t,self.ref)
        r=self.ref.copy()
        r.meta['reference_definition']='different'
        with self.assertRaises(ValueError): m.validate_input(self.table,r)

    def test_pca_covariance_all_points_without_scaling(self):
        # Known diagonal scatter; the long X point must not be clipped.
        points=np.array([[100.,0,0],[-100.,0,0],[0,3,0],[0,-3,0],[0,0,1],[0,0,-1]])+[7,8,9]
        fit=m.spatial_pca(points)
        np.testing.assert_allclose(fit['mean'],[7,8,9])
        np.testing.assert_allclose(fit['eigenvalues'],[4000,3.6,.4])
        np.testing.assert_allclose(fit['axes'],np.eye(3))
        self.assertEqual(fit['n'],6)
        for group,mask in m.populations(self.table).items():
            xyz=np.array([m.data(self.table,q)[mask] for q in m.c.POSITION[:3]]).T
            p=self.fits[group]
            covariance=np.cov(xyz,rowvar=False,ddof=1)
            np.testing.assert_allclose(p['axes']@covariance@p['axes'].T,np.diag(p['eigenvalues']),atol=1e-10)
            self.assertAlmostEqual(p['variance_ratios'].sum(),1.)
            np.testing.assert_allclose(p['lengths']**2,p['eigenvalues'])

    def test_orthonormal_right_handed_and_deterministic_signs(self):
        for p in self.fits.values():
            axes=p['axes']
            np.testing.assert_allclose(axes@axes.T,np.eye(3),atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(axes),1.)
            for i in (0,1): self.assertGreater(axes[i,np.argmax(np.abs(axes[i]))],0)
            for flips in [[-1,1,1],[1,-1,-1],[-1,-1,-1]]:
                np.testing.assert_allclose(m.orient_axes(axes*np.array(flips)[:,None]),axes,atol=1e-12)
        xyz=np.array([m.data(self.table,q) for q in m.c.POSITION[:3]]).T
        np.testing.assert_allclose(m.spatial_pca(xyz[::-1])['axes'],self.fits['full']['axes'],atol=1e-12)

    def test_fixed_basis_uses_only_lt_positions(self):
        t=self.table.copy()
        c=t['homogenized_class']=='C'
        for q in m.c.POSITION[:3]: t[q][c]+=1e5
        for q in m.c.VELOCITY[:3]: t[q]=np.arange(len(t))*1000*u.km/u.s
        fits,_=m.pca_populations(t)
        np.testing.assert_array_equal(fits['L+T']['axes'],self.basis)
        self.assertEqual(fits['L+T']['n'],516)

    def test_projection_reconstruction_and_norms(self):
        for old,new,norm in [(m.c.POSITION,m.S,'r_perp_ext_pc'),(m.c.VELOCITY,m.U,'u_perp_ext_kms')]:
            array,valid=m.c.vectors(self.table,old)
            projected=np.array([m.data(self.result,q) for q in new])
            np.testing.assert_allclose(projected[:,valid],self.basis@array[:,valid],atol=1e-12)
            np.testing.assert_allclose(self.basis.T@projected[:,valid],array[:,valid],atol=1e-12)
            np.testing.assert_allclose(m.data(self.result,norm)[valid],np.sqrt(projected[1,valid]**2+projected[2,valid]**2))
        # The canonical projection retains the cluster origin, not the PCA mean.
        mask=m.populations(self.table)['L+T']
        np.testing.assert_allclose(np.mean(np.array([m.data(self.result,q)[mask] for q in m.S]),axis=1),
                                   self.basis@self.fits['L+T']['mean'],atol=1e-12)

    def test_ecsv_basis_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'pca.ecsv';t=Path(d)/'output.ecsv'
            self.pca.write(p);self.result.write(t)
            pc=Table.read(p);ext=pc[pc['canonical_projection_basis']]
            basis=np.array([ext[q] for q in ['e_x','e_y','e_z']]).T
            np.testing.assert_array_equal(basis,self.basis)
            m.validate_projection(self.table,Table.read(t),basis)

    def test_summary_sides_and_radial_reuse(self):
        full,groups=m.summaries(self.result)
        self.assertEqual(len(groups),6*len(m.SUMMARY_UNITS))
        for r in groups:
            vals=m.data(self.result,r['quantity'])[self.result[r['group_column']]==r['group']]
            vals=vals[np.isfinite(vals)]
            self.assertEqual(r['n_valid'],len(vals))
            np.testing.assert_allclose([r['p16'],r['median'],r['p84']],np.percentile(vals,[16,50,84]))
        sides,radial=m.side_and_radial_summaries(self.result)
        for field,labels in m.a.GROUPS.items():
            for group,n in labels.items():
                rows=sides[(sides['group_column']==field)&(sides['group']==group)]
                self.assertEqual(sum(rows['n']),n)
        for r in radial:
            self.assertEqual(r['n_outward']+r['n_inward']+r['n_zero'],r['n_radial'])
            self.assertAlmostEqual(r['fraction_outward'],r['n_outward']/r['n_radial'])
            self.assertAlmostEqual(r['fraction_inward'],r['n_inward']/r['n_radial'])
        for q in m.c.DECOMPOSITION:
            np.testing.assert_array_equal(self.result[q],self.table[q])

    def test_correlations_and_pairwise_availability(self):
        result=m.correlation([1,2,3,4,np.nan],[2,4,6,8,0])
        self.assertEqual(result['n'],4)
        self.assertAlmostEqual(result['pearson_r'],1.)
        self.assertAlmostEqual(result['spearman_rho'],1.)
        self.assertTrue(np.isnan(m.correlation([1,1,1],[2,3,4])['pearson_r']))
        self.assertTrue(np.isnan(m.correlation([1],[2])['spearman_rho']))
        t=m.correlations(self.result)
        self.assertEqual(len(t),25)
        expected={'full':886,'C':594,'L':111,'T':181,'L+T':292}
        for r in t:
            self.assertEqual(r['n'],expected[r['population']])
            self.assertTrue(-1<=r['pearson_r']<=1)
            self.assertTrue(-1<=r['spearman_rho']<=1)
            self.assertTrue(0<=r['pearson_p_two_sided']<=1)
            self.assertTrue(0<=r['spearman_p_two_sided']<=1)

    def test_plots_coordinates_full_extents_and_ecdfs(self):
        specs=[(m.S[i],m.S[j],'homogenized_class') for i,j in [(0,1),(0,2),(1,2)]]
        specs+=[('s1_ext_pc','r_perp_ext_pc',f) for f in m.a.GROUPS]
        specs+=[(x,y,'homogenized_class') for x,y in m.PAIRS[:4]]
        for x,y,field in specs:
            fig,ax=m.scatter(self.result,x,y,field)
            for group,collection in zip(m.a.GROUPS[field],ax.collections):
                valid=(self.result[field]==group)&np.isfinite(m.data(self.result,x))&np.isfinite(m.data(self.result,y))
                np.testing.assert_allclose(collection.get_offsets()[:,0],m.data(self.result,x)[valid])
                np.testing.assert_allclose(collection.get_offsets()[:,1],m.data(self.result,y)[valid])
            for q,limits in [(x,ax.get_xlim()),(y,ax.get_ylim())]:
                self.assertLessEqual(limits[0],np.nanmin(m.data(self.result,q)))
                self.assertGreaterEqual(limits[1],np.nanmax(m.data(self.result,q)))
            self.assertEqual(ax.get_xlabel(),m.LABELS[x]);self.assertEqual(ax.get_ylabel(),m.LABELS[y])
            if x in m.S and y in m.S:self.assertEqual(ax.get_aspect(),1.)
            m.plt.close(fig)
        for q in ['s1_ext_pc','r_perp_ext_pc','u1_ext_kms']:
            fig,ax=m.ecdf(self.result,q)
            for group,line in zip(m.a.GROUPS['homogenized_class'],ax.lines):
                v=m.data(self.result,q)[self.result['homogenized_class']==group]
                v=np.sort(v[np.isfinite(v)])
                np.testing.assert_array_equal(line.get_xdata()[1:],v)
                self.assertEqual(line.get_ydata()[-1],1.)
            m.plt.close(fig)

    def test_axis_visualization(self):
        fig,axes=m.axis_visualization(self.result,self.basis)
        for ax,(i,j) in zip(axes,[(0,1),(0,2),(1,2)]):
            line=ax.lines[-1].get_xydata()
            direction=np.array([self.basis[0,i],self.basis[0,j]])
            self.assertAlmostEqual(np.linalg.det(np.array([line[1],direction])),0.,places=10)
            self.assertEqual(ax.get_aspect(),1.)
            self.assertEqual(ax.get_xlim(),axes[0].get_xlim())
        m.plt.close(fig)

    def test_pipeline_preserves_prior_products_and_never_recomputes_reference(self):
        before=m.protected_hashes()
        with tempfile.TemporaryDirectory(dir=m.ROOT) as d:
            with patch.object(m,'OUTPUT',Path(d)/'tables'),patch.object(m,'FIGURES',Path(d)/'figures'):
                with patch.object(m,'generate_plots',return_value=[]),patch('builtins.print'):
                    with patch.object(m.c,'reference_frame',side_effect=AssertionError('Reference must not be recomputed')):
                        m.main()
        self.assertEqual(before,m.protected_hashes())


if __name__=='__main__':
    unittest.main()
