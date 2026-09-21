"""Project 07 geometric invariants, axial statistics, and tensor conventions."""
from pathlib import Path
import sys
import unittest
import numpy as np
from astropy import units as u

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/project07'))
import geometry_support as g
frame=g.load_module('test07a','src/project07/01_clustercentric_frame.py')
tidal=g.load_module('test07d','src/project07/04_tidal_axis_comparison.py')


class Project07Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table=g.read(g.GEOMETRY); cls.axes=g.read(g.AXES); cls.boot=g.read(g.BOOT)
        cls.ref=g.read(g.REFERENCE)

    def test_canonical_identity_and_classes(self):
        old=g.read(g.ZONES)
        self.assertEqual(len(self.table),1456)
        for q in ['gaia_dr3_source_id','homogenized_class','catalogue_status','spatial_zone']:
            np.testing.assert_array_equal(self.table[q],old[q])
        self.assertEqual(dict(zip(*np.unique(self.table['homogenized_class'],return_counts=True))),{'C':940,'L':184,'T':332})

    def test_frozen_upstream_hashes(self): g.protect()

    def test_orthonormal_left_handed_and_prograde(self):
        B,pos,vel=g.basis(self.ref)
        np.testing.assert_allclose(B.T@B,np.eye(3),atol=1e-14)
        self.assertAlmostEqual(np.linalg.det(B),-1.)
        np.testing.assert_allclose(np.cross(B[:,0],B[:,1]),-B[:,2],atol=1e-14)
        self.assertGreater(vel@B[:,1],0)
        self.assertGreater(pos@B[:,0],0)

    def test_radius_and_inverse_projection(self):
        B,pos,vel=g.basis(self.ref); local=g.xyz(self.table)
        np.testing.assert_allclose(np.linalg.norm(local,axis=1),g.arr(self.table,'r_cluster_pc',u.pc),atol=1e-8,rtol=0)
        source=g.read(g.GC)
        gc=np.column_stack([g.arr(source,q,u.kpc) for q in ['x_gc_kpc','y_gc_kpc','z_gc_kpc']])
        np.testing.assert_allclose(local@B.T/1000+pos,gc,atol=1e-12,rtol=0)

    def test_zone_counts(self):
        m=g.groups(self.table)
        self.assertEqual([int(m[x].sum()) for x in ['inner','intermediate','outer','outside_rJ']],[1005,290,161,451])
        self.assertEqual(int(np.sum(m['inner']&m['C'])),850)
        old=g.read(g.ZONES)
        np.testing.assert_array_equal(self.table['spatial_zone'],old['spatial_zone'])

    def test_covariances_eigenpairs_and_order(self):
        for group,mask in g.groups(self.table).items():
            for method in ['covariance','spatial_sign']:
                rows=self.axes[(self.axes['population']==group)&(self.axes['method']==method)]
                w=np.asarray(rows['eigenvalue']); V=np.array([rows[q] for q in ['e_radial','e_prograde','e_vertical']])
                S=np.array([[rows[f'scatter_{i}{j}'][0] for j in range(3)] for i in range(3)])
                np.testing.assert_allclose(S,S.T,atol=1e-12)
                np.testing.assert_allclose(V.T@V,np.eye(3),atol=1e-12)
                np.testing.assert_allclose(S@V,V*w,atol=1e-9,rtol=1e-12)
                self.assertTrue(np.all(np.diff(w)<=0)); self.assertTrue(np.all(w>=0))
                self.assertTrue(0<=rows['c_a'][0]<=rows['b_a'][0]<=1)
                if method=='covariance': np.testing.assert_allclose(S,np.cov(g.xyz(self.table)[mask],rowvar=False,ddof=1),atol=1e-10)

    def test_pca_rotation_and_degeneracy(self):
        rng=np.random.default_rng(42); X=rng.normal(size=(100,3))*[8,2,1]
        Q,_=np.linalg.qr(rng.normal(size=(3,3)))
        a=g.fit(X); b=g.fit(X@Q)
        np.testing.assert_allclose(a['eigenvalues'],b['eigenvalues'],rtol=1e-12)
        self.assertLess(float(g.acute(a['vectors'][:,0],Q@b['vectors'][:,0])),1e-5)
        isotropic=np.vstack([np.eye(3),-np.eye(3)])
        self.assertFalse(g.fit(isotropic)['major_defined'])

    def test_spatial_sign_retains_all_rows(self):
        rng=np.random.default_rng(8); X=rng.normal(size=(101,3))*[10,2,1]
        X[0]=[0,1e6,0]
        a=g.fit(X); b=g.fit(X,robust=True)
        self.assertEqual(b['n'],len(X))
        self.assertGreater(float(g.acute(a['vectors'][:,0],b['vectors'][:,0])),60)
        self.assertLessEqual(np.trace(b['matrix']),1+1e-12)

    def test_acute_angles_and_axial_sign_statistics(self):
        self.assertAlmostEqual(float(g.acute([1,0,0],[-1,0,0])),0)
        self.assertAlmostEqual(float(g.acute([1,0,0],[0,1,0])),90)
        samples=np.array([[1.,0,0],[-1,0,0],[1,0,0]])
        s=g.axial_summary(samples,np.array([1.,0,0]))
        self.assertAlmostEqual(s['concentration'],1)
        np.testing.assert_allclose(s['projector_mean'],np.diag([1,0,0]))

    def test_bootstrap_reproducibility_and_validity(self):
        X=g.xyz(self.table)[g.groups(self.table)['T']]
        a,gap=g.bootstrap(X,19,n=20); b,_=g.bootstrap(X,19,n=20)
        np.testing.assert_array_equal(a,b)
        vectors=np.column_stack([self.boot[q] for q in ['e_radial','e_prograde','e_vertical']])
        self.assertTrue(np.isfinite(vectors).all())
        np.testing.assert_allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-12)
        self.assertEqual(len(self.boot),9*g.N_BOOT)
        self.assertTrue(np.all((self.boot['deviation_deg']>=0)&(self.boot['deviation_deg']<=90)))
        self.assertTrue(np.all(self.boot['major_gap']>1e-8))

    def test_direction_definitions(self):
        dirs=g.directions(self.ref); B,pos,vel=g.basis(self.ref)
        np.testing.assert_allclose(B@dirs['galactic_centre_toward'],-pos/np.linalg.norm(pos),atol=1e-14)
        np.testing.assert_allclose(B@dirs['orbital_velocity'],vel/np.linalg.norm(vel),atol=1e-14)
        self.assertGreater(float(g.acute(dirs['prograde'],dirs['orbital_velocity'])),0)
        for v in dirs.values(): self.assertAlmostEqual(np.linalg.norm(v),1)

    def test_direction_and_tidal_angle_bounds(self):
        for path in [g.DATA/'stock2_directional_angles.ecsv',g.DATA/'stock2_tidal_axis_angles.ecsv',g.DATA/'stock2_tidal_alignment_bootstrap.ecsv']:
            t=g.read(path)
            for q in t.colnames:
                if q.endswith('_deg'): self.assertTrue(np.all(np.isfinite(t[q])&(t[q]>=0)&(t[q]<=90)))

    def test_geometric_halfspaces(self):
        t=g.read(g.DATA/'stock2_geometric_halfspaces.ecsv'); dirs=g.directions(self.ref)
        proj=g.xyz(self.table)@dirs['orbital_velocity']
        np.testing.assert_allclose(proj,t['orbital_projection_pc'],atol=1e-10)
        np.testing.assert_array_equal(proj>1e-10,t['orbital_geometric_side']=='leading_side')
        counts=g.read(g.DATA/'stock2_geometric_side_counts.ecsv')
        for group,mask in g.groups(self.table).items():
            for partition in ['orbital_geometric_side','radial_geometric_side']:
                rows=counts[(counts['population']==group)&(counts['partition']==partition)]
                self.assertEqual(sum(rows['N_side']),int(mask.sum()))

    def test_tidal_tensor_and_force_derivative(self):
        T,Tgc,w,V=tidal.tidal_axes(self.ref); B,pos,vel=g.basis(self.ref)
        np.testing.assert_allclose(B@T@B.T,Tgc,atol=1e-9)
        np.testing.assert_allclose(T@V,V*w,atol=1e-9)
        np.testing.assert_allclose(V.T@V,np.eye(3),atol=1e-12)
        self.assertTrue(np.all(np.diff(w)<=0)); self.assertGreater(w[0],0); self.assertLess(w[-1],0)
        analytic,numerical,rel=g.p04.finite_difference_audit(np.hypot(*pos[:2]),pos[2])
        np.testing.assert_allclose(-T[0,0],analytic[0],rtol=1e-12)
        self.assertLess(rel.max(),5e-5)

    def test_outputs(self):
        names=['stock2_clustercentric_geometry','stock2_morphology_axes','stock2_orientation_bootstrap','stock2_tidal_tensor_axes','stock2_direction_vectors','stock2_directional_angles','stock2_geometric_halfspaces','stock2_geometric_side_counts','stock2_tidal_axis_angles','stock2_tidal_alignment_bootstrap']
        paths=[g.DATA/(name+'.ecsv') for name in names]
        paths += [g.RESULTS/name for name in ['project07_summary.txt','project07_metadata.json','upstream_integrity.json']]
        for folder,name in [('morphology','radial_tangential'),('morphology','radial_vertical'),('morphology','tangential_vertical'),('morphology','outer_population'),('bootstrap','orientation_distribution'),('directional_geometry','principal_axis_orientation')]:
            paths += [g.RESULTS/folder/(name+'.'+ext) for ext in ['png','pdf']]
        for p in paths: self.assertTrue(p.is_file(),str(p))


if __name__=='__main__': unittest.main()
