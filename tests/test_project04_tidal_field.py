import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'src/project04/03_local_tidal_field.py'
spec=importlib.util.spec_from_file_location('project04c',SCRIPT)
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


@unittest.skipIf(p.MWPotential2014 is None,'galpy not installed')
class TidalFieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference,cls.orbit,cls.summary=p.load_inputs()
        cls.axis=p.fixed_axis_gc(cls.reference)
        cls.state=p.interpolate_state(cls.orbit,0.)

    def test_fixed_inputs_and_present_state(self):
        ref=np.array([p.scalar(self.reference,q) for q in ['x_gc_kpc','y_gc_kpc','z_gc_kpc','vx_gc_kms','vy_gc_kms','vz_gc_kms']])
        np.testing.assert_allclose(self.state,ref,rtol=0,atol=3e-8)
        self.assertAlmostEqual(np.linalg.norm(self.axis),1.,places=12)

    def test_tensor_symmetry_and_eigensystem(self):
        T,_,basis=p.tidal_tensor_at(*self.state[:3])
        np.testing.assert_allclose(T,T.T,atol=1e-11)
        np.testing.assert_allclose(basis.T@basis,np.eye(3),atol=2e-14)
        self.assertGreater(np.linalg.det(basis),0.999999999)
        w,V=p.deterministic_eigensystem(T)
        self.assertTrue(np.all(np.diff(w)<=0))
        np.testing.assert_allclose(V.T@V,np.eye(3),atol=2e-12)
        np.testing.assert_allclose(T@V,V*w[None,:],rtol=1e-11,atol=1e-9)

    def test_force_derivative_audit(self):
        R=np.hypot(self.state[0],self.state[1]);z=self.state[2]
        analytic,numeric,rel=p.finite_difference_audit(R,z)
        self.assertLess(float(np.max(rel)),5e-5)
        np.testing.assert_allclose(analytic,numeric,rtol=5e-5,atol=1e-6)

    def test_sample_epochs_and_angles(self):
        rows=[]
        for tm in p.SAMPLE_TIMES_MYR:
            rr,T,b=p.one_epoch(float(tm),p.interpolate_state(self.orbit,float(tm)),self.axis)
            rows.extend(rr)
        self.assertEqual(len(rows),3*len(p.SAMPLE_TIMES_MYR))
        self.assertTrue(all(0<=r['angle_to_fixed_axis_deg']<=90 for r in rows))
        self.assertEqual(len([r for r in rows if r['time_myr']==0]),3)


if __name__=='__main__': unittest.main()
