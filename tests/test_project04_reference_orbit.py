import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('p04b',ROOT/'src/project04/02_reference_orbit.py')
m=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(m)


class ReferenceOrbitTests(unittest.TestCase):
    def test_coordinate_bridge_roundtrip(self):
        R=np.array([8.2,7.9]); phi=np.array([0.2,-2.4]); z=np.array([.01,-.02])
        vR=np.array([20.,-5.]); vT=np.array([230.,215.]); vz=np.array([-6.,3.])
        x,y,zz,vx,vy,vzz=m.galpy_cyl_to_astropy_cart(R,phi,z,vR,vT,vz)
        self.assertTrue(np.allclose(np.hypot(x,y),R))
        self.assertTrue(np.allclose(zz,z)); self.assertTrue(np.allclose(vzz,vz))
        self.assertTrue(np.allclose((x*vx+y*vy)/R,vR))
        self.assertTrue(np.allclose((-y*vx+x*vy)/R,-vT))

    def test_fixed_configuration(self):
        self.assertAlmostEqual(m.RO.to_value(m.u.kpc),8.0)
        self.assertAlmostEqual(m.VO.to_value(m.u.km/m.u.s),220.0)
        self.assertEqual(m.METHOD,'dop853')
        self.assertEqual(m.N_HALF,4001)

    @unittest.skipIf(m.galpy is None or not m.INPUT.exists() or not m.PCA.exists(),'galpy or Project 04A/03D product unavailable')
    def test_integration_invariants(self):
        ref=m.load_reference(); axis=m.load_fixed_axis(); orbit=m.build_orbit(ref)
        t,raw=m.integrate_two_sided(orbit); traj=m.trajectory_table(t,raw)
        diag=m.orbital_diagnostics(traj,axis); m.validate(ref,traj,diag)
        self.assertEqual(len(traj),2*m.N_HALF-1)
        self.assertLessEqual(diag['relative_energy_span'],1e-7)
        self.assertGreater(diag['rap_kpc'],diag['rperi_kpc'])
        self.assertGreaterEqual(diag['eccentricity'],0.)
        self.assertLess(diag['eccentricity'],1.)


if __name__=='__main__': unittest.main()
