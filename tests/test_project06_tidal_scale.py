"""Physical identities, unit checks, geometric counts and upstream protection."""
from pathlib import Path
import importlib.util
import json
import sys
import unittest
import numpy as np
from astropy import units as u
from astropy.constants import G
from galpy.potential import rtide

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/project06'))
import support as s
spec=importlib.util.spec_from_file_location('extent06',ROOT/'src/project06/03_candidate_extent.py')
e=importlib.util.module_from_spec(spec); spec.loader.exec_module(e)


class Project06Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref,cls.orbit,cls.spatial=s.validate_inputs()
        cls.field=s.read(s.FIELD); cls.jacobi=s.read(s.JACOBI)
        cls.zones=s.read(s.DATA/'stock2_candidate_tidal_zones.ecsv')
        cls.variation=s.read(s.DATA/'stock2_orbital_jacobi_radius.ecsv')

    def test_canonical_identity_and_original_labels(self):
        self.assertEqual(len(self.zones),1456)
        for key in ['gaia_dr3_source_id','homogenized_class','catalogue_status']:
            np.testing.assert_array_equal(self.zones[key],self.spatial[key])

    def test_protected_upstream_hashes(self):
        s.check_run_protection()
        h=json.loads((s.RESULTS/'upstream_integrity.json').read_text())['sha256']
        for p in [s.REFERENCE,s.ORBIT,s.REF03,s.SPATIAL]: self.assertIn(str(p.relative_to(ROOT)),h)

    def test_literature_mass_definition(self):
        mass=self.jacobi.meta['mass']
        self.assertEqual(mass['value'],4000.)
        self.assertEqual(mass['role'],'literature baseline')
        self.assertTrue(mass['approximate'])
        self.assertIsNone(mass['quoted_uncertainty'])
        self.assertEqual(mass['doi'],'10.3847/1538-3881/abc61a')
        self.assertEqual(mass['comparison_tidal_radius_pc'],22.65)

    def test_field_identity_and_units(self):
        f=self.field
        d=f['denominator'].quantity
        np.testing.assert_allclose(d.to_value(s.DUNIT),(4*f['Omega_c'].quantity**2-f['kappa'].quantity**2).to_value(s.DUNIT),rtol=1e-12)
        np.testing.assert_allclose(d.to_value(s.DUNIT),(f['Omega_c'].quantity**2-f['Phi_RR_midplane'].quantity).to_value(s.DUNIT),rtol=1e-12)
        self.assertTrue(np.all(d>0*s.DUNIT))
        for q in f.colnames:
            self.assertIsNotNone(f[q].unit); self.assertTrue(np.isfinite(s.arr(f,q)).all())
        self.assertEqual(f['z_jacobi_kpc'][0],0.)
        self.assertAlmostEqual(f['z_actual_kpc'][0],self.ref['z_gc_kpc'][0])

    def test_against_native_galpy_midplane_rtide(self):
        R=float(self.field['R_kpc'][0])
        mass_n=(G*4000*u.Msun/(s.VO*u.km/u.s)**2/(s.RO*u.kpc)).to_value(u.dimensionless_unscaled)
        independent=rtide(s.MWPotential2014,R/s.RO,0.,M=mass_n,use_physical=False)*s.RO*1000
        self.assertAlmostEqual(self.jacobi['r_J_now'][0],independent,places=10)

    def test_dimensional_conversion_and_kepler_limit(self):
        # For a point-mass galaxy D=3 GM_gal/R^3, so rJ=R*(Mcl/(3Mgal))^(1/3).
        R=8*u.kpc; mg=1e11*u.Msun; mc=4000*u.Msun
        d=3*G*mg/R**3
        expected=(R*(mc/(3*mg))**(1/3)).to_value(u.pc)
        self.assertAlmostEqual(s.jacobi(mc,d).to_value(u.pc),expected,places=10)
        self.assertAlmostEqual(s.jacobi(mc.to(u.kg),d.to(u.s**-2)).value,expected,places=10)

    def test_invalid_denominator_rejected(self):
        for d in [0.,-1.,np.nan,np.inf]:
            with self.assertRaises(ValueError): s.jacobi(4000*u.Msun,d*s.DUNIT)
        with self.assertRaises(ValueError): s.jacobi(-1*u.Msun,1*s.DUNIT)

    def test_mass_scaling(self):
        grid=s.read(s.DATA/'stock2_jacobi_mass_sensitivity.ecsv')
        self.assertTrue(np.all(np.diff(grid['r_J'])>0))
        ratio=np.asarray(grid['r_J'])/grid['r_J'][2]
        np.testing.assert_allclose(ratio,(np.asarray(grid['mass'])/4000)**(1/3),rtol=1e-14)
        self.assertAlmostEqual(grid['r_J'][2],self.jacobi['r_J_now'][0])

    def test_zone_boundaries_and_missing(self):
        labels,valid=e.zones(np.array([0,9.99,10,19.99,20,np.nan,-1]),10)
        np.testing.assert_array_equal(labels,[e.ZONE_LABELS[0],e.ZONE_LABELS[0],e.ZONE_LABELS[1],e.ZONE_LABELS[1],e.ZONE_LABELS[2],'unavailable','unavailable'])
        self.assertEqual(valid.sum(),5)

    def test_3d_radius_and_no_membership_labels(self):
        xyz=np.column_stack([s.arr(self.zones,q,u.pc) for q in ['dx_cluster_pc','dy_cluster_pc','dz_cluster_pc']])
        np.testing.assert_allclose(np.linalg.norm(xyz,axis=1),s.arr(self.zones,'r_cluster_pc',u.pc),atol=1e-9)
        for label in set(self.zones['spatial_zone']):
            for forbidden in ('member','bound','escape'): self.assertNotIn(forbidden,label.lower())
        expected,_=e.zones(np.linalg.norm(xyz,axis=1),float(self.jacobi['r_J_now'][0]))
        np.testing.assert_array_equal(expected,self.zones['spatial_zone'])

    def test_group_count_sums(self):
        counts=s.read(s.DATA/'stock2_candidate_tidal_zone_counts.ecsv')
        for group in np.unique(counts['group']):
            rows=counts[counts['group']==group]
            self.assertEqual(sum(rows['N_zone']),rows['N_calculable'][0])
            self.assertEqual(rows['N_total'][0],rows['N_calculable'][0]+rows['N_unavailable'][0])
            self.assertAlmostEqual(sum(rows['fraction_of_calculable']),1.)
        cls=counts[np.isin(counts['group'],['C','L','T'])]
        self.assertEqual(sum(cls['N_zone']),1456)

    def test_orbital_scale_and_strongest_field(self):
        r=s.arr(self.variation,'r_J',u.pc); d=s.arr(self.variation,'denominator')
        self.assertEqual(len(r),4001); self.assertTrue(np.isfinite(r).all())
        self.assertTrue(np.all(r>0)); self.assertLessEqual(r.min(),r[2000]); self.assertLessEqual(r[2000],r.max())
        self.assertAlmostEqual(r[2000],self.jacobi['r_J_now'][0],places=10)
        self.assertEqual(np.argmin(r),np.argmax(d))
        np.testing.assert_allclose(r**3*d,r[2000]**3*d[2000],rtol=1e-13)
        np.testing.assert_array_equal(self.variation['time_myr'],self.orbit['time_myr'])

    def test_literature_radius_only_comparison(self):
        d=self.field['denominator'].quantity[0]
        value=s.jacobi(4000*u.Msun,d).value
        self.assertAlmostEqual(self.jacobi['r_J_now'][0],value)
        self.assertAlmostEqual(self.jacobi['signed_difference'][0],value-22.65)
        self.assertAlmostEqual(self.jacobi['fractional_difference'][0],(value-22.65)/22.65)

    def test_required_products(self):
        data=['stock2_tidal_field','stock2_jacobi_radius','stock2_jacobi_mass_sensitivity',
              'stock2_candidate_tidal_zones','stock2_candidate_tidal_zone_counts','stock2_orbital_jacobi_radius']
        paths=[s.DATA/(name+'.ecsv') for name in data]
        paths += [s.RESULTS/name for name in ['project06_summary.txt','project06_metadata.json','upstream_integrity.json']]
        for folder,name in [('jacobi_radius','mass_sensitivity'),('candidate_extent','candidate_xy'),
                            ('candidate_extent','radial_distribution'),('candidate_extent','class_radial_distribution'),
                            ('orbital_variation','jacobi_time'),('orbital_variation','jacobi_R')]:
            paths += [s.RESULTS/folder/(name+'.'+ext) for ext in ('png','pdf')]
        for p in paths: self.assertTrue(p.is_file(),str(p))


if __name__=='__main__': unittest.main()
