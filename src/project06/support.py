"""Frozen inputs, physical conventions, and IO shared by Project 06 stages."""
from pathlib import Path
import hashlib
import importlib.util
import json
import platform
import astropy
import galpy
import numpy as np
import scipy
from astropy import units as u
from astropy.constants import G
from astropy.table import Table
from galpy.potential import (MWPotential2014, evaluateRforces, evaluatezforces,
                            evaluateR2derivs, omegac, epifreq, verticalfreq)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'data/interim/project06'
RESULTS = ROOT/'results/project06'
REFERENCE = ROOT/'data/interim/project05/stock2_cluster_reference.ecsv'
ORBIT = ROOT/'data/interim/project05/stock2_galactic_orbit.ecsv'
SPATIAL = ROOT/'data/interim/project03/stock2_cluster_relative_phase_space.ecsv'
MASTER = ROOT/'data/processed/stock2_literature_master.ecsv'
REF03 = ROOT/'data/interim/project03/stock2_reference_frame.ecsv'
FIELD = DATA/'stock2_tidal_field.ecsv'
JACOBI = DATA/'stock2_jacobi_radius.ecsv'
MASS_GRID = np.array([2000.,3000.,4000.,5000.,6000.])
RO, VO = 8., 220.
FUNIT = (u.km/u.s)**2/u.kpc
DUNIT = (u.km/u.s/u.kpc)**2
FREQUNIT = u.km/u.s/u.kpc
G_VALUE = G.to_value(u.pc*(u.km/u.s)**2/u.Msun)
MASS = dict(value=4000., unit='solMass', role='literature baseline', approximate=True,
            definition='Present-day cluster mass inferred from present-day mass-function analysis; not a sum of repository stars or an initial mass.',
            reference='Ye et al. (2021), AJ 161, 8', doi='10.3847/1538-3881/abc61a',
            title='Diagnosing Open Cluster Stock 2: Member Candidates and Mass Distribution with Gaia DR2 and LAMOST',
            member_candidates=1325, catalogue='Gaia DR2', quoted_uncertainty=None,
            uncertainty_status='No formal mass uncertainty supplied in the adopted provenance; none inferred.',
            core_radius_pc=3.97, comparison_tidal_radius_pc=22.65,
            provenance_status='Adopted from user-specified literature result; primary full text could not be retrieved in this session. Detailed mass-function corrections and tidal-radius derivation not independently audited.',
            comparison_role='22.65 pc is comparison only, never a Jacobi input; same-paper quantities need not be statistically independent.')
MODEL = dict(name='galpy MWPotential2014', galpy_version='1.12.0', ro_kpc=RO, vo_kms=VO,
             potential_convention='Phi is potential per unit test mass; acceleration=-grad(Phi).',
             formula='r_J=(G*M/D)^(1/3); D=Omega_c(R,0)^2-Phi_RR(R,0)=4*Omega_c(R,0)^2-kappa(R,0)^2',
             omega_definition='Circular equilibrium frequency sqrt(-F_R(R,0)/R), NOT the actual orbit angular rate v_phi/R.',
             approximation='Local circular-equivalent midplane Jacobi scale at instantaneous cylindrical R. Not an exact Jacobi boundary on an eccentric, vertically oscillating orbit.',
             vertical_treatment='Actual R,z forces and Phi_RR are auxiliary diagnostics. Jacobi denominator always uses z=0. Stored orbit z is not inserted into midplane frequency identities.',
             mass_evolution='4000 solMass held fixed throughout the stored orbit; no mass loss.')
SOFTWARE = dict(python=platform.python_version(), astropy=astropy.__version__, galpy=galpy.__version__,
                numpy=np.__version__, scipy=scipy.__version__)


def read(path): return Table.read(path, format='ascii.ecsv')

def arr(t, key, unit=None): return np.asarray(t[key].quantity.to_value(unit or t[key].unit),float)

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def hashes(paths): return {str(p.relative_to(ROOT)):sha(p) for p in paths}

def write_json(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def save(table,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    table.write(path,format='ascii.ecsv',overwrite=True)
    return read(path)

def metadata(stage, inputs):
    return dict(project=stage, mass=MASS, model=MODEL, software=SOFTWARE, inputs=hashes(inputs))

def check_hashes(snapshot):
    for name,h in snapshot.items():
        if sha(ROOT/name)!=h: raise ValueError(f'Protected file changed: {name}')

def upstream_snapshot():
    paths=[]
    for folder in ('data','results','src','tests','docs'):
        paths += [p for p in (ROOT/folder).rglob('*') if p.is_file() and
                  'project06' not in str(p.relative_to(ROOT)) and '__pycache__' not in p.parts]
    return hashes(paths)

def validate_inputs():
    if galpy.__version__!='1.12.0': raise ValueError('Requires baseline galpy 1.12.0')
    ref,orbit,spatial,master=map(read,(REFERENCE,ORBIT,SPATIAL,MASTER))
    if len(ref)!=1 or len(orbit)!=4001: raise ValueError('Unexpected baseline shapes')
    config=orbit.meta['integration']
    expected=dict(method='dop853',start_myr=-500.,stop_myr=500.,output_step_myr=.25,samples=4001,rtol=1e-11,atol=1e-11)
    for key,value in expected.items():
        if config[key]!=value: raise ValueError(f'Baseline configuration changed: {key}')
    for key,value in [('ro_kpc',RO),('vo_kms',VO)]:
        if orbit.meta['potential'][key]!=value: raise ValueError('Potential scales differ')
    if orbit.meta['potential']['name']!='galpy MWPotential2014': raise ValueError('Potential differs')
    if ref.meta['adopted_parameters']!=orbit.meta['adopted_parameters']: raise ValueError('Frame mismatch')
    if orbit.meta['inputs'][str(REFERENCE.relative_to(ROOT))]!=sha(REFERENCE): raise ValueError('Reference hash mismatch')
    t=arr(orbit,'time_myr',u.Myr)
    np.testing.assert_array_equal(t,np.linspace(-500,500,4001))
    for q in ['x_gc_kpc','y_gc_kpc','z_gc_kpc','vx_gc_kms','vy_gc_kms','vz_gc_kms']:
        np.testing.assert_allclose(arr(orbit,q)[2000],arr(ref,q)[0],rtol=0,atol=1e-9)
    if len(master)!=1456 or len(spatial)!=1456 or len(set(master['gaia_dr3_source_id']))!=1456: raise ValueError('Canonical count changed')
    # Existing order and labels must match, not just the population count.
    for q in ('gaia_dr3_source_id','homogenized_class','catalogue_status'):
        np.testing.assert_array_equal(spatial[q],master[q])
    for q,counts in [('homogenized_class',{'C':940,'L':184,'T':332}),('catalogue_status',{'shared':885,'kos_only':178,'risbud_only':393})]:
        actual={str(k):int(n) for k,n in zip(*np.unique(master[q],return_counts=True))}
        if actual!=counts: raise ValueError('Canonical labels changed')
    old=read(REF03)
    for axis in ('x','y','z','vx','vy','vz'):
        key=axis+'_ref_'+('kms' if axis.startswith('v') else 'pc')
        if spatial.meta[key]!=float(old[key][0]): raise ValueError('Spatial centre changed')
    if ref.meta['inputs'][str(REF03.relative_to(ROOT))]!=sha(REF03): raise ValueError('05/03 reference mismatch')
    return ref,orbit,spatial


def field_at(R,z):
    """Analytic galpy field, with separate actual-position and midplane fields."""
    if not np.isfinite(R+z) or R<=0: raise ValueError('Invalid cylindrical position')
    rn,zn=R/RO,z/RO
    kw=dict(use_physical=False)
    om=float(omegac(MWPotential2014,rn,**kw))*VO/RO
    kap=float(epifreq(MWPotential2014,rn,**kw))*VO/RO
    nu=float(verticalfreq(MWPotential2014,rn,**kw))*VO/RO
    rr=float(evaluateR2derivs(MWPotential2014,rn,0.,**kw))*(VO/RO)**2
    fr=float(evaluateRforces(MWPotential2014,rn,0.,**kw))*VO**2/RO
    d=om**2-rr
    np.testing.assert_allclose(d,4*om**2-kap**2,rtol=1e-12,atol=1e-9)
    np.testing.assert_allclose(om**2,-fr/R,rtol=1e-12)
    values=dict(R_kpc=R,z_actual_kpc=z,z_jacobi_kpc=0.,
                F_R_actual=float(evaluateRforces(MWPotential2014,rn,zn,**kw))*VO**2/RO,
                F_z_actual=float(evaluatezforces(MWPotential2014,rn,zn,**kw))*VO**2/RO,
                Phi_RR_actual=float(evaluateR2derivs(MWPotential2014,rn,zn,**kw))*(VO/RO)**2,
                F_R_midplane=fr,Phi_RR_midplane=rr,Omega_c=om,kappa=kap,nu=nu,denominator=d)
    if not np.all(np.isfinite(list(values.values()))) or d<=0: raise ValueError('Invalid field / nonpositive Jacobi denominator')
    return values

FIELD_UNITS=dict(R_kpc=u.kpc,z_actual_kpc=u.kpc,z_jacobi_kpc=u.kpc,
                 F_R_actual=FUNIT,F_z_actual=FUNIT,Phi_RR_actual=DUNIT,
                 F_R_midplane=FUNIT,Phi_RR_midplane=DUNIT,Omega_c=FREQUNIT,kappa=FREQUNIT,nu=FREQUNIT,denominator=DUNIT)

def field_table(rows):
    table=Table(rows=rows)
    for key,unit in FIELD_UNITS.items(): table[key].unit=unit
    return table

def jacobi(mass,denominator):
    m=u.Quantity(mass).to(u.Msun); d=u.Quantity(denominator).to(DUNIT)
    if not np.all(np.isfinite(m.value)) or not np.all(np.isfinite(d.value)) or np.any(m<=0*u.Msun) or np.any(d<=0*DUNIT):
        raise ValueError('Positive finite mass and denominator required')
    return np.cbrt((G*m/d).to_value(u.pc**3))*u.pc

def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.grid':True,'grid.alpha':.18,'savefig.dpi':220})
    return plt

def savefig(fig,folder,name):
    folder.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'): fig.savefig(folder/f'{name}.{ext}',bbox_inches='tight')


def check_run_protection():
    manifest=RESULTS/'upstream_integrity.json'
    check_hashes(json.loads(manifest.read_text())['sha256'])
