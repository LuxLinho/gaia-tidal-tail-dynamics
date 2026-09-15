"""Project 04B: integrate the fixed Stock 2 reference orbit in MWPotential2014.

This stage integrates only the one-row Project 03C/04A Stock 2 reference.
No source-by-source orbit integration or membership inference is performed.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import numpy as np
from astropy import units as u
from astropy.table import Table
from scipy.signal import find_peaks

try:
    import galpy
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014
except Exception:  # keep module importable for static/unit checks before dependency install
    galpy = None
    Orbit = None
    MWPotential2014 = None

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'data/interim/project04/stock2_galactocentric_reference.ecsv'
PCA = ROOT / 'data/interim/project03/stock2_morphology_pca.ecsv'
OUTPUT = ROOT / 'data/interim/project04'
FIGURES = ROOT / 'results/project04/reference_orbit'

# MWPotential2014 is defined in galpy natural units with these reference scales.
RO = 8.0 * u.kpc
VO = 220.0 * u.km / u.s
PAST = 1.0 * u.Gyr
FUTURE = 1.0 * u.Gyr
N_HALF = 4001
METHOD = 'dop853'

POTENTIAL_META = {
    'name': 'galpy MWPotential2014',
    'reference': 'Bovy 2015, ApJS 216, 29',
    'galpy_docs': 'https://docs.galpy.org/en/latest/tutorials/potentials/milky_way_potentials.html',
    'ro_kpc': 8.0,
    'vo_kms': 220.0,
    'components': [
        {'type': 'PowerSphericalPotentialwCutoff', 'alpha': 1.8, 'rc_over_ro': 1.9/8.0, 'normalize': 0.05},
        {'type': 'MiyamotoNagaiPotential', 'a_over_ro': 3.0/8.0, 'b_over_ro': 0.28/8.0, 'normalize': 0.60},
        {'type': 'NFWPotential', 'a_over_ro': 16.0/8.0, 'normalize': 0.35},
    ],
    'scope': 'Baseline axisymmetric reference potential for Project 04B; potential robustness is deferred to a later project.',
}


def values(table, name):
    return np.asarray(table[name].quantity.value, dtype=float)


def load_reference(path=INPUT):
    table = Table.read(path, format='ascii.ecsv')
    if len(table) != 1:
        raise ValueError('Project 04A reference product must contain exactly one row')
    required = ['x_gc_kpc','y_gc_kpc','z_gc_kpc','vx_gc_kms','vy_gc_kms','vz_gc_kms',
                'R_gc_kpc','phi_gc_deg','v_R_gc_kms','v_phi_gc_kms']
    missing = [q for q in required if q not in table.colnames]
    if missing:
        raise ValueError(f'Missing Project 04A reference columns: {missing}')
    return table


def load_fixed_axis(path=PCA):
    table = Table.read(path, format='ascii.ecsv')
    # Project 03D stores 3 rows/components for each population. Accept the established schema.
    mask = np.asarray(table['population']).astype(str) == 'L+T'
    rows = table[mask]
    if len(rows) != 3:
        raise ValueError('Expected three L+T PCA component rows')
    if 'component' in rows.colnames:
        order = np.argsort(np.asarray(rows['component'], dtype=int))
        rows = rows[order]
    candidates = [
        ('e_x','e_y','e_z'),
        ('vector_x','vector_y','vector_z'),
        ('eigenvector_x','eigenvector_y','eigenvector_z'),
        ('axis_x','axis_y','axis_z'),
    ]
    for cols in candidates:
        if all(c in rows.colnames for c in cols):
            axis = np.array([float(rows[c][0]) for c in cols])
            break
    else:
        # The existing Project 03D product is known to place vector components in these columns.
        vector_cols = [c for c in rows.colnames if c.lower().endswith(('_x','_y','_z'))]
        raise ValueError(f'Cannot identify PCA vector columns; available vector-like fields: {vector_cols}')
    norm = np.linalg.norm(axis)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError('Invalid stored L+T PCA axis')
    return axis / norm


def astropy_to_galpy_cyl(reference):
    """Map Project 04A Astropy Galactocentric convention into galpy cylindrical ICs.

    Astropy has the Sun on -X and solar rotation near +Y.  For the axisymmetric
    galpy potential we reflect X: x_g=-x_a, y_g=y_a, z_g=z_a.  Therefore
    phi_g=pi-phi_a and vT_g=-vphi_a, while R, vR, z, vz are unchanged.
    """
    R = float(reference['R_gc_kpc'][0]) * u.kpc
    vR = float(reference['v_R_gc_kms'][0]) * u.km/u.s
    vT = -float(reference['v_phi_gc_kms'][0]) * u.km/u.s
    z = float(reference['z_gc_kpc'][0]) * u.kpc
    vz = float(reference['vz_gc_kms'][0]) * u.km/u.s
    phi_a = np.deg2rad(float(reference['phi_gc_deg'][0]))
    phi_g = np.arctan2(np.sin(phi_a), -np.cos(phi_a)) * u.rad
    return [R, vR, vT, z, vz, phi_g]


def galpy_cyl_to_astropy_cart(R, phi, z, vR, vT, vz):
    """Return Astropy-04A Cartesian axes from galpy cylindrical arrays."""
    R = np.asarray(R, float); phi = np.asarray(phi, float); z = np.asarray(z, float)
    vR = np.asarray(vR, float); vT = np.asarray(vT, float); vz = np.asarray(vz, float)
    xg = R*np.cos(phi); yg = R*np.sin(phi)
    vxg = vR*np.cos(phi) - vT*np.sin(phi)
    vyg = vR*np.sin(phi) + vT*np.cos(phi)
    return (-xg, yg, z, -vxg, vyg, vz)


def build_orbit(reference):
    if Orbit is None:
        raise RuntimeError('Project 04B requires galpy. Install it in the project environment with: python -m pip install galpy')
    return Orbit(astropy_to_galpy_cyl(reference), ro=RO, vo=VO)


def integrate_two_sided(orbit):
    # Each integration begins from t=0 so the supplied reference is the present-day state.
    past_t = np.linspace(0.0, -PAST.to_value(u.Gyr), N_HALF) * u.Gyr
    future_t = np.linspace(0.0, FUTURE.to_value(u.Gyr), N_HALF) * u.Gyr
    op = Orbit(astropy_to_galpy_cyl(load_reference()), ro=RO, vo=VO)
    of = Orbit(astropy_to_galpy_cyl(load_reference()), ro=RO, vo=VO)
    op.integrate(past_t, MWPotential2014, method=METHOD)
    of.integrate(future_t, MWPotential2014, method=METHOD)

    def sample(o, t):
        return dict(
            R=np.asarray(o.R(t, use_physical=True), float),
            phi=np.asarray(o.phi(t, use_physical=True), float),
            z=np.asarray(o.z(t, use_physical=True), float),
            vR=np.asarray(o.vR(t, use_physical=True), float),
            vT=np.asarray(o.vT(t, use_physical=True), float),
            vz=np.asarray(o.vz(t, use_physical=True), float),
            E=np.asarray(o.E(t, pot=MWPotential2014, use_physical=True), float),
        )
    p, f = sample(op, past_t), sample(of, future_t)
    # Reverse past to increasing chronological time and remove its duplicate t=0.
    time = np.concatenate([past_t.to_value(u.Myr)[::-1][:-1], future_t.to_value(u.Myr)])
    combined = {}
    for key in p:
        combined[key] = np.concatenate([p[key][::-1][:-1], f[key]])
    return time, combined


def trajectory_table(time_myr, raw):
    x,y,z,vx,vy,vz = galpy_cyl_to_astropy_cart(raw['R'],raw['phi'],raw['z'],raw['vR'],raw['vT'],raw['vz'])
    table = Table()
    table['time_myr'] = time_myr * u.Myr
    table['x_gc_kpc'] = x * u.kpc
    table['y_gc_kpc'] = y * u.kpc
    table['z_gc_kpc'] = z * u.kpc
    table['R_gc_kpc'] = np.hypot(x,y) * u.kpc
    table['r_gc_kpc'] = np.sqrt(x*x+y*y+z*z) * u.kpc
    table['phi_gc_deg'] = np.degrees(np.arctan2(y,x)) * u.deg
    table['vx_gc_kms'] = vx * u.km/u.s
    table['vy_gc_kms'] = vy * u.km/u.s
    table['vz_gc_kms'] = vz * u.km/u.s
    R = np.hypot(x,y)
    table['v_R_gc_kms'] = ((x*vx+y*vy)/R) * u.km/u.s
    table['v_phi_gc_kms'] = ((-y*vx+x*vy)/R) * u.km/u.s
    table['v_gc_kms'] = np.sqrt(vx*vx+vy*vy+vz*vz) * u.km/u.s
    table['energy_galpy_kms2'] = raw['E'] * (u.km/u.s)**2
    table.meta.update(project='04B', potential=POTENTIAL_META, galpy_version=getattr(galpy,'__version__','unknown'),
                      integration_method=METHOD, past_gyr=PAST.to_value(u.Gyr), future_gyr=FUTURE.to_value(u.Gyr),
                      n_half=N_HALF, coordinate_bridge='x_galpy=-x_astropy; y,z unchanged; vT_galpy=-vphi_astropy')
    return table


def unit(v):
    v=np.asarray(v,float); n=np.linalg.norm(v)
    return v/n if np.isfinite(n) and n>0 else np.full(3,np.nan)


def unsigned_angle(a,b):
    dot=np.clip(abs(float(np.dot(unit(a),unit(b)))),-1.0,1.0)
    return float(np.degrees(np.arccos(dot)))


def interpolate_vector(table, t_myr, fields):
    t=values(table,'time_myr')
    return np.array([np.interp(t_myr,t,values(table,q)) for q in fields])


def chord_direction(table, half_window_myr):
    minus=interpolate_vector(table,-half_window_myr,['x_gc_kpc','y_gc_kpc','z_gc_kpc'])
    plus=interpolate_vector(table,+half_window_myr,['x_gc_kpc','y_gc_kpc','z_gc_kpc'])
    return unit(plus-minus)


def orbital_diagnostics(table, axis):
    t=values(table,'time_myr')
    R=values(table,'R_gc_kpc'); z=values(table,'z_gc_kpc')
    rperi=float(np.min(R)); rap=float(np.max(R))
    ecc=(rap-rperi)/(rap+rperi)
    zmax=float(np.max(np.abs(z)))
    minima,_=find_peaks(-R, distance=max(2,int(100.0/(np.median(np.diff(t))))))
    radial_period=float(np.median(np.diff(t[minima]))) if len(minima)>=2 else np.nan
    phi=np.unwrap(np.radians(values(table,'phi_gc_deg')))
    # Median instantaneous azimuthal period is robust for a nearly circular disk orbit.
    dphi_dt=np.gradient(phi,t)
    good=np.isfinite(dphi_dt)&(np.abs(dphi_dt)>0)
    az_period=float(np.median(2*np.pi/np.abs(dphi_dt[good]))) if good.any() else np.nan
    i0=int(np.argmin(np.abs(t)))
    r0=np.array([values(table,q)[i0] for q in ['x_gc_kpc','y_gc_kpc','z_gc_kpc']])
    v0=np.array([values(table,q)[i0] for q in ['vx_gc_kms','vy_gc_kms','vz_gc_kms']])
    L=unit(np.cross(r0,v0))
    rows=[]
    for w in (5.0,20.0,50.0):
        d=chord_direction(table,w)
        rows.append((w,*d,unsigned_angle(axis,d)))
    E=values(table,'energy_galpy_kms2')
    energy_span=float(np.ptp(E))
    energy_scale=max(abs(float(np.median(E))),1.0)
    return dict(rperi_kpc=rperi,rap_kpc=rap,eccentricity=ecc,zmax_kpc=zmax,
                radial_period_myr=radial_period,azimuthal_period_myr=az_period,
                energy_span_kms2=energy_span,relative_energy_span=energy_span/energy_scale,
                angular_momentum_unit=L,axis_to_orbit_plane_normal_deg=unsigned_angle(axis,L),chords=rows)


def validate(reference, trajectory, diagnostics):
    if len(trajectory) != 2*N_HALF-1:
        raise ValueError('Unexpected trajectory length')
    t=values(trajectory,'time_myr')
    if not np.all(np.diff(t)>0) or np.count_nonzero(np.isclose(t,0.0,atol=1e-12))!=1:
        raise ValueError('Two-sided time grid invalid')
    i0=int(np.argmin(np.abs(t)))
    for q in ['x_gc_kpc','y_gc_kpc','z_gc_kpc','vx_gc_kms','vy_gc_kms','vz_gc_kms']:
        trajectory_value = trajectory[q][i0]
        reference_value = reference[q][0]
        if hasattr(trajectory_value, 'to_value'):
            trajectory_value = trajectory_value.to_value(trajectory[q].unit)
        elif hasattr(trajectory_value, 'value'):
            trajectory_value = trajectory_value.value
        if hasattr(reference_value, 'to_value'):
            reference_value = reference_value.to_value(reference[q].unit)
        elif hasattr(reference_value, 'value'):
            reference_value = reference_value.value
        np.testing.assert_allclose(float(trajectory_value), float(reference_value), rtol=0, atol=3e-8)
    if not (diagnostics['rperi_kpc']>0 and diagnostics['rap_kpc']>=diagnostics['rperi_kpc']):
        raise ValueError('Invalid peri/apocentre')
    if not (0<=diagnostics['eccentricity']<1):
        raise ValueError('Invalid eccentricity')
    if diagnostics['zmax_kpc']<0:
        raise ValueError('Invalid zmax')
    # DOP853 should conserve energy much better than this loose regression threshold.
    if diagnostics['relative_energy_span']>1e-7:
        raise ValueError(f'Energy conservation failed: {diagnostics["relative_energy_span"]:.3e}')


def products(reference, trajectory, diagnostics, axis):
    summary=Table()
    scalar_units={
        'rperi_kpc':u.kpc,'rap_kpc':u.kpc,'eccentricity':None,'zmax_kpc':u.kpc,
        'radial_period_myr':u.Myr,'azimuthal_period_myr':u.Myr,
        'energy_span_kms2':(u.km/u.s)**2,'relative_energy_span':None,
        'axis_to_orbit_plane_normal_deg':u.deg,
    }
    for q,unit0 in scalar_units.items():
        summary[q]=[diagnostics[q]] if unit0 is None else [diagnostics[q]]*unit0
    summary['e1_ext_x']=[axis[0]]; summary['e1_ext_y']=[axis[1]]; summary['e1_ext_z']=[axis[2]]
    L=diagnostics['angular_momentum_unit']
    summary['orbit_Lhat_x']=[L[0]]; summary['orbit_Lhat_y']=[L[1]]; summary['orbit_Lhat_z']=[L[2]]
    summary.meta.update(potential=POTENTIAL_META,reference_source=str(INPUT.relative_to(ROOT)),
                        reference_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
                        pca_source=str(PCA.relative_to(ROOT)),pca_sha256=hashlib.sha256(PCA.read_bytes()).hexdigest())
    align=Table(rows=diagnostics['chords'],names=['half_window_myr','chord_x','chord_y','chord_z','axis_alignment_deg'])
    align['half_window_myr'].unit=u.Myr; align['axis_alignment_deg'].unit=u.deg
    align.meta.update(description='Sign-invariant angle between fixed Project 03D L+T major axis and finite orbit chord from -window to +window around present.')
    return summary,align


def make_plots(trajectory, reference):
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True,exist_ok=True)
    t=values(trajectory,'time_myr')
    x=values(trajectory,'x_gc_kpc'); y=values(trajectory,'y_gc_kpc'); z=values(trajectory,'z_gc_kpc'); R=values(trajectory,'R_gc_kpc')
    refs=(float(reference['x_gc_kpc'][0]),float(reference['y_gc_kpc'][0]),float(reference['z_gc_kpc'][0]))
    figs=[]
    fig,ax=plt.subplots(figsize=(7,7)); ax.plot(x,y,lw=1); ax.scatter([0],[0],marker='*',s=80,label='Galactic centre'); ax.scatter([refs[0]],[refs[1]],s=30,label='Stock 2 now'); ax.set_xlabel('X [kpc]'); ax.set_ylabel('Y [kpc]'); ax.set_aspect('equal',adjustable='box'); ax.legend(); fig.tight_layout(); figs.append(('reference_orbit_xy',fig))
    fig,ax=plt.subplots(figsize=(8,5)); ax.plot(t,R,lw=1); ax.axvline(0,lw=.8); ax.set_xlabel('Time from present [Myr]'); ax.set_ylabel('R [kpc]'); fig.tight_layout(); figs.append(('reference_orbit_R_time',fig))
    fig,ax=plt.subplots(figsize=(8,5)); ax.plot(t,z*1000,lw=1); ax.axvline(0,lw=.8); ax.axhline(0,lw=.6); ax.set_xlabel('Time from present [Myr]'); ax.set_ylabel('Z [pc]'); fig.tight_layout(); figs.append(('reference_orbit_Z_time',fig))
    for name,fig in figs:
        fig.savefig(FIGURES/f'{name}.png',dpi=180); fig.savefig(FIGURES/f'{name}.pdf'); plt.close(fig)


def report(reference,diagnostics,align,outputs):
    lines=['Project 04B — Stock 2 reference orbit integration','='*64,'',
           'Scope: fixed Project 03C/04A Stock 2 reference only; no source-by-source integration.',
           'Potential: galpy MWPotential2014 (Bovy 2015), baseline axisymmetric model.',
           f'galpy version: {getattr(galpy,"__version__","unknown")}',
           f'Integration: {METHOD}, -{PAST.to_value(u.Gyr):g} to +{FUTURE.to_value(u.Gyr):g} Gyr, {2*N_HALF-1} stored samples.', '',
           f'R_peri: {diagnostics["rperi_kpc"]:.6f} kpc',
           f'R_apo: {diagnostics["rap_kpc"]:.6f} kpc',
           f'eccentricity: {diagnostics["eccentricity"]:.6f}',
           f'Z_max: {diagnostics["zmax_kpc"]*1000:.3f} pc',
           f'radial period (sampled minima): {diagnostics["radial_period_myr"]:.3f} Myr',
           f'azimuthal period (median instantaneous): {diagnostics["azimuthal_period_myr"]:.3f} Myr',
           f'relative energy span: {diagnostics["relative_energy_span"]:.3e}','',
           'Fixed Project 03D L+T axis versus finite integrated orbit chords:']
    for row in align:
        lines.append(f'  +/-{float(row["half_window_myr"]):g} Myr: {float(row["axis_alignment_deg"]):.6f} deg')
    lines += ['',f'Angle between fixed L+T axis and orbital angular-momentum normal: {diagnostics["axis_to_orbit_plane_normal_deg"]:.6f} deg',
              'An angle near 90 deg here means the spatial axis lies near the instantaneous orbital plane; it does not establish tidal origin.','',
              'Interpretation limits:',
              '- The instantaneous orbit tangent is exactly the instantaneous velocity direction, so Project 04A already measured that angle.',
              '- Finite orbit chords add curvature context but remain model-dependent.',
              '- MWPotential2014 is one baseline potential. Potential robustness is deferred.',
              '- No membership, bound/unbound, Jacobi-radius, action, or individual-star orbit inference is performed.','',
              'Invariants:','fixed reference used unchanged: True','one reference orbit only: True','no canonical source table modified: True','no source membership inference: True','no source-by-source orbit integration: True',
              f'energy conservation threshold passed: {diagnostics["relative_energy_span"]<=1e-7}',
              'PROJECT 04B STATUS: PASS','', 'Outputs:']
    lines += [str(p.relative_to(ROOT)) for p in outputs]
    return '\n'.join(lines)+'\n'


def main():
    reference=load_reference(); axis=load_fixed_axis()
    # Protect all prior project products from accidental writes.
    protected=[]
    for p in list((ROOT/'data/interim/project03').glob('*'))+list((ROOT/'data/interim/project04').glob('stock2_galactocentric_*')):
        if p.is_file(): protected.append((p,hashlib.sha256(p.read_bytes()).hexdigest()))
    orbit=build_orbit(reference)
    time,raw=integrate_two_sided(orbit)
    trajectory=trajectory_table(time,raw)
    diagnostics=orbital_diagnostics(trajectory,axis)
    validate(reference,trajectory,diagnostics)
    summary,align=products(reference,trajectory,diagnostics,axis)
    OUTPUT.mkdir(parents=True,exist_ok=True); FIGURES.mkdir(parents=True,exist_ok=True)
    paths=[]
    for name,table in [('stock2_reference_orbit',trajectory),('stock2_reference_orbit_summary',summary),('stock2_reference_orbit_alignments',align)]:
        path=OUTPUT/f'{name}.ecsv'; table.write(path,format='ascii.ecsv',overwrite=True); paths.append(path)
    make_plots(trajectory,reference)
    paths += sorted(FIGURES.glob('*.png'))+sorted(FIGURES.glob('*.pdf'))
    for p,h in protected:
        if hashlib.sha256(p.read_bytes()).hexdigest()!=h: raise RuntimeError(f'Prior product changed: {p}')
    text=report(reference,diagnostics,align,paths+[FIGURES/'reference_orbit_report.txt'])
    report_path=FIGURES/'reference_orbit_report.txt'; report_path.write_text(text); print(text,end='')


if __name__=='__main__':
    main()
