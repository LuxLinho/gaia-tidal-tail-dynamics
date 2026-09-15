"""Project 04C: local Galactic gravitational tidal field around Stock 2.

Uses the same fixed MWPotential2014 baseline as Project 04B.  This stage computes
T_ij = -d2 Phi / dx_i dx_j (the inertial gravitational acceleration-gradient
/ tidal tensor), not an effective rotating-frame tensor.  No source membership,
Jacobi radius, escape criterion, or source-by-source orbit inference is performed.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
from astropy import units as u
from astropy.table import Table

try:
    import galpy
    from galpy.potential import (
        MWPotential2014,
        evaluateRforces,
        evaluateR2derivs,
        evaluateRzderivs,
        evaluatez2derivs,
    )
except Exception:  # keep importable for static checks
    galpy = None
    MWPotential2014 = None
    evaluateRforces = evaluateR2derivs = evaluateRzderivs = evaluatez2derivs = None

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / 'data/interim/project04/stock2_galactocentric_reference.ecsv'
ORBIT = ROOT / 'data/interim/project04/stock2_reference_orbit.ecsv'
ORBIT_SUMMARY = ROOT / 'data/interim/project04/stock2_reference_orbit_summary.ecsv'
OUTPUT = ROOT / 'data/interim/project04'
FIGURES = ROOT / 'results/project04/local_tidal_field'

RO_KPC = 8.0
VO_KMS = 220.0
SAMPLE_TIMES_MYR = np.array([-50., -20., -5., 0., 5., 20., 50.])
FD_STEP_KPC = 1.0e-4  # 0.1 pc, only for an independent derivative audit

POTENTIAL_META = {
    'name': 'galpy MWPotential2014',
    'reference': 'Bovy 2015, ApJS 216, 29',
    'ro_kpc': RO_KPC,
    'vo_kms': VO_KMS,
    'scope': 'Same baseline axisymmetric potential as Project 04B.',
}


def arr(table, name):
    col = table[name]
    if getattr(col, 'unit', None) is not None:
        return np.asarray(col.quantity.value, dtype=float)
    return np.asarray(col, dtype=float)


def scalar(table, name):
    return float(arr(table, name)[0])


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n <= 0:
        raise ValueError('Cannot normalize invalid vector')
    return v / n


def unsigned_angle(a, b):
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(unit(a), unit(b))), 0., 1.))))


def directed_angle(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(unit(a), unit(b)), -1., 1.))))


def load_inputs(reference_path=REFERENCE, orbit_path=ORBIT, summary_path=ORBIT_SUMMARY):
    reference = Table.read(reference_path, format='ascii.ecsv')
    orbit = Table.read(orbit_path, format='ascii.ecsv')
    summary = Table.read(summary_path, format='ascii.ecsv')
    if len(reference) != 1 or len(summary) != 1:
        raise ValueError('Expected one-row Project 04A reference and 04B summary')
    if len(orbit) < 3:
        raise ValueError('Project 04B orbit is missing or too short')
    required_ref = ['x_gc_kpc','y_gc_kpc','z_gc_kpc','vx_gc_kms','vy_gc_kms','vz_gc_kms','e1_ext_gc']
    missing = [q for q in required_ref if q not in reference.colnames]
    if missing:
        raise ValueError(f'Missing Project 04A reference fields: {missing}')
    for q in ['time_myr','x_gc_kpc','y_gc_kpc','z_gc_kpc','vx_gc_kms','vy_gc_kms','vz_gc_kms']:
        if q not in orbit.colnames:
            raise ValueError(f'Missing Project 04B orbit field: {q}')
    return reference, orbit, summary


def fixed_axis_gc(reference):
    raw = reference['e1_ext_gc'][0]
    if hasattr(raw, 'value'):
        raw = raw.value
    axis = np.asarray(raw, dtype=float).reshape(3)
    return unit(axis)


def local_basis(x, y):
    R = float(np.hypot(x, y))
    if not np.isfinite(R) or R <= 0:
        raise ValueError('R=0 has undefined local cylindrical basis')
    eR = np.array([x/R, y/R, 0.])
    ephi = np.array([-y/R, x/R, 0.])
    eZ = np.array([0., 0., 1.])
    basis = np.column_stack([eR, ephi, eZ])  # local components -> global Astropy GC Cartesian
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=2e-14)
    if np.linalg.det(basis) < 0.999999999999:
        raise ValueError('Local basis is not right-handed')
    return basis


def natural_derivatives(R_kpc, z_kpc):
    if MWPotential2014 is None:
        raise RuntimeError('Project 04C requires galpy in the active environment')
    Rn = float(R_kpc) / RO_KPC
    zn = float(z_kpc) / RO_KPC
    # All calls deliberately use galpy natural units; scaling is explicit below.
    FR = float(evaluateRforces(MWPotential2014, Rn, zn, use_physical=False))
    RR = float(evaluateR2derivs(MWPotential2014, Rn, zn, use_physical=False))
    RZ = float(evaluateRzderivs(MWPotential2014, Rn, zn, use_physical=False))
    ZZ = float(evaluatez2derivs(MWPotential2014, Rn, zn, use_physical=False))
    return FR, RR, RZ, ZZ


def tidal_tensor_at(x_kpc, y_kpc, z_kpc):
    """Return gravitational tidal tensor in Astropy-04A Cartesian axes.

    T = grad(acceleration) = - Hessian(Phi), units (km/s)^2/kpc^2.
    For an axisymmetric Phi(R,z), the potential Hessian in the orthonormal
    (e_R,e_phi,e_Z) basis is [[Phi_RR,0,Phi_Rz],[0,Phi_R/R,0],[Phi_Rz,0,Phi_ZZ]].
    Since F_R=-Phi_R, Phi_R/R = -F_R/R.
    """
    R = float(np.hypot(x_kpc, y_kpc))
    if R <= 0:
        raise ValueError('Cannot evaluate cylindrical Hessian at R=0')
    FRn, RRn, RZn, ZZn = natural_derivatives(R, z_kpc)
    scale = (VO_KMS ** 2) / (RO_KPC ** 2)  # (km/s)^2/kpc^2 per natural second derivative
    phi_rr = RRn * scale
    phi_rz = RZn * scale
    phi_zz = ZZn * scale
    # Phi_R/R: FR_n is in vo^2/ro, and R_n=R/ro -> same vo^2/ro^2 scale.
    phi_phiphi = (-FRn / (R / RO_KPC)) * scale
    H_local = np.array([[phi_rr, 0., phi_rz], [0., phi_phiphi, 0.], [phi_rz, 0., phi_zz]])
    basis = local_basis(x_kpc, y_kpc)
    H_global = basis @ H_local @ basis.T
    T_global = -H_global
    T_global = 0.5 * (T_global + T_global.T)
    return T_global, H_local, basis


def deterministic_eigensystem(tensor):
    """Eigenvalues descending; deterministic signs; right-handed eigenbasis columns."""
    w, V = np.linalg.eigh(np.asarray(tensor, dtype=float))
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    for j in range(3):
        v = V[:, j]
        idx = int(np.argmax(np.abs(v)))
        if v[idx] < 0:
            V[:, j] *= -1
    # Sign of an eigenvector is arbitrary; enforce a right-handed triad by flipping e3 only.
    if np.linalg.det(V) < 0:
        V[:, 2] *= -1
    np.testing.assert_allclose(V.T @ V, np.eye(3), atol=2e-12)
    np.testing.assert_allclose(tensor @ V, V * w[None, :], rtol=1e-11, atol=1e-9)
    return w, V


def finite_difference_audit(R_kpc, z_kpc):
    """Independent check of cylindrical Hessian derivatives from forces."""
    h = FD_STEP_KPC
    def forces(R, z):
        Rn, zn = R/RO_KPC, z/RO_KPC
        FR = float(evaluateRforces(MWPotential2014, Rn, zn, use_physical=False)) * VO_KMS**2 / RO_KPC
        # z-force import avoided: dPhi/dz can be checked through symmetry of Rz using R-force only;
        # RR and Rz are enough to audit the nontrivial radial block used here.
        return FR
    dFR_dR = (forces(R_kpc+h, z_kpc) - forces(R_kpc-h, z_kpc)) / (2*h)
    dFR_dz = (forces(R_kpc, z_kpc+h) - forces(R_kpc, z_kpc-h)) / (2*h)
    _, RRn, RZn, _ = natural_derivatives(R_kpc, z_kpc)
    scale = VO_KMS**2/RO_KPC**2
    analytic = np.array([RRn*scale, RZn*scale])
    numerical = np.array([-dFR_dR, -dFR_dz])  # Hessian Phi = - derivative(force)
    rel = np.abs(analytic-numerical)/np.maximum(np.abs(analytic), 1e-12)
    return analytic, numerical, rel


def interpolate_state(orbit, time_myr):
    t = arr(orbit, 'time_myr')
    if time_myr < t.min() or time_myr > t.max():
        raise ValueError('Requested time outside stored Project 04B orbit')
    fields = ['x_gc_kpc','y_gc_kpc','z_gc_kpc','vx_gc_kms','vy_gc_kms','vz_gc_kms']
    return np.array([np.interp(time_myr, t, arr(orbit, q)) for q in fields], dtype=float)


def one_epoch(time_myr, state, fixed_axis):
    x,y,z,vx,vy,vz = state
    T, Hlocal, basis = tidal_tensor_at(x,y,z)
    eigenvalues, eigenvectors = deterministic_eigensystem(T)
    eR,ephi,eZ = basis[:,0],basis[:,1],basis[:,2]
    tangent = unit([vx,vy,vz])
    labels = ['lambda1','lambda2','lambda3']
    rows=[]
    for j,label in enumerate(labels):
        ev = eigenvectors[:,j]
        rows.append(dict(
            time_myr=time_myr, component=j+1, eigen_label=label,
            eigenvalue_kms2_kpc2=eigenvalues[j],
            e_x=ev[0],e_y=ev[1],e_z=ev[2],
            sign_class='stretching' if eigenvalues[j] > 0 else ('compressive' if eigenvalues[j] < 0 else 'neutral'),
            angle_to_fixed_axis_deg=unsigned_angle(ev,fixed_axis),
            angle_to_radial_deg=unsigned_angle(ev,eR),
            angle_to_azimuthal_deg=unsigned_angle(ev,ephi),
            angle_to_vertical_deg=unsigned_angle(ev,eZ),
            angle_to_velocity_deg=unsigned_angle(ev,tangent),
        ))
    return rows, T, basis


def current_tensor_table(T, eigen_rows, audit):
    labels=['x','y','z']
    rows=[]
    for i in range(3):
        for j in range(3):
            rows.append(dict(row_axis=labels[i],col_axis=labels[j],tidal_kms2_kpc2=T[i,j]))
    table=Table(rows=rows)
    table['tidal_kms2_kpc2'].unit=(u.km/u.s)**2/u.kpc**2
    table.meta.update(
        definition='T_ij = partial a_i / partial x_j = -partial_i partial_j Phi in inertial Galactocentric Cartesian axes.',
        interpretation='Positive eigenvalue = differential stretching along that eigenvector; negative = differential compression. This is NOT the rotating-frame effective tidal tensor.',
        potential=POTENTIAL_META,
        finite_difference_step_kpc=FD_STEP_KPC,
        finite_difference_max_relative_error=float(np.max(audit[2])),
    )
    return table


def eigen_table(rows):
    table=Table(rows=rows)
    table['time_myr'].unit=u.Myr
    table['eigenvalue_kms2_kpc2'].unit=(u.km/u.s)**2/u.kpc**2
    for q in [c for c in table.colnames if c.endswith('_deg')]: table[q].unit=u.deg
    table.meta.update(
        eigen_order='Descending eigenvalue at each epoch. Signs deterministic by largest-absolute Cartesian component; third vector may be flipped to preserve right-handedness.',
        axis_angle='Unsigned 0-90 degree alignment angle because PCA/eigenvector signs are arbitrary.',
        fixed_axis_source='Project 04A e1_ext_gc, itself an Astropy rotation of the fixed Project 03D L+T PCA major axis; no refit.',
    )
    return table


def summary_table(reference, T, rows, audit):
    now=[r for r in rows if abs(r['time_myr'])<1e-12]
    stretch=[r for r in now if r['eigenvalue_kms2_kpc2']>0]
    strongest=max(now,key=lambda r:r['eigenvalue_kms2_kpc2'])
    table=Table()
    table['x_gc_kpc']=[scalar(reference,'x_gc_kpc')]*u.kpc
    table['y_gc_kpc']=[scalar(reference,'y_gc_kpc')]*u.kpc
    table['z_gc_kpc']=[scalar(reference,'z_gc_kpc')]*u.kpc
    table['trace_tidal_kms2_kpc2']=[float(np.trace(T))]*(u.km/u.s)**2/u.kpc**2
    table['n_positive_eigenvalues']=[len(stretch)]
    for r in now:
        n=r['component']
        table[f'lambda{n}_kms2_kpc2']=[r['eigenvalue_kms2_kpc2']]*(u.km/u.s)**2/u.kpc**2
        table[f'lambda{n}_axis_alignment_deg']=[r['angle_to_fixed_axis_deg']]*u.deg
    table['strongest_tidal_axis_component']=[int(strongest['component'])]
    table['strongest_tidal_axis_alignment_deg']=[strongest['angle_to_fixed_axis_deg']]*u.deg
    table['fd_max_relative_error']=[float(np.max(audit[2]))]
    table.meta.update(potential=POTENTIAL_META, definition='Present-day inertial gravitational tidal tensor summary.')
    return table


def validate(reference, orbit, T, rows, audit, fixed_axis):
    np.testing.assert_allclose(T, T.T, atol=1e-11)
    now=[r for r in rows if abs(r['time_myr'])<1e-12]
    if len(now)!=3: raise ValueError('Expected three present-day tidal eigenmodes')
    if not all(0 <= r['angle_to_fixed_axis_deg'] <= 90 for r in rows): raise ValueError('Invalid unsigned angle')
    if np.max(audit[2]) > 5e-5: raise ValueError(f'Finite-difference derivative audit failed: {np.max(audit[2]):.3e}')
    # Exact present state in 04B must agree with fixed 04A reference to the tolerance already used in 04B.
    state0=interpolate_state(orbit,0.)
    ref=np.array([scalar(reference,q) for q in ['x_gc_kpc','y_gc_kpc','z_gc_kpc','vx_gc_kms','vy_gc_kms','vz_gc_kms']])
    np.testing.assert_allclose(state0,ref,rtol=0,atol=3e-8)
    np.testing.assert_allclose(np.linalg.norm(fixed_axis),1.,atol=1e-12)


def make_plots(eigs, fixed_axis):
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True,exist_ok=True)
    outputs=[]
    t=arr(eigs,'time_myr')
    for quantity,ylabel,name in [
        ('eigenvalue_kms2_kpc2',r'Tidal eigenvalue [(km/s)$^2$/kpc$^2$]','tidal_eigenvalues_time'),
        ('angle_to_fixed_axis_deg','Alignment with fixed L+T axis [deg]','tidal_axis_alignment_time')]:
        fig,ax=plt.subplots(figsize=(8,5))
        for comp in (1,2,3):
            m=np.asarray(eigs['component'])==comp
            ax.plot(t[m],arr(eigs,quantity)[m],marker='o',label=f'lambda{comp}')
        if quantity=='eigenvalue_kms2_kpc2': ax.axhline(0,lw=.7)
        ax.set_xlabel('Time from present [Myr]'); ax.set_ylabel(ylabel); ax.legend(); fig.tight_layout()
        for ext in ('png','pdf'):
            p=FIGURES/f'{name}.{ext}'; fig.savefig(p,dpi=180 if ext=='png' else None); outputs.append(p)
        plt.close(fig)
    # Present-day local direction diagram in a simple 2D projection of global XY.
    now=eigs[np.isclose(arr(eigs,'time_myr'),0.)]
    fig,ax=plt.subplots(figsize=(7,7))
    origin=np.zeros(2)
    ax.quiver(*origin,fixed_axis[0],fixed_axis[1],angles='xy',scale_units='xy',scale=1,label='L+T major axis')
    for row in now:
        ax.quiver(*origin,float(row['e_x']),float(row['e_y']),angles='xy',scale_units='xy',scale=1,label=f"tidal e{int(row['component'])}")
    ax.set_xlim(-1.1,1.1);ax.set_ylim(-1.1,1.1);ax.set_aspect('equal');ax.set_xlabel('Galactocentric X component');ax.set_ylabel('Galactocentric Y component');ax.legend();fig.tight_layout()
    for ext in ('png','pdf'):
        p=FIGURES/f'present_tidal_directions.{ext}'; fig.savefig(p,dpi=180 if ext=='png' else None); outputs.append(p)
    plt.close(fig)
    return outputs


def report(reference, summary, eigs, audit, outputs):
    now=eigs[np.isclose(arr(eigs,'time_myr'),0.)]
    lines=['Project 04C — Local Galactic gravitational tidal field','='*67,'',
           'Scope: fixed Stock 2 reference and its Project 04B reference orbit only.',
           'Potential: galpy MWPotential2014, identical baseline to Project 04B.',
           f'galpy version: {getattr(galpy,"__version__","unknown")}',
           'Tensor definition: T_ij = partial a_i/partial x_j = -partial_i partial_j Phi.',
           'This is the inertial gravitational tidal tensor, NOT the rotating-frame effective tidal tensor.','',
           'Present-day tidal eigenmodes (eigenvalues descending):']
    for row in now:
        lines.append(f"  lambda{int(row['component'])}: {float(row['eigenvalue_kms2_kpc2']):.6f} (km/s)^2/kpc^2 [{row['sign_class']}]; "
                     f"axis alignment with L+T = {float(row['angle_to_fixed_axis_deg']):.6f} deg; "
                     f"radial/azimuthal/vertical = {float(row['angle_to_radial_deg']):.3f}/{float(row['angle_to_azimuthal_deg']):.3f}/{float(row['angle_to_vertical_deg']):.3f} deg")
    lines += ['',f"Strongest-tidal-axis alignment with fixed L+T: {float(summary['strongest_tidal_axis_alignment_deg'][0]):.6f} deg",
              f"Number of positive (stretching) eigenvalues now: {int(summary['n_positive_eigenvalues'][0])}",
              f"Tidal trace now: {float(summary['trace_tidal_kms2_kpc2'][0]):.6f} (km/s)^2/kpc^2",
              '', 'Independent derivative audit:',
              f'  analytic Phi_RR / Phi_Rz: {audit[0][0]:.9f}, {audit[0][1]:.9f}',
              f'  finite-difference values: {audit[1][0]:.9f}, {audit[1][1]:.9f}',
              f'  maximum relative discrepancy: {float(np.max(audit[2])):.3e}',
              '', 'Evolution along the fixed Project 04B orbit:']
    for tm in SAMPLE_TIMES_MYR:
        rr=eigs[np.isclose(arr(eigs,'time_myr'),tm)]
        strongest=rr[np.argmax(arr(rr,'eigenvalue_kms2_kpc2'))]
        lines.append(f"  {tm:+.0f} Myr: strongest-axis/L+T = {float(strongest['angle_to_fixed_axis_deg']):.6f} deg; "
                     f"lambda_max = {float(strongest['eigenvalue_kms2_kpc2']):.6f} (km/s)^2/kpc^2")
    lines += ['', 'Interpretation limits:',
              '- Positive eigenvalue means differential gravitational stretching; negative means compression.',
              '- This calculation does not include centrifugal/Coriolis terms of a cluster-comoving rotating frame.',
              '- Alignment or non-alignment does not by itself establish or reject tidal-tail origin.',
              '- MWPotential2014 is one baseline potential; potential robustness remains deferred.',
              '- No Jacobi radius, escape criterion, bound/unbound label, membership update, or individual-star orbit inference is performed.',
              '', 'Invariants:',
              'fixed Project 04A reference used unchanged: True',
              'fixed Project 03D L+T axis used through Project 04A e1_ext_gc: True',
              'Project 04B reference orbit used unchanged: True',
              'same MWPotential2014 baseline as Project 04B: True',
              'no canonical source table modified: True',
              'no membership inference: True',
              'no source-by-source orbit integration: True',
              'no effective rotating-frame tide/Jacobi calculation: True',
              'PROJECT 04C STATUS: PASS','', 'Outputs:']
    lines += [str(p.relative_to(ROOT)) for p in outputs]
    return '\n'.join(lines)+'\n'


def main():
    reference, orbit, orbit_summary = load_inputs()
    fixed_axis = fixed_axis_gc(reference)
    protected=[]
    for p in list((ROOT/'data/interim/project03').glob('*')) + list((ROOT/'data/interim/project04').glob('*')):
        if p.is_file(): protected.append((p,hashlib.sha256(p.read_bytes()).hexdigest()))

    state0 = interpolate_state(orbit,0.)
    R0=np.hypot(state0[0],state0[1]); z0=state0[2]
    audit=finite_difference_audit(R0,z0)
    all_rows=[]; current_T=None
    for tm in SAMPLE_TIMES_MYR:
        state=interpolate_state(orbit,float(tm))
        rows,T,_=one_epoch(float(tm),state,fixed_axis)
        all_rows.extend(rows)
        if tm==0: current_T=T
    validate(reference,orbit,current_T,all_rows,audit,fixed_axis)
    eigs=eigen_table(all_rows)
    tensor=current_tensor_table(current_T,[r for r in all_rows if r['time_myr']==0],audit)
    summary=summary_table(reference,current_T,all_rows,audit)

    OUTPUT.mkdir(parents=True,exist_ok=True);FIGURES.mkdir(parents=True,exist_ok=True)
    outputs=[]
    for name,table in [('stock2_local_tidal_tensor',tensor),('stock2_tidal_eigensystem',eigs),('stock2_tidal_field_summary',summary)]:
        p=OUTPUT/f'{name}.ecsv';table.write(p,format='ascii.ecsv',overwrite=True);outputs.append(p)
    outputs += make_plots(eigs,fixed_axis)

    for p,h in protected:
        if hashlib.sha256(p.read_bytes()).hexdigest()!=h:
            raise RuntimeError(f'Prior product changed: {p}')
    report_path=FIGURES/'local_tidal_field_report.txt'
    text=report(reference,summary,eigs,audit,outputs+[report_path]);report_path.write_text(text)
    print(text,end='')


if __name__=='__main__':
    main()
