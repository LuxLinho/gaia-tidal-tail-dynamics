"""05B: two-sided single-reference integration in the established baseline."""
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
from common import *

SPAN_MYR = 500.0
STEP_MYR = 0.25
RTOL = ATOL = 1e-11
CONFIG = dict(method='dop853', start_myr=-SPAN_MYR, stop_myr=SPAN_MYR,
              output_step_myr=STEP_MYR, samples=4001, rtol=RTOL, atol=ATOL,
              internal_step='adaptive; output cadence is not internal solver step',
              initialization='independent past and future integrations each start from supplied t=0 state')


def integrate(reference, step=STEP_MYR):
    n = int(round(SPAN_MYR / step))
    if step <= 0 or not np.isclose(n * step, SPAN_MYR):
        raise ValueError('Step must divide the half-window')
    halves = []
    for sign in (-1, 1):
        times = sign * np.linspace(0, SPAN_MYR, n + 1) * u.Myr
        orbit = Orbit(p04b.astropy_to_galpy_cyl(reference), ro=p04b.RO, vo=p04b.VO)
        orbit.integrate(times, MWPotential2014, method=CONFIG['method'], rtol=RTOL, atol=ATOL, progressbar=False)
        raw = {key: np.asarray(getattr(orbit, method)(times, use_physical=True), float)
               for key, method in [('R', 'R'), ('phi', 'phi'), ('z', 'z'), ('vR', 'vR'), ('vT', 'vT'), ('vz', 'vz')]}
        raw['E'] = np.asarray(orbit.E(times, pot=MWPotential2014, use_physical=True), float)
        halves.append((times.to_value(u.Myr), raw))
    past, future = halves
    time = np.r_[past[0][:0:-1], future[0]]
    raw = {key: np.r_[past[1][key][:0:-1], future[1][key]] for key in past[1]}
    table = p04b.trajectory_table(time, raw)
    table.meta = dict(project='05B', potential={**p04b.POTENTIAL_META, 'scope': 'Project 05 baseline; same model as 04B/04C', 'non_default_parameters': {}},
                      integration={**CONFIG, 'output_step_myr': step, 'samples': len(time)},
                      software=VERSIONS, adopted_parameters=reference.meta['adopted_parameters'],
                      reference_definition=reference.meta['reference_definition'],
                      coordinate_bridge='x_galpy=-x_astropy; y,z unchanged; vx_galpy=-vx_astropy; vT_galpy=-vphi_astropy')
    table['Lz_gc_kpc_kms'] = (array(table, 'x_gc_kpc') * array(table, 'vy_gc_kms') -
                              array(table, 'y_gc_kpc') * array(table, 'vx_gc_kms')) * u.kpc * u.km / u.s
    return table


def validate(reference, trajectory):
    t = array(trajectory, 'time_myr', u.Myr)
    if np.count_nonzero(t == 0) != 1 or not np.all(np.diff(t) > 0):
        raise ValueError('Invalid time grid')
    np.testing.assert_allclose(t[[0, -1]], [-SPAN_MYR, SPAN_MYR], atol=1e-12, rtol=0)
    np.testing.assert_allclose(np.diff(t), trajectory.meta['integration']['output_step_myr'], atol=1e-10, rtol=0)
    for q in trajectory.colnames:
        if trajectory[q].unit is None or not np.all(np.isfinite(array(trajectory, q))):
            raise ValueError(f'Invalid or unitless trajectory: {q}')
    i = np.flatnonzero(t == 0)[0]
    for q in POSITION + VELOCITY:
        np.testing.assert_allclose(array(trajectory, q)[i], array(reference, q)[0], rtol=0, atol=1e-9)
    # A trapezoidal position increment must agree with the stored velocities;
    # this detects jumps and sign/time reversal errors at the two-sided join.
    pos = np.column_stack([array(trajectory, q, u.kpc) for q in POSITION])
    vel = np.column_stack([array(trajectory, q, u.kpc/u.Myr) for q in VELOCITY])
    error = np.linalg.norm(np.diff(pos, axis=0) - (vel[1:] + vel[:-1]) / 2 * np.diff(t)[:, None], axis=1)
    continuity = float(error.max())
    if continuity > 1e-5:
        raise ValueError('Trajectory/velocity continuity failed')
    energy = array(trajectory, 'energy_galpy_kms2')
    energy_error = float(np.max(np.abs(energy - energy[i])) / abs(energy[i]))
    lz = array(trajectory, 'Lz_gc_kpc_kms')
    lz_error = float(np.max(np.abs(lz - lz[i])) / abs(lz[i]))
    if energy_error > 1e-7 or lz_error > 1e-7:
        raise ValueError('Conservation tolerance failed')
    return dict(status='PASS', max_relative_energy_drift=energy_error,
                max_relative_Lz_drift=lz_error, max_position_increment_residual_kpc=continuity)


def main():
    before = protected_hashes()
    reference = read(REFERENCE)
    module('project05a', 'src/project05/01_cluster_reference.py').validate(reference)
    trajectory = integrate(reference)
    trajectory.meta['inputs'] = provenance([REFERENCE, REF04, ROOT / 'src/project04/02_reference_orbit.py'])
    trajectory.meta['validation'] = validate(reference, trajectory)
    trajectory = save(trajectory, ORBIT)
    validate(reference, trajectory)
    check_protected(before)
    write_json(RESULTS / 'orbit_integration_metadata.json', trajectory.meta)
    print(trajectory.meta['validation'])


if __name__ == '__main__':
    main()
