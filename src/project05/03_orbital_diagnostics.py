"""05C: cylindrical-radius diagnostics and figures for one reference orbit."""
from scipy.signal import find_peaks
from common import *

FIGURES = RESULTS / 'orbital_diagnostics'
FIGURE_NAMES = ['cluster_reference_orbit_xy', 'cluster_reference_orbit_RZ',
                'cluster_reference_orbit_R_time', 'cluster_reference_orbit_Z_time']


def period(t, signal):
    """Median same-phase peak spacing; require at least two complete cycles."""
    peaks, _ = find_peaks(signal)
    if len(peaks) < 3:
        return None, int(len(peaks))
    return float(np.median(np.diff(t[peaks]))), int(len(peaks))


def diagnostics(orbit):
    t = array(orbit, 'time_myr')
    i = np.flatnonzero(t == 0)[0]
    R, z = array(orbit, 'R_gc_kpc'), array(orbit, 'z_gc_kpc')
    peri, apo = float(R.min()), float(R.max())
    radial, nrad = period(t, -R)
    vertical, nz = period(t, z)
    phi = np.unwrap(np.radians(array(orbit, 'phi_gc_deg')))
    turns = abs(phi[-1] - phi[0]) / (2 * np.pi)
    az = float((t[-1] - t[0]) / turns) if turns >= 2 else None
    return dict(R_now_kpc=float(R[i]), z_now_kpc=float(z[i]),
                speed_now_kms=float(array(orbit, 'v_gc_kms')[i]),
                pericentre_kpc=peri, apocentre_kpc=apo, eccentricity=(apo-peri)/(apo+peri),
                z_max_kpc=float(np.max(np.abs(z))), z_min_kpc=float(z.min()), z_peak_kpc=float(z.max()),
                Lz_now_kpc_kms=float(array(orbit, 'Lz_gc_kpc_kms')[i]),
                radial_period_myr=radial, vertical_period_myr=vertical, azimuthal_period_myr=az,
                radial_minima_count=nrad, vertical_maxima_count=nz,
                azimuthal_turns=float(turns))


def validate(d):
    if not (0 < d['pericentre_kpc'] < d['apocentre_kpc'] and 0 <= d['eccentricity'] < 1):
        raise ValueError('Invalid radial diagnostics')
    if d['z_max_kpc'] < abs(d['z_now_kpc']):
        raise ValueError('Invalid vertical envelope')


def make_plots(orbit, d):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    t = array(orbit, 'time_myr'); i = np.flatnonzero(t == 0)[0]
    x, y = array(orbit, 'x_gc_kpc'), array(orbit, 'y_gc_kpc')
    R, z = array(orbit, 'R_gc_kpc'), array(orbit, 'z_gc_kpc') * 1000
    with plt.rc_context({'font.size': 11, 'axes.grid': True, 'grid.alpha': .18, 'savefig.dpi': 220}):
        for name, a, b, xlabel, ylabel in [
            (FIGURE_NAMES[0], x, y, 'X [kpc]', 'Y [kpc]'),
            (FIGURE_NAMES[1], R, z, 'Cylindrical R [kpc]', 'Z [pc]'),
            (FIGURE_NAMES[2], t, R, 'Time from present [Myr]', 'Cylindrical R [kpc]'),
            (FIGURE_NAMES[3], t, z, 'Time from present [Myr]', 'Z [pc]')]:
            fig, ax = plt.subplots(figsize=(7.5, 6 if name.endswith('xy') else 4.8), layout='constrained')
            ax.plot(a[:i+1], b[:i+1], color='#4c72b0', lw=1.2, label='Past')
            ax.plot(a[i:], b[i:], color='#dd8452', lw=1.2, label='Future')
            ax.scatter(a[i], b[i], color='black', s=32, zorder=5, label='Stock 2 now')
            if name.endswith('xy'):
                ax.scatter(0, 0, marker='*', s=100, color='#555555', label='Galactic centre')
                ax.set_aspect('equal', adjustable='box')
            if name.endswith('time'):
                ax.axvline(0, color='grey', lw=.7)
            if name.endswith('R_time'):
                for value in (d['pericentre_kpc'], d['apocentre_kpc']):
                    ax.axhline(value, color='grey', ls=':', lw=.8)
            if name.endswith('Z_time') or name.endswith('RZ'):
                ax.axhline(0, color='grey', lw=.7)
            ax.set(xlabel=xlabel, ylabel=ylabel, title='Stock 2 cluster reference orbit')
            ax.legend(loc='upper center', bbox_to_anchor=(.5, -.17), ncol=2, frameon=False)
            for ext in ('png', 'pdf'):
                fig.savefig(FIGURES / f'{name}.{ext}')
            plt.close(fig)


def main():
    before = protected_hashes()
    counts = population_invariants()
    reference, orbit = read(REFERENCE), read(ORBIT)
    b = module('project05b', 'src/project05/02_galactic_orbit.py')
    if orbit.meta['inputs'][str(REFERENCE.relative_to(ROOT))] != sha(REFERENCE):
        raise ValueError('Orbit was generated with a different reference file')
    checks = b.validate(reference, orbit)
    d = diagnostics(orbit); validate(d)
    # Resolution audit in the SAME potential, not a potential-robustness study.
    fine = b.integrate(reference, step=b.STEP_MYR / 2)
    b.validate(reference, fine)
    refined = diagnostics(fine)
    delta = {key: abs(d[key] - refined[key]) for key in ('pericentre_kpc', 'apocentre_kpc', 'z_max_kpc', 'eccentricity')}
    if any(value > 1e-5 for value in delta.values()):
        raise ValueError(f'Sampled extrema not resolved: {delta}')
    checks['resolution_audit'] = dict(step_myr=b.STEP_MYR / 2, absolute_differences=delta,
                                     tolerance=1e-5, status='PASS')
    meta = dict(project='05C', radial_definition='Cylindrical R=hypot(X,Y); finite-window sampled extrema',
                eccentricity_definition='(R_apo-R_peri)/(R_apo+R_peri)',
                period_definition='Radial: median successive R-minimum spacing; vertical: median successive Z-maximum spacing; >=3 extrema required. Azimuthal: window duration / unwrapped turns, >=2 turns required. Unavailable values are null.',
                Lz_convention='X*Vy-Y*Vx in inherited Astropy axes; prograde solar rotation has negative Lz',
                inputs=provenance([REFERENCE, ORBIT]), reference=reference.meta,
                integration=orbit.meta['integration'], potential=orbit.meta['potential'],
                software=VERSIONS, validation=checks, invariants=counts)
    summary = Table()
    for key, value in d.items():
        unit = u.kpc*u.km/u.s if key.endswith('kpc_kms') else u.kpc if key.endswith('_kpc') else u.km/u.s if key.endswith('_kms') else u.Myr if key.endswith('_myr') else u.dimensionless_unscaled
        summary[key] = [np.nan if value is None else value] * unit
    summary.meta = meta
    save(summary, DATA / 'stock2_orbital_diagnostics.ecsv')
    write_json(RESULTS / 'orbital_diagnostics.json', {'diagnostics': d, 'metadata': meta})
    make_plots(orbit, d)
    check_protected(before)
    write_json(RESULTS / 'upstream_integrity.json', {'status': 'PASS', 'sha256': before})
    lines = ['Project 05 — Stock 2 Galactic Orbit', '=' * 55,
             reference.meta['reference_definition'],
             'N position = 940; N velocity = 594 (existing C population).',
             'Potential: galpy MWPotential2014, ro=8 kpc, vo=220 km/s; unchanged from 04B/04C.',
             f'Software: {VERSIONS}', f'Integration: {orbit.meta["integration"]}',
             'Frame: frozen Project 04A parameters; see cluster_reference_metadata.json.',
             'Present-day Galactocentric state:']
    lines += [f'  {q}: {reference[q][0]:.12g} {reference[q].unit}' for q in POSITION + VELOCITY]
    lines += ['', meta['radial_definition'], meta['eccentricity_definition'], meta['Lz_convention'], meta['period_definition']]
    lines += [f'{key}: {value}' for key, value in d.items()]
    lines += [f'Validation: {checks}', 'Canonical population invariant: 1456 unchanged; full 6D subset: 886.',
              f'All {len(before)} upstream files unchanged by SHA256: PASS.',
              'PROJECT 05 STATUS: PASS', '',
              'Interpretation: in this baseline model the reference follows a mildly eccentric disk orbit,',
              'remaining within about 0.12 kpc of the Galactic plane over the sampled window.',
              'This deterministic reference has separate position/velocity contributor sets; it is not a fitted centre of mass.',
              'No uncertainty propagation, candidate classification, membership cuts, potential robustness,',
              'or evidence of tidal disruption or confirmed tails is claimed. Sampled extrema are window dependent.']
    (RESULTS / 'project05_summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
