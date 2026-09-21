"""Focused Project 05 reference-construction audit; not a new main stage."""
import json
import numpy as np
from astropy import units as u
import common as c

OUT = c.RESULTS / 'reference_sensitivity'
B = c.module('sensitivity05b', 'src/project05/02_galactic_orbit.py')
D = c.module('sensitivity05c', 'src/project05/03_orbital_diagnostics.py')
P = c.p04a.c
INPUT = P.INPUT
KEYS = ['R_now_kpc', 'z_now_kpc', 'speed_now_kms', 'pericentre_kpc',
        'apocentre_kpc', 'eccentricity', 'z_max_kpc', 'Lz_now_kpc_kms',
        'radial_period_myr', 'vertical_period_myr', 'azimuthal_period_myr']
CONVENTIONS = '''Astropy uses the inherited right-handed Galactocentric frame: Sun at negative X,
solar rotation near positive Y, Z north. The galpy Galactic convention used here
is left-handed relative to these physical axes: x_g=-x_A, y_g=y_A, z_g=z_A,
and vx_g=-vx_A, vy_g=vy_A, vz_g=vz_A. The matrix diag(-1,1,1) has determinant -1:
a reflection, not a proper rotation. phi_g=pi-phi_A modulo 2pi, vR_g=vR_A,
vT_g=-vphi_A. Numerically Lz_A=X_A*Vy_A-Y_A*Vx_A=R*vphi_A;
galpy's scalar Lz_g=R*vT_g=-Lz_A. Angular momentum is an axial vector,
so its components under reflection obey L_g=det(M)*M*L_A, not M*L_A.
Negative Astropy Lz is expected for this prograde orbit (X<0 and Vy>0).
Positive galpy vT, positive galpy Orbit.Lz(), and increasing galpy azimuth
confirm prograde rotation. No signs are changed to relabel the results.
This bridge is used with the existing axisymmetric potential; no new rotation,
solar-frame assumptions, or potential model is introduced.'''


def coherent_reference(table):
    xyz, pos = P.vectors(table, P.b.POSITION)
    vel, complete_vel = P.vectors(table, P.b.VELOCITY)
    mask = (np.asarray(table['homogenized_class']) == 'C') & pos & complete_vel
    if not mask.any():
        raise ValueError('No complete 6D C sources')
    # Compute medians in the SAME barycentric Galactic axes as 03C. Medians
    # are not rotation-equivariant; do not take medians after the 04A transform.
    stored = P.reference_frame(table[mask])
    for fields, vectors in [(P.REF_POSITION, xyz), (P.REF_VELOCITY, vel)]:
        np.testing.assert_allclose([stored[q][0] for q in fields], np.median(vectors[:, mask], axis=1), rtol=0, atol=0)
    frame, config = c.p04a.load_frame()
    ref = c.p04a.transform_reference(stored, frame)
    ref.meta.update(adopted_parameters=config,
                    reference_definition='Diagnostic only: existing C class AND complete finite 6D vectors; both component-wise medians from the same sources in 03B axes, then 04A transform.')
    return ref, stored, mask


def comparison(baseline, alternative, unit):
    if baseline is None or alternative is None:
        return dict(baseline=baseline, comparison=alternative, unit=unit,
                    signed_difference=None, absolute_difference=None, absolute_relative_difference=None)
    delta = float(alternative - baseline)
    return dict(baseline=float(baseline), comparison=float(alternative), unit=unit,
                signed_difference=delta, absolute_difference=abs(delta),
                absolute_relative_difference=abs(delta)/abs(baseline) if baseline != 0 else None)


def convention_audit(ref, orbit):
    matrix = np.diag([-1., 1., 1.])
    pos = np.array([ref[q][0] for q in c.POSITION])
    vel = np.array([ref[q][0] for q in c.VELOCITY])
    gp, gv = matrix @ pos, matrix @ vel
    np.testing.assert_allclose(np.cross(gp, gv), np.linalg.det(matrix)*matrix@np.cross(pos, vel), atol=1e-10, rtol=0)
    native = c.p04b.build_orbit(ref)
    np.testing.assert_allclose([native.x(), native.y(), native.z()], gp, atol=1e-10, rtol=0)
    np.testing.assert_allclose([native.vx(), native.vy(), native.vz()], gv, atol=1e-10, rtol=0)
    lz = float(np.cross(pos, vel)[2])
    np.testing.assert_allclose(native.Lz(), -lz, atol=1e-9, rtol=0)
    np.testing.assert_allclose(native.vT(), -ref['v_phi_gc_kms'][0], atol=1e-10, rtol=0)
    np.testing.assert_allclose(native.vR(), ref['v_R_gc_kms'][0], atol=1e-10, rtol=0)
    phi = np.unwrap(np.arctan2(c.array(orbit, 'y_gc_kpc'), -c.array(orbit, 'x_gc_kpc')))
    prograde = bool(native.Lz() > 0 and native.vT() > 0 and
                    np.all(-c.array(orbit, 'v_phi_gc_kms') > 0) and np.all(np.diff(phi) > 0))
    if not prograde:
        raise ValueError('Prograde convention audit failed')
    return dict(status='PASS', reflection_determinant=float(np.linalg.det(matrix)),
                Astropy_Lz_kpc_kms=lz, galpy_Lz_kpc_kms=float(native.Lz()),
                Astropy_vphi_kms=float(ref['v_phi_gc_kms'][0]), galpy_vT_kms=float(native.vT()),
                physically_prograde=prograde, explanation=CONVENTIONS)


def assess(rows, baseline):
    # Explicit conservative review gates, not inferred uncertainties or membership
    # criteria: <1% on orbit scales, |delta e|<0.001. Near-plane z uses z_max
    # as physical scale rather than dividing only by the small instantaneous z.
    scales = {key: abs(baseline[key]) if baseline[key] is not None else None for key in KEYS}
    scales['z_now_kpc'] = baseline['z_max_kpc']
    normalized = {key: rows[key]['absolute_difference']/scale
                  for key, scale in scales.items() if key != 'eccentricity' and rows[key]['absolute_difference'] is not None and scale is not None and scale > 0}
    unavailable = [key for key in KEYS if rows[key]['absolute_difference'] is None]
    retain = not unavailable and all(value < .01 for value in normalized.values()) and rows['eccentricity']['absolute_difference'] < .001
    return dict(decision='RETAIN_FROZEN_REFERENCE' if retain else 'STOP_FOR_SCIENTIFIC_REVIEW',
                criteria='All available physical-scale normalized changes <1%; |delta eccentricity|<0.001. These conservative review gates are not statistical error bars. Any unavailable required comparison triggers review.',
                physical_scale_normalized_differences=normalized, unavailable=unavailable,
                maximum_normalized_quantity=max(normalized, key=normalized.get),
                maximum_normalized_difference=max(normalized.values()))


def run():
    # Protect the full original population, baseline outputs, README and previous
    # source/results. Only this audit output directory is excluded.
    paths = [c.ROOT/'README.md']
    for folder in ('data', 'results', 'src', 'tests', 'docs'):
        paths += [p for p in (c.ROOT/folder).rglob('*') if p.is_file() and
                  '__pycache__' not in p.parts and not p.is_relative_to(OUT)]
    before = c.provenance(paths)
    c.population_invariants()
    baseline, stored_orbit = c.read(c.REFERENCE), c.read(c.ORBIT)
    if c.galpy.__version__ != '1.12.0' or stored_orbit.meta['software']['galpy'] != '1.12.0':
        raise ValueError('Audit requires the baseline galpy 1.12.0')
    if stored_orbit.meta['integration'] != B.CONFIG:
        raise ValueError('Baseline integration configuration drift')
    if c.p04b.RO.to_value(u.kpc) != 8 or c.p04b.VO.to_value(u.km/u.s) != 220:
        raise ValueError('Potential scale drift')
    table = c.read(INPUT)
    if len(table) != 1456 or set(table['gaia_dr3_source_id']) != set(c.read(c.POPULATION)['gaia_dr3_source_id']):
        raise ValueError('Upstream canonical identity mismatch')
    # Verify that the upstream data still reproduces the frozen 03C reference.
    rebuilt, frozen = P.reference_frame(table), c.read(c.REF03)
    for q in P.REF_POSITION + P.REF_VELOCITY:
        np.testing.assert_allclose(rebuilt[q], frozen[q], atol=0, rtol=0)
    alternative, barycentric, mask = coherent_reference(table)
    if int(mask.sum()) != 594:
        raise ValueError('Expected 594 matched C sources; stop for review')
    alternative_orbit = B.integrate(alternative)
    for key in ('integration', 'potential', 'adopted_parameters', 'software'):
        if alternative_orbit.meta[key] != stored_orbit.meta[key]:
            raise ValueError(f'Baseline/comparison mismatch: {key}')
    validations = {'baseline': B.validate(baseline, stored_orbit),
                   'comparison': B.validate(alternative, alternative_orbit)}
    base_d, alt_d = D.diagnostics(stored_orbit), D.diagnostics(alternative_orbit)
    D.validate(base_d); D.validate(alt_d)
    initial = {q: comparison(baseline[q][0], alternative[q][0], str(baseline[q].unit)) for q in c.POSITION+c.VELOCITY}
    diagnostics = {q: comparison(base_d[q], alt_d[q], 'kpc km/s' if q.endswith('kpc_kms') else 'kpc' if q.endswith('_kpc') else 'km/s' if q.endswith('_kms') else 'Myr' if q.endswith('_myr') else 'dimensionless') for q in KEYS}
    assessment = assess(diagnostics, base_d)
    pos_shift = float(np.linalg.norm([initial[q]['signed_difference'] for q in c.POSITION])*1000)
    result = dict(sample_n=int(mask.sum()), source_ids=[str(x) for x in table['gaia_dr3_source_id'][mask]],
                  selection=alternative.meta['reference_definition'],
                  median_frame='Project 03B barycentric Galactic Cartesian; before 04A frame rotation/translation',
                  inputs=c.provenance([INPUT, c.REF03, c.REF04, c.REFERENCE, c.ORBIT, c.p04a.CONFIG,
                                       c.ROOT/'src/project05/02_galactic_orbit.py', c.ROOT/'src/project05/03_orbital_diagnostics.py',
                                       c.ROOT/'src/project05/reference_sensitivity.py']),
                  configuration={q: stored_orbit.meta[q] for q in ('integration','potential','adopted_parameters','software')},
                  comparison_barycentric_reference={q: dict(value=float(barycentric[q][0]), unit=str(barycentric[q].unit)) for q in P.REF_POSITION+P.REF_VELOCITY},
                  initial_state=initial, position_shift_pc=pos_shift, orbital_diagnostics=diagnostics,
                  conventions={'baseline': convention_audit(baseline, stored_orbit), 'comparison': convention_audit(alternative, alternative_orbit)},
                  validation=validations, assessment=assessment,
                  caveats=['One diagnostic reference-construction test, not a membership test or uncertainty estimate.',
                           'Radial/vertical periods are sampled-extremum estimates: identical values do not imply exact identical frequencies; cadence is 0.25 Myr.',
                           'Relative difference is abs(comparison-baseline)/abs(baseline); zero baselines yield null.',
                           'Instantaneous z has a small denominator; assess its absolute shift relative to the vertical envelope as well.'])
    c.check_protected(before)
    result['preservation'] = dict(status='PASS', protected_sha256=before, canonical_sources=1456,
                                  frozen_reference_unchanged=True, README_unchanged=True)
    c.write_json(OUT/'reference_state_sensitivity.json', result)
    def fmt(value):
        return 'unavailable' if value is None else f'{value:.12g}'

    lines = ['Project 05 — focused reference-state sensitivity audit',
             f'Matched complete-6D C sample: {int(mask.sum())}; frozen counts remain 940/594.',
             alternative.meta['reference_definition'], f'Initial position displacement: {pos_shift:.9g} pc',
             'All velocities unchanged; all integration, potential and coordinate configuration matched.',
             '', 'quantity | baseline | comparison | signed delta | absolute relative delta | unit']
    for rows in (initial, diagnostics):
        for key, row in rows.items():
            lines.append(f"{key} | {fmt(row['baseline'])} | {fmt(row['comparison'])} | {fmt(row['signed_difference'])} | {fmt(row['absolute_relative_difference'])} | {row['unit']}")
        lines.append('')
    lines += [CONVENTIONS, json.dumps(result['conventions']['baseline'], indent=2),
              'Scientific assessment:', json.dumps(assessment, indent=2),
              'Changes are negligible for the mildly eccentric, near-plane prograde disk-orbit conclusions.' if assessment['decision']=='RETAIN_FROZEN_REFERENCE' else 'STOP: differences require scientific review; baseline was not changed.',
              *result['caveats'], f'Protected files unchanged: {len(before)}; canonical 1456 unchanged. No commit or push.']
    (OUT/'reference_state_sensitivity.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    if assessment['decision'] != 'RETAIN_FROZEN_REFERENCE':
        raise RuntimeError('STOP: reference sensitivity requires scientific review; see saved results')
    return result


if __name__ == '__main__':
    run()
