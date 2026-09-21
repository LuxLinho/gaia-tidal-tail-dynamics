"""05A: export the existing 03C/04A reference; never refit membership."""
from astropy.coordinates import SkyCoord, Galactic, CartesianRepresentation, CartesianDifferential
from common import *


def build_reference():
    counts = population_invariants()
    stored, gc = read(REF03), read(REF04)
    if len(stored) != 1 or len(gc) != 1:
        raise ValueError('Expected one upstream reference row')
    frame, config = p04a.load_frame()
    if config != gc.meta['adopted_parameters'] or sha(REF03) != gc.meta['reference_sha256']:
        raise ValueError('04A frame or 03C reference provenance mismatch')
    # Observable quantities describe the existing Cartesian reference, not medians
    # of observables. Only this one fixed state is back-transformed.
    xyz = u.Quantity([stored[q].quantity[0] for q in ('x_ref_pc', 'y_ref_pc', 'z_ref_pc')])
    vel = u.Quantity([stored[q].quantity[0] for q in ('vx_ref_kms', 'vy_ref_kms', 'vz_ref_kms')])
    sky = SkyCoord(Galactic(CartesianRepresentation(xyz, differentials=CartesianDifferential(vel)))).icrs
    ref = gc[POSITION + VELOCITY + ['R_gc_kpc', 'phi_gc_deg', 'v_R_gc_kms', 'v_phi_gc_kms', 'v_gc_kms']].copy()
    for key, quantity in [('ra', sky.ra), ('dec', sky.dec), ('distance', sky.distance.to(u.pc)),
                          ('pmra', sky.pm_ra_cosdec), ('pmdec', sky.pm_dec), ('radial_velocity', sky.radial_velocity)]:
        ref[key] = [quantity.value] * quantity.unit
    for key in ('n_c_total', 'n_position_contributors', 'n_velocity_contributors', 'reference_population', 'reference_statistic'):
        ref[key] = stored[key]
    # Independent one-state round-trip verifies that observable export describes
    # the exact inherited Galactocentric state, without recomputing the catalogue.
    back = sky.transform_to(frame)
    for fields, vector, unit in [(POSITION, back.cartesian.xyz, u.kpc), (VELOCITY, back.velocity.d_xyz, u.km/u.s)]:
        np.testing.assert_allclose([ref[q][0] for q in fields], vector.to_value(unit), rtol=0, atol=1e-9)
    ref.meta.update(project='05A', reference_definition=stored.meta['reference_definition'],
                    observable_definition='ICRS back-transform of fixed 03C Cartesian reference; pmra includes cos(dec).',
                    present_epoch='t=0 is the inherited Gaia DR3 reference state, not propagated to execution date',
                    inputs=provenance([REF03, REF04, POPULATION, p04a.CONFIG]), software=VERSIONS,
                    invariants=counts, adopted_parameters=config)
    validate(ref)
    return ref


def validate(ref):
    if len(ref) != 1 or not ref.meta.get('reference_definition'):
        raise ValueError('Missing single-reference provenance')
    for q in POSITION + VELOCITY + ['ra', 'dec', 'distance', 'pmra', 'pmdec', 'radial_velocity']:
        if ref[q].unit is None or not np.all(np.isfinite(array(ref, q))):
            raise ValueError(f'Invalid or unitless reference quantity: {q}')
    if ref['distance'][0] <= 0:
        raise ValueError('Reference distance must be positive')
    if ref.meta['adopted_parameters'] != p04a.load_frame()[1]:
        raise ValueError('Frozen coordinate frame mismatch')


def main():
    before = protected_hashes()
    ref = save(build_reference(), REFERENCE)
    validate(ref)
    check_protected(before)
    write_json(RESULTS / 'cluster_reference_metadata.json', ref.meta)
    lines = ['Project 05A — Cluster Reference State', ref.meta['reference_definition'],
             'Position contributors: 940; velocity contributors: 594; no new selection.']
    lines += [f'{q}: {ref[q][0]} {ref[q].unit or ""}' for q in ref.colnames]
    lines += ['Canonical population: 1456 unchanged; validation: PASS']
    (RESULTS / 'cluster_reference_summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
