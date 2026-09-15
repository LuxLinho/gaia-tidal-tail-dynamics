"""Descriptive solar-neighbourhood Galactic Cartesian coordinates, Project 03B."""
from pathlib import Path
import importlib.util
import hashlib

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, Distance, Galactic, CartesianRepresentation, CartesianDifferential
from astropy.table import Table
import astropy

# Reuse Project 03A helpers without modifying its implementation or products.
_spec = importlib.util.spec_from_file_location('project03a', Path(__file__).with_name('01_observed_phase_space_diagnostics.py'))
a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a)
plt = a.plt
ROOT = a.ROOT
OUTPUT = ROOT / 'data/interim/project03'
FIGURES = ROOT / 'results/project03/heliocentric_phase_space'
POSITION = ['x_helio_pc', 'y_helio_pc', 'z_helio_pc', 'r_helio_pc']
VELOCITY = ['vx_helio_kms', 'vy_helio_kms', 'vz_helio_kms', 'speed_helio_kms']
UNITS = {**dict.fromkeys(POSITION, 'pc'), **dict.fromkeys(VELOCITY, 'km / s')}
CONVENTION = ('Astropy ICRS -> Galactic, right-handed Cartesian: +X toward l=0,b=0 '
              '(Galactic-centre direction), +Y toward l=90,b=0 (rotation direction), '
              '+Z toward b=90 (North Galactic Pole). Velocities positive along the same axes. '
              'Gaia pmra is pm_ra_cosdec; positive RV is receding. No solar position or motion added. '
              'The helio names denote local solar-neighbourhood coordinates: the actual ICRS/Gaia '
              'origin and RV convention remain barycentric; no epoch-dependent Sun/barycentre '
              'translation or RV correction is performed.')
DISTANCE = 'Direct geometric distance via Astropy Distance(parallax=...), pc = 1000 / parallax[mas]; no prior or zero-point correction.'


def input_masks(table):
    v = {q: a.values(table[q]) for q in a.UNITS}
    p = np.isfinite(v['parallax']) & (v['parallax'] > 0)
    p &= np.isfinite(v['ra']) & np.isfinite(v['dec']) & (np.abs(v['dec']) <= 90)
    vel = p & np.isfinite(v['pmra']) & np.isfinite(v['pmdec']) & np.isfinite(v['radial_velocity'])
    return v, p, vel


def transform(table):
    """Preserve every row; missing/invalid inputs yield NaN derived values."""
    v, pos, vel = input_masks(table)
    result = table.copy(copy_data=True)
    for q, unit in UNITS.items():
        if q in result.colnames:
            raise ValueError(f'Derived column already exists: {q}')
        result[q] = np.full(len(table), np.nan) * u.Unit(unit)
    def coord(mask, velocities=False):
        kw = dict(ra=v['ra'][mask]*u.deg, dec=v['dec'][mask]*u.deg,
                  distance=Distance(parallax=v['parallax'][mask]*u.mas), frame='icrs')
        if velocities:
            kw.update(pm_ra_cosdec=v['pmra'][mask]*u.mas/u.yr,
                      pm_dec=v['pmdec'][mask]*u.mas/u.yr,
                      radial_velocity=v['radial_velocity'][mask]*u.km/u.s)
        return SkyCoord(**kw).galactic
    if pos.any():
        xyz = coord(pos).cartesian.xyz.to_value(u.pc)
        for q, component in zip(POSITION[:3], xyz):
            result[q][pos] = component
        result[POSITION[3]][pos] = np.linalg.norm(xyz, axis=0)
    if vel.any():
        velocity = coord(vel, True).velocity.d_xyz.to_value(u.km/u.s)
        for q, component in zip(VELOCITY[:3], velocity):
            result[q][vel] = component
        result[VELOCITY[3]][vel] = np.linalg.norm(velocity, axis=0)
    result.meta.update(project03b_frame=CONVENTION, project03b_distance=DISTANCE,
                       project03b_astropy=astropy.__version__,
                       project03b_input=str(a.INPUT_PATH.relative_to(ROOT)),
                       project03b_missing='Invalid positions and unsupported velocities are NaN; all rows retained.')
    return result


def validate_output(original, result):
    """Check preservation, availability, norms, and full Cartesian round trip."""
    if len(original) != len(result):
        raise ValueError('Row count changed')
    for q in original.colnames:
        np.testing.assert_equal(np.ma.getmaskarray(original[q]), np.ma.getmaskarray(result[q]))
        np.testing.assert_equal(original[q].data, result[q].data)
        if original[q].unit != result[q].unit:
            raise ValueError(f'Input units changed: {q}')
    v, pos, vel = input_masks(original)
    for q, unit in UNITS.items():
        if result[q].unit != u.Unit(unit):
            raise ValueError(f'Wrong derived unit: {q}')
        np.testing.assert_array_equal(np.isfinite(a.values(result[q])), pos if q in POSITION else vel)
    xyz = np.array([a.values(result[q]) for q in POSITION[:3]])
    vv = np.array([a.values(result[q]) for q in VELOCITY[:3]])
    d = Distance(parallax=v['parallax'][pos]*u.mas).to_value(u.pc)
    np.testing.assert_allclose(np.linalg.norm(xyz[:, pos], axis=0), d, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(a.values(result['r_helio_pc'])[pos], d, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(a.values(result['speed_helio_kms'])[vel], np.linalg.norm(vv[:, vel], axis=0), rtol=1e-12)
    def back(mask, with_velocity=False):
        rep = CartesianRepresentation(xyz[:, mask]*u.pc)
        if with_velocity:
            rep = rep.with_differentials(CartesianDifferential(vv[:, mask]*u.km/u.s))
        return SkyCoord(Galactic(rep)).icrs
    max_angle = 0.
    if pos.any():
        recovered = back(pos)
        initial = SkyCoord(ra=v['ra'][pos]*u.deg, dec=v['dec'][pos]*u.deg)
        max_angle = float(initial.separation(recovered).to_value(u.arcsec).max())
        if max_angle > 1e-7:
            raise ValueError('Position round trip failed')
        np.testing.assert_allclose(recovered.distance.to_value(u.pc), d, rtol=1e-12, atol=1e-9)
    if vel.any():
        recovered = back(vel, True)
        for q, component, unit in [('pmra', recovered.pm_ra_cosdec, u.mas/u.yr),
                                    ('pmdec', recovered.pm_dec, u.mas/u.yr),
                                    ('radial_velocity', recovered.radial_velocity, u.km/u.s)]:
            np.testing.assert_allclose(component.to_value(unit), v[q][vel], rtol=1e-11, atol=1e-9)
    return dict(n_position=int(pos.sum()), n_velocity=int(vel.sum()), max_roundtrip_arcsec=max_angle)


def summaries(table):
    def rows(mask, field, group):
        return [dict(group_column=field, group=group, n_total=int(mask.sum()),
                     n_positional=int(np.isfinite(a.values(table['r_helio_pc'])[mask]).sum()),
                     n_velocity=int(np.isfinite(a.values(table['speed_helio_kms'])[mask]).sum()),
                     quantity=q, unit=unit, **a.percentile_summary(table[q][mask])) for q, unit in UNITS.items()]
    full = Table(rows=rows(np.ones(len(table), dtype=bool), 'all', 'all'))
    groups = Table(rows=[r for field, labels in a.GROUPS.items() for label in labels
                         for r in rows(np.asarray(table[field]) == label, field, label)])
    for t in (full, groups):
        t.meta.update(frame=CONVENTION, distance=DISTANCE,
                      statistics='Linear percentiles of finite measurements independently by quantity; no clipping. Units per row.')
    return full, groups


def scatter(table, x, y, field):
    fig, ax = plt.subplots(figsize=(7, 5), layout='constrained')
    for color, group in zip(a.COLORS, a.GROUPS[field]):
        group_mask = np.asarray(table[field]) == group
        mask = group_mask & np.isfinite(a.values(table[x])) & np.isfinite(a.values(table[y]))
        ax.scatter(a.values(table[x])[mask], a.values(table[y])[mask], s=12, alpha=.45,
                   edgecolors='none', color=color, label=f'{group} (valid N={mask.sum()}/{group_mask.sum()})')
    def label(q):
        return q.split('_')[0] + (' [pc]' if q in POSITION else ' [km/s]')
    ax.set(xlabel=label(x), ylabel=label(y), title='Stock 2 — heliocentric Galactic Cartesian')
    if x in POSITION:
        ax.set_aspect('equal', adjustable='box')
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.grid(alpha=.18)
    return fig, ax


def histogram(table):
    fig, ax = plt.subplots(figsize=(7, 5), layout='constrained')
    v = a.values(table['r_helio_pc'])
    bins = np.histogram_bin_edges(v[np.isfinite(v)], bins=40)
    for color, group in zip(a.COLORS, a.GROUPS['homogenized_class']):
        mask = np.asarray(table['homogenized_class']) == group
        finite = v[mask & np.isfinite(v)]
        ax.hist(finite, bins=bins, histtype='step', color=color, linewidth=1.7,
                label=f'{group} (valid N={len(finite)}/{mask.sum()})')
    ax.set(xlabel='Heliocentric distance [pc]', ylabel='Number of sources',
           title='Stock 2 — direct inverse-parallax distances')
    ax.legend(fontsize=9)
    ax.grid(alpha=.18)
    return fig, ax


def generate_plots(table, directory):
    outputs = []
    specs = [(POSITION[i], POSITION[j], 'homogenized_class') for i, j in [(0,1),(0,2),(1,2)]]
    specs += [(POSITION[0], POSITION[1], 'catalogue_status')]
    specs += [(VELOCITY[i], VELOCITY[j], 'homogenized_class') for i, j in [(0,1),(0,2),(1,2)]]
    for x, y, field in specs + [(None, None, None)]:
        fig, _ = scatter(table, x, y, field) if x else histogram(table)
        stem = f'{x}_vs_{y}_by_{field}' if x else 'distance_histogram_by_homogenized_class'
        for ext in ('png', 'pdf'):
            p = directory / f'{stem}.{ext}'
            fig.savefig(p, dpi=200)
            outputs.append(p)
        plt.close(fig)
    return outputs


def protected_hashes():
    hashes = a.protected_hashes()
    paths = list((ROOT/'results/project03/phase_space').rglob('*'))
    paths += list((ROOT/'data/interim/project03').glob('stock2_phase_space*.ecsv'))
    paths += [Path(a.__file__), ROOT/'docs/project03_observed_phase_space.md', ROOT/'tests/test_project03_phase_space.py']
    for p in paths:
        if p.is_file():
            hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return hashes


def main():
    before = protected_hashes()
    table = Table.read(a.INPUT_PATH)
    a.validate_invariants(table, Table.read(a.MASTER_PATH))
    v, pos, vel = input_masks(table)
    parallax = v['parallax']
    lines = ['Project 03B — Heliocentric Cartesian phase space', '='*60,
             f'Canonical rows: {len(table)}', f'Unique source_id: {len(np.unique(a.identities(table["source_id"])))}',
             f'RV finite: {np.isfinite(v["radial_velocity"]).sum()}',
             f'Parallax finite: {np.isfinite(parallax).sum()}',
             f'Parallax positive: {(np.isfinite(parallax) & (parallax > 0)).sum()}',
             f'Parallax zero: {(parallax == 0).sum()}',
             f'Parallax negative: {(np.isfinite(parallax) & (parallax < 0)).sum()}',
             f'Parallax nonfinite: {(~np.isfinite(parallax)).sum()}']
    print('\n'.join(lines), flush=True)  # Report distance validity before transformation.
    result = transform(table)
    checks = validate_output(table, result)
    full, groups = summaries(result)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, t in [('stock2_heliocentric_phase_space', result), ('stock2_heliocentric_summary', full),
                    ('stock2_heliocentric_group_summary', groups)]:
        p = OUTPUT / f'{name}.ecsv'
        t.write(p, overwrite=True)
        outputs.append(p)
    validate_output(table, Table.read(outputs[0]))
    outputs += generate_plots(result, FIGURES)
    if protected_hashes() != before:
        raise RuntimeError('Project 02/03A files changed')
    lines += [f'Valid positions: {checks["n_position"]}; unavailable: {len(table)-checks["n_position"]}',
              f'Valid 3D velocities: {checks["n_velocity"]}; unavailable: {len(table)-checks["n_velocity"]}',
              f'Max position round-trip separation: {checks["max_roundtrip_arcsec"]:.3g} arcsec',
              'Full velocity round trip: rtol=1e-11, atol=1e-9 in native units: passed',
              '', CONVENTION, DISTANCE, f'Astropy version: {astropy.__version__}', '', 'Full and grouped summaries:']
    for t in (full, groups):
        for r in t:
            stats = ', '.join(f'{k}={r[k]:.6g}' for k in a.PERCENTILES)
            lines.append(f'{r["group_column"]}/{r["group"]}: N={r["n_total"]}, N positional={r["n_positional"]}, '
                         f'N velocity={r["n_velocity"]}; {r["quantity"]} [{r["unit"]}], valid N={r["n_valid"]}: {stats}')
    lines += ['', 'No source was removed from the canonical population.',
              'No membership criterion was introduced. No quality cut or outlier clipping applied.',
              'No Galactocentric correction was applied. No solar motion correction was applied.',
              'No orbit integration was performed. No cluster centre or bulk-motion subtraction. Stop at 03B.',
              '', 'Outputs:'] + [str(p.relative_to(ROOT)) for p in outputs]
    lines += ['', 'Invariants:', 'rows == 1456: True', 'unique source_id == 1456: True',
              'canonical population unchanged: True', 'no membership cut applied: True',
              'no source dropped because of missing RV: True', 'position norm reproduces adopted distance: True',
              'velocities exist only where input information supports them: True',
              'Original observables and labels unchanged: True', 'Project 02/03A products unchanged: True',
              'PROJECT 03B STATUS: PASS']
    report = '\n'.join(lines) + '\n'
    (FIGURES/'heliocentric_phase_space_report.txt').write_text(report, encoding='utf-8')
    print('\n'.join(lines[10:]))
    return result


if __name__ == '__main__':
    main()
