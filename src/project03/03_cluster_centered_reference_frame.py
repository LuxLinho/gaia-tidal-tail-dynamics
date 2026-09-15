"""Project 03C: a fixed component-median reference from the existing C class."""
from pathlib import Path
import hashlib
import importlib.util

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, Galactic, CartesianRepresentation
from astropy.table import Table

_spec = importlib.util.spec_from_file_location('project03b', Path(__file__).with_name('02_heliocentric_phase_space.py'))
b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b)
a = b.a
plt = a.plt
ROOT = a.ROOT
INPUT = ROOT / 'data/interim/project03/stock2_heliocentric_phase_space.ecsv'
OUTPUT = ROOT / 'data/interim/project03'
FIGURES = ROOT / 'results/project03/cluster_relative_phase_space'
POSITION = ['dx_cluster_pc', 'dy_cluster_pc', 'dz_cluster_pc', 'r_cluster_pc']
VELOCITY = ['dvx_cluster_kms', 'dvy_cluster_kms', 'dvz_cluster_kms', 'dv_cluster_kms']
DECOMPOSITION = ['v_radial_cluster_rel_kms', 'v_tangential_cluster_rel_kms']
UNITS = {**dict.fromkeys(POSITION, 'pc'), **dict.fromkeys(VELOCITY + DECOMPOSITION, 'km / s')}
REF_POSITION = ['x_ref_pc', 'y_ref_pc', 'z_ref_pc']
REF_VELOCITY = ['vx_ref_kms', 'vy_ref_kms', 'vz_ref_kms']
LABELS = dict(zip(POSITION + VELOCITY, [r'$\Delta X$ [pc]', r'$\Delta Y$ [pc]',
    r'$\Delta Z$ [pc]', r'$r_{\rm cluster}$ [pc]', r'$\Delta V_x$ [km/s]',
    r'$\Delta V_y$ [km/s]', r'$\Delta V_z$ [km/s]', r'$|\Delta V|$ [km/s]']))
CONVENTION = ('Project 03B Astropy Galactic Cartesian axes unchanged: right-handed, '
              '+X toward l=0,b=0, +Y toward l=90,b=0, +Z toward the North Galactic Pole. '
              'Project 03B *_helio_* fields have a solar-system barycentric origin. '
              'Cluster-relative components subtract the fixed C component medians without rotation. '
              'No solar-motion or Galactocentric correction.')
METHOD = ('Existing homogenized_class == C; component-wise median of all complete Cartesian '
          'position vectors and, separately, all complete Cartesian velocity vectors. '
          'No clipping, quality selection, iteration, or new membership inference.')


def vectors(table, columns):
    array = np.array([a.values(table[q]) for q in columns[:3]])
    return array, np.all(np.isfinite(array), axis=0)


def reference_frame(table):
    """Define position and velocity using separate complete-vector C contributors."""
    xyz, pos = vectors(table, b.POSITION)
    velocity, vel = vectors(table, b.VELOCITY)
    c = np.asarray(table['homogenized_class']) == 'C'
    if not (c & pos).any() or not (c & vel).any():
        raise ValueError('No valid C contributors for reference position or velocity')
    position_ref = np.median(xyz[:, c & pos], axis=1)
    velocity_ref = np.median(velocity[:, c & vel], axis=1)
    ref = Table(dict(reference_population=['C'], reference_statistic=['component-wise median'],
                     n_c_total=[int(c.sum())], n_position_contributors=[int((c & pos).sum())],
                     n_velocity_contributors=[int((c & vel).sum())]))
    for q, value in zip(REF_POSITION, position_ref):
        ref[q] = [value] * u.pc
    for q, value in zip(REF_VELOCITY, velocity_ref):
        ref[q] = [value] * u.km/u.s
    ref['r_ref_pc'] = [np.linalg.norm(position_ref)] * u.pc
    ref['speed_ref_kms'] = [np.linalg.norm(velocity_ref)] * u.km/u.s
    sky = SkyCoord(Galactic(CartesianRepresentation(position_ref*u.pc)))
    for q, value in [('l_ref_deg', sky.l.deg), ('b_ref_deg', sky.b.deg),
                     ('ra_ref_deg', sky.icrs.ra.deg), ('dec_ref_deg', sky.icrs.dec.deg)]:
        ref[q] = [value] * u.deg
    ref.meta.update(coordinate_convention=CONVENTION, reference_definition=METHOD,
                    input=str(INPUT.relative_to(ROOT)),
                    sky_reference='Descriptive back-transform of Cartesian median; not a sky-coordinate average.')
    return ref


def relative_phase_space(table, ref):
    """Translate the full table; never estimate unavailable Cartesian velocities."""
    result = table.copy(copy_data=True)
    for q, unit in UNITS.items():
        if q in result.colnames:
            raise ValueError(f'Derived column already exists: {q}')
        result[q] = np.full(len(table), np.nan) * u.Unit(unit)
    for original, derived, refs in [(b.POSITION, POSITION, REF_POSITION), (b.VELOCITY, VELOCITY, REF_VELOCITY)]:
        array, valid = vectors(table, original)
        offset = array[:, valid] - np.array([ref[q][0] for q in refs])[:, None]
        for q, component in zip(derived[:3], offset):
            result[q][valid] = component
        result[derived[3]][valid] = np.linalg.norm(offset, axis=0)
    displacement, pos = vectors(result, POSITION)
    dv, vel = vectors(result, VELOCITY)
    r = a.values(result['r_cluster_pc'])
    valid = pos & vel & (r > 0)
    direction = displacement[:, valid] / r[valid]
    radial = np.sum(dv[:, valid] * direction, axis=0)
    # Norm of the orthogonal residual is equivalent to sqrt(v^2-v_radial^2)
    # without cancellation for nearly radial motion.
    tangential = np.linalg.norm(dv[:, valid] - radial * direction, axis=0)
    result[DECOMPOSITION[0]][valid] = radial
    result[DECOMPOSITION[1]][valid] = tangential
    result.meta.update(project03c_frame=CONVENTION, project03c_reference=METHOD,
                       project03c_reference_product='stock2_reference_frame.ecsv',
                       project03c_decomposition='Signed outward radial velocity and nonnegative tangential speed relative to instantaneous cluster displacement; not Gaia radial_velocity. Both NaN at exact reference position or if either vector is missing.')
    for q in REF_POSITION + REF_VELOCITY:
        result.meta[q] = float(ref[q][0])
    return result


def validate_relative(original, result, ref):
    """Validate reference, exact row/column preservation, offsets, and availability."""
    if len(original) != len(result):
        raise ValueError('Row count changed')
    for q in original.colnames:
        np.testing.assert_equal(original[q].data, result[q].data)
        np.testing.assert_array_equal(np.ma.getmaskarray(original[q]), np.ma.getmaskarray(result[q]))
        if original[q].unit != result[q].unit:
            raise ValueError(f'Original units changed: {q}')
    expected_ref = reference_frame(original)
    for q in expected_ref.colnames:
        np.testing.assert_equal(ref[q].data, expected_ref[q].data)
        if ref[q].unit != expected_ref[q].unit:
            raise ValueError(f'Reference units changed: {q}')
    for original_fields, derived, refs in [(b.POSITION, POSITION, REF_POSITION), (b.VELOCITY, VELOCITY, REF_VELOCITY)]:
        array, valid = vectors(original, original_fields)
        expected = array[:, valid] - np.array([ref[q][0] for q in refs])[:, None]
        for q, row in zip(derived[:3], expected):
            np.testing.assert_allclose(a.values(result[q])[valid], row, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(a.values(result[derived[3]])[valid], np.linalg.norm(expected, axis=0), rtol=1e-12, atol=1e-12)
        for q in derived:
            np.testing.assert_array_equal(np.isfinite(a.values(result[q])), valid)
    for q, unit in UNITS.items():
        if result[q].unit != u.Unit(unit):
            raise ValueError(f'Derived units changed: {q}')
    displacement, pos = vectors(result, POSITION)
    dv, vel = vectors(result, VELOCITY)
    r = a.values(result['r_cluster_pc'])
    valid = pos & vel & (r > 0)
    for q in DECOMPOSITION:
        np.testing.assert_array_equal(np.isfinite(a.values(result[q])), valid)
    radial = a.values(result[DECOMPOSITION[0]])[valid]
    tangent = a.values(result[DECOMPOSITION[1]])[valid]
    np.testing.assert_allclose(radial, np.sum(dv[:, valid]*displacement[:, valid]/r[valid], axis=0), atol=1e-12)
    np.testing.assert_allclose(radial**2 + tangent**2, a.values(result['dv_cluster_kms'])[valid]**2, rtol=1e-12, atol=1e-12)
    if np.any(tangent < 0):
        raise ValueError('Negative tangential speed')
    return dict(n_position=int(pos.sum()), n_velocity=int(vel.sum()),
                n_decomposition=int(valid.sum()), n_zero_radius=int((pos & (r == 0)).sum()))


def validate_canonical(table):
    a.validate_invariants(table, Table.read(a.MASTER_PATH))
    # Read-only consistency check against the original Project 02 observations.
    b.validate_output(Table.read(a.INPUT_PATH), table)
    _, valid = vectors(table, b.VELOCITY)
    if valid.sum() != 886:
        raise ValueError('Expected 886 complete Cartesian velocities')
    np.testing.assert_array_equal(valid, np.isfinite(a.values(table['radial_velocity'])))


def summaries(table):
    def rows(mask, field, group):
        return [dict(group_column=field, group=group, n_total=int(mask.sum()),
                     n_position=int(np.isfinite(a.values(table['r_cluster_pc'])[mask]).sum()),
                     n_velocity=int(np.isfinite(a.values(table['dv_cluster_kms'])[mask]).sum()),
                     quantity=q, unit=unit, **a.percentile_summary(table[q][mask])) for q, unit in UNITS.items()]
    overall = Table(rows=rows(np.ones(len(table), dtype=bool), 'all', 'all'))
    grouped = Table(rows=[r for field, groups in a.GROUPS.items() for group in groups
                         for r in rows(np.asarray(table[field]) == group, field, group)])
    for t in (overall, grouped):
        t.meta.update(coordinate_convention=CONVENTION, reference_definition=METHOD,
                      statistics='All finite measurements, linear percentiles; no clipping. Units specified per row.',
                      reference_product='stock2_reference_frame.ecsv')
    return overall, grouped


def draw_scatter(ax, table, x, y, field, origin=True):
    for color, group in zip(a.COLORS, a.GROUPS[field]):
        group_mask = np.asarray(table[field]) == group
        valid = group_mask & np.isfinite(a.values(table[x])) & np.isfinite(a.values(table[y]))
        ax.scatter(a.values(table[x])[valid], a.values(table[y])[valid], s=12, alpha=.45,
                   color=color, edgecolors='none', label=f'{group} (N={valid.sum()}/{group_mask.sum()})')
    if origin:
        ax.plot(0, 0, marker='+', color='black', markersize=9, markeredgewidth=1.2,
                linestyle='none', label='C reference origin', zorder=5)
    ax.set(xlabel=LABELS[x], ylabel=LABELS[y])
    ax.grid(alpha=.18)
    if x in POSITION[:3] and y in POSITION[:3]:
        ax.set_aspect('equal', adjustable='box')


def scatter(table, x, y, field):
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    draw_scatter(ax, table, x, y, field, origin=x != 'r_cluster_pc')
    ax.set_title('Stock 2 — relative to the existing C reference')
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1))
    return fig, ax


def ecdf(table, quantity):
    fig, ax = plt.subplots(figsize=(7, 5), layout='constrained')
    for color, group in zip(a.COLORS, a.GROUPS['homogenized_class']):
        mask = np.asarray(table['homogenized_class']) == group
        values = a.values(table[quantity])[mask]
        values = np.sort(values[np.isfinite(values)])
        if len(values):
            ax.step(np.r_[values[0], values], np.arange(len(values)+1)/len(values),
                    where='post', color=color, linewidth=1.7,
                    label=f'{group} (valid N={len(values)}/{mask.sum()})')
    ax.set(xlabel=LABELS[quantity], ylabel='Cumulative fraction within each class',
           title='Stock 2 — empirical cumulative distributions', ylim=(0, 1.02))
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=.18)
    return fig, ax


def reference_overview(table):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), layout='constrained')
    xyz, valid = vectors(table, POSITION)
    extent = max(1., float(np.abs(xyz[:, valid]).max()) * 1.05)
    for ax, (i, j) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
        draw_scatter(ax, table, POSITION[i], POSITION[j], 'homogenized_class')
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent))
    axes[1].legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.18), ncol=2)
    fig.suptitle('Stock 2 reference frame — all 1,456 literature sources; common scale in pc')
    return fig, axes


def generate_plots(table, directory):
    outputs = []
    def save(fig, stem):
        for extension in ('png', 'pdf'):
            path = directory / f'{stem}.{extension}'
            fig.savefig(path, dpi=200)
            outputs.append(path)
        plt.close(fig)
    specs = [(POSITION[i], POSITION[j], 'homogenized_class') for i, j in [(0,1),(0,2),(1,2)]]
    specs += [(POSITION[0], POSITION[1], 'catalogue_status')]
    specs += [(VELOCITY[i], VELOCITY[j], 'homogenized_class') for i, j in [(0,1),(0,2),(1,2)]]
    specs += [('r_cluster_pc', 'dv_cluster_kms', 'homogenized_class')]
    for x, y, field in specs:
        fig, _ = scatter(table, x, y, field)
        save(fig, f'{x}_vs_{y}_by_{field}')
    for q in ('r_cluster_pc', 'dv_cluster_kms'):
        fig, _ = ecdf(table, q)
        save(fig, f'{q}_ecdf_by_homogenized_class')
    fig, _ = reference_overview(table)
    save(fig, 'reference_frame_spatial_overview')
    return outputs


def protected_hashes():
    hashes = b.protected_hashes()
    paths = list((ROOT/'results/project03/heliocentric_phase_space').rglob('*'))
    paths += list((ROOT/'data/interim/project03').glob('stock2_heliocentric*.ecsv'))
    paths += [Path(b.__file__), ROOT/'docs/project03_heliocentric_phase_space.md', ROOT/'tests/test_project03_heliocentric.py']
    for p in paths:
        if p.is_file():
            hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return hashes


def make_report(table, ref, checks, overall, grouped, outputs):
    lines = ['Project 03C — Stock 2 cluster-centered reference frame', '='*65,
             'Reference definition: already-existing literature C class; no new membership inference.', METHOD,
             f'C total: {ref["n_c_total"][0]}; position contributors: {ref["n_position_contributors"][0]}; '
             f'velocity contributors: {ref["n_velocity_contributors"][0]}']
    for q in REF_POSITION + REF_VELOCITY + ['r_ref_pc', 'speed_ref_kms', 'l_ref_deg', 'b_ref_deg', 'ra_ref_deg', 'dec_ref_deg']:
        lines.append(f'{q}: {ref[q][0]:.10f} [{ref[q].unit}]')
    lines += ['', CONVENTION, '', f'Canonical rows: {len(table)}',
              f'Unique source IDs: {len(np.unique(a.identities(table["source_id"])))}',
              f'Valid positions: {checks["n_position"]}; valid velocities: {checks["n_velocity"]}; '
              f'missing velocities: {len(table)-checks["n_velocity"]}',
              f'Radial/tangential decomposition: {checks["n_decomposition"]}; exact reference-position rows: {checks["n_zero_radius"]}',
              'Radial relative velocity is positive outward from the reference; tangential speed is nonnegative.',
              'These are spatial cluster-relative components, not Gaia line-of-sight radial_velocity.',
              '', 'Full and grouped descriptive summaries (no clipping):']
    for t in (overall, grouped):
        for row in t:
            stats = ', '.join(f'{key}={row[key]:.6g}' for key in a.PERCENTILES)
            lines.append(f'{row["group_column"]}/{row["group"]}: total N={row["n_total"]}, '
                         f'N position={row["n_position"]}, N velocity={row["n_velocity"]}; '
                         f'{row["quantity"]} [{row["unit"]}], valid N={row["n_valid"]}: {stats}')
    lines += ['', 'No source removed. Original observations and labels preserved.',
              'No quality cuts, clipping, membership criterion, centre iteration, or structure classification.',
              'No Galactocentric transformation, solar-motion correction, or orbit integration. Stop after 03C.',
              'ECDFs show fractions within each class with available N stated. All finite extrema retained.',
              '', 'Outputs:'] + [str(p.relative_to(ROOT)) for p in outputs]
    lines += ['', 'Invariants:', 'rows == 1456: True', 'unique source_id == 1456: True',
              'C total == 940: True', 'L total == 184: True', 'T total == 332: True',
              'velocity availability remains 886: True', 'sources without RV retain NaN relative velocities: True',
              'no source removed: True', 'original observables unchanged: True', 'original labels unchanged: True',
              'reference position uses only existing C sources: True',
              'reference velocity uses only existing C sources with valid velocities: True',
              'no clipping used: True', 'no membership criterion introduced: True',
              'no Galactocentric transformation performed: True', 'no orbit integration performed: True',
              'Project 02/03A/03B products unchanged: True', 'PROJECT 03C STATUS: PASS']
    return '\n'.join(lines) + '\n'


def main():
    before = protected_hashes()
    table = Table.read(INPUT)
    validate_canonical(table)  # Before reference calculation or analysis.
    ref = reference_frame(table)
    result = relative_phase_space(table, ref)
    checks = validate_relative(table, result, ref)
    validate_canonical(result)
    overall, grouped = summaries(result)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    outputs = []
    products = [('stock2_cluster_relative_phase_space', result), ('stock2_reference_frame', ref),
                ('stock2_cluster_relative_summary', overall), ('stock2_cluster_relative_group_summary', grouped)]
    for stem, product in products:
        path = OUTPUT / f'{stem}.ecsv'
        product.write(path, overwrite=True)
        outputs.append(path)
    validate_relative(table, Table.read(outputs[0]), Table.read(outputs[1]))
    outputs += generate_plots(result, FIGURES)
    if protected_hashes() != before:
        raise RuntimeError('Protected Project 02/03A/03B files changed')
    report_path = FIGURES / 'cluster_reference_frame_report.txt'
    outputs.append(report_path)
    report = make_report(result, ref, checks, overall, grouped, outputs)
    report_path.write_text(report, encoding='utf-8')
    print(report)
    return result, ref


if __name__ == '__main__':
    main()
