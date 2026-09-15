"""Project 03A: descriptive Gaia observables, without population selection."""
from pathlib import Path
import hashlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from astropy import units as u

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / 'data/interim/project02/stock2_gaia_corrected_flux_excess.ecsv'
MASTER_PATH = ROOT / 'data/processed/stock2_literature_master.ecsv'
INTERIM_DIR = ROOT / 'data/interim/project03'
RESULTS_DIR = ROOT / 'results/project03/phase_space'
UNITS = dict(ra='deg', dec='deg', parallax='mas', pmra='mas / yr',
             pmdec='mas / yr', radial_velocity='km / s')
LABELS = dict(ra='RA [deg]', dec='Dec [deg]', parallax='Parallax [mas]',
              pmra=r'pmra ($\mu_{\alpha *}$) [mas yr$^{-1}$]',
              pmdec=r'pmdec ($\mu_\delta$) [mas yr$^{-1}$]',
              radial_velocity=r'Radial velocity [km s$^{-1}$]')
GROUPS = {'catalogue_status': {'shared': 885, 'kos_only': 178, 'risbud_only': 393},
          'homogenized_class': {'C': 940, 'L': 184, 'T': 332}}
PERCENTILES = dict(median=50, p05=5, p16=16, p84=84, p95=95, min=0, max=100)
COLORS = ['#0072B2', '#D55E00', '#009E73']


def values(column):
    """Keep missing measurements missing, including masked finite payloads."""
    return np.asarray(np.ma.asarray(column, dtype=float).filled(np.nan))


def percentile_summary(column):
    a = values(column)
    a = a[np.isfinite(a)]
    return {'n_valid': len(a), **{k: float(np.percentile(a, p)) if len(a) else np.nan
                                for k, p in PERCENTILES.items()}}


def identities(column):
    # Never pass Gaia identifiers through floating point.
    if np.any(np.ma.getmaskarray(column)) or column.dtype.kind == 'f':
        raise ValueError('Masked or floating-point source identifiers')
    return np.asarray([int(str(x)) for x in column], dtype=np.int64)


def validate_invariants(table, master):
    ids = identities(table['source_id'])
    mid = identities(master['gaia_dr3_source_id'])
    if len(table) != 1456 or len(np.unique(ids)) != 1456:
        raise ValueError('Expected rows == 1456 and unique source_id == 1456')
    if len(master) != 1456 or len(np.unique(mid)) != 1456 or set(ids) != set(mid):
        raise ValueError('Canonical population changed')
    if not np.array_equal(ids, identities(table['gaia_dr3_source_id'])):
        raise ValueError('Gaia/literature identity mismatch')
    order = np.argsort(ids)
    morder = np.argsort(mid)
    for field, expected in GROUPS.items():
        if np.any(np.ma.getmaskarray(table[field])):
            raise ValueError(f'Missing labels: {field}')
        labels = np.asarray(table[field], dtype=str)
        if dict(zip(*np.unique(labels, return_counts=True))) != expected:
            raise ValueError(f'Unexpected group counts: {field}')
        if not np.array_equal(labels[order], np.asarray(master[field], dtype=str)[morder]):
            raise ValueError(f'Canonical labels changed: {field}')
    for field, unit in UNITS.items():
        if table[field].unit is None or u.Unit(table[field].unit) != u.Unit(unit):
            raise ValueError(f'Unexpected units: {field}')
    rv = np.isfinite(values(table['radial_velocity']))
    if rv.sum() != 886 or not np.array_equal(rv, np.asarray(table['has_rv'], dtype=bool)):
        raise ValueError('RV availability differs from Project 02B')
    return ids


def summary_tables(table):
    def rows(mask, kind, group):
        n_rv = int(np.isfinite(values(table['radial_velocity'])[mask]).sum())
        return [dict(group_column=kind, group=group, n_total=int(mask.sum()), n_rv=n_rv,
                     quantity=q, unit=unit, **percentile_summary(table[q][mask]))
                for q, unit in UNITS.items()]
    full = Table(rows=rows(np.ones(len(table), dtype=bool), 'all', 'all'))
    grouped = Table(rows=[row for field, groups in GROUPS.items() for group in groups
                          for row in rows(np.asarray(table[field]) == group, field, group)])
    for result in (full, grouped):
        result.meta.update(input=str(INPUT_PATH.relative_to(ROOT)),
                           statistics='Linear percentiles of finite measurements; no clipping.',
                           units='Each row uses the unit column for all percentile/min/max values.',
                           population='Unchanged canonical 1456 sources; existing labels reused.')
    return full, grouped


def scatter(table, x, y, field):
    fig, ax = plt.subplots(figsize=(7, 5), layout='constrained')
    for color, group in zip(COLORS, GROUPS[field]):
        mask = np.asarray(table[field]) == group
        ax.scatter(values(table[x])[mask], values(table[y])[mask], s=12, alpha=.45,
                   edgecolors='none', color=color, label=f'{group} (N={mask.sum()})')
    ax.set(xlabel=LABELS[x], ylabel=LABELS[y], title='Stock 2 — canonical literature population')
    ax.legend(fontsize=9)
    ax.grid(alpha=.18)
    return fig, ax


def histogram(table, quantity):
    fig, ax = plt.subplots(figsize=(7, 5), layout='constrained')
    a = values(table[quantity])
    bins = np.histogram_bin_edges(a[np.isfinite(a)], bins=40)
    for color, group in zip(COLORS, GROUPS['homogenized_class']):
        selected = a[np.asarray(table['homogenized_class']) == group]
        finite = selected[np.isfinite(selected)]
        ax.hist(finite, bins=bins, histtype='step', linewidth=1.7, color=color,
                label=f'{group} (valid N={len(finite)}/{len(selected)})')
    ax.set(xlabel=LABELS[quantity], ylabel='Number of sources',
           title=f'Stock 2 — {np.isfinite(a).sum()} available / {len(table)} sources')
    ax.legend(fontsize=9)
    ax.grid(alpha=.18)
    return fig, ax


def generate_plots(table, directory):
    paths = []
    specs = [('ra', 'dec', f) for f in GROUPS] + [('pmra', 'pmdec', f) for f in GROUPS]
    specs += [('parallax', pm, 'homogenized_class') for pm in ('pmra', 'pmdec')]
    for x, y, field in specs:
        fig, _ = scatter(table, x, y, field)
        stem = f'{x}_vs_{y}_by_{field}'
        for extension in ('png', 'pdf'):
            path = directory / f'{stem}.{extension}'
            fig.savefig(path, dpi=200)
            paths.append(path)
        plt.close(fig)
    for q in ('parallax', 'radial_velocity'):
        fig, _ = histogram(table, q)
        for extension in ('png', 'pdf'):
            path = directory / f'{q}_histogram_by_homogenized_class.{extension}'
            fig.savefig(path, dpi=200)
            paths.append(path)
        plt.close(fig)
    return paths


def protected_hashes():
    paths = [MASTER_PATH]
    for directory in ('data/interim/project02', 'data/raw/gaia/project02', 'results/project02'):
        paths += [p for p in (ROOT / directory).rglob('*') if p.is_file()]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    before = protected_hashes()
    table, master = Table.read(INPUT_PATH), Table.read(MASTER_PATH)
    validate_invariants(table, master)  # Must precede any analysis or output.
    snapshot = table.copy(copy_data=True)
    full, grouped = summary_tables(table)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, result in [('stock2_phase_space_summary', full),
                         ('stock2_phase_space_group_summary', grouped)]:
        path = INTERIM_DIR / f'{name}.ecsv'
        result.write(path, overwrite=True)
        outputs.append(path)
    outputs += generate_plots(table, RESULTS_DIR)
    validate_invariants(table, master)
    for c in table.colnames:
        np.testing.assert_equal(np.ma.getmaskarray(table[c]), np.ma.getmaskarray(snapshot[c]))
        np.testing.assert_equal(table[c].data, snapshot[c].data)
    if protected_hashes() != before:
        raise RuntimeError('Protected input/Project 02 products changed')
    lines = ['Project 03A — Observed astrometric phase-space diagnostics',
             '=' * 65, 'Canonical population size: 1456', 'Unique source_id count: 1456',
             'RV availability: 886 / 1456', '', 'Existing catalogue and class sizes:']
    for field, groups in GROUPS.items():
        for group, count in groups.items():
            row = grouped[(grouped['group_column'] == field) & (grouped['group'] == group)][0]
            lines.append(f'{field}: {group}: N={count}, N RV={row["n_rv"]}')
    lines += ['', 'Full population summaries:']
    for r in full:
        lines.append(f'{r["quantity"]} [{r["unit"]}]: valid N={r["n_valid"]}, '
                     f'median={r["median"]:.6g}, p16–p84={r["p16"]:.6g}–{r["p84"]:.6g}, '
                     f'min–max={r["min"]:.6g}–{r["max"]:.6g}')
    lines += ['', 'No filtering applied. Missing measurements remain missing; finite values only',
              'for each statistic/histogram. No source is removed from the population.',
              'All plot axes cover the complete finite range. No new centre is defined.',
              '', 'Outputs:'] + [str(p.relative_to(ROOT)) for p in outputs]
    lines += ['', 'Invariants:', 'rows == 1456: True', 'unique source_id == 1456: True',
              'canonical population unchanged: True', 'canonical labels unchanged: True',
              'Project 01/02 input and Project 02 products unchanged: True',
              'no membership cut applied: True', 'no quality cut applied: True',
              'no outlier clipping applied: True', 'PROJECT 03A STATUS: PASS']
    report = '\n'.join(lines) + '\n'
    (RESULTS_DIR / 'phase_space_report.txt').write_text(report, encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
