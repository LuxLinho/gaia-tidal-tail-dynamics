# Project 03A — Observed astrometric phase-space diagnostics

## Input and scientific scope

Uses the latest Project 02E product directly:
`data/interim/project02/stock2_gaia_corrected_flux_excess.ecsv`.
The existing `catalogue_status` and `homogenized_class` fields are reused.
The Project 01 literature master is read only to verify identities and labels;
no new sample is constructed. All 1456 unique sources remain included.

This is observational characterization only. No membership, quality, RUWE,
parallax, proper-motion, RV, photometry, or outlier selection is applied.
No centre, corrected observable, dynamical coordinate, orbit, or new class
is introduced. Project 03B is outside scope.

## Run from repository root

```sh
MPLCONFIGDIR=/private/tmp/stock2-mpl .venv/bin/python -m unittest discover -s tests -v
MPLCONFIGDIR=/private/tmp/stock2-mpl .venv/bin/python src/project03/01_observed_phase_space_diagnostics.py
```

Seven tests passed. Tests cover masked/nonfinite measurements and extreme
values, real-data invariants, duplicate/missing/replaced sources, label swaps
that preserve counts, summary counts and ECSV round trips, scatter coordinates
and limits, and histogram count conservation. No earlier test suite was present.

## Outputs

- `data/interim/project03/stock2_phase_space_summary.ecsv`: six observable rows.
- `data/interim/project03/stock2_phase_space_group_summary.ecsv`: 36 rows,
  covering six observables for each of three provenance groups and three classes.
- `results/project03/phase_space/phase_space_report.txt`: console report and invariants.
- Eight figures, each in PNG and PDF, in `results/project03/phase_space/`:
  - `ra_vs_dec_by_catalogue_status`
  - `ra_vs_dec_by_homogenized_class`
  - `pmra_vs_pmdec_by_catalogue_status`
  - `pmra_vs_pmdec_by_homogenized_class`
  - `parallax_vs_pmra_by_homogenized_class`
  - `parallax_vs_pmdec_by_homogenized_class`
  - `parallax_histogram_by_homogenized_class`
  - `radial_velocity_histogram_by_homogenized_class`

Each summary row contains total N, RV N, valid N, median, p05, p16, p84,
p95, min, max, quantity, and units. Statistics use NumPy linear percentiles
of finite measurements independently per quantity. Missing values are never
replaced by zero. Units are recorded per row because quantities have different
units. Histograms share 40 equal-width bins over the full finite sample range.
Scatter x and y are explicitly identified by the axis labels (filename order is
x then y). Gaia pmra is mu_alpha*cos(dec), without another cosine correction.

RA is kept in its original 0–360 degree convention, including linear percentile
summaries. The wrap boundary stretches the sky plots; it is not a physical gap.
No optional proper-motion offset plot is generated because no new centre is needed.

## Verified results

- Rows and unique source IDs: 1456 each.
- RV available: 886.
- Provenance total/RV: shared 885/580; kos_only 178/75; risbud_only 393/231.
- Class total/RV: C 940/594; L 184/111; T 332/181.
- Full sample median: parallax 2.67572 mas; pmra 15.7976 mas/yr;
  pmdec -13.7913 mas/yr; RV 8.78903 km/s.

All eight PNG figures were visually inspected together after generation, and
both ECSV files were read back. Axes, labels, units, group sizes, missing data,
and extreme-value coverage were checked. RV spans -96.0155 to 55.7351 km/s;
these values remain visible. Tests independently check scatter coordinates and
histogram count conservation. Source identity and per-source labels agree with
the canonical master. Input and existing Project 02 product hashes agree before
and after execution. Existing uncommitted Project 01/02 changes were preserved.

Added source: `src/project03/01_observed_phase_space_diagnostics.py`.
Added tests: `tests/test_project03_phase_space.py`.
Added documentation: this file. No existing source file was changed.

PROJECT 03A STATUS: PASS
