# Project 03B — Heliocentric Cartesian phase space

## Input and conventions

The canonical input remains `data/interim/project02/stock2_gaia_corrected_flux_excess.ecsv`.
Project 03A generates summaries, not an alternative canonical population. Its
validation and percentile helpers are imported without modifying its files.
Existing per-source provenance and C/L/T labels are preserved exactly.

Astropy transforms ICRS to Galactic Cartesian coordinates, with right-handed axes:
X toward l=0,b=0 (Galactic-centre direction), Y toward l=90,b=0 (Galactic rotation
direction), Z toward the North Galactic Pole. Positions and radius are in pc;
velocities and speed are in km/s. Velocity signs follow the corresponding positive
axes. Gaia pmra is supplied as pm_ra_cosdec without a second cosine factor;
positive radial velocity means receding.

The requested `helio` column names use the conventional local solar-neighbourhood
interpretation. Precisely, Astropy Galactic has the solar-system barycentre as its
origin; the original Gaia barycentric convention is preserved. There is no
Sun/barycentre translation, epoch propagation, or RV correction. See the official
[Galactic frame documentation](https://docs.astropy.org/en/stable/api/astropy.coordinates.Galactic.html)
and [velocity guide](https://docs.astropy.org/en/stable/coordinates/velocities.html).
No Galactocentric position, solar motion, or cluster bulk motion is introduced.

Distances use `Distance(parallax=...)`, equivalent to 1000/parallax[mas] in pc.
There is no distance prior or parallax zero-point correction. Nonfinite, zero,
or negative parallax and invalid sky positions yield missing derived positions
and velocities while retaining the row. Missing proper motion prevents velocity
calculation but does not prevent a valid position. Missing RV leaves all four
velocity fields NaN. These are computational availability conditions, not
membership or quality selection.

## Added files

- `src/project03/02_heliocentric_phase_space.py`: reusable transformation,
  validation, summaries, plots, and report.
- `tests/test_project03_heliocentric.py`: nine tests.
- `docs/project03_heliocentric_phase_space.md`: this implementation record.

No pre-existing source or documentation file was modified.

## Commands run

From the repository root:

```sh
MPLCONFIGDIR=/private/tmp/stock2-mpl .venv/bin/python -m unittest discover -s tests -v
MPLCONFIGDIR=/private/tmp/stock2-mpl .venv/bin/python src/project03/02_heliocentric_phase_space.py
```

The diagnostic command was run with console output captured in
`/private/tmp/stock2-03b-console.txt`. Read-only repository status checks,
ECSV inspection, and a temporary montage of all eight PNGs were also performed.

## Generated products

Under `data/interim/project03/`:

- `stock2_heliocentric_phase_space.ecsv`: 1456 rows, all original columns plus
  x_helio_pc, y_helio_pc, z_helio_pc, r_helio_pc, vx_helio_kms, vy_helio_kms,
  vz_helio_kms, speed_helio_kms. Derived units and conventions are in the ECSV.
- `stock2_heliocentric_summary.ecsv`: 8 observable summaries.
- `stock2_heliocentric_group_summary.ecsv`: 48 rows for six existing groups/classes.

Each summary gives total N, positional N, velocity N, valid N, median, p05,
p16, p84, p95, min, max, and units per row. Percentiles use all finite values
independently by quantity, with no clipping.

Under `results/project03/heliocentric_phase_space/`, PNG and PDF versions of:

- `x_helio_pc_vs_y_helio_pc_by_homogenized_class`
- `x_helio_pc_vs_z_helio_pc_by_homogenized_class`
- `y_helio_pc_vs_z_helio_pc_by_homogenized_class`
- `x_helio_pc_vs_y_helio_pc_by_catalogue_status`
- `vx_helio_kms_vs_vy_helio_kms_by_homogenized_class`
- `vx_helio_kms_vs_vz_helio_kms_by_homogenized_class`
- `vy_helio_kms_vs_vz_helio_kms_by_homogenized_class`
- `distance_histogram_by_homogenized_class`

The same directory contains `heliocentric_phase_space_report.txt`, with all
full-sample and group statistics, conventions, output paths, and final invariants.
Existing repository ignore rules exclude interim ECSV products from Git by default.

## Validation results

All 16 tests passed: nine Project 03B tests plus all seven unchanged Project 03A
tests. Coverage includes canonical row/ID/observable preservation, missing RV,
invalid and masked parallaxes, missing astrometry, units, distance norm, ECSV
round trips, full-sample coordinate and velocity round trips, axis/sign checks
using Astropy cardinal directions, summary counts, plot coordinates and complete
ranges, histogram conservation, and protected-product hashes around pipeline execution.

- 1456 rows and 1456 unique source IDs.
- Finite and positive parallaxes: 1456. Zero, negative, and nonfinite: 0 each.
- Positions: 1456. Velocities: 886. Missing-RV rows retained: 570.
- Distance range: 317.3372–502.0963 pc.
- Position norm agrees at rtol=1e-12, atol=1e-9 pc.
- Full position round-trip maximum angular error: 2.76e-10 arcsec.
- Full velocity round trip agrees at rtol=1e-11, atol=1e-9 in native units.

The main ECSV was read back and all original columns, masks, and units compared.
Both summary tables were inspected. All eight final PNGs were visually inspected
for axes, units, group counts, full finite ranges, and missing values. Position
projections use equal physical scaling, so the small Z extent appears as a narrow
panel. Scatter legends are outside the data panels. No extremes were cropped.
Protected Project 02 and Project 03A file hashes were unchanged during execution.

No source removal, membership criterion, quality cut, outlier rejection, cluster
centre, bulk-motion subtraction, Galactocentric correction, solar-motion correction,
or orbit integration was performed. Project 03C was not started.

PROJECT 03B STATUS: PASS
