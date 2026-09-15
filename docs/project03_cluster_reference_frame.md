# Project 03C — Stock 2 cluster-centered reference frame

## Definition and preserved inputs

Input: `data/interim/project03/stock2_heliocentric_phase_space.ecsv`.
All original Project 03B columns, units, masks, row order, Gaia observables,
source identities, and existing provenance and C/L/T labels are preserved.

The reference uses only the already-existing `homogenized_class == C` population.
Position and velocity are computed separately as component-wise medians of all
complete finite Cartesian vectors in that class. No quality cut, clipping,
iteration, optimization, or new membership inference is performed. The input
contains 940 C sources with positions and 594 C sources with velocities.

The Galactic Cartesian axes remain those of Project 03B: right-handed X toward
l=0,b=0, Y toward l=90,b=0, and Z toward the North Galactic Pole. The inherited
`*_helio_*` names retain compatibility; their actual origin is the solar-system
barycentre in Astropy Galactic. Project 03C subtracts the adopted reference values
without rotating the axes. No solar-motion or Galactocentric correction is applied.

## Adopted reference

| Quantity | Value | Units |
|---|---:|---|
| X_ref | -257.0582356893 | pc |
| Y_ref | 271.7037408556 | pc |
| Z_ref | -10.3984938308 | pc |
| Vx_ref | -30.6513762878 | km/s |
| Vy_ref | -17.5197042716 | km/s |
| Vz_ref | -14.3806970059 | km/s |
| r_ref | 374.1790854719 | pc |
| V_ref | 38.1215339733 | km/s |
| Galactic l | 133.4134427428 | deg |
| Galactic b | -1.5924634672 | deg |
| ICRS RA | 33.8917184222 | deg |
| ICRS Dec | 59.5563102729 | deg |

The sky coordinates are an Astropy back-transform of the Cartesian reference,
not a mean of RA/Dec or a second definition of the reference.

## Relative quantities

The enriched output adds dx_cluster_pc, dy_cluster_pc, dz_cluster_pc, and their
Euclidean norm r_cluster_pc; dvx_cluster_kms, dvy_cluster_kms, dvz_cluster_kms,
and their Euclidean norm dv_cluster_kms. All 1456 rows remain. The 570 sources
without RV retain NaN in all relative-velocity fields.

Optional spatial decomposition is included as v_radial_cluster_rel_kms and
v_tangential_cluster_rel_kms. The signed radial component is the dot product of
relative velocity with the unit cluster displacement vector (positive outward).
Tangential speed is the norm of the orthogonal residual, mathematically equivalent
to sqrt(relative_speed^2 - radial_component^2) but avoiding cancellation near
purely radial motion. Both quantities are NaN at exactly zero displacement or
when either vector is missing. They are not Gaia line-of-sight radial_velocity.
There are 886 valid decompositions and zero exact-reference-position cases.

## Class summaries

| Class | Total / positions | Velocities | Median r_cluster [pc] | Median relative speed [km/s] |
|---|---:|---:|---:|---:|
| C | 940 | 594 | 8.147600 | 2.146601 |
| L | 184 | 111 | 24.583761 | 2.891856 |
| T | 332 | 181 | 34.011715 | 3.076190 |

| Provenance | Total / positions | Velocities | Median r_cluster [pc] | Median relative speed [km/s] |
|---|---:|---:|---:|---:|
| shared | 885 | 580 | 8.783629 | 2.048606 |
| kos_only | 178 | 75 | 14.763004 | 3.140774 |
| risbud_only | 393 | 231 | 36.961161 | 3.101665 |

These describe existing populations and do not assign membership probabilities,
choose a preferred catalogue, or classify spatial structures.

## Added files

- `src/project03/03_cluster_centered_reference_frame.py`
- `tests/test_project03_cluster_reference.py`
- `docs/project03_cluster_reference_frame.md`

No pre-existing source, test, or documentation file was edited.

## Commands run

From the repository root:

```sh
MPLCONFIGDIR=/private/tmp/stock2-mpl .venv/bin/python -m unittest discover -s tests -v
MPLCONFIGDIR=/private/tmp/stock2-mpl .venv/bin/python src/project03/03_cluster_centered_reference_frame.py > /private/tmp/stock2-03c-console.txt
```

Read-only repository status and input inspection were performed first. Generated
ECSV products were read back and a temporary montage was created for visual review.

## Generated products

Under `data/interim/project03/`:

- `stock2_cluster_relative_phase_space.ecsv`: 1456 rows; original Project 03B
  columns plus ten relative quantities.
- `stock2_reference_frame.ecsv`: one row, units, method, counts, reference vectors,
  norms, back-transformed coordinates, and coordinate convention.
- `stock2_cluster_relative_summary.ecsv`: ten observable summaries.
- `stock2_cluster_relative_group_summary.ecsv`: 60 rows containing both C/L/T and
  catalogue-provenance summaries, identified by group_column and group.

Summary rows contain total N, positional N, velocity N, valid N, median, p05, p16,
p84, p95, min, and max. Units are explicit per row. All finite measurements are
used, with linear percentiles and no clipping. Existing Git ignore rules exclude
interim ECSV files by default.

Under `results/project03/cluster_relative_phase_space/`, each in PNG and PDF:

- `dx_cluster_pc_vs_dy_cluster_pc_by_homogenized_class`
- `dx_cluster_pc_vs_dz_cluster_pc_by_homogenized_class`
- `dy_cluster_pc_vs_dz_cluster_pc_by_homogenized_class`
- `dx_cluster_pc_vs_dy_cluster_pc_by_catalogue_status`
- `dvx_cluster_kms_vs_dvy_cluster_kms_by_homogenized_class`
- `dvx_cluster_kms_vs_dvz_cluster_kms_by_homogenized_class`
- `dvy_cluster_kms_vs_dvz_cluster_kms_by_homogenized_class`
- `r_cluster_pc_vs_dv_cluster_kms_by_homogenized_class`
- `r_cluster_pc_ecdf_by_homogenized_class`
- `dv_cluster_kms_ecdf_by_homogenized_class`
- `reference_frame_spatial_overview`

The same directory contains `cluster_reference_frame_report.txt`, with the full
reference, overall and group percentile summaries, contributor counts, output
paths, and explicit required invariants.

ECDFs show cumulative fractions within each class, with available sample sizes
stated. Position panels use equal physical scales and show the reference origin.
The additional three-panel overview uses identical limits in all projections to
compare the full spatial extent without stretching individual axes. Scatter
legends remain outside the data regions. No selection line or fitted axis is drawn.

## Verification

27 tests passed, zero failures: eleven new Project 03C tests plus the sixteen
unchanged Project 03A/03B tests. Tests cover canonical preservation, reference
medians and C-only influence, extreme contributors and incomplete vectors,
arithmetic, norms, missing-RV propagation, units, ECSV serialization, sky
back-transformation, signed radial/tangential behavior and zero radius,
independent summary statistics, plotted coordinates and ranges, origin markers,
ECDF normalization, common overview scales, and prior-product protection.

The main and reference tables were reloaded and revalidated by the production
pipeline. Full and group summaries were read back for inspection. All eleven
final PNGs were visually inspected for units, axes, legends, origins, full ranges,
and class counts. Project 02/03A/03B product hashes were unchanged across the run.

All 1456 unique sources, 1456 positions, and 886 velocities remain. Existing
C/L/T counts are 940/184/332. Original observations and labels remain unchanged.
No source classification, membership selection, clipping, Galactocentric
transformation, solar-motion correction, or orbit integration was performed.
Project 03D was not started.

PROJECT 03C STATUS: PASS
