# Project 03D — Cluster-relative morphology and phase-space coherence

## Scope and inputs

Reads `data/interim/project03/stock2_cluster_relative_phase_space.ecsv` and
`data/interim/project03/stock2_reference_frame.ecsv` unchanged. All 1456 sources,
940/184/332 C/L/T labels, 1456 positions, and 886 velocities remain. The 570
missing-RV rows retain NaN projected velocities. The Project 03C reference is
read, checked against stored metadata and coordinate subtraction, and never
recomputed. Stored spatial radial/tangential components are reused unchanged.

Inherited Project 03B `*_helio_*` fields use Astropy Galactic's solar-system
barycentric origin. Project 03D retains the Project 03C cluster-relative origin.
No source selection, membership field, threshold, clipping, Galactocentric
transformation, orbit, action, or potential model is introduced. No Project 04 work.

## PCA method and fixed basis

For full/C/L/T/L+T separately, subtract that population's coordinate mean and
perform SVD on the N-by-3 position matrix in pc. Eigenvalues are squared singular
values divided by N-1 (sample covariance). No coordinate standardization or
outlier removal is applied. Square roots of eigenvalues are RMS axis-length
proxies, not hard boundaries. Covariance axis ratios are square roots of the
corresponding eigenvalue ratios.

Eigenvalues are ordered descending. For e1 and e2, the largest-absolute Cartesian
component is made positive; a tie uses the first index in X,Y,Z order. e3 is
cross(e1,e2), so the basis is right-handed. This sign convention is deterministic;
axes within an exactly degenerate eigenspace would not have a unique geometric
orientation. The observed eigenvalues are distinct.

Only the existing 516 L+T positions determine the canonical projection basis.
Velocity information and C positions have no influence on this basis.

| Population | PC1 fraction | PC2 fraction | PC3 fraction | Axis 1/2 | Axis 2/3 | Axis 1/3 |
|---|---:|---:|---:|---:|---:|---:|
| Full | 0.713579 | 0.247770 | 0.038652 | 1.697059 | 2.531860 | 4.296715 |
| C | 0.773243 | 0.123794 | 0.102963 | 2.499238 | 1.096504 | 2.740426 |
| L | 0.833449 | 0.138946 | 0.027605 | 2.449152 | 2.243539 | 5.494767 |
| T | 0.788862 | 0.162332 | 0.048806 | 2.204438 | 1.823751 | 4.020345 |
| L+T | 0.771467 | 0.201372 | 0.027161 | 1.957308 | 2.722894 | 5.329542 |

Fixed L+T unit vectors, as rows in original Project 03C Cartesian axes:

- e1 = (0.6515182608824263, 0.7585157819803061, 0.0133328250398672)
- e2 = (0.7585035322824300, -0.6516311224769061, 0.0070193827766425)
- e3 = (0.0140123963623609, 0.0055397388289368, -0.9998864755770489)

Every canonical position is projected as s = basis @ delta_r; every available
velocity as u = basis @ delta_v. The population mean is subtracted only while
fitting PCA, not when calculating canonical s coordinates. Thus s=0 stays at the
stored Stock 2 reference. No separate velocity PCA is computed.

## Projected diagnostics

| Class | Median abs(s1) [pc] | Median r_perp [pc] | Median abs(u1) [km/s] | Median transverse speed [km/s] |
|---|---:|---:|---:|---:|
| C | 3.451413 | 6.475006 | 0.520640 | 2.059578 |
| L | 13.638032 | 16.177746 | 0.967788 | 2.455110 |
| T | 19.985093 | 18.409665 | 0.680644 | 2.766451 |

The spatial PCA describes elongation of the literature-defined populations.
It does not establish a tidal origin. All full-range plots retain extreme values.
The geometric major-axis line is drawn through the stored reference origin to
show orientation; it is not a fitted locus, boundary, or membership criterion.

### s1 versus u1 correlations

| Population | N | Pearson r | Spearman rho |
|---|---:|---:|---:|
| Full | 886 | 0.043676 | 0.062103 |
| C | 594 | 0.037491 | 0.024339 |
| L | 111 | 0.181360 | 0.384293 |
| T | 181 | 0.162908 | 0.355220 |
| L+T | 292 | 0.028601 | 0.126847 |

Global along-axis correlations are weak, while L and T separately show larger
rank correlations. This does not establish gravitational association. The
machine-readable table includes five relationships for all five populations:
s1/u1, abs(s1)/abs(u1), r_cluster/relative speed, r_perp/transverse speed, and
s1/relative speed. Every row records its paired finite N and two-sided p-values.

Pearson p-values use SciPy's normal uncorrelated null. Spearman p-values are
asymptotic and can be inaccurate for smaller samples, especially N<=500.
These are descriptive diagnostics, without multiple-comparison correction,
measurement-error propagation, or shared-reference uncertainty modelling.
They are not complete physical significance tests and are never used to classify
sources. Sources: [SciPy Pearson documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html)
and [SciPy Spearman documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html).

### Reused cluster-relative radial behavior

| Class | N radial | Median radial [km/s] | Outward fraction | Inward fraction |
|---|---:|---:|---:|---:|
| C | 594 | -0.046421 | 0.484848 | 0.515152 |
| L | 111 | 0.036584 | 0.504505 | 0.495495 |
| T | 181 | 0.613996 | 0.629834 | 0.370166 |

Fractions use finite radial measurements as the denominator. Positive means
motion outward from the fixed reference, not escaping or unbound. Zero radial
velocities and zero s1 positions are separately counted in the output tables.

## Added files and commands

- `src/project03/04_relative_morphology_coherence.py`
- `tests/test_project03_morphology.py`
- `docs/project03_relative_morphology.md`

No existing source, test, documentation, or Project 03A–03C product was edited.

Commands run from the repository root:

```sh
MPLCONFIGDIR=/private/tmp/stock2-mpl .venv/bin/python -m unittest discover -s tests -v
MPLCONFIGDIR=/private/tmp/stock2-mpl .venv/bin/python src/project03/04_relative_morphology_coherence.py > /private/tmp/stock2-03d-console.txt
```

Read-only repository/input inspection and ECSV readback were also performed.
Temporary montages supported visual inspection of all thirteen PNG figures.

## Generated tables

Under `data/interim/project03/`:

- `stock2_relative_morphology.ecsv`: 1456 rows; all Project 03C columns plus
  s1_ext_pc, s2_ext_pc, s3_ext_pc, r_perp_ext_pc, u1_ext_kms, u2_ext_kms,
  u3_ext_kms, and u_perp_ext_kms. Stored radial/tangential fields are inherited.
- `stock2_morphology_pca.ecsv`: 15 component rows for five populations, including
  counts, means, eigenvalues, variance fractions, vectors, lengths, axis ratios,
  and identification of the L+T canonical projection basis.
- `stock2_morphology_summary.ecsv`: 16 full-population quantity summaries.
- `stock2_morphology_group_summary.ecsv`: 96 rows for C/L/T and provenance,
  including signed and absolute projected coordinates, velocities, and norms.
- `stock2_morphology_side_summary.ecsv`: 18 rows for six groups and negative,
  positive, and zero s1; counts, median abs(s1), maximum abs(s1).
- `stock2_morphology_radial_summary.ecsv`: six groups, radial counts, fractions,
  median/p16/p84 radial velocity, and median tangential speed.
- `stock2_morphology_correlations.ecsv`: 25 relationships with N, Pearson,
  Spearman, two-sided p-values, and method caveats.

Units, PCA sign convention, projection method, and exact input/reference file
hashes are stored in the relevant ECSV metadata. Interim ECSV products follow
the repository's existing Git ignore rules.

## Generated figures and report

Under `results/project03/relative_morphology/`, PNG and PDF versions of:

- `s1_ext_pc_vs_s2_ext_pc_by_homogenized_class`
- `s1_ext_pc_vs_s3_ext_pc_by_homogenized_class`
- `s2_ext_pc_vs_s3_ext_pc_by_homogenized_class`
- `s1_ext_pc_vs_r_perp_ext_pc_by_homogenized_class`
- `s1_ext_pc_vs_u1_ext_kms_by_homogenized_class`
- `abs_s1_ext_pc_vs_abs_u1_ext_kms_by_homogenized_class`
- `r_cluster_pc_vs_dv_cluster_kms_by_homogenized_class`
- `r_perp_ext_pc_vs_u_perp_ext_kms_by_homogenized_class`
- `s1_ext_pc_vs_r_perp_ext_pc_by_catalogue_status`
- `s1_ext_pc_ecdf_by_homogenized_class`
- `r_perp_ext_pc_ecdf_by_homogenized_class`
- `u1_ext_kms_ecdf_by_homogenized_class`
- `extended_population_pca_axis_original_coordinates`

The same directory contains `relative_morphology_report.txt`, with prominent
key diagnostics, full percentile summaries, side counts, correlations and
p-values, radial fractions, output paths, and all required invariants.

ECDFs are explicitly labelled as within-class fractions and report available
sample sizes. Spatial projections have equal physical scales. Scatter legends
are outside data panels. The axis figure uses the requested descriptive label:
“major spatial principal axis of the literature-defined L+T population”.

## Validation

Final complete suite: 39 tests passed, zero failures (12 new Project 03D tests
and all 27 unchanged earlier tests). An initial test fixture accidentally made
all C positions identical; it was corrected to translate C positions while
retaining their scatter, so it tests L+T independence without undefined C PCA.

Tests cover row/ID/label preservation, reference metadata mismatch detection,
PCA covariance without scaling or clipping, orthonormality and right-handedness,
deterministic signs and row-order stability, basis independence from C and
velocities, uncentred canonical projection/reconstruction, norms, missing RV,
units, ECSV basis round trips, independent group statistics, radial reuse and
fractions, correlations and paired counts, plotted coordinates and full ranges,
ECDF completeness, axis visualization, protected-file hashes, and a pipeline
check that fails if Project 03C reference calculation is invoked.

The enriched and PCA ECSV files were reloaded and their projections revalidated.
All seven output tables were read for inspection. All thirteen figures were
visually inspected for labels, units, ranges, class/provenance counts, and axis
orientation. Project 03A/03B/03C file hashes were unchanged across execution.

PROJECT 03D STATUS: PASS
