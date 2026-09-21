# Project 06 — Galactic Tidal Field & Jacobi Radius

Four separate stages reuse the reviewed Project 05 orbit and frozen 03C→04A
reference. The canonical 1456-source population and inherited labels are never
changed. This is a geometric/environment diagnostic, not an escape classifier.

## Mass provenance

The user-specified literature baseline is **approximately 4000 solar masses**:
Ye et al. (2021), *Diagnosing Open Cluster Stock 2: Member Candidates and Mass
Distribution with Gaia DR2 and LAMOST*, AJ 161, 8,
[DOI 10.3847/1538-3881/abc61a](https://doi.org/10.3847/1538-3881/abc61a).
The supplied provenance describes 1325 Gaia DR2 member candidates and a
present-day mass-function analysis. It is neither initial mass nor a sum of the
repository sample. The supplied core radius is 3.97 pc and tidal radius 22.65 pc.
No formal mass uncertainty is adopted or invented. Primary full text could not
be retrieved in this session; detailed mass-function corrections and the original
tidal-radius derivation have not been independently audited. This limitation is
recorded in the machine-readable mass metadata.

The 22.65 pc radius is a **comparison only**, never an input or calibration target.
It is not necessarily statistically independent of the mass from the same paper.
Definitions, membership samples, Galactic models and derivation methods may
explain differences. Later Stock 2 catalogues may adopt other tidal radii; no
later numeric radius is adopted or treated as independently verified here.
The 2000/3000/4000/5000/6000 solar-mass grid is a sensitivity experiment, **not**
an uncertainty interval.

## 06A — Galactic tidal field

Use installed galpy 1.12.0, unmodified MWPotential2014, with the existing
ro=8 kpc, vo=220 km/s normalization. Potential Phi is per unit test mass and
acceleration is minus its gradient. Analytic `evaluateRforces`, `evaluatezforces`
and `evaluateR2derivs` give actual-position F_R, F_z and Phi_RR at the current
(R,z). These are explicitly labelled auxiliary local-field quantities.

For the Jacobi calculation use a consistent **local circular-equivalent midplane
approximation** at the current cylindrical radius R:

    Omega_c² = -F_R(R,0)/R
    kappa² = Phi_RR(R,0) + 3 Omega_c²
    D = Omega_c² - Phi_RR(R,0) = 4 Omega_c² - kappa²
    r_J = (G M / D)^(1/3)

`omegac`, `epifreq` and `verticalfreq` are galpy's midplane small-oscillation
frequencies. Omega_c is the circular equilibrium frequency, not the actual
cluster angular rate v_phi/R. No off-plane derivative is mixed with these
midplane identities. Stored z_jacobi=0 and z_actual carry distinct meanings.
Frequencies use km/s/kpc, force uses (km/s)^2/kpc, and D/Phi_RR use
(km/s)^2/kpc^2. Natural galpy results are explicitly scaled by vo²/ro for
forces, vo²/ro² for derivatives, and vo/ro for frequencies. G is taken from
`astropy.constants.G`, with units preserved through the cube root.

This is a standard circular-orbit scale evaluated locally along a mildly
eccentric, nonplanar orbit. It is **not** an exact conserved Jacobi boundary
on the actual orbit. Vertical modulation and noncircular rotating-frame terms
are not included in this baseline approximation. The actual field at (R,z) is
saved to avoid conflating it with the midplane model used for r_J. Small |z|
and e≈0.097 motivate a diagnostic approximation but do not make it exact.
See the installed galpy frequency APIs and
[galpy potential tutorial](https://docs.galpy.org/en/v1.12.0/tutorials/potentials/introduction.html).
An independent numerical check uses galpy `rtide` **at z=0 only**, where its
spherical radial derivative and this cylindrical midplane expression coincide.
No general off-plane equivalence is assumed.

## 06B — Present scale and mass sensitivity

Save every ingredient, the adopted mass and r_J, plus signed/absolute difference
from 22.65 pc and fractional difference `(r_J-22.65)/22.65`. At fixed D,
`r_J proportional to M^(1/3)`. The literature mass has no assigned error bar.
The comparison does not tune the field, mass or Galactic frame.

## 06C — Candidate extent

Reuse `stock2_cluster_relative_phase_space.ecsv` and its frozen 03C centre.
The spatial norm is the inherited **3D** separation, independently checked from
(dx,dy,dz). All canonical source IDs, row order, C/L/T and catalogue provenance
are preserved. Geometric zones are `r < r_J`, `r_J <= r < 2 r_J`, and
`r >= 2 r_J`; missing positions would remain `unavailable`. Counts and fractions
use calculable positions within each full/class/provenance group, with total and
missing counts recorded explicitly. No quality or membership cuts are added.

The XY plot projects the 3D sample in the original 03B Galactic Cartesian axes;
circles indicate projected spherical scale guides, not membership boundaries
or a spherical model of the actual Roche surface. Zone assignments always use
3D radii. All sources and the full radial range remain visible. Histograms and
class ECDFs use 3D radius, not sky-projected separation. Candidate distance errors
and the inherited inverse-parallax assumptions are not propagated here.

## 06D — Orbital variation

Evaluate all 4001 stored Project 05 points, -500 to +500 Myr at 0.25 Myr spacing.
No orbit reintegration. Mass remains 4000 solar masses. The field table stores
actual R,z diagnostics and the midplane denominator at each R. r_J(t) therefore
varies with R, not directly with z; this limitation is explicit in both metadata
and plots. Report min/max/median/current r_J, and r_J at the sampled global
cylindrical pericentre/apocentre with their times and R. The strongest effective
midplane denominator must coincide with the smallest r_J.

## Reproduction and outputs

From the repository root:

```sh
MPLCONFIGDIR=/private/tmp/stock2-mpl .venv/bin/python src/project06/01_tidal_field.py
MPLCONFIGDIR=/private/tmp/stock2-mpl .venv/bin/python src/project06/02_jacobi_radius.py
MPLCONFIGDIR=/private/tmp/stock2-mpl .venv/bin/python src/project06/03_candidate_extent.py
MPLCONFIGDIR=/private/tmp/stock2-mpl .venv/bin/python src/project06/04_orbital_variation.py
MPLCONFIGDIR=/private/tmp/stock2-mpl .venv/bin/python -m unittest discover -s tests -v
```

Six unit-bearing ECSV files under `data/interim/project06/`:
`stock2_tidal_field`, `stock2_jacobi_radius`, `stock2_jacobi_mass_sensitivity`,
`stock2_candidate_tidal_zones`, `stock2_candidate_tidal_zone_counts`, and
`stock2_orbital_jacobi_radius`.

`results/project06/` contains the summary, metadata and upstream SHA256 manifest.
Six PNG/PDF pairs: `jacobi_radius/mass_sensitivity`,
`candidate_extent/candidate_xy`, `candidate_extent/radial_distribution`,
`candidate_extent/class_radial_distribution`, `orbital_variation/jacobi_time`,
`orbital_variation/jacobi_R`. Numerical results and validation are recorded in
`project06_summary.txt`, `project06_metadata.json`, and `verification.md`.
Intermediate ECSV files follow the existing ignore rules and are reproducible
locally; no ignore rules are changed.

## Validation and interpretation limits

Check finite positive D and r_J, both denominator identities, physical-unit
conversion, galpy midplane rtide agreement, analytic point-mass limit, monotonic
cube-root mass scaling, exact zone boundaries/missing-data handling, class sums,
time-zero consistency, strongest-field/minimum-radius correspondence, all
required outputs, and upstream file hashes. A legacy Project 05 sensitivity
test is maintained to protect scientific baseline data without permanently
freezing the README against authorized later-stage status updates. Historical
Project 05 audit products remain unchanged.

No source-level escape/boundness claim, confirmed tail, disruption history,
membership optimization, candidate orbit, mass-loss history, or alternative
Galactic potential is inferred. Geometric position outside r_J alone does not
establish any of those. Uncertainty propagation remains future work; this
Project 06 implements the user-specified tidal-environment scope.

## Numerical results and review

The current local midplane frequencies are Omega_c=26.115074765,
kappa=34.914724181, and nu=70.635587774 km/s/kpc. Phi_RR=-826.953425365
and D=1508.950555359 (km/s/kpc)^2. At the actual (R,z), F_R=-5717.40373018
and F_z=-55.14352306 (km/s)^2/kpc; actual Phi_RR=-826.916808909
(km/s/kpc)^2 is saved as an auxiliary quantity, not used in the denominator.

The present r_J is **22.506883190 pc**. Compared with 22.65 pc, the signed
difference is -0.143116810 pc (-0.6318623%); the absolute difference is
0.143116810 pc. Agreement is not an independent validation of either mass
or the physical accuracy of the circular approximation.

| Sensitivity mass [solar masses] | r_J [pc] |
|---|---:|
| 2000 | 17.863725 |
| 3000 | 20.448860 |
| 4000 | 22.506883 |
| 5000 | 24.244805 |
| 6000 | 25.763950 |

| Population | r < r_J | r_J <= r < 2r_J | r >= 2r_J |
|---|---:|---:|---:|
| All 1456 | 1005 (69.02%) | 290 (19.92%) | 161 (11.06%) |
| C (940) | 850 (90.43%) | 75 (7.98%) | 15 (1.60%) |
| L (184) | 78 (42.39%) | 62 (33.70%) | 44 (23.91%) |
| T (332) | 77 (23.19%) | 153 (46.08%) | 102 (30.72%) |

All 1456 positions are calculable. Catalogue-provenance counts and fractions
are also stored in the ECSV summary. Orbital r_J spans **21.721921706–24.952077825
pc**, with median **23.565541674 pc**. The sampled global pericentre is at
-402.0 Myr, R=7.969363343 kpc, r_J=21.721921706 pc; sampled apocentre is at
+438.75 Myr, R=9.676913611 kpc, r_J=24.952077825 pc. Peak-to-peak scale
variation is 14.35% of the present scale. The strongest effective denominator
occurs at the smallest radius/scale in this model.

The full suite passes **80 tests**, including 14 new Project 06 tests.
Six PNG plots were visually checked; PDF pairs are also present. See the
verification record for the protected-file audit and complete file inventory.
No commit or push was performed.
