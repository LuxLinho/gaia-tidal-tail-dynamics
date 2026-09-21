# Project 05 — Stock 2 Galactic Orbit

## Scope and relationship to Project 04

05A exports the existing single cluster reference, 05B integrates it, and 05C
measures the orbital environment. Repository inspection found that Project 04B
already introduced MWPotential2014 and integrated the reference over ±1 Gyr;
04C evaluated its local tidal field. Thus the supplied specification's description
of Project 05 as the first potential stage does not match repository history.
Project 05 preserves that history and baseline, adding the requested auditable
products over ±500 Myr. Earlier source transformations and Gaia retrieval are
not repeated. The 1456-source population is never rewritten.

## Reference and inputs

- `data/interim/project03/stock2_reference_frame.ecsv`: existing 03C definition.
- `data/interim/project04/stock2_galactocentric_reference.ecsv`: exact fixed 04A
  state also used by 04B and 04C; copied without refitting.
- `data/interim/project04/stock2_galactocentric_phase_space.ecsv`: read-only
  population/count audit (1456 unique, 886 6D, C/L/T 940/184/332,
  shared/Kos-only/Risbud-only 885/178/393).
- `src/project04/galactocentric_parameters.json`: frozen frame configuration.

The reference is the component-wise Cartesian median of the already-defined C
population: 940 complete position vectors and, separately, 594 complete velocity
vectors. No clipping or new membership selection. It is a deterministic reference,
not a newly fitted centre of mass. RA, Dec, distance, pmra (including cos Dec),
pmdec and RV are an ICRS back-transform of this single inherited 03C state, not
separate medians of observed quantities. A single-state round-trip checks agreement
with 04A. No per-source coordinate transformation is rerun. t=0 represents the
inherited Gaia DR3 reference state, not the date the software is executed.

The frame remains Astropy Galactocentric: galcen_distance=8.122 kpc,
z_sun=20.8 pc, galcen_v_sun=[12.9,245.6,7.78] km/s,
galcen_coord ICRS=(266.4051,-28.936175) deg, roll=0 deg.
All parameters, versions and input SHA256 hashes are stored in output metadata.

## Potential and integration

Reuse installed galpy 1.12.0 and its unmodified `MWPotential2014` (Bovy 2015).
No new dependency and no custom force law. Model scales are ro=8 kpc and
vo=220 km/s, distinct from the solar coordinate-frame assumptions.
Components: cutoff power-law bulge (alpha=1.8, rc=1.9 kpc, normalization=.05),
Miyamoto–Nagai disk (a=3 kpc, b=.28 kpc, normalization=.60), and NFW halo
(a=16 kpc, normalization=.35). Normalizations are circular-force fractions at ro;
there are no non-default potential parameters. See the
[galpy potential documentation](https://docs.galpy.org/en/latest/reference/potential.html)
and [integration API](https://docs.galpy.org/en/latest/reference/orbitint.html).

Integration uses adaptive `dop853`, rtol=atol=1e-11 in galpy natural coordinates,
with independent integrations from t=0 to -500 and +500 Myr. Stored cadence is
0.25 Myr (4001 merged rows with one exact t=0), not the adaptive internal step.
The established bridge is x_galpy=-x_Astropy, vx_galpy=-vx_Astropy, with y,z and
vy,vz unchanged; vT_galpy=-vphi_Astropy. All saved coordinates are restored to
Astropy axes. Lz=X Vy-Y Vx is therefore negative for local prograde rotation.

## Diagnostics and validation

Pericentre/apocentre use **cylindrical R**, not spherical radius;
e=(R_apo-R_peri)/(R_apo+R_peri). z_max=max(abs(z)). These are sampled extrema
over the stated interval. A 0.125 Myr rerun in the same potential checks extrema
and eccentricity differences <1e-5 (kpc or dimensionless respectively). This is
numerical resolution validation, not potential robustness.

Radial/vertical periods use median successive R-minimum/Z-maximum spacings,
requiring at least three extrema. Azimuthal period is duration divided by total
unwrapped turns, requiring two turns. Unavailable estimates are JSON null and
ECSV NaN, explicitly described in metadata. Periods are descriptive estimates,
not exact fundamental frequencies; no uncertainties are inferred.

Checks cover finite unit-bearing 6D states, inherited frame and reference
provenance, t=0 agreement, ordered time coverage, position/velocity continuity,
energy and Lz relative drift <1e-7, physical diagnostic bounds, serialized
products, and upstream SHA256 preservation. Energy is meaningfully conserved
because this baseline is time-independent; numerical conservation is not exact.

## Run and products

From the repository root, in the existing environment:

```sh
MPLCONFIGDIR=/private/tmp/stock2-mpl .venv/bin/python src/project05/01_cluster_reference.py
MPLCONFIGDIR=/private/tmp/stock2-mpl .venv/bin/python src/project05/02_galactic_orbit.py
MPLCONFIGDIR=/private/tmp/stock2-mpl .venv/bin/python src/project05/03_orbital_diagnostics.py
MPLCONFIGDIR=/private/tmp/stock2-mpl .venv/bin/python -m unittest discover -s tests -v
```

`data/interim/project05/` contains `stock2_cluster_reference.ecsv`,
`stock2_galactic_orbit.ecsv`, and `stock2_orbital_diagnostics.ecsv`, with units.
`results/project05/` contains `cluster_reference_metadata.json`,
`cluster_reference_summary.txt`, `orbit_integration_metadata.json`,
`orbital_diagnostics.json`, `upstream_integrity.json`, and `project05_summary.txt`.
The summary and JSON carry numerical results and validation evidence.
`results/project05/orbital_diagnostics/` contains PNG/PDF pairs:
`cluster_reference_orbit_xy`, `cluster_reference_orbit_RZ`,
`cluster_reference_orbit_R_time`, `cluster_reference_orbit_Z_time`.
XY has equal scales; RZ intentionally labels Z in pc and R in kpc to resolve
vertical oscillations and should not be read as an equal-aspect orbit geometry.
Existing ignore rules exclude interim data; no ignore rules are changed.

## Interpretation limits

This is one deterministic cluster reference orbit in one smooth, static,
axisymmetric potential. No candidate orbits, new membership cuts, candidate
classification, uncertainty propagation or potential robustness are included.
A disk-confined, mildly eccentric reference trajectory is not evidence of tidal
disruption, confirmed tails, or escape of individual stars. The inherited
separate position/velocity contributor sets and finite integration window remain
scientific caveats.

## Verified results

All 59 unittest tests passed (13 Project 05 tests; 46 pre-existing tests), with
no skips or failures. Four PNG figures were visually inspected. The verification
record and full test log are under `results/project05/`.

| Quantity | Value |
|---|---:|
| R now | 8.383461 kpc |
| z now | 11.059883 pc |
| Galactocentric speed now | 228.865899 km/s |
| Cylindrical pericentre | 7.969363 kpc |
| Cylindrical apocentre | 9.676914 kpc |
| Eccentricity | 0.09676547 |
| Maximum absolute z | 103.990988 pc |
| Lz (Astropy convention) | -1906.265144 kpc km/s |
| Radial period estimate | 186.75 Myr |
| Vertical period estimate | 94.75 Myr |
| Mean azimuthal period estimate | 250.983833 Myr |

Maximum relative energy/Lz drift is 9.17e-12 / 3.72e-12. Halving output cadence
changes radial extrema by less than 0.004 pc. These are numerical checks, not
physical uncertainty estimates. All 263 protected upstream files retained their
SHA256 values. The orbit is mildly eccentric and stays close to the disk plane
in this baseline model.

## Reference-state sensitivity

A focused audit uses exactly the 594 existing C-class sources with complete
finite 6D vectors for BOTH position and velocity component-wise medians.
Medians are computed in the original 03B barycentric Galactic axes before
transforming the single reference through the frozen 04A frame. Component-wise
medians are not generally rotation invariant. No membership definition changes.

Run `MPLCONFIGDIR=/private/tmp/stock2-mpl .venv/bin/python src/project05/reference_sensitivity.py`.
The audit imports unchanged 05B integration and 05C diagnostics, requiring
galpy 1.12.0 and identical software, potential, frame and integration metadata:
MWPotential2014, ro=8 kpc, vo=220 km/s, ±500 Myr, 0.25 Myr output cadence,
DOP853, rtol=atol=1e-11. Results are saved under
`results/project05/reference_sensitivity/reference_state_sensitivity.{txt,json}`.
JSON includes all 594 source IDs as strings, input hashes, both initial states,
unit-bearing diagnostics, signed/absolute/relative differences, convention
checks and protected-file hashes. The alternative never replaces the baseline.

| Quantity | Frozen baseline | Coherent 594 | Signed change |
|---|---:|---:|---:|
| x [kpc] | -8.379057121 | -8.378994624 | +0.000062497 |
| y [kpc] | 0.271703995 | 0.271714133 | +0.000010138 |
| z [pc] | 11.059883 | 10.962475 | -0.097408 |
| vx [km/s] | -17.788100 | -17.788100 | 0 |
| vy [km/s] | 228.080345 | 228.080345 | 0 |
| vz [km/s] | -6.522223 | -6.522223 | 0 |
| R now [kpc] | 8.383461177 | 8.383399041 | -0.000062135 |
| Speed now [km/s] | 228.865899 | 228.865899 | roundoff only |
| Pericentre [kpc] | 7.969363343 | 7.969290484 | -0.000072859 |
| Apocentre [kpc] | 9.676913611 | 9.676834836 | -0.000078775 |
| Eccentricity | 0.096765469 | 0.096765965 | +0.000000496 |
| z_max [pc] | 103.990988 | 103.973971 | -0.017017 |
| Astropy Lz [kpc km/s] | -1906.265144 | -1906.250710 | +0.014435 |
| Radial period [Myr] | 186.75 | 186.75 | 0 at stored cadence |
| Vertical period [Myr] | 94.75 | 94.75 | 0 at stored cadence |
| Azimuthal period [Myr] | 250.983833 | 250.981529 | -0.002304 |

The initial position displacement is about 0.1162 pc; velocity is unchanged.
Among the nonzero envelope/period/Lz/speed/R quantities the largest fractional
change is z_max, 0.01636%. Instantaneous z changes by 0.88073% of its small
present value, but only 0.09367% of the 104 pc vertical envelope. Radial extrema
shift by <0.079 pc (<0.001%); eccentricity changes by 4.96e-7. These changes
are negligible for the mildly eccentric, near-plane, prograde disk-orbit
conclusions. Retain the frozen reference; this audit does not establish an
optimal centre or estimate physical uncertainty. Identical sampled radial and
vertical periods do not imply identical fundamental frequencies.

Explicit conservative review gates are <1% changes relative to physical orbit
scales (instantaneous z is normalized to baseline z_max), and absolute
eccentricity change <0.001. These are review triggers, not membership cuts or
statistical confidence limits. A failed gate or unavailable required comparison
writes the audit and stops for scientific review, without changing the baseline.

**Conventions.** Astropy is right-handed, with the Sun on negative X and
rotation near positive Y. The galpy Galactic convention used here is
left-handed relative to these physical axes: `(x,y,z)_g=(-x,y,z)_A`, with the
same reflection applied to velocity. The matrix has determinant -1: a reflection,
not a proper rotation. `phi_g=pi-phi_A` modulo 2pi, `vR_g=vR_A`, and
`vT_g=-vphi_A`. Stored `Lz_A=X*Vy-Y*Vx=R*vphi_A` is negative for local prograde
rotation; galpy's scalar `Lz_g=R*vT_g=-Lz_A` is positive. Angular momentum is
an axial vector, so under reflection its numerical components obey
`L_g=det(M)*M*L_A`, not the polar-vector rule. No signs were changed.

Independent galpy `Orbit.x/y/z/vx/vy/vz`, `vR`, `vT`, and `Lz` calls verify
initialization. Both trajectories have positive galpy vT and increasing
unwrapped galpy azimuth throughout the interval. Baseline galpy vT is
+227.384025 km/s and Lz is +1906.265144 kpc km/s: physically prograde.
The reflection is appropriate for this existing axisymmetric potential; an
arbitrary non-axisymmetric model would also require its orientation transformed.
See [galpy conventions](https://docs.galpy.org/en/v1.10.0/getting_started.html)
and [Astropy's frame](https://docs.astropy.org/en/stable/api/astropy.coordinates.Galactocentric.html).
Installed galpy 1.12.0 is the numerical authority for this audit.
