# Project 07 — Cluster-centric Morphology & Tidal Geometry

## Scope and inherited inputs

Describe the existing Stock 2 populations in physical space and compare their
axes with Galactic/orbital directions and the inertial Galactic tidal tensor.
The 1456-source catalogue, C/L/T=940/184/332, provenance, frozen 03C→04A centre,
Project 05 orbit and Project 06 Jacobi scale/zones remain unchanged.

Inputs: `data/interim/project04/stock2_galactocentric_phase_space.ecsv`,
`data/interim/project05/stock2_cluster_reference.ecsv`, and Project 06
`stock2_candidate_tidal_zones.ecsv` / `stock2_jacobi_radius.ecsv`.
The 04C analytic tensor implementation is reused read-only. No angular RA/Dec
PCA, distance recalculation, candidate orbit or new membership cut is performed.

## 07A — Local physical frame

For frozen Astropy Galactocentric reference (X,Y,Z), R=hypot(X,Y), define
basis columns

    e_R   = (X/R, Y/R, 0)       outward cylindrical radial
    e_pro = (Y/R, -X/R, 0)      local prograde tangential
    e_Z   = (0, 0, 1)          Galactic north

This preferred outward/prograde/north basis is **left-handed** relative to
Astropy's right-handed Cartesian axes: B^T B=I, det(B)=-1,
e_R cross e_pro=-e_Z. This is intentional, not a sign error. The local prograde
axis is minus Astropy's increasing-azimuth e_phi, consistent with reviewed
Project 05 negative Astropy Lz and positive galpy Lz.

For each star, local coordinates are `B^T (r_star-r_reference)` in pc. These are
linear projections on fixed axes, not differences in cylindrical radius or arc
length. Source identities/order, labels and existing 06 zones are checked
against input tables. Norms must match inherited 3D separation to 1e-8 pc.
This uses existing transformed positions; no parallax or sky-coordinate
transformation is rerun. Display circles show projected scale references, not
physical membership boundaries or spherical Roche surfaces.

## 07B — Shape, radial leverage, and bootstrap

Populations: full, C, L, T, combined L+T, inner r<r_J, intermediate
r_J<=r<2r_J, outer r>=2r_J, and all outside r_J. Shells and classes are
conditional descriptive groups; the catalogue itself is never pruned.

Standard PCA uses unweighted mean-centred sample covariance with ddof=1.
Eigenvalues are descending; a,b,c are their square roots (RMS one-sigma spatial
widths, not ellipsoid endpoints), with b/a and c/a. PCA group centroids do not
replace the frozen coordinate origin. A relative top eigenvalue gap below
1e-8 flags an undefined major direction; the pipeline stops rather than
inventing an orientation. Eigenvector display signs make their largest absolute
component positive; inference uses axes without signs.

The radial-leverage comparison uses **spatial-sign scatter**: component-median
centring, then normalize every nonzero displacement to unit length. Zero vectors
contribute zero, and all rows remain in the denominator. This bounds radial
leverage; it is a dimensionless angular-shape comparison, **not a robust spatial
size estimate**. Component medians are not fully rotation-equivariant; this
comparison is defined specifically in the documented local physical basis.
The standard/robust angle and shell dependence are reported, not hidden.

Additionally report the share of covariance trace supplied by the top ceil(1% N)
centred squared distances and the largest major-axis change among all
leave-one-out refits. These influence calculations do not remove catalogue rows
or adopt any leave-one-out solution. They distinguish orientation sensitivity
from simply noticing that a few distant objects contribute variance.

Bootstrap: 1000 independent with-replacement refits per fixed population,
base seed 20260921, population seeds base+index in the documented group order.
All 9000 unit major axes and eigenvalue gaps are saved. Axial consensus comes
from the largest eigenvector of mean(v v^T); do not average signed vectors.
Deviations use arccos(abs(dot)) and report 50/68/95 percentiles around the
original axis. Directional and tidal alignment distributions report
2.5/16/50/84/97.5 percentiles. These measure conditional catalogue-sampling
stability, **not** Gaia measurement uncertainty, a formal significance test, or
an isotropic-null rejection. Bootstrap does not eliminate systematic selection
or distance errors. Broad distributions and robust/shell rotation must temper
any apparent point-estimate alignment.

## 07C — Galactic/orbital geometry

Save explicit local and Astropy components for outward/inward **planar** radial,
true 3D Galactic-centre toward/away, prograde tangent, instantaneous velocity,
and vertical directions. The true GC direction includes the small reference
height, and is not silently identified with planar -e_R. Tangent and normalized
full 3D velocity are distinct. Major-axis angles use acute axial dot products
in [0,90] degrees. Opposite GC directions have the same axial angle.

Leading-side/trailing-side geometry is the sign of delta-position dot the
instantaneous velocity unit vector; radial inward/outward uses delta dot e_R.
A 1e-10 pc on-plane tolerance is explicit. These half-spaces are not the
literature L/T definitions and are never interpreted as escaped or dynamically
confirmed tail members. Full/class/shell counts retain on-plane cases.

## 07D — Local tidal principal axes

At the actual present (R,z), compute the **inertial gravitational tensor**

    T_ij = partial a_i / partial x_j = -partial_i partial_j Phi.

Phi is Galactic potential per unit test mass. Positive eigenvalues stretch;
negative ones compress. For the orthonormal radial/azimuthal/vertical basis,
the Hessian has entries Phi_RR, Phi_R/R=-F_R/R, Phi_ZZ and symmetric Phi_RZ.
Use the unchanged Project 04C analytic galpy derivatives and MWPotential2014,
galpy 1.12.0, ro=8 kpc, vo=220 km/s. This agrees with the Project 06 environment.
No homemade potential or derivative approximation enters the production tensor.
A force finite-difference audit is used only as an independent test.

Transform this symmetric rank-2 tensor with `T_local=B^T T_Astropy B`.
This rule remains valid for det(B)=-1; it is not the axial-vector cross-product
rule for angular momentum. Units are (km/s/kpc)^2, eigenvalues descending.
Eigenvectors are provided in both bases. Every spatial component is compared
with each tidal axis, with major-axis bootstrap distributions also saved.

This is **not** an effective rotating-frame tensor: no centrifugal or Coriolis
terms. Its eigenvalues are not Project 06's effective circular-midplane Jacobi
denominator. The tensor is at actual z, while the inherited Jacobi scale is a
midplane approximation; these are separate diagnostics. A local tensor is not
assumed constant over the full extended sample. See
[galpy potential machinery](https://docs.galpy.org/en/v1.12.0/tutorials/potentials/introduction.html).

## Run and outputs

From the repository root, with existing dependencies:

```sh
MPLCONFIGDIR=/private/tmp/stock2-mpl .venv/bin/python src/project07/01_clustercentric_frame.py
MPLCONFIGDIR=/private/tmp/stock2-mpl .venv/bin/python src/project07/02_morphology.py
MPLCONFIGDIR=/private/tmp/stock2-mpl .venv/bin/python src/project07/03_directional_geometry.py
MPLCONFIGDIR=/private/tmp/stock2-mpl .venv/bin/python src/project07/04_tidal_axis_comparison.py
MPLCONFIGDIR=/private/tmp/stock2-mpl .venv/bin/python -m unittest discover -s tests -v
```

Ten ECSV tables in `data/interim/project07/`: clustercentric_geometry,
morphology_axes, orientation_bootstrap, direction_vectors, directional_angles,
geometric_halfspaces, geometric_side_counts, tidal_tensor_axes, tidal_axis_angles,
and tidal_alignment_bootstrap (all filenames prefixed `stock2_`). The morphology
row unit fields explicitly distinguish covariance pc²/RMS pc from dimensionless
spatial-sign scatter. Inputs/software/frame definitions are saved in metadata.

Results include summary, JSON metadata, shape/influence/axial-bootstrap summaries,
upstream hashes and verification record. Six PNG/PDF pairs: three local
projections, outer population, principal-axis comparison, bootstrap orientation
CDF. Vector display lengths are arbitrary and documented; projected components
are never renormalized to imply stronger in-plane alignment. Existing interim
Git ignore rules are unchanged.

## Interpretation limits

Elongation and acute angles quantify morphology. Geometric alignment alone
does not establish stripping, an escaped/unbound candidate, confirmed tails,
or a unique Galactic-tide mechanism. Class definitions, distance systematics,
reference assumptions and baseline potential are inherited; their uncertainty
and robustness are not inferred by a catalogue bootstrap. All alignment claims
must account for the robust comparison, eigenvalue gaps and radial dependence.
