# Project 04C — Local Galactic gravitational tidal field

## Scope

Project 04C uses the fixed Project 04A Stock 2 Galactocentric reference, the fixed Project 03D L+T major spatial axis as stored by Project 04A (`e1_ext_gc`), and the Project 04B reference orbit. It does not alter any prior product.

The baseline Galactic potential remains `galpy.potential.MWPotential2014` with the Project 04B natural-unit scales `ro=8 kpc`, `vo=220 km/s`.

## Tensor definition

The project computes the inertial gravitational tidal tensor

\[
T_{ij}=\frac{\partial a_i}{\partial x_j}=-\frac{\partial^2\Phi}{\partial x_i\partial x_j}.
\]

A positive eigenvalue is differential stretching along that eigenvector; a negative eigenvalue is differential compression. This is deliberately **not** the effective tidal tensor in a cluster-comoving rotating frame. Centrifugal/Coriolis terms, Jacobi radii, and escape criteria are outside Project 04C.

For an axisymmetric potential, the potential Hessian in the orthonormal local cylindrical basis is

\[
H_\Phi = \begin{pmatrix}
\Phi_{RR}&0&\Phi_{Rz}\\
0&\Phi_R/R&0\\
\Phi_{Rz}&0&\Phi_{zz}
\end{pmatrix},
\]

and `T=-H_Phi`. Second derivatives come from galpy's analytic potential derivatives in natural units and are explicitly converted to `(km/s)^2/kpc^2`.

## Orbit sampling

The field is evaluated at `-50, -20, -5, 0, +5, +20, +50 Myr` along the already-stored Project 04B reference orbit. The fixed present-day L+T spatial axis is not refit at any epoch.

## Numerical checks

- tensor symmetry;
- orthonormal, right-handed local basis;
- eigenpair reconstruction;
- exact Project 04A / Project 04B present-state agreement;
- independent finite-difference audit of `Phi_RR` and `Phi_Rz` from radial forces;
- no modification of any Project 01–04B product.

## Products

- `data/interim/project04/stock2_local_tidal_tensor.ecsv`
- `data/interim/project04/stock2_tidal_eigensystem.ecsv`
- `data/interim/project04/stock2_tidal_field_summary.ecsv`
- `results/project04/local_tidal_field/`

## Interpretation guardrails

Geometric alignment with a tidal eigenvector is descriptive and does not by itself establish a tidal-tail origin. Non-alignment likewise does not reject a tidal origin. The calculation uses one smooth axisymmetric potential and excludes the rotating-frame effective tide and source-by-source dynamics.
