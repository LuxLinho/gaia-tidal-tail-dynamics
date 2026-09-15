# Project 04B — Stock 2 reference orbit integration

Project 04B integrates only the fixed one-row Stock 2 reference from Project 03C/04A. It does not integrate the 886 individual stellar velocities and does not alter membership or C/L/T labels.

The baseline Galactic potential is `galpy.potential.MWPotential2014` (Bovy 2015), with its defining scales frozen at `ro=8 kpc` and `vo=220 km/s`. The model contains a cutoff power-law bulge, Miyamoto–Nagai disk, and NFW halo with the standard 0.05/0.60/0.35 circular-force normalization fractions at the reference radius. This is a baseline axisymmetric model, not the final potential choice; robustness to alternative Galactic potentials is deferred.

The Project 04A Astropy Galactocentric convention places the Sun at negative X and has solar rotation near positive Y. The axisymmetric galpy orbit is initialized through the explicit bridge `x_galpy=-x_astropy`, `y_galpy=y_astropy`, `z_galpy=z_astropy`, implying `phi_galpy=pi-phi_astropy` and `vT_galpy=-vphi_astropy`. Every stored trajectory point is transformed back into the Project 04A Cartesian convention. The present-day trajectory sample is required to reconstruct the fixed Project 04A reference numerically.

The orbit is integrated independently backward and forward from the present for 1 Gyr each with `dop853`, with 4001 samples per half and a single shared t=0 row. Stored diagnostics include numerical pericentre, apocentre, eccentricity, maximum vertical height, a radial-period estimate from successive R minima, an azimuthal-period estimate from the median instantaneous angular rate, and energy-conservation diagnostics.

The instantaneous orbit tangent is mathematically identical to the instantaneous velocity direction, so Project 04B does not treat that as new evidence beyond Project 04A. Instead it compares the fixed Project 03D L+T spatial major axis with finite integrated orbit chords spanning ±5, ±20, and ±50 Myr around the present. These are sign-invariant geometric alignment diagnostics. They are model-dependent and do not establish a tidal origin.

No actions, Jacobi radius, escape energy, bound/unbound classification, source-by-source integration, or membership inference is performed in Project 04B.

## Dependency

Project 04B adds `galpy` as a dynamics dependency. Install in the existing project environment with:

```sh
python -m pip install galpy
```

Then run:

```sh
python src/project04/02_reference_orbit.py
```

Generated products are written under `data/interim/project04/` and `results/project04/reference_orbit/`.

References: Bovy (2015), *galpy: A Python Library for Galactic Dynamics*, ApJS 216, 29; galpy MWPotential2014 documentation.
