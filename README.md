# 🔭 Stock 2 Star Cluster Dynamics

### Gaia DR3 phase-space and dynamical analysis of the extended Stock 2 population

![Status](https://img.shields.io/badge/status-active%20research-2ea44f)
![Gaia](https://img.shields.io/badge/data-Gaia%20DR3-4c72b0)
![Python](https://img.shields.io/badge/python-3.x-3776ab)
![Astropy](https://img.shields.io/badge/Astropy-8.0.1-e67e22)
![Sample](https://img.shields.io/badge/canonical%20sample-1456-6f42c1)
![Stage](https://img.shields.io/badge/current%20stage-Project%2006-f0ad4e)

> **Are the literature-defined extended and tidal-tail candidates around Stock 2 dynamically consistent with the cluster in Gaia 6D phase space and Galactic dynamics?**

A reproducible investigation of the extended stellar population associated with **Stock 2**, combining literature catalogues, **Gaia DR3 astrometry and radial velocities**, Galactocentric phase-space modelling, and progressively more stringent dynamical tests.

The project is built around three principles:

**provenance · reproducibility · evidence-bounded interpretation**

---

## 🔭 Scientific Question

Open clusters gradually lose stars through internal evolution and interaction with the Galactic tidal field. Gaia has made extended structures around nearby clusters increasingly visible, but spatial or proper-motion similarity alone does not demonstrate a common dynamical origin.

This project therefore asks:

> **Are the published extended / tidal-tail candidate populations around Stock 2 genuinely consistent with the cluster once they are tested in full phase space and Galactic dynamics?**

The analysis follows the chain:

    Literature candidates
            ↓
    Catalogue reconstruction
            ↓
    Gaia DR3 measurements
            ↓
    Observed phase space
            ↓
    Cartesian phase space
            ↓
    Galactocentric dynamics
            ↓
    Orbit / action analysis
            ↓
    Uncertainty propagation
            ↓
    Robustness tests
            ↓
    Astrophysical interpretation

---

## 🛰️ Data Foundation

The project starts from two published Stock 2 candidate catalogues and reconstructs their complete source-level provenance.

### Canonical literature population

| Quantity | Value |
|---|---:|
| Literature catalogue rows | **2341** |
| Unique Gaia DR3 sources | **1456** |
| Shared between catalogues | **885** |
| Kos-only | **178** |
| Risbud-only | **393** |
| Catalogue Jaccard similarity | **0.6078** |

Homogenized literature classes:

| Class | N |
|---|---:|
| C | **940** |
| L | **184** |
| T | **332** |

> **Canonical invariant:** `N = 1456`

The canonical population is preserved throughout the diagnostic stages. Quality diagnostics, missing radial velocities, or later dynamical measurements do not silently redefine the original literature sample.

---

## 📡 Gaia DR3 Coverage

All canonical sources are recovered through exact Gaia DR3 `source_id` matching.

    Canonical sources       1456
    Exact Gaia matches      1456
    Identity audit          PASS

### 6D completeness

| Population | RV completeness |
|---|---:|
| Full canonical sample | **60.85%** |
| Shared sources | **65.59%** |
| Risbud-only | **58.78%** |
| Kos-only | **42.13%** |

    Sources with radial velocity      886
    Sources with full 6D data         886

No source is removed merely because it lacks radial velocity information.

---

## 🔬 Gaia Quality Diagnostics

Gaia quality indicators are treated first as **diagnostics**, not automatic membership cuts.

Representative astrometric statistics:

    RUWE median                    1.0111
    RUWE p95                       1.4792
    RUWE maximum                  11.43

    parallax_error median          0.02309 mas
    pmra_error median              0.01972 mas yr⁻¹
    pmdec_error median             0.02369 mas yr⁻¹

    parallax_over_error median   116.94

Photometric diagnostics include the colour-corrected BP/RP flux-excess statistic:

    C* valid              1456 / 1456
    median                 0.000545
    p05                   -0.006196
    p95                    0.028644
    p99                    0.292119
    maximum                1.190211

No photometric membership threshold is imposed at this stage.

---

## 🧭 Phase-Space Analysis

The project progressively moves from directly observed quantities to physically interpretable stellar phase space.

### ✅ Project 03A — Observed phase space

The complete literature population is examined in:

- RA
- Dec
- parallax
- proper motion in RA
- proper motion in Dec
- radial velocity where available

No clipping, membership pruning, or outlier rejection is introduced.

### ✅ Project 03B — Barycentric Cartesian phase space

Gaia observables are transformed into a local Cartesian representation.

This stage connects Gaia measurement space to later dynamical analysis while keeping the original membership information untouched.

---

## 🌌 Galactic Dynamics

### ✅ Project 04A — Galactocentric transformation

The full 6D-capable population is transformed into a frozen Galactocentric frame.

Adopted parameters:

    galcen_distance = 8.122 kpc
    z_sun           = 20.8 pc

    galcen_v_sun =
        [12.9, 245.6, 7.78] km s⁻¹

    Galactic-centre ICRS:
        RA  = 266.4051 deg
        Dec = -28.936175 deg

    roll = 0 deg

Reference environment:

    Astropy 8.0.1

Freezing these assumptions ensures that downstream numerical results cannot change silently because of coordinate-default updates.

### ✅ Projects 04B–04C

The subsequent Project 04 stages establish the dynamical comparison framework required before orbit-level modelling.

Detailed interpretation is intentionally separated from the coordinate transformation itself so that later scientific claims remain traceable to explicit dynamical tests.

---

## ✅ Project 05 — Stock 2 Galactic Orbit: COMPLETE

Three reproducible stages export the inherited 03C/04A cluster reference (940 C
position contributors; 594 C velocity contributors), integrate one orbit over
−500 to +500 Myr, and generate orbital diagnostics. The baseline remains the
`galpy.MWPotential2014` already introduced in 04B/04C; no earlier products are
replaced. DOP853 uses a 0.25 Myr output cadence with explicit tolerances.

Cylindrical pericentre/apocentre: **7.96936 / 9.67691 kpc**;
eccentricity: **0.096765**; maximum |Z|: **103.991 pc**.
These describe the cluster reference environment, not candidate membership.
The canonical **1456** sources and existing labels remain unchanged.

See [Project 05 methods and reproduction](docs/project05_stock2_galactic_orbit.md),
[numerical summary](results/project05/project05_summary.txt), and
[verification record](results/project05/verification.md).

---

## ✅ Project 06 — Galactic Tidal Field & Jacobi Radius: COMPLETE

Four auditable stages evaluate the Galactic field, the present Jacobi scale,
the existing candidates’ geometric extent, and variation along the stored
Project 05 orbit. The adopted mass is the approximate **4000 solar-mass**
literature baseline from Ye et al. (2021), not a sum of repository stars.

In the explicit circular-equivalent midplane approximation, **r_J now =
22.5069 pc**, compared with Ye et al.’s 22.65 pc. At fixed mass the scale ranges
from **21.7219 to 24.9521 pc** along the stored orbit. All **1456** sources remain;
spatial zones are descriptive and do not determine membership or escape.
The 2000–6000 solar-mass sensitivity grid is not an uncertainty interval.

See [Project 06 methods and limitations](docs/project06_tidal_field_jacobi_radius.md),
[numerical results](results/project06/project06_summary.txt), and
[verification](results/project06/verification.md). Uncertainty propagation
remains future work; this stage follows the requested tidal-environment scope.

---

## 🧠 Sample Philosophy

Three concepts are deliberately kept separate:

    literature candidate
            ≠
    high-quality Gaia measurement
            ≠
    dynamically validated association

A star can appear in a published Stock 2 catalogue without being dynamically consistent with the cluster.

Likewise, an apparently unusual candidate may become consistent once observational uncertainties are propagated.

For this reason, derived samples must always remain traceable to the **1,456-source canonical population**.

---

## 🧪 Reproducibility

Every major analysis stage follows the same logic:

    input
      ↓
    transformation / analysis
      ↓
    validation
      ↓
    machine-readable output
      ↓
    diagnostic figures
      ↓
    scientific interpretation

Particular attention is given to:

- Gaia source identity
- catalogue provenance
- coordinate-frame definitions
- physical units
- missing RV information
- Gaia covariance
- Solar position and motion
- software versions
- Galactic-potential assumptions
- Monte Carlo configuration

---

## 🗂️ Repository Structure

    .
    ├── data/
    │   ├── raw/
    │   ├── interim/
    │   │   ├── project01/
    │   │   ├── project02/
    │   │   ├── project03/
    │   │   └── project04/
    │   └── processed/
    │
    ├── notebooks/
    │   ├── project00/
    │   ├── project01/
    │   ├── project02/
    │   ├── project03/
    │   └── project04/
    │
    ├── results/
    │   ├── project01/
    │   ├── project02/
    │   ├── project03/
    │   └── project04/
    │
    ├── figures/
    ├── scripts/
    ├── src/
    ├── docs/
    ├── report/
    ├── tests/
    └── README.md

---

## 🛣️ Research Roadmap

| Stage | Description | Status |
|---|---|---|
| Project 00 | Catalogue feasibility | ✅ Complete |
| Project 01 | Literature reconstruction | ✅ Complete |
| Project 02 | Gaia DR3 data foundation | ✅ Complete |
| Project 03 | Observed / local phase space | ✅ Complete |
| Project 04A | Galactocentric transformation | ✅ Complete |
| Project 04B | Dynamical comparison | ✅ Complete |
| Project 04C | Extended dynamical diagnostics | ✅ Complete |
| Project 05 | Stock 2 Galactic Orbit | ✅ COMPLETE |
| Project 06 | Galactic Tidal Field & Jacobi Radius | ✅ COMPLETE |
| Project 07 | Galactic-potential robustness | ⏳ Planned |
| Project 08 | Catalogue robustness | ⏳ Planned |
| Final | Astrophysical interpretation | ⏳ Planned |

---

## 🔮 Planned Analysis

The next stages extend the deterministic phase-space analysis into uncertainty-aware Galactic dynamics.

Project 05 now provides deterministic cluster-reference eccentricity, radial
extrema, vertical excursion, angular momentum and energy diagnostics. Future
work includes uncertainty propagation, relative cluster–candidate orbital
behaviour and orbital actions where appropriate.

Gaia measurement uncertainties and covariance information will be propagated using Monte Carlo sampling.

Key conclusions will subsequently be tested under alternative plausible Milky Way potential models.

---

## ⚠️ Interpretation Boundaries

The terms **extended population** and **tidal-tail candidate** describe hypotheses inherited from the literature.

They do not by themselves demonstrate that a star:

- formed in Stock 2;
- remains gravitationally bound to Stock 2;
- escaped from Stock 2;
- belongs to a confirmed tidal tail;
- shares a uniquely demonstrated common origin with the cluster.

Such interpretations require progressively stronger dynamical evidence.

---

## 🎯 Scientific Goal

The final product is intended to provide a candidate-level answer to several related questions:

1. Where do independent Stock 2 catalogues agree?
2. Where do they disagree?
3. How coherent are their stars in Gaia phase space?
4. Which candidate populations remain dynamically consistent?
5. How much do Gaia uncertainties affect those conclusions?
6. Are the results stable against catalogue choice?
7. Are they stable against Galactic-potential assumptions?

The aim is not merely to produce another membership list.

It is to construct a **reproducible dynamical assessment of the extended Stock 2 population**.

---

## 📚 Project Context

This repository is part of:

### **Galactic Research Notes · 银河研究手记**

    Galactic Archaeology
            ↓
    Star Cluster Dynamics
            ↓
    Tidal Structure

These projects investigate how stellar populations preserve information about their formation, dynamical evolution, disruption, and redistribution throughout the Milky Way.

---

## 📖 Citation

If this repository or its derived products contribute to scientific work, please cite:

- the relevant original Stock 2 literature catalogues;
- Gaia DR3;
- the corresponding archived release or future publication associated with this project.

A formal citation entry will be added with the manuscript / archival release.

---

## 🛰️ Data Acknowledgement

This work makes use of data from the **European Space Agency (ESA) mission Gaia**, processed by the Gaia Data Processing and Analysis Consortium (DPAC).

Original catalogue authors retain credit for their published candidate samples. This repository reconstructs and dynamically analyses those populations rather than replacing the source catalogues.

---

### 🚧 Current status

**Active research · Project 06 — Galactic Tidal Field & Jacobi Radius: COMPLETE**

Next major phase:

> **uncertainty propagation → robustness → astrophysical interpretation**
