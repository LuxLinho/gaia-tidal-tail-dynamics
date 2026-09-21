from pathlib import Path
import json

import numpy as np
from astropy.table import Table
from astropy import units as u

from support import (
    MODELS,
    DATA,
    RESULTS,
    integrate_symmetric,
    extrema_summary,
    frozen_reference_state,
    save,
)

all_orbits = []
summaries = []

for name in MODELS:
    print(f"Integrating {name} ...")

    orbit = integrate_symmetric(name)

    if len(orbit) != 4001:
        raise RuntimeError(f"{name}: expected 4001 orbit rows, got {len(orbit)}")

    t = np.asarray(orbit["time_myr"], dtype=float)

    if not np.any(np.isclose(t, 0.0, atol=1e-12)):
        raise RuntimeError(f"{name}: t=0 missing")

    numeric = [
        "x_galpy_kpc",
        "y_galpy_kpc",
        "z_kpc",
        "R_kpc",
        "vx_galpy_kms",
        "vy_galpy_kms",
        "vz_kms",
        "vR_kms",
        "vT_kms",
    ]

    for q in numeric:
        if not np.all(np.isfinite(np.asarray(orbit[q], dtype=float))):
            raise RuntimeError(f"{name}: non-finite values in {q}")

    summary = extrema_summary(orbit)

    if not summary["rperi_kpc"] < summary["rap_kpc"]:
        raise RuntimeError(f"{name}: invalid radial extrema")

    if not 0 <= summary["eccentricity"] < 1:
        raise RuntimeError(f"{name}: invalid eccentricity")

    path = DATA / f"stock2_orbit_{name}.ecsv"
    save(orbit, path)

    summaries.append(
        (
            name,
            summary["rperi_kpc"],
            summary["rap_kpc"],
            summary["eccentricity"],
            summary["zmax_kpc"],
        )
    )

summary_table = Table(
    rows=summaries,
    names=[
        "model",
        "rperi_kpc",
        "rap_kpc",
        "eccentricity",
        "zmax_kpc",
    ],
)

summary_table["rperi_kpc"].unit = u.kpc
summary_table["rap_kpc"].unit = u.kpc
summary_table["zmax_kpc"].unit = u.kpc

save(
    summary_table,
    DATA / "stock2_orbit_summary_by_potential.ecsv"
)

baseline = summary_table[
    summary_table["model"] == "MWPotential2014"
][0]

rows = []
for row in summary_table:
    rows.append(
        (
            str(row["model"]),
            float(row["rperi_kpc"] - baseline["rperi_kpc"]),
            float(row["rap_kpc"] - baseline["rap_kpc"]),
            float(row["eccentricity"] - baseline["eccentricity"]),
            float(row["zmax_kpc"] - baseline["zmax_kpc"]),
            float(
                (row["rperi_kpc"] - baseline["rperi_kpc"])
                / baseline["rperi_kpc"]
            ),
            float(
                (row["rap_kpc"] - baseline["rap_kpc"])
                / baseline["rap_kpc"]
            ),
            float(
                (row["eccentricity"] - baseline["eccentricity"])
                / baseline["eccentricity"]
            ),
            float(
                (row["zmax_kpc"] - baseline["zmax_kpc"])
                / baseline["zmax_kpc"]
            ),
        )
    )

delta = Table(
    rows=rows,
    names=[
        "model",
        "delta_rperi_kpc",
        "delta_rap_kpc",
        "delta_eccentricity",
        "delta_zmax_kpc",
        "frac_rperi",
        "frac_rap",
        "frac_eccentricity",
        "frac_zmax",
    ],
)

save(delta, DATA / "stock2_orbit_robustness.ecsv")

out = RESULTS / "orbit_robustness"
out.mkdir(parents=True, exist_ok=True)

with open(out / "orbit_robustness.txt", "w") as f:
    f.write("Project 08B — Orbit robustness\n")
    f.write("=" * 60 + "\n\n")
    f.write(str(summary_table))
    f.write("\n\nRelative to MWPotential2014\n")
    f.write(str(delta))
    f.write("\n")

print(summary_table)
print()
print(delta)
print("\n08B PASS")
