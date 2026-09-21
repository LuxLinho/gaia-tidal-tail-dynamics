import numpy as np
from astropy.table import Table
from astropy import units as u

from support import (
    MODELS,
    DATA,
    RESULTS,
    read,
    save,
    circular_tidal_quantities,
    CLUSTER_MASS_MSUN,
)

rows_all = []
rows_summary = []

for name in MODELS:
    print(f"Jacobi calculation: {name}")

    orbit = read(DATA / f"stock2_orbit_{name}.ecsv")

    t = np.asarray(orbit["time_myr"], dtype=float)
    R = np.asarray(orbit["R_kpc"], dtype=float)

    q = circular_tidal_quantities(name, R)

    vc = np.asarray(q["vcirc_kms"], dtype=float)
    Omega = np.asarray(q["Omega_kms_kpc"], dtype=float)
    kappa = np.asarray(q["kappa_kms_kpc"], dtype=float)
    D = np.asarray(q["denominator_kms2_kpc2"], dtype=float)
    rj = np.asarray(q["rj_kpc"], dtype=float)

    if not np.all(np.isfinite(rj)):
        raise RuntimeError(f"{name}: non-finite Jacobi radius")

    if not np.all(rj > 0):
        raise RuntimeError(f"{name}: non-positive Jacobi radius")

    for i in range(len(t)):
        rows_all.append(
            (
                name,
                float(t[i]),
                float(R[i]),
                float(vc[i]),
                float(Omega[i]),
                float(kappa[i]),
                float(D[i]),
                float(rj[i] * 1000.0),
            )
        )

    i0 = int(np.argmin(np.abs(t)))
    iperi = int(np.argmin(R))
    iapo = int(np.argmax(R))

    rows_summary.append(
        (
            name,
            float(R[i0]),
            float(Omega[i0]),
            float(kappa[i0]),
            float(D[i0]),
            float(rj[i0] * 1000.0),
            float(np.min(rj) * 1000.0),
            float(np.max(rj) * 1000.0),
            float(np.median(rj) * 1000.0),
            float(rj[iperi] * 1000.0),
            float(rj[iapo] * 1000.0),
        )
    )

all_table = Table(
    rows=rows_all,
    names=[
        "model",
        "time_myr",
        "R_kpc",
        "vcirc_kms",
        "Omega_kms_kpc",
        "kappa_kms_kpc",
        "denominator_kms2_kpc2",
        "rj_pc",
    ],
)

all_table["time_myr"].unit = u.Myr
all_table["R_kpc"].unit = u.kpc
all_table["vcirc_kms"].unit = u.km/u.s
all_table["Omega_kms_kpc"].unit = u.km/u.s/u.kpc
all_table["kappa_kms_kpc"].unit = u.km/u.s/u.kpc
all_table["denominator_kms2_kpc2"].unit = (u.km/u.s/u.kpc)**2
all_table["rj_pc"].unit = u.pc

summary = Table(
    rows=rows_summary,
    names=[
        "model",
        "R_now_kpc",
        "Omega_now_kms_kpc",
        "kappa_now_kms_kpc",
        "denominator_now_kms2_kpc2",
        "rj_now_pc",
        "rj_min_pc",
        "rj_max_pc",
        "rj_median_pc",
        "rj_peri_pc",
        "rj_apo_pc",
    ],
)

summary["R_now_kpc"].unit = u.kpc
summary["Omega_now_kms_kpc"].unit = u.km/u.s/u.kpc
summary["kappa_now_kms_kpc"].unit = u.km/u.s/u.kpc
summary["denominator_now_kms2_kpc2"].unit = (u.km/u.s/u.kpc)**2

for q in [
    "rj_now_pc",
    "rj_min_pc",
    "rj_max_pc",
    "rj_median_pc",
    "rj_peri_pc",
    "rj_apo_pc",
]:
    summary[q].unit = u.pc

summary.meta["cluster_mass_msun"] = CLUSTER_MASS_MSUN
summary.meta["definition"] = (
    "r_J=[G M/(4 Omega_c^2-kappa^2)]^(1/3), "
    "local circular midplane approximation"
)

save(
    all_table,
    DATA / "stock2_jacobi_robustness.ecsv",
)

save(
    summary,
    DATA / "stock2_jacobi_summary_by_potential.ecsv",
)

baseline = summary[
    summary["model"] == "MWPotential2014"
][0]

comparison_rows = []

for row in summary:
    comparison_rows.append(
        (
            str(row["model"]),
            float(row["rj_now_pc"]),
            float(
                row["rj_now_pc"] - baseline["rj_now_pc"]
            ),
            float(
                (row["rj_now_pc"] - baseline["rj_now_pc"])
                / baseline["rj_now_pc"]
            ),
            float(row["rj_min_pc"]),
            float(row["rj_max_pc"]),
            float(
                row["rj_max_pc"] - row["rj_min_pc"]
            ),
        )
    )

comparison = Table(
    rows=comparison_rows,
    names=[
        "model",
        "rj_now_pc",
        "delta_rj_now_pc",
        "frac_rj_now",
        "rj_min_pc",
        "rj_max_pc",
        "rj_peak_to_peak_pc",
    ],
)

for q in [
    "rj_now_pc",
    "delta_rj_now_pc",
    "rj_min_pc",
    "rj_max_pc",
    "rj_peak_to_peak_pc",
]:
    comparison[q].unit = u.pc

save(
    comparison,
    DATA / "stock2_jacobi_comparison.ecsv",
)

out = RESULTS / "jacobi_robustness"
out.mkdir(parents=True, exist_ok=True)

with open(out / "jacobi_robustness.txt", "w") as f:
    f.write("Project 08C — Jacobi / tidal-scale robustness\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"Fixed cluster mass: {CLUSTER_MASS_MSUN:.1f} Msun\n\n")
    f.write("Summary by Galactic potential\n")
    f.write("-----------------------------\n")
    f.write(str(summary))
    f.write("\n\nRelative to MWPotential2014\n")
    f.write("--------------------------\n")
    f.write(str(comparison))
    f.write("\n")

print(summary)
print()
print(comparison)
print("\n08C PASS")
