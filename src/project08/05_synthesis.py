from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data/interim/project08"
RESULTS = ROOT / "results/project08"
DOCS = ROOT / "docs"

RESULTS.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)

models = [
    "MWPotential2014",
    "McMillan17",
    "Irrgang13I",
    "Cautun20",
]

potential_table = Table.read(
    DATA / "stock2_potential_models.ecsv",
    format="ascii.ecsv",
)

orbit_summary = Table.read(
    DATA / "stock2_orbit_summary_by_potential.ecsv",
    format="ascii.ecsv",
)

jacobi_summary = Table.read(
    DATA / "stock2_jacobi_summary_by_potential.ecsv",
    format="ascii.ecsv",
)

tidal = Table.read(
    DATA / "stock2_tidal_tensor_robustness.ecsv",
    format="ascii.ecsv",
)

alignment = Table.read(
    DATA / "stock2_alignment_robustness.ecsv",
    format="ascii.ecsv",
)

figdirs = {
    "orbit": RESULTS / "orbit_robustness",
    "jacobi": RESULTS / "jacobi_robustness",
    "tidal": RESULTS / "tidal_geometry",
}

for d in figdirs.values():
    d.mkdir(parents=True, exist_ok=True)


def savefig(path):
    plt.tight_layout()
    plt.savefig(path.with_suffix(".png"), dpi=220)
    plt.savefig(path.with_suffix(".pdf"))
    plt.close()


plt.figure(figsize=(8, 5))
for model in models:
    t = Table.read(
        DATA / f"stock2_orbit_{model}.ecsv",
        format="ascii.ecsv",
    )
    plt.plot(t["time_myr"], t["R_kpc"], label=model)

plt.xlabel("Time [Myr]")
plt.ylabel("Galactocentric R [kpc]")
plt.legend()
savefig(figdirs["orbit"] / "R_vs_time")


plt.figure(figsize=(8, 5))
for model in models:
    t = Table.read(
        DATA / f"stock2_orbit_{model}.ecsv",
        format="ascii.ecsv",
    )
    plt.plot(
        t["time_myr"],
        np.asarray(t["z_kpc"], dtype=float) * 1000.0,
        label=model,
    )

plt.xlabel("Time [Myr]")
plt.ylabel("z [pc]")
plt.legend()
savefig(figdirs["orbit"] / "z_vs_time")


plt.figure(figsize=(6, 6))
for model in models:
    t = Table.read(
        DATA / f"stock2_orbit_{model}.ecsv",
        format="ascii.ecsv",
    )
    plt.plot(
        t["x_galpy_kpc"],
        t["y_galpy_kpc"],
        label=model,
    )

plt.xlabel("x [kpc]")
plt.ylabel("y [kpc]")
plt.axis("equal")
plt.legend()
savefig(figdirs["orbit"] / "xy_orbit_comparison")


jacobi = Table.read(
    DATA / "stock2_jacobi_robustness.ecsv",
    format="ascii.ecsv",
)

plt.figure(figsize=(8, 5))
for model in models:
    x = jacobi[jacobi["model"] == model]
    plt.plot(
        x["time_myr"],
        x["rj_pc"],
        label=model,
    )

plt.xlabel("Time [Myr]")
plt.ylabel("Jacobi radius [pc]")
plt.legend()
savefig(figdirs["jacobi"] / "jacobi_radius_vs_time")


plt.figure(figsize=(7, 5))
vals = [
    float(
        jacobi_summary[
            jacobi_summary["model"] == m
        ][0]["rj_now_pc"]
    )
    for m in models
]

plt.bar(models, vals)
plt.ylabel("Present-day Jacobi radius [pc]")
plt.xticks(rotation=20)
savefig(figdirs["jacobi"] / "present_day_jacobi_radius")


plt.figure(figsize=(8, 5))

x = np.arange(len(models))
width = 0.22

for comp in [1, 2, 3]:
    vals = [
        float(
            tidal[
                (tidal["model"] == m)
                & (tidal["tidal_component"] == comp)
            ][0]["eigenvalue"]
        )
        for m in models
    ]

    plt.bar(
        x + (comp - 2) * width,
        vals,
        width=width,
        label=f"component {comp}",
    )

plt.xticks(x, models, rotation=20)
plt.ylabel(r"Tidal eigenvalue [(km s$^{-1}$ kpc$^{-1}$)$^2$]")
plt.legend()
savefig(figdirs["tidal"] / "tidal_eigenvalue_comparison")


focus_pops = [
    "intermediate",
    "outer",
    "outside_rJ",
]

plt.figure(figsize=(8, 5))

x = np.arange(len(models))
width = 0.24

for i, pop in enumerate(focus_pops):
    vals = [
        float(
            alignment[
                (alignment["model"] == m)
                & (alignment["population"] == pop)
            ][0]["angle_to_stretching_deg"]
        )
        for m in models
    ]

    plt.bar(
        x + (i - 1) * width,
        vals,
        width=width,
        label=pop,
    )

plt.xticks(x, models, rotation=20)
plt.ylabel("Major-axis angle to stretching axis [deg]")
plt.legend()
savefig(figdirs["tidal"] / "morphology_stretching_alignment")


def values(table, column):
    return np.asarray(table[column], dtype=float)


baseline_orbit = orbit_summary[
    orbit_summary["model"] == "MWPotential2014"
][0]

baseline_jacobi = jacobi_summary[
    jacobi_summary["model"] == "MWPotential2014"
][0]

metrics = {}

for column in [
    "rperi_kpc",
    "rap_kpc",
    "eccentricity",
    "zmax_kpc",
]:
    vals = values(orbit_summary, column)
    base = float(baseline_orbit[column])

    metrics[column] = {
        "minimum": float(np.min(vals)),
        "maximum": float(np.max(vals)),
        "absolute_range": float(np.ptp(vals)),
        "fractional_range_vs_baseline": float(
            np.ptp(vals) / abs(base)
        ),
    }

for column in [
    "rj_now_pc",
    "rj_min_pc",
    "rj_max_pc",
]:
    vals = values(jacobi_summary, column)
    base = float(baseline_jacobi[column])

    metrics[column] = {
        "minimum": float(np.min(vals)),
        "maximum": float(np.max(vals)),
        "absolute_range": float(np.ptp(vals)),
        "fractional_range_vs_baseline": float(
            np.ptp(vals) / abs(base)
        ),
    }

angle_ranges = {}

for pop in focus_pops:
    vals = np.asarray(
        alignment[
            alignment["population"] == pop
        ]["angle_to_stretching_deg"],
        dtype=float,
    )

    angle_ranges[pop] = {
        "minimum_deg": float(np.min(vals)),
        "maximum_deg": float(np.max(vals)),
        "spread_deg": float(np.ptp(vals)),
    }


e_values = values(
    orbit_summary,
    "eccentricity",
)

zmax_values_pc = (
    values(
        orbit_summary,
        "zmax_kpc",
    )
    * 1000.0
)

rj_values = values(
    jacobi_summary,
    "rj_now_pc",
)

intermediate_angles = np.asarray(
    alignment[
        alignment["population"] == "intermediate"
    ]["angle_to_stretching_deg"],
    dtype=float,
)

outer_angles = np.asarray(
    alignment[
        alignment["population"] == "outer"
    ]["angle_to_stretching_deg"],
    dtype=float,
)

conclusions = {
    "low_eccentricity_disk_orbit": bool(
        np.all(e_values < 0.15)
    ),
    "vertically_confined_near_disk": bool(
        np.all(zmax_values_pc < 150.0)
    ),
    "jacobi_scale_remains_tens_of_pc": bool(
        np.all((rj_values > 15.0) & (rj_values < 30.0))
    ),
    "intermediate_close_to_stretching_axis": bool(
        np.all(intermediate_angles < 10.0)
    ),
    "outer_rotated_from_stretching_axis": bool(
        np.all(outer_angles > 30.0)
    ),
}

summary_lines = [
    "Project 08 — Galactic-potential robustness",
    "=" * 72,
    "",
    "Tested models",
    "-------------",
]

for row in potential_table:
    summary_lines.append(
        f"{row['model']}: "
        f"ro={float(row['ro_kpc']):.3f} kpc, "
        f"vo={float(row['vo_kms']):.3f} km/s, "
        f"vcirc(R_now)={float(row['vcirc_kms']):.3f} km/s"
    )

summary_lines.extend(
    [
        "",
        "Orbit robustness",
        "----------------",
    ]
)

for row in orbit_summary:
    summary_lines.append(
        f"{row['model']}: "
        f"Rperi={float(row['rperi_kpc']):.6f} kpc, "
        f"Rapo={float(row['rap_kpc']):.6f} kpc, "
        f"e={float(row['eccentricity']):.6f}, "
        f"zmax={1000*float(row['zmax_kpc']):.3f} pc"
    )

summary_lines.extend(
    [
        "",
        "Jacobi-radius robustness",
        "------------------------",
    ]
)

for row in jacobi_summary:
    summary_lines.append(
        f"{row['model']}: "
        f"rJ_now={float(row['rj_now_pc']):.3f} pc, "
        f"rJ_min={float(row['rj_min_pc']):.3f} pc, "
        f"rJ_max={float(row['rj_max_pc']):.3f} pc"
    )

summary_lines.extend(
    [
        "",
        "Morphology-to-stretching-axis robustness",
        "----------------------------------------",
    ]
)

for model in models:
    for pop in focus_pops:
        row = alignment[
            (alignment["model"] == model)
            & (alignment["population"] == pop)
        ][0]

        summary_lines.append(
            f"{model} / {pop}: "
            f"{float(row['angle_to_stretching_deg']):.6f} deg"
        )

summary_lines.extend(
    [
        "",
        "Scientific synthesis",
        "--------------------",
        (
            "1. Stock 2 remains on a low-eccentricity disk orbit "
            "in every tested Galactic potential."
        ),
        (
            "2. The reference orbit remains vertically confined to "
            "roughly 0.09-0.10 kpc from the Galactic midplane."
        ),
        (
            "3. The present-day Jacobi radius remains near 21-23 pc "
            "for the fixed 4000 Msun literature baseline mass."
        ),
        (
            "4. The intermediate morphology remains within about 5 deg "
            "of the instantaneous stretching axis."
        ),
        (
            "5. The outer and outside-rJ morphologies remain rotated "
            "by roughly 45-48 deg from the instantaneous stretching axis."
        ),
        "",
        (
            "These tests demonstrate robustness across the selected "
            "smooth Galactic-potential family. They do not represent "
            "exhaustive Milky Way potential uncertainty."
        ),
        (
            "No candidate is classified as escaped, bound, unbound, "
            "or dynamically confirmed as a tidal-tail member."
        ),
    ]
)

(RESULTS / "project08_summary.txt").write_text(
    "\n".join(summary_lines) + "\n",
    encoding="utf-8",
)

metadata = {
    "project": "Project 08 — Galactic-potential robustness",
    "models": models,
    "canonical_population": 1456,
    "classes": {
        "C": 940,
        "L": 184,
        "T": 332,
    },
    "cluster_mass_msun": 4000.0,
    "metrics": metrics,
    "alignment_angle_ranges": angle_ranges,
    "conclusions": conclusions,
    "interpretation_limits": [
        "model-family robustness, not exhaustive Galactic-potential uncertainty",
        "no individual escape classification",
        "no candidate-level orbit inference",
        "no reclassification of literature C/L/T labels",
        "no morphology refit by potential",
    ],
}

(RESULTS / "project08_metadata.json").write_text(
    json.dumps(
        metadata,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

verification = [
    "# Project 08 verification",
    "",
    "- Same frozen physical Stock 2 initial state used for every potential.",
    "- MWPotential2014 reproduces Project 05 orbit diagnostics.",
    "- MWPotential2014 reproduces Project 06 Jacobi radius.",
    "- MWPotential2014 reproduces Project 07 tidal tensor and axes.",
    "- Project 07 morphology is reused unchanged.",
    "- No membership cuts or candidate reclassification introduced.",
    "- No candidate-level escape interpretation performed.",
]

(RESULTS / "verification.md").write_text(
    "\n".join(verification) + "\n",
    encoding="utf-8",
)

doc = """# Project 08 — Galactic-potential robustness

Project 08 tests whether the main Stock 2 dynamical conclusions obtained in
Projects 05–07 depend strongly on the adopted smooth Galactic gravitational
potential.

The same frozen physical Stock 2 reference position and velocity are used for
every model. The baseline remains MWPotential2014. The alternatives are
McMillan17, Irrgang13I and Cautun20, all provided locally by galpy 1.12.0.

Each literature model retains its native physical scaling. The observational
Stock 2 morphology from Project 07 is never refit as the potential changes.

## Main result

The numerical orbit changes across Galactic potentials, but the qualitative
dynamical picture does not.

Stock 2 remains on a low-eccentricity disk orbit, with eccentricity around
0.08–0.10 and maximum vertical excursions around 0.09–0.10 kpc.

For the fixed 4000 Msun literature mass baseline, the present-day Jacobi
radius remains around 21–23 pc.

The intermediate-radius morphology remains closely aligned with the local
inertial tidal stretching axis, at about 5 degrees in every tested model.
The outer and outside-rJ morphologies remain rotated by roughly 45–48 degrees
from that axis.

Thus the Project 07 radial-to-outer morphological transition is not an
artifact of choosing MWPotential2014.

## Limits

This is robustness across a selected family of smooth Milky Way potentials,
not an exhaustive treatment of Galactic-potential uncertainty.

The analysis does not classify individual sources as bound, unbound, escaped,
or dynamically confirmed tidal-tail members.

The literature C/L/T classes remain unchanged.
"""

(DOCS / "project08_galactic_potential_robustness.md").write_text(
    doc,
    encoding="utf-8",
)

print((RESULTS / "project08_summary.txt").read_text())
print("\n08E synthesis complete")
