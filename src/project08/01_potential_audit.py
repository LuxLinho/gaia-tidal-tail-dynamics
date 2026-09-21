from astropy.table import Table
from astropy import units as u

from support import (
    MODELS,
    DATA,
    RESULTS,
    frozen_reference_state,
    local_force_quantities,
    model_object,
    save,
)

state = frozen_reference_state()

rows = []

for name, cfg in MODELS.items():
    pot = model_object(name)

    comps = list(pot)
    axisymmetric = all(not bool(getattr(p, "isNonAxi", False)) for p in comps)

    local = local_force_quantities(
        name,
        state["R_kpc"],
        state["z_kpc"],
    )

    rows.append(
        (
            name,
            bool(cfg["baseline"]),
            float(cfg["ro"]),
            float(cfg["vo"]),
            bool(axisymmetric),
            cfg["provenance"],
            local["F_R_kms2_kpc"],
            local["F_z_kms2_kpc"],
            local["vcirc_kms"],
            local["Omega_kms_kpc"],
        )
    )

table = Table(
    rows=rows,
    names=[
        "model",
        "baseline",
        "ro_kpc",
        "vo_kms",
        "axisymmetric",
        "provenance",
        "F_R_kms2_kpc",
        "F_z_kms2_kpc",
        "vcirc_kms",
        "Omega_kms_kpc",
    ],
)

table["ro_kpc"].unit = u.kpc
table["vo_kms"].unit = u.km/u.s
table["F_R_kms2_kpc"].unit = (u.km/u.s)**2/u.kpc
table["F_z_kms2_kpc"].unit = (u.km/u.s)**2/u.kpc
table["vcirc_kms"].unit = u.km/u.s
table["Omega_kms_kpc"].unit = u.km/u.s/u.kpc

table.meta["principle"] = (
    "Same frozen physical Stock 2 initial state for every potential; "
    "each literature potential retains its native physical ro/vo scaling."
)

save(table, DATA / "stock2_potential_models.ecsv")

RESULTS.mkdir(parents=True, exist_ok=True)

with open(RESULTS / "potential_comparison" / "potential_audit.txt", "w") as f:
    f.write("Project 08A — Potential audit\n")
    f.write("=" * 50 + "\n\n")
    f.write(str(table))
    f.write("\n")

print(table)
print("\n08A PASS")
