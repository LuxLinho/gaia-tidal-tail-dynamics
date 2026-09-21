import numpy as np
from astropy.table import Table
from astropy import units as u

from support import (
    MODELS,
    DATA,
    RESULTS,
    frozen_reference_state,
    tidal_tensor_local,
    acute_axis_angle_deg,
    read,
    save,
)

PROJECT07_AXES = (
    DATA.parent
    / "project07"
    / "stock2_morphology_axes.ecsv"
)

PROJECT07_TIDAL = (
    DATA.parent
    / "project07"
    / "stock2_tidal_tensor_axes.ecsv"
)

morph = read(PROJECT07_AXES)
p07_tidal = read(PROJECT07_TIDAL)

state = frozen_reference_state()

R = float(state["R_kpc"])
z = float(state["z_kpc"])

populations = [
    "full",
    "C",
    "L",
    "T",
    "extended_LT",
    "inner",
    "intermediate",
    "outer",
    "outside_rJ",
]

tensor_rows = []
alignment_rows = []

baseline_T = None
baseline_evals = None
baseline_evecs = None

for name in MODELS:
    print(f"Tidal tensor: {name}")

    T, evals, evecs = tidal_tensor_local(
        name,
        R,
        z,
    )

    for j in range(3):
        vec = evecs[:, j]

        interpretation = (
            "stretching"
            if evals[j] > 0
            else "compression"
        )

        tensor_rows.append(
            (
                name,
                j + 1,
                float(evals[j]),
                float(vec[0]),
                float(vec[1]),
                float(vec[2]),
                interpretation,
                float(T[0, 0]),
                float(T[0, 1]),
                float(T[0, 2]),
                float(T[1, 0]),
                float(T[1, 1]),
                float(T[1, 2]),
                float(T[2, 0]),
                float(T[2, 1]),
                float(T[2, 2]),
            )
        )

    stretch = evecs[:, 0]

    for pop in populations:
        rows = morph[
            (morph["population"] == pop)
            & (morph["method"] == "covariance")
            & (morph["component"] == 1)
        ]

        if len(rows) != 1:
            raise RuntimeError(
                f"Expected exactly one covariance major axis "
                f"for population={pop}; got {len(rows)}"
            )

        row = rows[0]

        major = np.array(
            [
                float(row["e_radial"]),
                float(row["e_prograde"]),
                float(row["e_vertical"]),
            ]
        )

        angle = acute_axis_angle_deg(
            major,
            stretch,
        )

        alignment_rows.append(
            (
                name,
                pop,
                "covariance",
                1,
                angle,
                float(stretch[0]),
                float(stretch[1]),
                float(stretch[2]),
                float(major[0]),
                float(major[1]),
                float(major[2]),
            )
        )

    if name == "MWPotential2014":
        baseline_T = T.copy()
        baseline_evals = evals.copy()
        baseline_evecs = evecs.copy()

tensor_table = Table(
    rows=tensor_rows,
    names=[
        "model",
        "tidal_component",
        "eigenvalue",
        "e_radial",
        "e_prograde",
        "e_vertical",
        "interpretation",
        "T_00",
        "T_01",
        "T_02",
        "T_10",
        "T_11",
        "T_12",
        "T_20",
        "T_21",
        "T_22",
    ],
)

tensor_table["eigenvalue"].unit = (u.km/u.s/u.kpc)**2

for q in [
    "T_00",
    "T_01",
    "T_02",
    "T_10",
    "T_11",
    "T_12",
    "T_20",
    "T_21",
    "T_22",
]:
    tensor_table[q].unit = (u.km/u.s/u.kpc)**2

alignment_table = Table(
    rows=alignment_rows,
    names=[
        "model",
        "population",
        "morphology_method",
        "spatial_component",
        "angle_to_stretching_deg",
        "stretch_e_radial",
        "stretch_e_prograde",
        "stretch_e_vertical",
        "major_e_radial",
        "major_e_prograde",
        "major_e_vertical",
    ],
)

alignment_table["angle_to_stretching_deg"].unit = u.deg

save(
    tensor_table,
    DATA / "stock2_tidal_tensor_robustness.ecsv",
)

save(
    alignment_table,
    DATA / "stock2_alignment_robustness.ecsv",
)

p07_evals = np.array(
    [
        float(
            p07_tidal[
                p07_tidal["component"] == component
            ][0]["eigenvalue"]
        )
        for component in [1, 2, 3]
    ]
)

p07_vecs = np.array(
    [
        [
            float(
                p07_tidal[
                    p07_tidal["component"] == component
                ][0]["e_radial"]
            ),
            float(
                p07_tidal[
                    p07_tidal["component"] == component
                ][0]["e_prograde"]
            ),
            float(
                p07_tidal[
                    p07_tidal["component"] == component
                ][0]["e_vertical"]
            ),
        ]
        for component in [1, 2, 3]
    ]
).T

eval_diff = baseline_evals - p07_evals

axis_diff = np.array(
    [
        acute_axis_angle_deg(
            baseline_evecs[:, i],
            p07_vecs[:, i],
        )
        for i in range(3)
    ]
)

print("\nProject 07 baseline reproduction")
print("--------------------------------")
print("Project07 eigenvalues:", p07_evals)
print("Project08 eigenvalues:", baseline_evals)
print("difference:", eval_diff)
print("axis-angle differences [deg]:", axis_diff)

if not np.allclose(
    baseline_evals,
    p07_evals,
    rtol=1e-8,
    atol=1e-6,
):
    raise RuntimeError(
        "MWPotential2014 tidal eigenvalues do not reproduce Project 07"
    )

if not np.all(axis_diff < 1e-5):
    raise RuntimeError(
        "MWPotential2014 tidal axes do not reproduce Project 07"
    )

for row in alignment_table:
    angle = float(row["angle_to_stretching_deg"])
    if not (0.0 <= angle <= 90.0):
        raise RuntimeError(
            f"Invalid acute alignment angle: {angle}"
        )

out = RESULTS / "tidal_geometry"
out.mkdir(parents=True, exist_ok=True)

focus = alignment_table[
    np.isin(
        np.asarray(
            alignment_table["population"],
            dtype=str,
        ),
        [
            "intermediate",
            "outer",
            "outside_rJ",
        ],
    )
]

with open(
    out / "tidal_alignment_robustness.txt",
    "w",
) as f:
    f.write(
        "Project 08D — Tidal-axis / morphology-alignment robustness\n"
    )
    f.write("=" * 72 + "\n\n")

    f.write(
        "Morphology is frozen from Project 07 covariance PCA.\n"
    )
    f.write(
        "Only the Galactic-potential tidal tensor is changed.\n\n"
    )

    f.write("Tidal tensor eigenaxes\n")
    f.write("----------------------\n")
    f.write(str(
        tensor_table[
            [
                "model",
                "tidal_component",
                "eigenvalue",
                "e_radial",
                "e_prograde",
                "e_vertical",
                "interpretation",
            ]
        ]
    ))
    f.write("\n\n")

    f.write(
        "Key morphology-to-stretching-axis angles\n"
    )
    f.write("---------------------------------------\n")
    f.write(str(
        focus[
            [
                "model",
                "population",
                "angle_to_stretching_deg",
            ]
        ]
    ))
    f.write("\n\n")

    f.write(
        "Project 07 baseline reproduction\n"
    )
    f.write("--------------------------------\n")
    f.write(
        "Eigenvalue differences: "
        + np.array2string(
            eval_diff,
            precision=12,
        )
        + "\n"
    )
    f.write(
        "Axis-angle differences [deg]: "
        + np.array2string(
            axis_diff,
            precision=12,
        )
        + "\n"
    )

print()
print(
    tensor_table[
        [
            "model",
            "tidal_component",
            "eigenvalue",
            "e_radial",
            "e_prograde",
            "e_vertical",
            "interpretation",
        ]
    ]
)

print()
print(
    focus[
        [
            "model",
            "population",
            "angle_to_stretching_deg",
        ]
    ]
)

print("\n08D PASS")
