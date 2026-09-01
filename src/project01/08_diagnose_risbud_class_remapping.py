from pathlib import Path
from collections import Counter

import numpy as np
from astropy.table import Table


ROOT = Path(__file__).resolve().parents[2]

RISBUD_INPUT = (
    ROOT
    / "data/raw/literature/risbud2025/risbud2025_stock2.ecsv"
)

JADHAV_INPUT = (
    ROOT
    / "data/interim/project01/jadhav2025_stock2.ecsv"
)

RESULTS_DIR = ROOT / "results/project01"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MATRIX_OUTPUT = (
    RESULTS_DIR / "risbud2025_original_vs_jadhav_class_matrix.csv"
)

MISMATCH_OUTPUT = (
    RESULTS_DIR / "risbud2025_class_mismatches.csv"
)

AUDIT_OUTPUT = (
    RESULTS_DIR / "risbud2025_class_remapping_audit.txt"
)


def original_class(row):
    """
    Recover the mutually exclusive published Risbud morphology class.
    """
    c = int(row["FwithinTidRad"])
    l = int(row["FleadingTail"])
    t = int(row["FtrailingTail"])

    total = c + l + t

    if total != 1:
        raise RuntimeError(
            f"Non-exclusive Risbud flags for GaiaDR3={row['GaiaDR3']}: "
            f"C={c}, L={l}, T={t}"
        )

    if c == 1:
        return "C"
    if l == 1:
        return "L"
    return "T"


def stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return "n=0"

    return (
        f"n={len(values)}, "
        f"median={np.median(values):.3f}, "
        f"min={np.min(values):.3f}, "
        f"max={np.max(values):.3f}"
    )


def main():
    risbud = Table.read(
        RISBUD_INPUT,
        format="ascii.ecsv",
    )

    jadhav = Table.read(
        JADHAV_INPUT,
        format="ascii.ecsv",
    )

    jadhav = jadhav[
        np.asarray(jadhav["Ref"]) == "Risbud2025"
    ]

    risbud_by_id = {
        int(row["GaiaDR3"]): row
        for row in risbud
    }

    # ---------------------------------------------------------
    # 1. Build original -> Jadhav transition matrix
    # ---------------------------------------------------------

    transitions = Counter()
    mismatch_rows = []

    for row in jadhav:
        sid = int(row["GaiaDR3"])

        rrow = risbud_by_id[sid]

        orig = original_class(rrow)
        new = str(row["Class"])

        transitions[(orig, new)] += 1

        if orig != new:
            mismatch_rows.append(
                (
                    sid,
                    orig,
                    new,
                    int(rrow["FCorrSamp"]),
                    float(row["r3d"]),
                    float(row["distFromO"]),
                    float(row["distAlongO"]),
                    float(row["deltaphiGC"]),
                )
            )

    # ---------------------------------------------------------
    # 2. Matrix output
    # ---------------------------------------------------------

    matrix = Table(
        names=[
            "risbud_original_class",
            "jadhav_class",
            "n_sources",
        ],
        dtype=[
            "U1",
            "U1",
            "i8",
        ],
    )

    for orig in ["C", "L", "T"]:
        for new in ["C", "L", "T"]:
            matrix.add_row(
                (
                    orig,
                    new,
                    transitions.get((orig, new), 0),
                )
            )

    matrix.write(
        MATRIX_OUTPUT,
        format="ascii.csv",
        overwrite=True,
    )

    # ---------------------------------------------------------
    # 3. Detailed mismatch table
    # ---------------------------------------------------------

    mismatch_table = Table(
        rows=mismatch_rows,
        names=[
            "gaia_dr3_source_id",
            "risbud_original_class",
            "jadhav_class",
            "risbud_corrected_sample",
            "jadhav_r3d_pc",
            "jadhav_dist_from_orbit_pc",
            "jadhav_dist_along_orbit_pc",
            "jadhav_delta_phi_gc",
        ],
    )

    mismatch_table.write(
        MISMATCH_OUTPUT,
        format="ascii.csv",
        overwrite=True,
    )

    # ---------------------------------------------------------
    # 4. Transition diagnostics
    # ---------------------------------------------------------

    lines = [
        "Project 01 — Risbud2025 -> Jadhav2025 class-remapping diagnostic",
        "=" * 72,
        "",
        "Transition matrix:",
        "  rows = Risbud original class",
        "  columns = Jadhav class",
        "",
    ]

    for orig in ["C", "L", "T"]:
        vals = [
            transitions.get((orig, new), 0)
            for new in ["C", "L", "T"]
        ]

        lines.append(
            f"  {orig}: C={vals[0]}, L={vals[1]}, T={vals[2]}"
        )

    n_total = len(jadhav)
    n_mismatch = len(mismatch_rows)

    lines.extend(
        [
            "",
            f"Total sources: {n_total}",
            f"Class matches: {n_total - n_mismatch}",
            f"Class mismatches: {n_mismatch}",
            f"Mismatch fraction: {n_mismatch / n_total:.4f}",
            "",
            "Mismatch transitions:",
        ]
    )

    for orig in ["C", "L", "T"]:
        for new in ["C", "L", "T"]:
            if orig == new:
                continue

            n = transitions.get((orig, new), 0)

            lines.append(
                f"  {orig} -> {new}: {n}"
            )

    # ---------------------------------------------------------
    # 5. Orbit-coordinate summaries for each transition
    # ---------------------------------------------------------

    lines.extend(
        [
            "",
            "Jadhav orbital-coordinate diagnostics by transition:",
        ]
    )

    for orig in ["C", "L", "T"]:
        for new in ["C", "L", "T"]:

            selected = []

            for row in jadhav:
                sid = int(row["GaiaDR3"])

                if (
                    original_class(risbud_by_id[sid]) == orig
                    and str(row["Class"]) == new
                ):
                    selected.append(row)

            if not selected:
                continue

            along = [
                float(row["distAlongO"])
                for row in selected
            ]

            away = [
                float(row["distFromO"])
                for row in selected
            ]

            r3d = [
                float(row["r3d"])
                for row in selected
            ]

            lines.extend(
                [
                    "",
                    f"  {orig} -> {new}:",
                    f"    distAlongO: {stats(along)}",
                    f"    distFromO:  {stats(away)}",
                    f"    r3d:        {stats(r3d)}",
                ]
            )

    lines.extend(
        [
            "",
            "Outputs:",
            f"  {MATRIX_OUTPUT.relative_to(ROOT)}",
            f"  {MISMATCH_OUTPUT.relative_to(ROOT)}",
        ]
    )

    AUDIT_OUTPUT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(lines))


if __name__ == "__main__":
    main()
