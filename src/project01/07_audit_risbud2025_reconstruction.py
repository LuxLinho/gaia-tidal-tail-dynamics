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

AUDIT_OUTPUT = (
    RESULTS_DIR / "risbud2025_reconstruction_audit.txt"
)


def id_set(table):
    return set(
        int(x)
        for x in table["GaiaDR3"]
    )


def compare(label, ids, target):
    return {
        "label": label,
        "n": len(ids),
        "shared": len(ids & target),
        "selected_only": len(ids - target),
        "jadhav_only": len(target - ids),
        "exact": ids == target,
    }


def main():
    risbud = Table.read(
        RISBUD_INPUT,
        format="ascii.ecsv",
    )

    jadhav = Table.read(
        JADHAV_INPUT,
        format="ascii.ecsv",
    )

    # -----------------------------------------------------
    # Basic source audit
    # -----------------------------------------------------

    raw_ids_array = np.asarray(
        risbud["GaiaDR3"],
        dtype=np.int64,
    )

    unique_ids, counts = np.unique(
        raw_ids_array,
        return_counts=True,
    )

    duplicate_ids = unique_ids[counts > 1]

    raw_ids = set(
        int(x)
        for x in unique_ids
    )

    jadhav_risbud = jadhav[
        np.asarray(jadhav["Ref"]) == "Risbud2025"
    ]

    jadhav_ids = id_set(jadhav_risbud)

    # -----------------------------------------------------
    # Original Risbud flags
    # -----------------------------------------------------

    leading = np.asarray(
        risbud["FleadingTail"],
        dtype=int,
    )

    trailing = np.asarray(
        risbud["FtrailingTail"],
        dtype=int,
    )

    within = np.asarray(
        risbud["FwithinTidRad"],
        dtype=int,
    )

    corrected = np.asarray(
        risbud["FCorrSamp"],
        dtype=int,
    )

    # Individual flag sets
    leading_ids = id_set(risbud[leading == 1])
    trailing_ids = id_set(risbud[trailing == 1])
    within_ids = id_set(risbud[within == 1])
    corrected_ids = id_set(risbud[corrected == 1])

    # Any morphological membership flag
    morphology_mask = (
        (leading == 1)
        | (trailing == 1)
        | (within == 1)
    )

    morphology_ids = id_set(
        risbud[morphology_mask]
    )

    # Corrected AND morphological
    corrected_morphology_mask = (
        morphology_mask
        & (corrected == 1)
    )

    corrected_morphology_ids = id_set(
        risbud[corrected_morphology_mask]
    )

    tests = [
        compare(
            "all raw Stock_2 rows",
            raw_ids,
            jadhav_ids,
        ),
        compare(
            "C/L/T union",
            morphology_ids,
            jadhav_ids,
        ),
        compare(
            "FCorrSamp == 1",
            corrected_ids,
            jadhav_ids,
        ),
        compare(
            "C/L/T union AND FCorrSamp == 1",
            corrected_morphology_ids,
            jadhav_ids,
        ),
    ]

    # -----------------------------------------------------
    # Flag-combination audit
    # -----------------------------------------------------

    combinations = Counter(
        (
            int(l),
            int(t),
            int(w),
            int(c),
        )
        for l, t, w, c in zip(
            leading,
            trailing,
            within,
            corrected,
        )
    )

    # -----------------------------------------------------
    # Test Jadhav C/L/T against original flags
    # -----------------------------------------------------

    risbud_by_id = {
        int(row["GaiaDR3"]): row
        for row in risbud
    }

    class_match = 0
    class_mismatch = []

    for row in jadhav_risbud:
        sid = int(row["GaiaDR3"])
        cls = str(row["Class"])

        original = risbud_by_id[sid]

        expected = None

        if int(original["FwithinTidRad"]) == 1:
            expected = "C"
        elif int(original["FleadingTail"]) == 1:
            expected = "L"
        elif int(original["FtrailingTail"]) == 1:
            expected = "T"

        if cls == expected:
            class_match += 1
        else:
            class_mismatch.append(
                (
                    sid,
                    cls,
                    expected,
                    int(original["FleadingTail"]),
                    int(original["FtrailingTail"]),
                    int(original["FwithinTidRad"]),
                    int(original["FCorrSamp"]),
                )
            )

    # -----------------------------------------------------
    # Report
    # -----------------------------------------------------

    lines = [
        "Project 01 — Risbud2025 original-catalogue reconstruction audit",
        "=" * 69,
        "",
        "Original Risbud2025 Stock 2:",
        f"  rows: {len(risbud)}",
        f"  unique Gaia DR3 IDs: {len(raw_ids)}",
        f"  duplicated Gaia IDs: {len(duplicate_ids)}",
        "",
        "Original flag counts:",
        f"  FleadingTail == 1: {len(leading_ids)}",
        f"  FtrailingTail == 1: {len(trailing_ids)}",
        f"  FwithinTidRad == 1: {len(within_ids)}",
        f"  FCorrSamp == 1: {len(corrected_ids)}",
        f"  C/L/T union: {len(morphology_ids)}",
        f"  corrected C/L/T union: {len(corrected_morphology_ids)}",
        "",
        "Jadhav2025 Risbud2025 subset:",
        f"  rows: {len(jadhav_risbud)}",
        f"  unique Gaia DR3 IDs: {len(jadhav_ids)}",
        "",
        "Selection reconstruction tests:",
    ]

    for result in tests:
        lines.extend(
            [
                f"  {result['label']}:",
                f"    selected: {result['n']}",
                f"    shared: {result['shared']}",
                f"    selected only: {result['selected_only']}",
                f"    Jadhav only: {result['jadhav_only']}",
                f"    exact match: {result['exact']}",
            ]
        )

    lines.extend(
        [
            "",
            "Flag combinations (L,T,Corr-radius,CorrSample):",
        ]
    )

    for combo, count in sorted(combinations.items()):
        lines.append(
            f"  {combo}: {count}"
        )

    lines.extend(
        [
            "",
            "Jadhav C/L/T reconstruction:",
            f"  matched original flags: {class_match}",
            f"  mismatched: {len(class_mismatch)}",
        ]
    )

    if class_mismatch:
        lines.append("")
        lines.append("First classification mismatches:")

        for item in class_mismatch[:20]:
            lines.append(
                "  "
                + " | ".join(str(x) for x in item)
            )

    AUDIT_OUTPUT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(lines))


if __name__ == "__main__":
    main()
