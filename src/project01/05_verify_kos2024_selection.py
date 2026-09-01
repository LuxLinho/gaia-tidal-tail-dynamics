from pathlib import Path

import numpy as np
from astropy.table import Table


ROOT = Path(__file__).resolve().parents[2]

KOS_INPUT = (
    ROOT
    / "data/raw/literature/kos2024/kos2024_stock2.ecsv"
)

JADHAV_INPUT = (
    ROOT
    / "data/interim/project01/jadhav2025_stock2.ecsv"
)

RESULTS_DIR = ROOT / "results/project01"
PROCESSED_DIR = ROOT / "data/processed"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_OUTPUT = (
    RESULTS_DIR / "kos2024_selection_reconstruction_audit.txt"
)

SELECTED_OUTPUT = (
    PROCESSED_DIR / "stock2_kos2024_pbint_gt_0p9.ecsv"
)


def compare_sets(label, selected_ids, jadhav_ids):
    shared = selected_ids & jadhav_ids
    selected_only = selected_ids - jadhav_ids
    jadhav_only = jadhav_ids - selected_ids

    return {
        "label": label,
        "n_selected": len(selected_ids),
        "n_shared": len(shared),
        "n_selected_only": len(selected_only),
        "n_jadhav_only": len(jadhav_only),
        "exact": selected_ids == jadhav_ids,
    }


def main():
    kos = Table.read(KOS_INPUT, format="ascii.ecsv")
    jadhav = Table.read(JADHAV_INPUT, format="ascii.ecsv")

    jadhav_kos = jadhav[
        np.asarray(jadhav["Ref"]) == "Kos2024"
    ]

    jadhav_ids = set(
        int(x)
        for x in jadhav_kos["GaiaDR3"]
    )

    # -----------------------------------------------------
    # Test documented Jadhav criterion
    # -----------------------------------------------------

    pbint = np.asarray(kos["pbint"], dtype=float)

    mask_gt = pbint > 0.9
    mask_ge = pbint >= 0.9

    kos_gt = kos[mask_gt]
    kos_ge = kos[mask_ge]

    ids_gt = set(
        int(x)
        for x in kos_gt["GaiaDR3"]
    )

    ids_ge = set(
        int(x)
        for x in kos_ge["GaiaDR3"]
    )

    result_gt = compare_sets(
        "pbint > 0.9",
        ids_gt,
        jadhav_ids,
    )

    result_ge = compare_sets(
        "pbint >= 0.9",
        ids_ge,
        jadhav_ids,
    )

    # -----------------------------------------------------
    # Distribution audit
    # -----------------------------------------------------

    finite = np.isfinite(pbint)

    quantiles = np.quantile(
        pbint[finite],
        [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0],
    )

    # -----------------------------------------------------
    # Preserve selected original Kos rows
    # -----------------------------------------------------

    kos_gt.write(
        SELECTED_OUTPUT,
        format="ascii.ecsv",
        overwrite=True,
    )

    # -----------------------------------------------------
    # Report
    # -----------------------------------------------------

    lines = [
        "Project 01 — Kos2024 selection reconstruction",
        "=" * 55,
        "",
        "Raw Stock 2 catalogue:",
        f"  rows: {len(kos)}",
        "",
        "Jadhav2025 Kos2024 subset:",
        f"  rows: {len(jadhav_kos)}",
        f"  unique source IDs: {len(jadhav_ids)}",
        "",
        "Documented criterion test:",
    ]

    for result in [result_gt, result_ge]:
        lines.extend(
            [
                f"  {result['label']}:",
                f"    selected: {result['n_selected']}",
                f"    shared with Jadhav: {result['n_shared']}",
                f"    selected only: {result['n_selected_only']}",
                f"    Jadhav only: {result['n_jadhav_only']}",
                f"    exact source-set match: {result['exact']}",
            ]
        )

    lines.extend(
        [
            "",
            "pbint distribution:",
            f"  min: {quantiles[0]:.6f}",
            f"  Q25: {quantiles[1]:.6f}",
            f"  median: {quantiles[2]:.6f}",
            f"  Q75: {quantiles[3]:.6f}",
            f"  P90: {quantiles[4]:.6f}",
            f"  P95: {quantiles[5]:.6f}",
            f"  P99: {quantiles[6]:.6f}",
            f"  max: {quantiles[7]:.6f}",
            "",
            "Selected original-table output:",
            f"  {SELECTED_OUTPUT.relative_to(ROOT)}",
        ]
    )

    AUDIT_OUTPUT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(lines))


if __name__ == "__main__":
    main()
