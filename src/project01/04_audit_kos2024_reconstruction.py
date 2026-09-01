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
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_OUTPUT = RESULTS_DIR / "kos2024_reconstruction_audit.txt"


def find_gaia_column(table):
    candidates = [
        "GaiaDR3",
        "Source",
        "source_id",
        "SourceID",
        "Gaia",
    ]

    for name in candidates:
        if name in table.colnames:
            return name

    raise RuntimeError(
        "Could not identify Gaia DR3 source-ID column.\n"
        f"Available columns: {table.colnames}"
    )


def main():
    if not KOS_INPUT.exists():
        raise FileNotFoundError(KOS_INPUT)

    if not JADHAV_INPUT.exists():
        raise FileNotFoundError(JADHAV_INPUT)

    kos = Table.read(KOS_INPUT, format="ascii.ecsv")
    jadhav = Table.read(JADHAV_INPUT, format="ascii.ecsv")

    # -----------------------------------------------------
    # Original Kos source IDs
    # -----------------------------------------------------

    gaia_col = find_gaia_column(kos)

    kos_ids_array = np.asarray(
        kos[gaia_col],
        dtype=np.int64,
    )

    kos_unique, kos_counts = np.unique(
        kos_ids_array,
        return_counts=True,
    )

    kos_duplicate_ids = kos_unique[kos_counts > 1]

    kos_ids = set(int(x) for x in kos_unique)

    # -----------------------------------------------------
    # Jadhav's Kos2024 subset
    # -----------------------------------------------------

    jadhav_kos = jadhav[
        np.asarray(jadhav["Ref"]) == "Kos2024"
    ]

    jadhav_ids = set(
        int(x)
        for x in jadhav_kos["GaiaDR3"]
    )

    # -----------------------------------------------------
    # Set comparison
    # -----------------------------------------------------

    shared = kos_ids & jadhav_ids
    kos_only = kos_ids - jadhav_ids
    jadhav_only = jadhav_ids - kos_ids
    union = kos_ids | jadhav_ids

    exact_match = kos_ids == jadhav_ids

    # -----------------------------------------------------
    # Audit
    # -----------------------------------------------------

    lines = [
        "Project 01 — Kos2024 original-catalogue reconstruction audit",
        "=" * 66,
        "",
        f"Detected Gaia-ID column: {gaia_col}",
        "",
        "Original Kos2024 Stock 2:",
        f"  rows: {len(kos)}",
        f"  unique Gaia DR3 IDs: {len(kos_ids)}",
        f"  duplicated Gaia IDs: {len(kos_duplicate_ids)}",
        "",
        "Jadhav2025 Kos2024 subset:",
        f"  rows: {len(jadhav_kos)}",
        f"  unique Gaia DR3 IDs: {len(jadhav_ids)}",
        "",
        "Source-set comparison:",
        f"  shared: {len(shared)}",
        f"  Kos-original only: {len(kos_only)}",
        f"  Jadhav only: {len(jadhav_only)}",
        f"  union: {len(union)}",
        f"  exact source-set match: {exact_match}",
    ]

    if kos_only:
        lines.extend(
            [
                "",
                "First Kos-original-only Gaia IDs:",
            ]
        )

        for source_id in sorted(kos_only)[:20]:
            lines.append(f"  {source_id}")

    if jadhav_only:
        lines.extend(
            [
                "",
                "First Jadhav-only Gaia IDs:",
            ]
        )

        for source_id in sorted(jadhav_only)[:20]:
            lines.append(f"  {source_id}")

    AUDIT_OUTPUT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(lines))


if __name__ == "__main__":
    main()
