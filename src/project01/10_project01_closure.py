from pathlib import Path

import numpy as np
from astropy.table import Table


ROOT = Path(__file__).resolve().parents[2]

LONG = ROOT / "data/processed/stock2_literature_long.ecsv"
MASTER = ROOT / "data/processed/stock2_literature_master.ecsv"

REPORT = ROOT / "results/project01/project01_closure.txt"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    require(LONG.exists(), f"Missing {LONG}")
    require(MASTER.exists(), f"Missing {MASTER}")

    long = Table.read(LONG, format="ascii.ecsv")
    master = Table.read(MASTER, format="ascii.ecsv")

    require(len(long) == 2341, "Unexpected long-table row count")
    require(len(master) == 1456, "Unexpected master-table row count")

    statuses = np.asarray(master["catalogue_status"])

    n_shared = int(np.sum(statuses == "shared"))
    n_kos_only = int(np.sum(statuses == "kos_only"))
    n_risbud_only = int(np.sum(statuses == "risbud_only"))

    require(n_shared == 885, "Unexpected shared count")
    require(n_kos_only == 178, "Unexpected Kos-only count")
    require(n_risbud_only == 393, "Unexpected Risbud-only count")

    classes = np.asarray(master["homogenized_class"])

    n_c = int(np.sum(classes == "C"))
    n_l = int(np.sum(classes == "L"))
    n_t = int(np.sum(classes == "T"))

    require(n_c == 940, "Unexpected C count")
    require(n_l == 184, "Unexpected L count")
    require(n_t == 332, "Unexpected T count")

    source_ids = np.asarray(
        master["gaia_dr3_source_id"],
        dtype=np.int64,
    )

    require(
        len(np.unique(source_ids)) == len(master),
        "Master table contains duplicate Gaia DR3 source IDs",
    )

    lines = [
        "Project 01 — Closure audit",
        "=" * 38,
        "",
        "All acceptance invariants passed.",
        "",
        f"Long rows: {len(long)}",
        f"Master sources: {len(master)}",
        "",
        f"Shared: {n_shared}",
        f"Kos-only: {n_kos_only}",
        f"Risbud-only: {n_risbud_only}",
        "",
        f"C: {n_c}",
        f"L: {n_l}",
        f"T: {n_t}",
        "",
        "PROJECT 01 STATUS: COMPLETE",
    ]

    REPORT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(lines))


if __name__ == "__main__":
    main()
