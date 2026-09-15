from pathlib import Path

import numpy as np
from astropy.table import Table


# ============================================================
# Paths
# ============================================================

INPUT_PATH = Path("data/processed/stock2_literature_master.ecsv")

INTERIM_DIR = Path("data/interim/project02")
RESULTS_DIR = Path("results/project02")

QUERY_ECSV = INTERIM_DIR / "gaia_dr3_source_query.ecsv"
QUERY_CSV = INTERIM_DIR / "gaia_dr3_source_query.csv"
AUDIT_PATH = RESULTS_DIR / "gaia_source_query_preparation.txt"


# ============================================================
# Expected Project 01 invariants
# ============================================================

EXPECTED_MASTER_ROWS = 1456
SOURCE_ID_COLUMN = "gaia_dr3_source_id"


def main():
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Project 01 canonical master table not found: {INPUT_PATH}"
        )

    table = Table.read(INPUT_PATH)

    # --------------------------------------------------------
    # Basic schema audit
    # --------------------------------------------------------

    n_rows = len(table)

    if SOURCE_ID_COLUMN not in table.colnames:
        raise KeyError(
            f"Required column '{SOURCE_ID_COLUMN}' is missing.\n"
            f"Available columns:\n{table.colnames}"
        )

    source_col = table[SOURCE_ID_COLUMN]

    # --------------------------------------------------------
    # Missing-value audit
    # --------------------------------------------------------

    if hasattr(source_col, "mask"):
        mask = np.asarray(source_col.mask)

        if mask.ndim == 0:
            n_missing = int(bool(mask))
        else:
            n_missing = int(np.sum(mask))
    else:
        n_missing = 0

    # --------------------------------------------------------
    # Convert IDs safely
    #
    # Gaia source_id is a 64-bit integer identifier.
    # Never convert through float because precision can be lost.
    # --------------------------------------------------------

    source_ids = []

    for i, value in enumerate(source_col):
        if np.ma.is_masked(value):
            continue

        try:
            source_id = int(str(value).strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid Gaia DR3 source_id at row {i}: {value!r}"
            ) from exc

        source_ids.append(source_id)

    source_ids = np.asarray(source_ids, dtype=np.int64)

    n_valid_ids = len(source_ids)
    n_unique_ids = len(np.unique(source_ids))
    n_duplicate_ids = n_valid_ids - n_unique_ids

    # --------------------------------------------------------
    # Expected invariants
    # --------------------------------------------------------

    invariant_rows = n_rows == EXPECTED_MASTER_ROWS
    invariant_no_missing = n_missing == 0
    invariant_all_ids_valid = n_valid_ids == n_rows
    invariant_unique = n_unique_ids == n_rows

    all_invariants = all(
        [
            invariant_rows,
            invariant_no_missing,
            invariant_all_ids_valid,
            invariant_unique,
        ]
    )

    # --------------------------------------------------------
    # Audit report
    # --------------------------------------------------------

    lines = [
        "Project 02A — Gaia DR3 source-query preparation",
        "=" * 56,
        "",
        f"Input table:              {INPUT_PATH}",
        f"Input rows:               {n_rows}",
        f"Expected rows:            {EXPECTED_MASTER_ROWS}",
        "",
        f"Source-ID column:         {SOURCE_ID_COLUMN}",
        f"Missing source IDs:       {n_missing}",
        f"Valid source IDs:         {n_valid_ids}",
        f"Unique source IDs:        {n_unique_ids}",
        f"Duplicate source IDs:     {n_duplicate_ids}",
        "",
        "Invariants:",
        f"  master rows == 1456:            {invariant_rows}",
        f"  missing source IDs == 0:        {invariant_no_missing}",
        f"  valid IDs == master rows:       {invariant_all_ids_valid}",
        f"  unique IDs == master rows:      {invariant_unique}",
        "",
        f"PREPARATION STATUS: {'PASS' if all_invariants else 'FAIL'}",
    ]

    audit_text = "\n".join(lines)

    print(audit_text)

    AUDIT_PATH.write_text(
        audit_text + "\n",
        encoding="utf-8",
    )

    if not all_invariants:
        raise RuntimeError(
            "Project 02A source-query preparation failed one or more "
            "invariants. Gaia retrieval must not proceed."
        )

    # --------------------------------------------------------
    # Canonical Gaia query manifest
    #
    # Deliberately minimal:
    # one row = one exact Gaia DR3 source_id to retrieve.
    # --------------------------------------------------------

    query_table = Table()
    query_table["gaia_dr3_source_id"] = source_ids

    query_table.write(
        QUERY_ECSV,
        format="ascii.ecsv",
        overwrite=True,
    )

    query_table.write(
        QUERY_CSV,
        format="ascii.csv",
        overwrite=True,
    )

    print("")
    print("Outputs:")
    print(f"  {QUERY_ECSV}")
    print(f"  {QUERY_CSV}")
    print(f"  {AUDIT_PATH}")


if __name__ == "__main__":
    main()
