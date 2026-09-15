from pathlib import Path

import numpy as np
from astropy.table import Table


# ============================================================
# Paths
# ============================================================

MANIFEST_PATH = Path(
    "data/interim/project02/gaia_dr3_source_query.ecsv"
)

GAIA_PATH = Path(
    "data/raw/gaia/project02/gaia_dr3_enrichment_raw.ecsv"
)

RESULTS_DIR = Path("results/project02")

AUDIT_PATH = RESULTS_DIR / "gaia_identity_audit.txt"
MISSING_PATH = RESULTS_DIR / "gaia_identity_missing.csv"
UNEXPECTED_PATH = RESULTS_DIR / "gaia_identity_unexpected.csv"
DUPLICATE_PATH = RESULTS_DIR / "gaia_identity_duplicates.csv"
MISMATCH_PATH = RESULTS_DIR / "gaia_identity_row_mismatches.csv"


# ============================================================
# Expected invariant
# ============================================================

EXPECTED_SOURCE_COUNT = 1456


def to_int64_array(column, column_name):
    """
    Convert a Gaia source-ID column safely to int64.

    Masked/null values are returned separately and are not silently
    converted.
    """

    values = []
    masked_indices = []

    for i, value in enumerate(column):
        if np.ma.is_masked(value):
            masked_indices.append(i)
            continue

        try:
            values.append(int(str(value).strip()))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid value in '{column_name}' at row {i}: {value!r}"
            ) from exc

    return np.asarray(values, dtype=np.int64), masked_indices


def write_id_table(ids, column_name, path):
    """
    Write a simple CSV table containing source IDs.
    """

    table = Table()
    table[column_name] = np.asarray(ids, dtype=np.int64)

    table.write(
        path,
        format="ascii.csv",
        overwrite=True,
    )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Input existence checks
    # --------------------------------------------------------

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Gaia query manifest not found: {MANIFEST_PATH}"
        )

    if not GAIA_PATH.exists():
        raise FileNotFoundError(
            f"Gaia raw enrichment table not found: {GAIA_PATH}"
        )

    manifest = Table.read(MANIFEST_PATH)
    gaia = Table.read(GAIA_PATH)

    # --------------------------------------------------------
    # Schema checks
    # --------------------------------------------------------

    if "gaia_dr3_source_id" not in manifest.colnames:
        raise KeyError(
            "Manifest is missing 'gaia_dr3_source_id'."
        )

    required_gaia_columns = [
        "input_source_id",
        "source_id",
    ]

    for column in required_gaia_columns:
        if column not in gaia.colnames:
            raise KeyError(
                f"Gaia enrichment table is missing '{column}'."
            )

    # --------------------------------------------------------
    # Manifest IDs
    # --------------------------------------------------------

    manifest_ids, manifest_masked = to_int64_array(
        manifest["gaia_dr3_source_id"],
        "gaia_dr3_source_id",
    )

    # --------------------------------------------------------
    # Returned input IDs
    #
    # These are the IDs copied from the TAP_UPLOAD table.
    # --------------------------------------------------------

    returned_input_ids, returned_input_masked = to_int64_array(
        gaia["input_source_id"],
        "input_source_id",
    )

    # --------------------------------------------------------
    # Gaia-resolved source IDs
    # --------------------------------------------------------

    returned_gaia_ids, returned_gaia_masked = to_int64_array(
        gaia["source_id"],
        "source_id",
    )

    # --------------------------------------------------------
    # Basic row counts
    # --------------------------------------------------------

    n_manifest = len(manifest)
    n_gaia_rows = len(gaia)

    # --------------------------------------------------------
    # Set-level comparisons
    # --------------------------------------------------------

    manifest_set = set(manifest_ids.tolist())
    returned_input_set = set(returned_input_ids.tolist())
    returned_gaia_set = set(returned_gaia_ids.tolist())

    missing_from_gaia = sorted(
        manifest_set - returned_gaia_set
    )

    unexpected_gaia = sorted(
        returned_gaia_set - manifest_set
    )

    missing_from_returned_input = sorted(
        manifest_set - returned_input_set
    )

    unexpected_returned_input = sorted(
        returned_input_set - manifest_set
    )

    # --------------------------------------------------------
    # Duplicate checks
    # --------------------------------------------------------

    unique_manifest_ids, manifest_counts = np.unique(
        manifest_ids,
        return_counts=True,
    )

    manifest_duplicate_ids = unique_manifest_ids[
        manifest_counts > 1
    ]

    unique_returned_input_ids, returned_input_counts = np.unique(
        returned_input_ids,
        return_counts=True,
    )

    returned_input_duplicate_ids = unique_returned_input_ids[
        returned_input_counts > 1
    ]

    unique_gaia_ids, gaia_counts = np.unique(
        returned_gaia_ids,
        return_counts=True,
    )

    gaia_duplicate_ids = unique_gaia_ids[
        gaia_counts > 1
    ]

    # --------------------------------------------------------
    # Row-by-row exact-match audit
    #
    # LEFT JOIN should return:
    #
    # input_source_id == source_id
    #
    # for every resolved row.
    # --------------------------------------------------------

    mismatch_rows = []

    for i, row in enumerate(gaia):
        input_id = row["input_source_id"]
        source_id = row["source_id"]

        if np.ma.is_masked(input_id):
            mismatch_rows.append(
                (i, None, None, "masked_input_source_id")
            )
            continue

        input_id_int = int(str(input_id).strip())

        if np.ma.is_masked(source_id):
            mismatch_rows.append(
                (
                    i,
                    input_id_int,
                    None,
                    "unresolved_gaia_source_id",
                )
            )
            continue

        source_id_int = int(str(source_id).strip())

        if input_id_int != source_id_int:
            mismatch_rows.append(
                (
                    i,
                    input_id_int,
                    source_id_int,
                    "input_source_id != source_id",
                )
            )

    # --------------------------------------------------------
    # Write forensic outputs
    # --------------------------------------------------------

    write_id_table(
        missing_from_gaia,
        "gaia_dr3_source_id",
        MISSING_PATH,
    )

    write_id_table(
        unexpected_gaia,
        "gaia_dr3_source_id",
        UNEXPECTED_PATH,
    )

    duplicate_table = Table(
        names=[
            "source_id",
            "duplicate_location",
        ],
        dtype=[
            np.int64,
            "U32",
        ],
    )

    for source_id in manifest_duplicate_ids:
        duplicate_table.add_row(
            (int(source_id), "manifest")
        )

    for source_id in returned_input_duplicate_ids:
        duplicate_table.add_row(
            (int(source_id), "returned_input")
        )

    for source_id in gaia_duplicate_ids:
        duplicate_table.add_row(
            (int(source_id), "gaia_result")
        )

    duplicate_table.write(
        DUPLICATE_PATH,
        format="ascii.csv",
        overwrite=True,
    )

    mismatch_table = Table(
        names=[
            "row_index",
            "input_source_id",
            "source_id",
            "reason",
        ],
        dtype=[
            np.int64,
            object,
            object,
            "U64",
        ],
    )

    for row in mismatch_rows:
        mismatch_table.add_row(row)

    mismatch_table.write(
        MISMATCH_PATH,
        format="ascii.csv",
        overwrite=True,
    )

    # --------------------------------------------------------
    # Invariants
    # --------------------------------------------------------

    invariant_manifest_rows = (
        n_manifest == EXPECTED_SOURCE_COUNT
    )

    invariant_gaia_rows = (
        n_gaia_rows == EXPECTED_SOURCE_COUNT
    )

    invariant_manifest_no_masked = (
        len(manifest_masked) == 0
    )

    invariant_returned_input_no_masked = (
        len(returned_input_masked) == 0
    )

    invariant_gaia_no_masked = (
        len(returned_gaia_masked) == 0
    )

    invariant_manifest_unique = (
        len(manifest_set) == EXPECTED_SOURCE_COUNT
    )

    invariant_returned_input_unique = (
        len(returned_input_set) == EXPECTED_SOURCE_COUNT
    )

    invariant_gaia_unique = (
        len(returned_gaia_set) == EXPECTED_SOURCE_COUNT
    )

    invariant_manifest_equals_returned_input = (
        manifest_set == returned_input_set
    )

    invariant_manifest_equals_gaia = (
        manifest_set == returned_gaia_set
    )

    invariant_no_missing = (
        len(missing_from_gaia) == 0
    )

    invariant_no_unexpected = (
        len(unexpected_gaia) == 0
    )

    invariant_no_row_mismatches = (
        len(mismatch_rows) == 0
    )

    all_invariants = all(
        [
            invariant_manifest_rows,
            invariant_gaia_rows,
            invariant_manifest_no_masked,
            invariant_returned_input_no_masked,
            invariant_gaia_no_masked,
            invariant_manifest_unique,
            invariant_returned_input_unique,
            invariant_gaia_unique,
            invariant_manifest_equals_returned_input,
            invariant_manifest_equals_gaia,
            invariant_no_missing,
            invariant_no_unexpected,
            invariant_no_row_mismatches,
        ]
    )

    # --------------------------------------------------------
    # Audit report
    # --------------------------------------------------------

    lines = [
        "Project 02A — Gaia DR3 identity audit",
        "=" * 50,
        "",
        "Inputs",
        "------",
        f"Manifest:                  {MANIFEST_PATH}",
        f"Gaia enrichment:           {GAIA_PATH}",
        "",
        "Row counts",
        "----------",
        f"Manifest rows:              {n_manifest}",
        f"Gaia returned rows:         {n_gaia_rows}",
        "",
        "Unique IDs",
        "----------",
        f"Manifest unique IDs:        {len(manifest_set)}",
        f"Returned input unique IDs:  {len(returned_input_set)}",
        f"Gaia source unique IDs:     {len(returned_gaia_set)}",
        "",
        "Masked / unresolved",
        "-------------------",
        f"Masked manifest IDs:        {len(manifest_masked)}",
        f"Masked returned input IDs:  {len(returned_input_masked)}",
        f"Unresolved Gaia source IDs: {len(returned_gaia_masked)}",
        "",
        "Set comparison",
        "--------------",
        f"Missing Gaia IDs:           {len(missing_from_gaia)}",
        f"Unexpected Gaia IDs:        {len(unexpected_gaia)}",
        (
            "Missing returned-input IDs:"
            f" {len(missing_from_returned_input)}"
        ),
        (
            "Unexpected input IDs:     "
            f"{len(unexpected_returned_input)}"
        ),
        "",
        "Duplicates",
        "----------",
        f"Manifest duplicate IDs:     {len(manifest_duplicate_ids)}",
        (
            "Returned-input duplicates:"
            f" {len(returned_input_duplicate_ids)}"
        ),
        f"Gaia-result duplicates:     {len(gaia_duplicate_ids)}",
        "",
        "Row-level identity",
        "------------------",
        f"Row mismatches:             {len(mismatch_rows)}",
        "",
        "Invariants",
        "----------",
        (
            "manifest rows == 1456:              "
            f"{invariant_manifest_rows}"
        ),
        (
            "Gaia rows == 1456:                  "
            f"{invariant_gaia_rows}"
        ),
        (
            "manifest IDs all present:           "
            f"{invariant_manifest_no_masked}"
        ),
        (
            "returned input IDs all present:     "
            f"{invariant_returned_input_no_masked}"
        ),
        (
            "Gaia source IDs all resolved:       "
            f"{invariant_gaia_no_masked}"
        ),
        (
            "manifest IDs unique:                "
            f"{invariant_manifest_unique}"
        ),
        (
            "returned input IDs unique:          "
            f"{invariant_returned_input_unique}"
        ),
        (
            "Gaia source IDs unique:             "
            f"{invariant_gaia_unique}"
        ),
        (
            "manifest set == returned-input set: "
            f"{invariant_manifest_equals_returned_input}"
        ),
        (
            "manifest set == Gaia source set:    "
            f"{invariant_manifest_equals_gaia}"
        ),
        (
            "missing Gaia IDs == 0:              "
            f"{invariant_no_missing}"
        ),
        (
            "unexpected Gaia IDs == 0:           "
            f"{invariant_no_unexpected}"
        ),
        (
            "row-level ID mismatches == 0:       "
            f"{invariant_no_row_mismatches}"
        ),
        "",
        (
            "PROJECT 02A IDENTITY STATUS: "
            f"{'PASS' if all_invariants else 'FAIL'}"
        ),
    ]

    report = "\n".join(lines)

    AUDIT_PATH.write_text(
        report + "\n",
        encoding="utf-8",
    )

    print(report)

    print()
    print("Forensic outputs:")
    print(f"  {MISSING_PATH}")
    print(f"  {UNEXPECTED_PATH}")
    print(f"  {DUPLICATE_PATH}")
    print(f"  {MISMATCH_PATH}")
    print(f"  {AUDIT_PATH}")

    if not all_invariants:
        raise RuntimeError(
            "Project 02A identity audit failed. "
            "Do not proceed to Gaia quality-control analysis."
        )


if __name__ == "__main__":
    main()
