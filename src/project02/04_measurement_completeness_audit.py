from pathlib import Path

import numpy as np
from astropy.table import Table, join


# ============================================================
# Paths
# ============================================================

LITERATURE_PATH = Path(
    "data/processed/stock2_literature_master.ecsv"
)

GAIA_PATH = Path(
    "data/raw/gaia/project02/gaia_dr3_enrichment_raw.ecsv"
)

INTERIM_DIR = Path("data/interim/project02")
RESULTS_DIR = Path("results/project02")

ENRICHED_PATH = (
    INTERIM_DIR / "stock2_gaia_measurement_completeness.ecsv"
)

SUMMARY_PATH = (
    RESULTS_DIR / "measurement_completeness_summary.csv"
)

CATALOGUE_BREAKDOWN_PATH = (
    RESULTS_DIR / "measurement_completeness_by_catalogue.csv"
)

CLASS_BREAKDOWN_PATH = (
    RESULTS_DIR / "measurement_completeness_by_class.csv"
)

REPORT_PATH = (
    RESULTS_DIR / "measurement_completeness_audit.txt"
)


EXPECTED_SOURCE_COUNT = 1456


# ============================================================
# Helpers
# ============================================================

def available(value):
    """
    True if an Astropy table value is present and finite when numeric.
    """

    if np.ma.is_masked(value):
        return False

    try:
        return bool(np.isfinite(value))
    except TypeError:
        return value is not None


def column_available(table, column_name):
    """
    Boolean array indicating whether a measurement is available.
    """

    if column_name not in table.colnames:
        raise KeyError(
            f"Required column '{column_name}' is missing."
        )

    return np.asarray(
        [available(value) for value in table[column_name]],
        dtype=bool,
    )


def percent(n, total):
    if total == 0:
        return np.nan
    return 100.0 * n / total


def add_summary_row(
    table,
    label,
    mask,
    flags,
    flag_order,
):
    n_total = int(np.sum(mask))

    values = [
        label,
        n_total,
    ]

    for flag_name in flag_order:
        n = int(np.sum(mask & flags[flag_name]))
        values.extend([n, percent(n, n_total)])

    table.add_row(values)


# ============================================================
# Main
# ============================================================

def main():
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Read inputs
    # --------------------------------------------------------

    if not LITERATURE_PATH.exists():
        raise FileNotFoundError(
            f"Literature master not found: {LITERATURE_PATH}"
        )

    if not GAIA_PATH.exists():
        raise FileNotFoundError(
            f"Gaia enrichment table not found: {GAIA_PATH}"
        )

    literature = Table.read(LITERATURE_PATH)
    gaia = Table.read(GAIA_PATH)

    if len(literature) != EXPECTED_SOURCE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_SOURCE_COUNT} literature rows, "
            f"found {len(literature)}."
        )

    if len(gaia) != EXPECTED_SOURCE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_SOURCE_COUNT} Gaia rows, "
            f"found {len(gaia)}."
        )

    # --------------------------------------------------------
    # Normalize join key
    # --------------------------------------------------------

    literature = literature.copy()
    gaia = gaia.copy()

    literature["source_id_join"] = np.asarray(
        [
            int(str(value).strip())
            for value in literature["gaia_dr3_source_id"]
        ],
        dtype=np.int64,
    )

    gaia["source_id_join"] = np.asarray(
        [
            int(str(value).strip())
            for value in gaia["source_id"]
        ],
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Join literature provenance + Gaia observables
    # --------------------------------------------------------

    combined = join(
        literature,
        gaia,
        keys="source_id_join",
        join_type="inner",
        metadata_conflicts="silent",
    )

    if len(combined) != EXPECTED_SOURCE_COUNT:
        raise RuntimeError(
            "Literature/Gaia enrichment join did not preserve "
            f"{EXPECTED_SOURCE_COUNT} rows; found {len(combined)}."
        )

    # --------------------------------------------------------
    # Individual measurement availability
    # --------------------------------------------------------

    flags = {}

    flags["has_ra"] = column_available(combined, "ra")
    flags["has_dec"] = column_available(combined, "dec")

    flags["has_parallax"] = column_available(
        combined,
        "parallax",
    )

    flags["has_pmra"] = column_available(
        combined,
        "pmra",
    )

    flags["has_pmdec"] = column_available(
        combined,
        "pmdec",
    )

    flags["has_g"] = column_available(
        combined,
        "phot_g_mean_mag",
    )

    flags["has_bp"] = column_available(
        combined,
        "phot_bp_mean_mag",
    )

    flags["has_rp"] = column_available(
        combined,
        "phot_rp_mean_mag",
    )

    flags["has_rv"] = column_available(
        combined,
        "radial_velocity",
    )

    flags["has_rv_error"] = column_available(
        combined,
        "radial_velocity_error",
    )

    # --------------------------------------------------------
    # Compound availability flags
    #
    # IMPORTANT:
    # These are measurement-completeness flags only.
    # They are NOT quality-passed samples.
    # --------------------------------------------------------

    flags["has_position"] = (
        flags["has_ra"]
        & flags["has_dec"]
    )

    flags["has_pm"] = (
        flags["has_pmra"]
        & flags["has_pmdec"]
    )

    flags["has_5d"] = (
        flags["has_ra"]
        & flags["has_dec"]
        & flags["has_parallax"]
        & flags["has_pmra"]
        & flags["has_pmdec"]
    )

    flags["has_gbp_rp"] = (
        flags["has_g"]
        & flags["has_bp"]
        & flags["has_rp"]
    )

    flags["has_rv_measurement"] = (
        flags["has_rv"]
        & flags["has_rv_error"]
    )

    flags["has_6d"] = (
        flags["has_5d"]
        & flags["has_rv_measurement"]
    )

    # --------------------------------------------------------
    # Store flags in combined table
    # --------------------------------------------------------

    for name, mask in flags.items():
        combined[name] = mask

    # --------------------------------------------------------
    # Save enriched interim table
    # --------------------------------------------------------

    combined.write(
        ENRICHED_PATH,
        format="ascii.ecsv",
        overwrite=True,
    )

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    summary_flag_order = [
        "has_position",
        "has_parallax",
        "has_pm",
        "has_5d",
        "has_gbp_rp",
        "has_rv",
        "has_rv_measurement",
        "has_6d",
    ]

    summary = Table(
        names=[
            "measurement",
            "n_available",
            "fraction_percent",
        ],
        dtype=[
            "U32",
            np.int64,
            np.float64,
        ],
    )

    for name in summary_flag_order:
        n = int(np.sum(flags[name]))

        summary.add_row(
            (
                name,
                n,
                percent(n, len(combined)),
            )
        )

    summary.write(
        SUMMARY_PATH,
        format="ascii.csv",
        overwrite=True,
    )

    # --------------------------------------------------------
    # Breakdown schema
    # --------------------------------------------------------

    breakdown_names = ["group", "n_total"]
    breakdown_dtypes = ["U64", np.int64]

    for name in summary_flag_order:
        breakdown_names.extend(
            [
                f"n_{name}",
                f"pct_{name}",
            ]
        )

        breakdown_dtypes.extend(
            [
                np.int64,
                np.float64,
            ]
        )

    # --------------------------------------------------------
    # Catalogue-status breakdown
    # --------------------------------------------------------

    catalogue_breakdown = Table(
        names=breakdown_names,
        dtype=breakdown_dtypes,
    )

    if "catalogue_status" not in combined.colnames:
        raise KeyError(
            "Expected literature provenance column "
            "'catalogue_status' is missing."
        )

    catalogue_values = sorted(
        set(str(x) for x in combined["catalogue_status"])
    )

    for group in catalogue_values:
        mask = np.asarray(
            [
                str(x) == group
                for x in combined["catalogue_status"]
            ],
            dtype=bool,
        )

        add_summary_row(
            catalogue_breakdown,
            group,
            mask,
            flags,
            summary_flag_order,
        )

    catalogue_breakdown.write(
        CATALOGUE_BREAKDOWN_PATH,
        format="ascii.csv",
        overwrite=True,
    )

    # --------------------------------------------------------
    # Homogenized-class breakdown
    # --------------------------------------------------------

    class_breakdown = Table(
        names=breakdown_names,
        dtype=breakdown_dtypes,
    )

    if "homogenized_class" not in combined.colnames:
        raise KeyError(
            "Expected literature provenance column "
            "'homogenized_class' is missing."
        )

    class_values = sorted(
        set(str(x) for x in combined["homogenized_class"])
    )

    for group in class_values:
        mask = np.asarray(
            [
                str(x) == group
                for x in combined["homogenized_class"]
            ],
            dtype=bool,
        )

        add_summary_row(
            class_breakdown,
            group,
            mask,
            flags,
            summary_flag_order,
        )

    class_breakdown.write(
        CLASS_BREAKDOWN_PATH,
        format="ascii.csv",
        overwrite=True,
    )

    # --------------------------------------------------------
    # Invariants
    # --------------------------------------------------------

    invariant_rows = (
        len(combined) == EXPECTED_SOURCE_COUNT
    )

    invariant_all_gaia_identity = (
        len(
            set(
                int(x)
                for x in combined["source_id_join"]
            )
        )
        == EXPECTED_SOURCE_COUNT
    )

    invariant_6d_subset_5d = bool(
        np.all(
            (~flags["has_6d"])
            | flags["has_5d"]
        )
    )

    invariant_rv_measurement_subset_rv = bool(
        np.all(
            (~flags["has_rv_measurement"])
            | flags["has_rv"]
        )
    )

    all_invariants = all(
        [
            invariant_rows,
            invariant_all_gaia_identity,
            invariant_6d_subset_5d,
            invariant_rv_measurement_subset_rv,
        ]
    )

    # --------------------------------------------------------
    # Text report
    # --------------------------------------------------------

    lines = [
        "Project 02B — Measurement completeness audit",
        "=" * 54,
        "",
        f"Total Gaia-resolved sources:       {len(combined)}",
        "",
        "Measurement availability",
        "------------------------",
    ]

    for name in summary_flag_order:
        n = int(np.sum(flags[name]))

        lines.append(
            f"{name:<26} "
            f"{n:>4} / {len(combined)} "
            f"({percent(n, len(combined)):6.2f}%)"
        )

    lines.extend(
        [
            "",
            "Important interpretation",
            "------------------------",
            (
                "has_5d and has_6d indicate measurement "
                "availability only."
            ),
            (
                "No astrometric, photometric, or RV quality "
                "threshold has been applied."
            ),
            (
                "No physical membership criterion has been "
                "applied."
            ),
            "",
            "Invariants",
            "----------",
            (
                "combined rows == 1456:                "
                f"{invariant_rows}"
            ),
            (
                "combined source IDs unique == 1456:   "
                f"{invariant_all_gaia_identity}"
            ),
            (
                "has_6d subset of has_5d:              "
                f"{invariant_6d_subset_5d}"
            ),
            (
                "has_rv_measurement subset of has_rv:  "
                f"{invariant_rv_measurement_subset_rv}"
            ),
            "",
            (
                "PROJECT 02B COMPLETENESS STATUS: "
                f"{'PASS' if all_invariants else 'FAIL'}"
            ),
            "",
            "Outputs:",
            f"  {ENRICHED_PATH}",
            f"  {SUMMARY_PATH}",
            f"  {CATALOGUE_BREAKDOWN_PATH}",
            f"  {CLASS_BREAKDOWN_PATH}",
        ]
    )

    report = "\n".join(lines)

    REPORT_PATH.write_text(
        report + "\n",
        encoding="utf-8",
    )

    print(report)

    if not all_invariants:
        raise RuntimeError(
            "Project 02B measurement completeness audit failed."
        )


if __name__ == "__main__":
    main()
