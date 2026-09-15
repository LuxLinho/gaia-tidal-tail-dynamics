from pathlib import Path

import numpy as np
from astropy.table import Table


# ============================================================
# Paths
# ============================================================

INPUT_PATH = Path(
    "data/interim/project02/stock2_gaia_astrometric_quality_flags.ecsv"
)

RESULTS_DIR = Path("results/project02")
INTERIM_DIR = Path("data/interim/project02")

OUTPUT_PATH = (
    INTERIM_DIR / "stock2_gaia_photometric_diagnostics.ecsv"
)

SUMMARY_PATH = (
    RESULTS_DIR / "photometric_diagnostics_summary.csv"
)

CATALOGUE_PATH = (
    RESULTS_DIR / "photometric_diagnostics_by_catalogue.csv"
)

CLASS_PATH = (
    RESULTS_DIR / "photometric_diagnostics_by_class.csv"
)

REPORT_PATH = (
    RESULTS_DIR / "photometric_diagnostics_audit.txt"
)


EXPECTED_SOURCE_COUNT = 1456


PERCENTILES = [
    ("min", 0.0),
    ("p01", 1.0),
    ("p05", 5.0),
    ("p16", 16.0),
    ("median", 50.0),
    ("p84", 84.0),
    ("p95", 95.0),
    ("p99", 99.0),
    ("max", 100.0),
]


DIAGNOSTICS = [
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "phot_g_mean_flux_over_error",
    "phot_bp_mean_flux_over_error",
    "phot_rp_mean_flux_over_error",
    "phot_bp_rp_excess_factor",
    "bp_rp",
    "bp_g",
    "g_rp",
]


def finite_array(column):
    values = []

    for value in column:
        if np.ma.is_masked(value):
            continue

        try:
            value = float(value)
        except (TypeError, ValueError):
            continue

        if np.isfinite(value):
            values.append(value)

    return np.asarray(values, dtype=float)


def summarize(table, column_name, mask=None):
    column = table[column_name]

    if mask is not None:
        column = column[mask]

    values = finite_array(column)

    result = {
        "n_valid": len(values),
        "n_missing": len(column) - len(values),
    }

    if len(values) == 0:
        for label, _ in PERCENTILES:
            result[label] = np.nan
        return result

    for label, percentile in PERCENTILES:
        result[label] = float(
            np.percentile(values, percentile)
        )

    return result


def build_rows(table, group_type, group, mask):
    rows = []

    for diagnostic in DIAGNOSTICS:
        stats = summarize(
            table,
            diagnostic,
            mask=mask,
        )

        row = [
            group_type,
            group,
            int(np.sum(mask)),
            diagnostic,
            stats["n_valid"],
            stats["n_missing"],
        ]

        for label, _ in PERCENTILES:
            row.append(stats[label])

        rows.append(row)

    return rows


def make_table(rows):
    names = [
        "group_type",
        "group",
        "n_group",
        "diagnostic",
        "n_valid",
        "n_missing",
    ] + [label for label, _ in PERCENTILES]

    dtypes = [
        "U32",
        "U64",
        np.int64,
        "U64",
        np.int64,
        np.int64,
    ] + [np.float64] * len(PERCENTILES)

    return Table(
        rows=rows,
        names=names,
        dtype=dtypes,
    )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input table not found: {INPUT_PATH}"
        )

    table = Table.read(INPUT_PATH)

    if len(table) != EXPECTED_SOURCE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_SOURCE_COUNT} rows, "
            f"found {len(table)}."
        )

    required = DIAGNOSTICS + [
        "source_id",
        "catalogue_status",
        "homogenized_class",
    ]

    for column in required:
        if column not in table.colnames:
            raise KeyError(
                f"Required column '{column}' is missing."
            )

    table.write(
        OUTPUT_PATH,
        format="ascii.ecsv",
        overwrite=True,
    )

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    full_mask = np.ones(
        len(table),
        dtype=bool,
    )

    overall_rows = build_rows(
        table,
        "overall",
        "all",
        full_mask,
    )

    make_table(overall_rows).write(
        SUMMARY_PATH,
        format="ascii.csv",
        overwrite=True,
    )

    # --------------------------------------------------------
    # Catalogue breakdown
    # --------------------------------------------------------

    catalogue_rows = []

    for group in sorted(
        set(str(x) for x in table["catalogue_status"])
    ):
        mask = np.asarray(
            [
                str(x) == group
                for x in table["catalogue_status"]
            ],
            dtype=bool,
        )

        catalogue_rows.extend(
            build_rows(
                table,
                "catalogue_status",
                group,
                mask,
            )
        )

    make_table(catalogue_rows).write(
        CATALOGUE_PATH,
        format="ascii.csv",
        overwrite=True,
    )

    # --------------------------------------------------------
    # Class breakdown
    # --------------------------------------------------------

    class_rows = []

    for group in sorted(
        set(str(x) for x in table["homogenized_class"])
    ):
        mask = np.asarray(
            [
                str(x) == group
                for x in table["homogenized_class"]
            ],
            dtype=bool,
        )

        class_rows.extend(
            build_rows(
                table,
                "homogenized_class",
                group,
                mask,
            )
        )

    make_table(class_rows).write(
        CLASS_PATH,
        format="ascii.csv",
        overwrite=True,
    )

    # --------------------------------------------------------
    # Compact report
    # --------------------------------------------------------

    g_stats = summarize(
        table,
        "phot_g_mean_flux_over_error",
    )

    bp_stats = summarize(
        table,
        "phot_bp_mean_flux_over_error",
    )

    rp_stats = summarize(
        table,
        "phot_rp_mean_flux_over_error",
    )

    excess_stats = summarize(
        table,
        "phot_bp_rp_excess_factor",
    )

    lines = [
        "Project 02E — Photometric diagnostics audit",
        "=" * 52,
        "",
        f"Total sources: {len(table)}",
        "",
        "Flux-over-error",
        "---------------",
        (
            "G median / p05:  "
            f"{g_stats['median']:.3f} / "
            f"{g_stats['p05']:.3f}"
        ),
        (
            "BP median / p05: "
            f"{bp_stats['median']:.3f} / "
            f"{bp_stats['p05']:.3f}"
        ),
        (
            "RP median / p05: "
            f"{rp_stats['median']:.3f} / "
            f"{rp_stats['p05']:.3f}"
        ),
        "",
        "BP/RP excess factor",
        "-------------------",
        (
            "median / p05 / p95: "
            f"{excess_stats['median']:.4f} / "
            f"{excess_stats['p05']:.4f} / "
            f"{excess_stats['p95']:.4f}"
        ),
        "",
        "Important interpretation",
        "------------------------",
        "No photometric quality threshold has been applied.",
        "No CMD membership selection has been applied.",
        "No source has been removed.",
        "",
        "Outputs:",
        f"  {OUTPUT_PATH}",
        f"  {SUMMARY_PATH}",
        f"  {CATALOGUE_PATH}",
        f"  {CLASS_PATH}",
        f"  {REPORT_PATH}",
    ]

    report = "\n".join(lines)

    REPORT_PATH.write_text(
        report + "\n",
        encoding="utf-8",
    )

    print(report)


if __name__ == "__main__":
    main()
