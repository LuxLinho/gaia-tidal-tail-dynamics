from pathlib import Path

import numpy as np
from astropy.table import Table


# ============================================================
# Paths
# ============================================================

INPUT_PATH = Path(
    "data/interim/project02/stock2_gaia_measurement_completeness.ecsv"
)

RESULTS_DIR = Path("results/project02")
INTERIM_DIR = Path("data/interim/project02")

ENRICHED_PATH = (
    INTERIM_DIR / "stock2_gaia_astrometric_diagnostics.ecsv"
)

OVERALL_SUMMARY_PATH = (
    RESULTS_DIR / "astrometric_diagnostics_summary.csv"
)

CATALOGUE_SUMMARY_PATH = (
    RESULTS_DIR / "astrometric_diagnostics_by_catalogue.csv"
)

CLASS_SUMMARY_PATH = (
    RESULTS_DIR / "astrometric_diagnostics_by_class.csv"
)

DISCRETE_SUMMARY_PATH = (
    RESULTS_DIR / "astrometric_discrete_diagnostics.csv"
)

REPORT_PATH = (
    RESULTS_DIR / "astrometric_diagnostics_audit.txt"
)


EXPECTED_SOURCE_COUNT = 1456


# ============================================================
# Continuous diagnostics to audit
# ============================================================

CONTINUOUS_COLUMNS = [
    "ruwe",
    "visibility_periods_used",
    "astrometric_n_good_obs_al",
    "astrometric_n_bad_obs_al",
    "astrometric_gof_al",
    "astrometric_excess_noise",
    "astrometric_excess_noise_sig",
    "ipd_frac_multi_peak",
    "ipd_gof_harmonic_amplitude",
    "parallax_error",
    "pmra_error",
    "pmdec_error",
    "parallax_over_error",
    "abs_pmra_over_error",
    "abs_pmdec_over_error",
]

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


# ============================================================
# Helpers
# ============================================================

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


def safe_ratio(numerator, denominator, absolute=False):
    out = np.full(len(numerator), np.nan, dtype=float)

    for i, (num, den) in enumerate(zip(numerator, denominator)):
        if np.ma.is_masked(num) or np.ma.is_masked(den):
            continue

        try:
            num = float(num)
            den = float(den)
        except (TypeError, ValueError):
            continue

        if not np.isfinite(num) or not np.isfinite(den):
            continue

        if den <= 0:
            continue

        if absolute:
            num = abs(num)

        out[i] = num / den

    return out


def summarize_column(table, column_name, mask=None):
    if column_name not in table.colnames:
        raise KeyError(
            f"Required diagnostic column '{column_name}' is missing."
        )

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


def build_summary_table(
    table,
    group_name,
    group_value,
    mask,
):
    rows = []

    n_group = int(np.sum(mask))

    for column_name in CONTINUOUS_COLUMNS:
        stats = summarize_column(
            table,
            column_name,
            mask=mask,
        )

        row = [
            group_name,
            group_value,
            n_group,
            column_name,
            stats["n_valid"],
            stats["n_missing"],
        ]

        for label, _ in PERCENTILES:
            row.append(stats[label])

        rows.append(row)

    return rows


def create_summary_table(rows):
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


# ============================================================
# Main
# ============================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Project 02B table not found: {INPUT_PATH}"
        )

    table = Table.read(INPUT_PATH)

    if len(table) != EXPECTED_SOURCE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_SOURCE_COUNT} rows, "
            f"found {len(table)}."
        )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "source_id",
        "parallax",
        "parallax_error",
        "pmra",
        "pmra_error",
        "pmdec",
        "pmdec_error",
        "ruwe",
        "visibility_periods_used",
        "astrometric_params_solved",
        "astrometric_n_good_obs_al",
        "astrometric_n_bad_obs_al",
        "astrometric_gof_al",
        "astrometric_excess_noise",
        "astrometric_excess_noise_sig",
        "ipd_frac_multi_peak",
        "ipd_gof_harmonic_amplitude",
        "duplicated_source",
        "catalogue_status",
        "homogenized_class",
    ]

    for column in required_columns:
        if column not in table.colnames:
            raise KeyError(
                f"Required column '{column}' is missing."
            )

    table = table.copy()

    # --------------------------------------------------------
    # Derived diagnostics
    #
    # These are descriptive only.
    # They are not membership criteria and not QC thresholds.
    # --------------------------------------------------------

    table["parallax_over_error"] = safe_ratio(
        table["parallax"],
        table["parallax_error"],
        absolute=False,
    )

    table["abs_pmra_over_error"] = safe_ratio(
        table["pmra"],
        table["pmra_error"],
        absolute=True,
    )

    table["abs_pmdec_over_error"] = safe_ratio(
        table["pmdec"],
        table["pmdec_error"],
        absolute=True,
    )

    table.write(
        ENRICHED_PATH,
        format="ascii.ecsv",
        overwrite=True,
    )

    # --------------------------------------------------------
    # Overall continuous-diagnostic summary
    # --------------------------------------------------------

    overall_rows = []

    full_mask = np.ones(
        len(table),
        dtype=bool,
    )

    overall_rows.extend(
        build_summary_table(
            table,
            "overall",
            "all",
            full_mask,
        )
    )

    overall_summary = create_summary_table(
        overall_rows
    )

    overall_summary.write(
        OVERALL_SUMMARY_PATH,
        format="ascii.csv",
        overwrite=True,
    )

    # --------------------------------------------------------
    # Catalogue breakdown
    # --------------------------------------------------------

    catalogue_rows = []

    catalogue_values = sorted(
        set(str(x) for x in table["catalogue_status"])
    )

    for group in catalogue_values:
        mask = np.asarray(
            [
                str(x) == group
                for x in table["catalogue_status"]
            ],
            dtype=bool,
        )

        catalogue_rows.extend(
            build_summary_table(
                table,
                "catalogue_status",
                group,
                mask,
            )
        )

    catalogue_summary = create_summary_table(
        catalogue_rows
    )

    catalogue_summary.write(
        CATALOGUE_SUMMARY_PATH,
        format="ascii.csv",
        overwrite=True,
    )

    # --------------------------------------------------------
    # Homogenized-class breakdown
    # --------------------------------------------------------

    class_rows = []

    class_values = sorted(
        set(str(x) for x in table["homogenized_class"])
    )

    for group in class_values:
        mask = np.asarray(
            [
                str(x) == group
                for x in table["homogenized_class"]
            ],
            dtype=bool,
        )

        class_rows.extend(
            build_summary_table(
                table,
                "homogenized_class",
                group,
                mask,
            )
        )

    class_summary = create_summary_table(
        class_rows
    )

    class_summary.write(
        CLASS_SUMMARY_PATH,
        format="ascii.csv",
        overwrite=True,
    )

    # --------------------------------------------------------
    # Discrete diagnostics
    # --------------------------------------------------------

    discrete = Table(
        names=[
            "diagnostic",
            "value",
            "count",
            "fraction_percent",
        ],
        dtype=[
            "U64",
            "U64",
            np.int64,
            np.float64,
        ],
    )

    # astrometric_params_solved
    aps_values = sorted(
        set(
            str(x)
            for x in table["astrometric_params_solved"]
        )
    )

    for value in aps_values:
        count = int(
            np.sum(
                [
                    str(x) == value
                    for x in table["astrometric_params_solved"]
                ]
            )
        )

        discrete.add_row(
            (
                "astrometric_params_solved",
                value,
                count,
                100.0 * count / len(table),
            )
        )

    # duplicated_source
    dup_values = sorted(
        set(
            str(x)
            for x in table["duplicated_source"]
        )
    )

    for value in dup_values:
        count = int(
            np.sum(
                [
                    str(x) == value
                    for x in table["duplicated_source"]
                ]
            )
        )

        discrete.add_row(
            (
                "duplicated_source",
                value,
                count,
                100.0 * count / len(table),
            )
        )

    discrete.write(
        DISCRETE_SUMMARY_PATH,
        format="ascii.csv",
        overwrite=True,
    )

    # --------------------------------------------------------
    # Basic invariant checks
    # --------------------------------------------------------

    invariant_rows = (
        len(table) == EXPECTED_SOURCE_COUNT
    )

    invariant_source_ids_unique = (
        len(
            set(
                int(str(x).strip())
                for x in table["source_id"]
            )
        )
        == EXPECTED_SOURCE_COUNT
    )

    invariant_ruwe_available = (
        len(finite_array(table["ruwe"]))
        > 0
    )

    invariant_parallax_error_available = (
        len(finite_array(table["parallax_error"]))
        == EXPECTED_SOURCE_COUNT
    )

    invariant_pmra_error_available = (
        len(finite_array(table["pmra_error"]))
        == EXPECTED_SOURCE_COUNT
    )

    invariant_pmdec_error_available = (
        len(finite_array(table["pmdec_error"]))
        == EXPECTED_SOURCE_COUNT
    )

    all_invariants = all(
        [
            invariant_rows,
            invariant_source_ids_unique,
            invariant_ruwe_available,
            invariant_parallax_error_available,
            invariant_pmra_error_available,
            invariant_pmdec_error_available,
        ]
    )

    # --------------------------------------------------------
    # Compact report
    # --------------------------------------------------------

    ruwe_stats = summarize_column(
        table,
        "ruwe",
    )

    parallax_error_stats = summarize_column(
        table,
        "parallax_error",
    )

    pmra_error_stats = summarize_column(
        table,
        "pmra_error",
    )

    pmdec_error_stats = summarize_column(
        table,
        "pmdec_error",
    )

    parallax_snr_stats = summarize_column(
        table,
        "parallax_over_error",
    )

    lines = [
        "Project 02C — Astrometric diagnostics audit",
        "=" * 54,
        "",
        f"Total sources: {len(table)}",
        "",
        "RUWE",
        "----",
        f"valid:   {ruwe_stats['n_valid']}",
        f"missing: {ruwe_stats['n_missing']}",
        f"median:  {ruwe_stats['median']:.4f}",
        f"p84:     {ruwe_stats['p84']:.4f}",
        f"p95:     {ruwe_stats['p95']:.4f}",
        f"p99:     {ruwe_stats['p99']:.4f}",
        f"max:     {ruwe_stats['max']:.4f}",
        "",
        "Astrometric uncertainties",
        "-------------------------",
        (
            "parallax_error median / p95: "
            f"{parallax_error_stats['median']:.6f} / "
            f"{parallax_error_stats['p95']:.6f}"
        ),
        (
            "pmra_error median / p95:     "
            f"{pmra_error_stats['median']:.6f} / "
            f"{pmra_error_stats['p95']:.6f}"
        ),
        (
            "pmdec_error median / p95:    "
            f"{pmdec_error_stats['median']:.6f} / "
            f"{pmdec_error_stats['p95']:.6f}"
        ),
        "",
        "Derived diagnostic",
        "------------------",
        (
            "parallax_over_error median / p05: "
            f"{parallax_snr_stats['median']:.4f} / "
            f"{parallax_snr_stats['p05']:.4f}"
        ),
        "",
        "Important interpretation",
        "------------------------",
        "No RUWE threshold has been applied.",
        "No visibility-period threshold has been applied.",
        "No duplicated-source exclusion has been applied.",
        "No astrometric-error threshold has been applied.",
        "No physical membership criterion has been applied.",
        "",
        "Invariants",
        "----------",
        f"rows == 1456:                    {invariant_rows}",
        (
            "source IDs unique == 1456:       "
            f"{invariant_source_ids_unique}"
        ),
        (
            "RUWE available for some sources: "
            f"{invariant_ruwe_available}"
        ),
        (
            "parallax_error complete:          "
            f"{invariant_parallax_error_available}"
        ),
        (
            "pmra_error complete:              "
            f"{invariant_pmra_error_available}"
        ),
        (
            "pmdec_error complete:             "
            f"{invariant_pmdec_error_available}"
        ),
        "",
        (
            "PROJECT 02C DIAGNOSTICS STATUS: "
            f"{'PASS' if all_invariants else 'FAIL'}"
        ),
        "",
        "Outputs:",
        f"  {ENRICHED_PATH}",
        f"  {OVERALL_SUMMARY_PATH}",
        f"  {CATALOGUE_SUMMARY_PATH}",
        f"  {CLASS_SUMMARY_PATH}",
        f"  {DISCRETE_SUMMARY_PATH}",
        f"  {REPORT_PATH}",
    ]

    report = "\n".join(lines)

    REPORT_PATH.write_text(
        report + "\n",
        encoding="utf-8",
    )

    print(report)

    if not all_invariants:
        raise RuntimeError(
            "Project 02C astrometric diagnostics audit failed."
        )


if __name__ == "__main__":
    main()
