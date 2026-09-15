from pathlib import Path

import numpy as np
from astropy.table import Table


# ============================================================
# Paths
# ============================================================

INPUT_PATH = Path(
    "data/interim/project02/stock2_gaia_photometric_diagnostics.ecsv"
)

INTERIM_DIR = Path("data/interim/project02")
RESULTS_DIR = Path("results/project02")

OUTPUT_PATH = (
    INTERIM_DIR / "stock2_gaia_corrected_flux_excess.ecsv"
)

SUMMARY_PATH = (
    RESULTS_DIR / "corrected_flux_excess_summary.csv"
)

CATALOGUE_PATH = (
    RESULTS_DIR / "corrected_flux_excess_by_catalogue.csv"
)

CLASS_PATH = (
    RESULTS_DIR / "corrected_flux_excess_by_class.csv"
)

REPORT_PATH = (
    RESULTS_DIR / "corrected_flux_excess_audit.txt"
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


# ============================================================
# Riello et al. (2021) corrected BP/RP flux excess
#
# C* = phot_bp_rp_excess_factor - f(BP-RP)
# ============================================================

def correct_flux_excess_factor(bp_rp, flux_excess):
    bp_rp = np.asarray(bp_rp, dtype=float)
    flux_excess = np.asarray(flux_excess, dtype=float)

    if bp_rp.shape != flux_excess.shape:
        raise ValueError(
            "bp_rp and flux_excess arrays must have the same shape."
        )

    correction = np.full(
        bp_rp.shape,
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(bp_rp)
        & np.isfinite(flux_excess)
    )

    blue = valid & (bp_rp < 0.5)

    middle = (
        valid
        & (bp_rp >= 0.5)
        & (bp_rp < 4.0)
    )

    red = valid & (bp_rp >= 4.0)

    x = bp_rp

    correction[blue] = (
        1.154360
        + 0.033772 * x[blue]
        + 0.032277 * x[blue] ** 2
    )

    correction[middle] = (
        1.162004
        + 0.011464 * x[middle]
        + 0.049255 * x[middle] ** 2
        - 0.005879 * x[middle] ** 3
    )

    correction[red] = (
        1.057572
        + 0.140537 * x[red]
    )

    corrected = flux_excess - correction
    corrected[~valid] = np.nan

    return corrected


def to_float_array(column):
    result = np.full(
        len(column),
        np.nan,
        dtype=float,
    )

    for i, value in enumerate(column):
        if np.ma.is_masked(value):
            continue

        try:
            value = float(value)
        except (TypeError, ValueError):
            continue

        if np.isfinite(value):
            result[i] = value

    return result


def summarize(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    result = {
        "n_valid": len(values),
    }

    for label, percentile in PERCENTILES:
        if len(values) == 0:
            result[label] = np.nan
        else:
            result[label] = float(
                np.percentile(
                    values,
                    percentile,
                )
            )

    return result


def build_group_table(table, group_column):
    output = Table(
        names=[
            "group",
            "n_total",
            "n_valid",
            "median",
            "p05",
            "p95",
            "p99",
        ],
        dtype=[
            "U64",
            np.int64,
            np.int64,
            np.float64,
            np.float64,
            np.float64,
            np.float64,
        ],
    )

    groups = sorted(
        set(
            str(x)
            for x in table[group_column]
        )
    )

    cstar = np.asarray(
        table["phot_bp_rp_excess_factor_corrected"],
        dtype=float,
    )

    for group in groups:
        mask = np.asarray(
            [
                str(x) == group
                for x in table[group_column]
            ],
            dtype=bool,
        )

        stats = summarize(
            cstar[mask]
        )

        output.add_row(
            (
                group,
                int(np.sum(mask)),
                stats["n_valid"],
                stats["median"],
                stats["p05"],
                stats["p95"],
                stats["p99"],
            )
        )

    return output


def main():
    INTERIM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    required = [
        "source_id",
        "bp_rp",
        "phot_bp_rp_excess_factor",
        "catalogue_status",
        "homogenized_class",
    ]

    for name in required:
        if name not in table.colnames:
            raise KeyError(
                f"Required column '{name}' is missing."
            )

    table = table.copy()

    bp_rp = to_float_array(
        table["bp_rp"]
    )

    raw_excess = to_float_array(
        table["phot_bp_rp_excess_factor"]
    )

    corrected = correct_flux_excess_factor(
        bp_rp,
        raw_excess,
    )

    table[
        "phot_bp_rp_excess_factor_corrected"
    ] = corrected

    table.write(
        OUTPUT_PATH,
        format="ascii.ecsv",
        overwrite=True,
    )

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    stats = summarize(corrected)

    summary = Table(
        names=[
            "diagnostic",
            "n_valid",
            "min",
            "p01",
            "p05",
            "p16",
            "median",
            "p84",
            "p95",
            "p99",
            "max",
        ],
        dtype=[
            "U64",
            np.int64,
        ] + [np.float64] * 9,
    )

    summary.add_row(
        (
            "phot_bp_rp_excess_factor_corrected",
            stats["n_valid"],
            stats["min"],
            stats["p01"],
            stats["p05"],
            stats["p16"],
            stats["median"],
            stats["p84"],
            stats["p95"],
            stats["p99"],
            stats["max"],
        )
    )

    summary.write(
        SUMMARY_PATH,
        format="ascii.csv",
        overwrite=True,
    )

    # --------------------------------------------------------
    # Group breakdowns
    # --------------------------------------------------------

    build_group_table(
        table,
        "catalogue_status",
    ).write(
        CATALOGUE_PATH,
        format="ascii.csv",
        overwrite=True,
    )

    build_group_table(
        table,
        "homogenized_class",
    ).write(
        CLASS_PATH,
        format="ascii.csv",
        overwrite=True,
    )

    # --------------------------------------------------------
    # Invariants
    # --------------------------------------------------------

    invariant_rows = (
        len(table) == EXPECTED_SOURCE_COUNT
    )

    invariant_complete = (
        stats["n_valid"]
        == EXPECTED_SOURCE_COUNT
    )

    all_invariants = (
        invariant_rows
        and invariant_complete
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    lines = [
        "Project 02E — Corrected BP/RP flux-excess audit",
        "=" * 58,
        "",
        f"Total sources: {len(table)}",
        "",
        "Corrected flux excess C*",
        "------------------------",
        f"valid:   {stats['n_valid']}",
        f"median:  {stats['median']:.6f}",
        f"p05:     {stats['p05']:.6f}",
        f"p16:     {stats['p16']:.6f}",
        f"p84:     {stats['p84']:.6f}",
        f"p95:     {stats['p95']:.6f}",
        f"p99:     {stats['p99']:.6f}",
        f"min:     {stats['min']:.6f}",
        f"max:     {stats['max']:.6f}",
        "",
        "Interpretation",
        "--------------",
        (
            "C* is the colour-corrected BP/RP flux-excess "
            "diagnostic."
        ),
        (
            "Well-behaved Gaia photometry is expected to "
            "lie near C* = 0."
        ),
        "No C* threshold has been applied.",
        "No CMD membership criterion has been applied.",
        "All 1456 sources remain in the canonical population.",
        "",
        "Invariants",
        "----------",
        f"rows == 1456:          {invariant_rows}",
        f"C* available == 1456: {invariant_complete}",
        "",
        (
            "PROJECT 02E CORRECTED-EXCESS STATUS: "
            f"{'PASS' if all_invariants else 'FAIL'}"
        ),
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

    if not all_invariants:
        raise RuntimeError(
            "Corrected flux-excess audit failed."
        )


if __name__ == "__main__":
    main()
