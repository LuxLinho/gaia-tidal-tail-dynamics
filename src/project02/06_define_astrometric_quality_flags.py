from pathlib import Path

import numpy as np
from astropy.table import Table


# ============================================================
# Paths
# ============================================================

INPUT_PATH = Path(
    "data/interim/project02/stock2_gaia_astrometric_diagnostics.ecsv"
)

INTERIM_DIR = Path("data/interim/project02")
RESULTS_DIR = Path("results/project02")

OUTPUT_PATH = (
    INTERIM_DIR / "stock2_gaia_astrometric_quality_flags.ecsv"
)

FLOW_PATH = (
    RESULTS_DIR / "astrometric_quality_flag_flow.csv"
)

CATALOGUE_PATH = (
    RESULTS_DIR / "astrometric_quality_by_catalogue.csv"
)

CLASS_PATH = (
    RESULTS_DIR / "astrometric_quality_by_class.csv"
)

REPORT_PATH = (
    RESULTS_DIR / "astrometric_quality_flags.txt"
)


EXPECTED_SOURCE_COUNT = 1456


# ============================================================
# Operational thresholds
#
# IMPORTANT:
#
# These thresholds define measurement-quality flags.
# They are NOT membership criteria.
#
# No source is deleted from the canonical master population.
# ============================================================

RUWE_MAX = 1.4
VISIBILITY_PERIODS_MIN = 10
GOF_AL_MAX = 3.0
EXCESS_NOISE_SIG_MAX = 2.0

IPD_FRAC_MULTI_PEAK_MAX = 2.0
IPD_GOF_HARMONIC_MAX = 0.1


# ============================================================
# Helpers
# ============================================================

def as_float_array(column):
    out = np.full(len(column), np.nan, dtype=float)

    for i, value in enumerate(column):
        if np.ma.is_masked(value):
            continue

        try:
            value = float(value)
        except (TypeError, ValueError):
            continue

        if np.isfinite(value):
            out[i] = value

    return out


def as_int_array(column):
    out = np.full(len(column), -999, dtype=np.int64)

    for i, value in enumerate(column):
        if np.ma.is_masked(value):
            continue

        try:
            out[i] = int(value)
        except (TypeError, ValueError):
            continue

    return out


def as_bool_array(column):
    result = np.zeros(len(column), dtype=bool)

    for i, value in enumerate(column):
        if np.ma.is_masked(value):
            continue

        if isinstance(value, (bool, np.bool_)):
            result[i] = bool(value)
            continue

        text = str(value).strip().lower()

        result[i] = text in {
            "true",
            "1",
            "t",
            "yes",
        }

    return result


def pct(n, total):
    if total == 0:
        return np.nan

    return 100.0 * n / total


def add_group_rows(
    output,
    table,
    group_column,
    flags,
):
    groups = sorted(
        set(str(x) for x in table[group_column])
    )

    for group in groups:
        group_mask = np.asarray(
            [
                str(x) == group
                for x in table[group_column]
            ],
            dtype=bool,
        )

        n_total = int(np.sum(group_mask))

        for flag_name, flag in flags.items():
            n_pass = int(
                np.sum(group_mask & flag)
            )

            output.add_row(
                (
                    group_column,
                    group,
                    flag_name,
                    n_total,
                    n_pass,
                    pct(n_pass, n_total),
                )
            )


# ============================================================
# Main
# ============================================================

def main():
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Astrometric diagnostics table not found: {INPUT_PATH}"
        )

    table = Table.read(INPUT_PATH)

    if len(table) != EXPECTED_SOURCE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_SOURCE_COUNT} sources, "
            f"found {len(table)}."
        )

    required = [
        "source_id",
        "astrometric_params_solved",
        "visibility_periods_used",
        "ruwe",
        "astrometric_gof_al",
        "astrometric_excess_noise_sig",
        "duplicated_source",
        "ipd_frac_multi_peak",
        "ipd_gof_harmonic_amplitude",
        "catalogue_status",
        "homogenized_class",
    ]

    for name in required:
        if name not in table.colnames:
            raise KeyError(
                f"Required column '{name}' is missing."
            )

    table = table.copy()

    # --------------------------------------------------------
    # Convert diagnostic values
    # --------------------------------------------------------

    aps = as_int_array(
        table["astrometric_params_solved"]
    )

    visibility = as_float_array(
        table["visibility_periods_used"]
    )

    ruwe = as_float_array(
        table["ruwe"]
    )

    gof = as_float_array(
        table["astrometric_gof_al"]
    )

    excess_sig = as_float_array(
        table["astrometric_excess_noise_sig"]
    )

    ipd_multi = as_float_array(
        table["ipd_frac_multi_peak"]
    )

    ipd_harmonic = as_float_array(
        table["ipd_gof_harmonic_amplitude"]
    )

    duplicated = as_bool_array(
        table["duplicated_source"]
    )

    # --------------------------------------------------------
    # Individual astrometric QC flags
    # --------------------------------------------------------

    flags = {}

    # Gaia DR3 31 = five-parameter solution
    # Gaia DR3 95 = six-parameter solution including pseudocolour.
    #
    # Both provide the position/parallax/proper-motion parameters
    # required for our 5D analysis.
    flags["has_full_astrometric_solution"] = np.isin(
        aps,
        [31, 95],
    )

    flags["pass_ruwe_1p4"] = (
        np.isfinite(ruwe)
        & (ruwe < RUWE_MAX)
    )

    # Gaia documentation warns that values below ~10 visibility
    # periods make parallaxes more vulnerable to calibration errors.
    flags["pass_visibility_ge10"] = (
        np.isfinite(visibility)
        & (visibility >= VISIBILITY_PERIODS_MIN)
    )

    flags["pass_gof_al_3"] = (
        np.isfinite(gof)
        & (gof < GOF_AL_MAX)
    )

    flags["pass_excess_noise_sig_2"] = (
        np.isfinite(excess_sig)
        & (excess_sig <= EXCESS_NOISE_SIG_MAX)
    )

    flags["pass_not_duplicated"] = (
        ~duplicated
    )

    # --------------------------------------------------------
    # IPD diagnostic flags
    #
    # These are kept separately because failed IPD diagnostics
    # may identify genuine binaries / multiples rather than
    # simple measurement failures.
    # --------------------------------------------------------

    flags["pass_ipd_multi_peak_2"] = (
        np.isfinite(ipd_multi)
        & (ipd_multi <= IPD_FRAC_MULTI_PEAK_MAX)
    )

    flags["pass_ipd_harmonic_0p1"] = (
        np.isfinite(ipd_harmonic)
        & (ipd_harmonic < IPD_GOF_HARMONIC_MAX)
    )

    # --------------------------------------------------------
    # Composite definitions
    #
    # Baseline:
    # Conservative measurement-quality requirement without
    # forcing IPD multiplicity diagnostics.
    #
    # Enhanced:
    # Adds IPD cleanliness requirements.
    # --------------------------------------------------------

    # Primary astrometric baseline:
    #
    # Keep the baseline focused on solution availability,
    # standard RUWE quality, sufficient observing epochs,
    # and Gaia duplicated-source status.
    #
    # GOF, excess-noise significance, and IPD diagnostics are
    # retained separately for sensitivity analysis because
    # they can respond to genuine unresolved multiplicity and
    # partly overlap with RUWE.
    flags["pass_astrometry_baseline"] = (
        flags["has_full_astrometric_solution"]
        & flags["pass_ruwe_1p4"]
        & flags["pass_visibility_ge10"]
        & flags["pass_not_duplicated"]
    )

    flags["pass_astrometry_gof_sensitivity"] = (
        flags["pass_astrometry_baseline"]
        & flags["pass_gof_al_3"]
    )

    flags["pass_astrometry_excess_noise_sensitivity"] = (
        flags["pass_astrometry_gof_sensitivity"]
        & flags["pass_excess_noise_sig_2"]
    )

    flags["pass_astrometry_enhanced"] = (
        flags["pass_astrometry_excess_noise_sensitivity"]
        & flags["pass_ipd_multi_peak_2"]
        & flags["pass_ipd_harmonic_0p1"]
    )

    # --------------------------------------------------------
    # Store flags
    # --------------------------------------------------------

    for name, values in flags.items():
        table[name] = values

    table.write(
        OUTPUT_PATH,
        format="ascii.ecsv",
        overwrite=True,
    )

    # --------------------------------------------------------
    # Overall flag flow
    # --------------------------------------------------------

    flow = Table(
        names=[
            "flag",
            "n_pass",
            "n_fail",
            "pct_pass",
        ],
        dtype=[
            "U64",
            np.int64,
            np.int64,
            np.float64,
        ],
    )

    for name, values in flags.items():
        n_pass = int(np.sum(values))
        n_fail = len(table) - n_pass

        flow.add_row(
            (
                name,
                n_pass,
                n_fail,
                pct(n_pass, len(table)),
            )
        )

    flow.write(
        FLOW_PATH,
        format="ascii.csv",
        overwrite=True,
    )

    # --------------------------------------------------------
    # Catalogue / class breakdown
    # --------------------------------------------------------

    breakdown_names = [
        "group_type",
        "group",
        "flag",
        "n_total",
        "n_pass",
        "pct_pass",
    ]

    breakdown_dtypes = [
        "U32",
        "U64",
        "U64",
        np.int64,
        np.int64,
        np.float64,
    ]

    catalogue_summary = Table(
        names=breakdown_names,
        dtype=breakdown_dtypes,
    )

    add_group_rows(
        catalogue_summary,
        table,
        "catalogue_status",
        flags,
    )

    catalogue_summary.write(
        CATALOGUE_PATH,
        format="ascii.csv",
        overwrite=True,
    )

    class_summary = Table(
        names=breakdown_names,
        dtype=breakdown_dtypes,
    )

    add_group_rows(
        class_summary,
        table,
        "homogenized_class",
        flags,
    )

    class_summary.write(
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

    invariant_enhanced_subset = bool(
        np.all(
            (~flags["pass_astrometry_enhanced"])
            | flags["pass_astrometry_baseline"]
        )
    )

    invariant_flags_length = all(
        len(values) == EXPECTED_SOURCE_COUNT
        for values in flags.values()
    )

    all_invariants = all(
        [
            invariant_rows,
            invariant_enhanced_subset,
            invariant_flags_length,
        ]
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    lines = [
        "Project 02D — Astrometric quality flags",
        "=" * 50,
        "",
        f"Total sources: {len(table)}",
        "",
        "Operational definitions",
        "-----------------------",
        f"RUWE < {RUWE_MAX}",
        (
            "visibility_periods_used >= "
            f"{VISIBILITY_PERIODS_MIN}"
        ),
        f"astrometric_gof_al < {GOF_AL_MAX}",
        (
            "astrometric_excess_noise_sig <= "
            f"{EXCESS_NOISE_SIG_MAX}"
        ),
        "duplicated_source == False",
        "",
        "Enhanced IPD diagnostics",
        "------------------------",
        (
            "ipd_frac_multi_peak <= "
            f"{IPD_FRAC_MULTI_PEAK_MAX}"
        ),
        (
            "ipd_gof_harmonic_amplitude < "
            f"{IPD_GOF_HARMONIC_MAX}"
        ),
        "",
        "Flag results",
        "------------",
    ]

    for name, values in flags.items():
        n_pass = int(np.sum(values))

        lines.append(
            f"{name:<32} "
            f"{n_pass:>4} / {len(table)} "
            f"({pct(n_pass, len(table)):6.2f}%)"
        )

    lines.extend(
        [
            "",
            "Interpretation",
            "--------------",
            (
                "pass_astrometry_baseline requires a full "
                "astrometric solution, RUWE < 1.4, at least 10 "
                "visibility periods, and duplicated_source == False."
            ),
            (
                "GOF_AL, excess-noise significance, and IPD "
                "diagnostics are retained as stricter sensitivity "
                "definitions rather than primary baseline gates."
            ),
            (
                "All 1456 literature sources remain in the "
                "canonical table regardless of flag status."
            ),
            (
                "No parallax or proper-motion consistency with "
                "Stock 2 has been required."
            ),
            "",
            "Invariants",
            "----------",
            f"rows == 1456:                  {invariant_rows}",
            (
                "enhanced subset of baseline:   "
                f"{invariant_enhanced_subset}"
            ),
            (
                "all flag arrays length 1456:   "
                f"{invariant_flags_length}"
            ),
            "",
            (
                "PROJECT 02D FLAG STATUS: "
                f"{'PASS' if all_invariants else 'FAIL'}"
            ),
            "",
            "Outputs:",
            f"  {OUTPUT_PATH}",
            f"  {FLOW_PATH}",
            f"  {CATALOGUE_PATH}",
            f"  {CLASS_PATH}",
            f"  {REPORT_PATH}",
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
            "Project 02D astrometric QC flag audit failed."
        )


if __name__ == "__main__":
    main()
