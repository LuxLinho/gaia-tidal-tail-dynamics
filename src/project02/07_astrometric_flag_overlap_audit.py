from pathlib import Path

import numpy as np
from astropy.table import Table


INPUT_PATH = Path(
    "data/interim/project02/stock2_gaia_astrometric_quality_flags.ecsv"
)

RESULTS_DIR = Path("results/project02")

OVERLAP_PATH = (
    RESULTS_DIR / "astrometric_flag_overlap.csv"
)

SEQUENTIAL_PATH = (
    RESULTS_DIR / "astrometric_flag_sequential_flow.csv"
)

REPORT_PATH = (
    RESULTS_DIR / "astrometric_flag_overlap_audit.txt"
)

EXPECTED_SOURCE_COUNT = 1456


FLAGS = [
    "has_full_astrometric_solution",
    "pass_ruwe_1p4",
    "pass_visibility_ge10",
    "pass_gof_al_3",
    "pass_excess_noise_sig_2",
    "pass_not_duplicated",
    "pass_ipd_multi_peak_2",
    "pass_ipd_harmonic_0p1",
]


def pct(n, total):
    if total == 0:
        return np.nan
    return 100.0 * n / total


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

    for flag in FLAGS:
        if flag not in table.colnames:
            raise KeyError(
                f"Required flag '{flag}' is missing."
            )

    flags = {
        name: np.asarray(table[name], dtype=bool)
        for name in FLAGS
    }

    # ========================================================
    # Pairwise failure overlap
    # ========================================================

    overlap = Table(
        names=[
            "flag_a",
            "flag_b",
            "n_fail_a",
            "n_fail_b",
            "n_fail_both",
            "pct_a_fail_also_b",
            "pct_b_fail_also_a",
        ],
        dtype=[
            "U64",
            "U64",
            np.int64,
            np.int64,
            np.int64,
            np.float64,
            np.float64,
        ],
    )

    for i, flag_a in enumerate(FLAGS):
        fail_a = ~flags[flag_a]
        n_fail_a = int(np.sum(fail_a))

        for flag_b in FLAGS[i + 1:]:
            fail_b = ~flags[flag_b]
            n_fail_b = int(np.sum(fail_b))

            both = fail_a & fail_b
            n_both = int(np.sum(both))

            overlap.add_row(
                (
                    flag_a,
                    flag_b,
                    n_fail_a,
                    n_fail_b,
                    n_both,
                    pct(n_both, n_fail_a),
                    pct(n_both, n_fail_b),
                )
            )

    overlap.write(
        OVERLAP_PATH,
        format="ascii.csv",
        overwrite=True,
    )

    # ========================================================
    # Candidate composite definitions
    # ========================================================

    candidate_definitions = []

    candidate_definitions.append(
        (
            "A_ruwe_only",
            flags["has_full_astrometric_solution"]
            & flags["pass_ruwe_1p4"],
        )
    )

    candidate_definitions.append(
        (
            "B_ruwe_visibility",
            flags["has_full_astrometric_solution"]
            & flags["pass_ruwe_1p4"]
            & flags["pass_visibility_ge10"],
        )
    )

    candidate_definitions.append(
        (
            "C_ruwe_visibility_notdup",
            flags["has_full_astrometric_solution"]
            & flags["pass_ruwe_1p4"]
            & flags["pass_visibility_ge10"]
            & flags["pass_not_duplicated"],
        )
    )

    candidate_definitions.append(
        (
            "D_plus_gof",
            flags["has_full_astrometric_solution"]
            & flags["pass_ruwe_1p4"]
            & flags["pass_visibility_ge10"]
            & flags["pass_not_duplicated"]
            & flags["pass_gof_al_3"],
        )
    )

    candidate_definitions.append(
        (
            "E_plus_excess_noise_sig",
            flags["has_full_astrometric_solution"]
            & flags["pass_ruwe_1p4"]
            & flags["pass_visibility_ge10"]
            & flags["pass_not_duplicated"]
            & flags["pass_gof_al_3"]
            & flags["pass_excess_noise_sig_2"],
        )
    )

    candidate_definitions.append(
        (
            "F_plus_ipd",
            flags["has_full_astrometric_solution"]
            & flags["pass_ruwe_1p4"]
            & flags["pass_visibility_ge10"]
            & flags["pass_not_duplicated"]
            & flags["pass_gof_al_3"]
            & flags["pass_excess_noise_sig_2"]
            & flags["pass_ipd_multi_peak_2"]
            & flags["pass_ipd_harmonic_0p1"],
        )
    )

    sequential = Table(
        names=[
            "definition",
            "n_pass",
            "n_fail",
            "pct_pass",
            "delta_from_previous",
        ],
        dtype=[
            "U64",
            np.int64,
            np.int64,
            np.float64,
            np.int64,
        ],
    )

    previous_n = EXPECTED_SOURCE_COUNT

    for name, mask in candidate_definitions:
        n_pass = int(np.sum(mask))

        sequential.add_row(
            (
                name,
                n_pass,
                EXPECTED_SOURCE_COUNT - n_pass,
                pct(n_pass, EXPECTED_SOURCE_COUNT),
                n_pass - previous_n,
            )
        )

        previous_n = n_pass

    sequential.write(
        SEQUENTIAL_PATH,
        format="ascii.csv",
        overwrite=True,
    )

    # ========================================================
    # Important pairwise overlaps for compact report
    # ========================================================

    ruwe_fail = ~flags["pass_ruwe_1p4"]
    gof_fail = ~flags["pass_gof_al_3"]
    excess_fail = ~flags["pass_excess_noise_sig_2"]
    duplicate_fail = ~flags["pass_not_duplicated"]

    n_ruwe_fail = int(np.sum(ruwe_fail))
    n_gof_fail = int(np.sum(gof_fail))
    n_excess_fail = int(np.sum(excess_fail))
    n_duplicate_fail = int(np.sum(duplicate_fail))

    ruwe_gof = int(np.sum(ruwe_fail & gof_fail))
    ruwe_excess = int(np.sum(ruwe_fail & excess_fail))
    gof_excess = int(np.sum(gof_fail & excess_fail))

    lines = [
        "Project 02D — Astrometric flag overlap audit",
        "=" * 56,
        "",
        f"Total sources: {EXPECTED_SOURCE_COUNT}",
        "",
        "Individual failures",
        "-------------------",
        f"RUWE fail:              {n_ruwe_fail}",
        f"GOF_AL fail:            {n_gof_fail}",
        f"Excess-noise-sig fail:  {n_excess_fail}",
        f"Duplicated-source fail: {n_duplicate_fail}",
        "",
        "Important overlaps",
        "------------------",
        f"RUWE fail AND GOF fail:              {ruwe_gof}",
        f"RUWE fail AND excess-noise fail:     {ruwe_excess}",
        f"GOF fail AND excess-noise fail:      {gof_excess}",
        "",
        "Candidate sequential definitions",
        "--------------------------------",
    ]

    for row in sequential:
        lines.append(
            f"{row['definition']:<30} "
            f"{int(row['n_pass']):>4} / "
            f"{EXPECTED_SOURCE_COUNT} "
            f"({float(row['pct_pass']):6.2f}%) "
            f"delta={int(row['delta_from_previous']):+d}"
        )

    lines.extend(
        [
            "",
            "Interpretation",
            "--------------",
            (
                "This audit does not freeze the final "
                "astrometric baseline."
            ),
            (
                "It quantifies how strongly each diagnostic "
                "changes the retained sample."
            ),
            (
                "No source is removed from the canonical "
                "1456-source population."
            ),
            "",
            f"Outputs:",
            f"  {OVERLAP_PATH}",
            f"  {SEQUENTIAL_PATH}",
            f"  {REPORT_PATH}",
        ]
    )

    report = "\n".join(lines)

    REPORT_PATH.write_text(
        report + "\n",
        encoding="utf-8",
    )

    print(report)


if __name__ == "__main__":
    main()
