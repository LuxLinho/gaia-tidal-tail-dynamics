from pathlib import Path

import numpy as np
from astropy.table import Table, MaskedColumn, join


ROOT = Path(__file__).resolve().parents[2]

KOS_INPUT = (
    ROOT
    / "data/processed/stock2_kos2024_pbint_gt_0p9.ecsv"
)

RISBUD_INPUT = (
    ROOT
    / "data/raw/literature/risbud2025/risbud2025_stock2.ecsv"
)

JADHAV_INPUT = (
    ROOT
    / "data/interim/project01/jadhav2025_stock2.ecsv"
)

OUTDIR = ROOT / "data/processed"
RESULTS_DIR = ROOT / "results/project01"

OUTDIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LONG_ECSV = OUTDIR / "stock2_literature_long.ecsv"
LONG_CSV = OUTDIR / "stock2_literature_long.csv"

MASTER_ECSV = OUTDIR / "stock2_literature_master.ecsv"
MASTER_CSV = OUTDIR / "stock2_literature_master.csv"

AUDIT = RESULTS_DIR / "project01_canonical_sample_audit.txt"


def risbud_original_class(row):
    c = int(row["FwithinTidRad"])
    l = int(row["FleadingTail"])
    t = int(row["FtrailingTail"])

    if c + l + t != 1:
        raise RuntimeError(
            f"Non-exclusive Risbud flags for {row['GaiaDR3']}"
        )

    if c == 1:
        return "C"
    if l == 1:
        return "L"
    return "T"


def get_jadhav_map(jadhav, ref):
    subset = jadhav[np.asarray(jadhav["Ref"]) == ref]

    return {
        int(row["GaiaDR3"]): row
        for row in subset
    }


def build_kos_long(kos, jadhav_map):
    rows = []

    for row in kos:
        sid = int(row["GaiaDR3"])
        jrow = jadhav_map[sid]

        rows.append(
            (
                sid,
                "Stock 2",
                "Kos2024",
                "Kos 2024",
                "high_probability_analysis_subset",
                "pbint > 0.9",
                "",
                "",
                str(jrow["Class"]),
                "Jadhav2025_distAlongO_pm10pc",
                float(row["pbint"]),
                float(row["pbbin"]),
                float(row["like"]),
                float(row["like6d"]),
                -1,
                -1,
                -1,
                -1,
                float(row["RA_ICRS"]),
                float(row["DE_ICRS"]),
                float(row["pmRA"]),
                float(row["pmDE"]),
                float(row["RV"]) if not np.ma.is_masked(row["RV"]) else np.nan,
            )
        )

    return rows


def build_risbud_long(risbud, jadhav_map):
    rows = []

    for row in risbud:
        sid = int(row["GaiaDR3"])
        jrow = jadhav_map[sid]

        original = risbud_original_class(row)

        rows.append(
            (
                sid,
                "Stock 2",
                "Risbud2025",
                "Risbud et al. 2025",
                "published_stock2_sample",
                "published catalogue inclusion",
                original,
                "Risbud2025_original_flags",
                str(jrow["Class"]),
                "Jadhav2025_distAlongO_pm10pc",
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                int(row["FleadingTail"]),
                int(row["FtrailingTail"]),
                int(row["FwithinTidRad"]),
                int(row["FCorrSamp"]),
                float(row["RA_ICRS"]),
                float(row["DE_ICRS"]),
                float(row["pmRA"]),
                float(row["pmDE"]),
                float(row["RV"]) if not np.ma.is_masked(row["RV"]) else np.nan,
            )
        )

    return rows


def main():
    kos = Table.read(KOS_INPUT, format="ascii.ecsv")
    risbud = Table.read(RISBUD_INPUT, format="ascii.ecsv")
    jadhav = Table.read(JADHAV_INPUT, format="ascii.ecsv")

    kos_jadhav = get_jadhav_map(jadhav, "Kos2024")
    risbud_jadhav = get_jadhav_map(jadhav, "Risbud2025")

    schema = [
        "gaia_dr3_source_id",
        "cluster_name",
        "catalogue_id",
        "catalogue_reference",
        "selection_stage",
        "selection_criterion",
        "original_class",
        "original_class_basis",
        "homogenized_class",
        "homogenized_class_basis",
        "kos_pbint",
        "kos_pbbin",
        "kos_like",
        "kos_like6d",
        "risbud_leading_flag",
        "risbud_trailing_flag",
        "risbud_within_tidal_radius_flag",
        "risbud_corrected_sample_flag",
        "ra_deg",
        "dec_deg",
        "pmra_masyr",
        "pmdec_masyr",
        "rv_kms",
    ]

    rows = []
    rows.extend(build_kos_long(kos, kos_jadhav))
    rows.extend(build_risbud_long(risbud, risbud_jadhav))

    long = Table(rows=rows, names=schema)

    long.sort(["gaia_dr3_source_id", "catalogue_id"])

    long.write(
        LONG_ECSV,
        format="ascii.ecsv",
        overwrite=True,
    )

    long.write(
        LONG_CSV,
        format="ascii.csv",
        overwrite=True,
    )

    # -----------------------------------------------------
    # Build one-row-per-source master table
    # -----------------------------------------------------

    all_ids = sorted(
        set(int(x) for x in long["gaia_dr3_source_id"])
    )

    master_rows = []

    for sid in all_ids:
        subset = long[
            np.asarray(long["gaia_dr3_source_id"]) == sid
        ]

        cats = set(str(x) for x in subset["catalogue_id"])

        in_kos = "Kos2024" in cats
        in_risbud = "Risbud2025" in cats

        kos_row = None
        risbud_row = None

        for row in subset:
            if str(row["catalogue_id"]) == "Kos2024":
                kos_row = row
            elif str(row["catalogue_id"]) == "Risbud2025":
                risbud_row = row

        if in_kos and in_risbud:
            source_status = "shared"
        elif in_kos:
            source_status = "kos_only"
        else:
            source_status = "risbud_only"

        homogenized_classes = sorted(
            set(str(x) for x in subset["homogenized_class"])
        )

        if len(homogenized_classes) != 1:
            raise RuntimeError(
                f"Homogenized class conflict for GaiaDR3={sid}: "
                f"{homogenized_classes}"
            )

        master_rows.append(
            (
                sid,
                int(in_kos),
                int(in_risbud),
                len(cats),
                source_status,
                homogenized_classes[0],
                (
                    float(kos_row["kos_pbint"])
                    if kos_row is not None
                    else np.nan
                ),
                (
                    str(risbud_row["original_class"])
                    if risbud_row is not None
                    else ""
                ),
                (
                    int(risbud_row["risbud_corrected_sample_flag"])
                    if risbud_row is not None
                    else -1
                ),
            )
        )

    master = Table(
        rows=master_rows,
        names=[
            "gaia_dr3_source_id",
            "in_kos2024",
            "in_risbud2025",
            "n_catalogues",
            "catalogue_status",
            "homogenized_class",
            "kos_pbint",
            "risbud_original_class",
            "risbud_corrected_sample_flag",
        ],
    )

    master.write(
        MASTER_ECSV,
        format="ascii.ecsv",
        overwrite=True,
    )

    master.write(
        MASTER_CSV,
        format="ascii.csv",
        overwrite=True,
    )

    # -----------------------------------------------------
    # Audit
    # -----------------------------------------------------

    n_long = len(long)
    n_master = len(master)

    n_shared = np.sum(
        np.asarray(master["catalogue_status"]) == "shared"
    )
    n_kos_only = np.sum(
        np.asarray(master["catalogue_status"]) == "kos_only"
    )
    n_risbud_only = np.sum(
        np.asarray(master["catalogue_status"]) == "risbud_only"
    )

    class_counts = {
        cls: int(
            np.sum(
                np.asarray(master["homogenized_class"]) == cls
            )
        )
        for cls in ["C", "L", "T"]
    }

    lines = [
        "Project 01 — Canonical literature sample audit",
        "=" * 52,
        "",
        f"Long-table rows: {n_long}",
        f"Unique master sources: {n_master}",
        "",
        "Catalogue composition:",
        f"  shared: {n_shared}",
        f"  Kos-only: {n_kos_only}",
        f"  Risbud-only: {n_risbud_only}",
        "",
        "Master homogenized C/L/T:",
        f"  C: {class_counts['C']}",
        f"  L: {class_counts['L']}",
        f"  T: {class_counts['T']}",
        "",
        "Expected invariants:",
        f"  long rows == 1063 + 1278: {n_long == 2341}",
        f"  master rows == 1456: {n_master == 1456}",
        f"  shared == 885: {n_shared == 885}",
        f"  Kos-only == 178: {n_kos_only == 178}",
        f"  Risbud-only == 393: {n_risbud_only == 393}",
        "",
        "Outputs:",
        f"  {LONG_ECSV.relative_to(ROOT)}",
        f"  {LONG_CSV.relative_to(ROOT)}",
        f"  {MASTER_ECSV.relative_to(ROOT)}",
        f"  {MASTER_CSV.relative_to(ROOT)}",
    ]

    AUDIT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(lines))


if __name__ == "__main__":
    main()
