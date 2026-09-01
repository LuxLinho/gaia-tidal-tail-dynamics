from pathlib import Path
from collections import Counter

import numpy as np
from astropy.table import Table


ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data/interim/project01/jadhav2025_stock2.ecsv"

PROCESSED_DIR = ROOT / "data/processed"
RESULTS_DIR = ROOT / "results/project01"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LONG_ECSV = PROCESSED_DIR / "stock2_jadhav_provenance_long.ecsv"
LONG_CSV = PROCESSED_DIR / "stock2_jadhav_provenance_long.csv"

SUMMARY_CSV = RESULTS_DIR / "project01_catalogue_summary.csv"
OVERLAP_CSV = RESULTS_DIR / "project01_overlap_summary.csv"
CLASS_MATRIX_CSV = RESULTS_DIR / "project01_shared_class_matrix.csv"
AUDIT_TXT = RESULTS_DIR / "project01_jadhav_provenance_audit.txt"


EXPECTED_REFS = {"Kos2024", "Risbud2025"}
VALID_CLASSES = {"C", "L", "T"}

CLASS_LABELS = {
    "C": "cluster",
    "L": "leading_tail",
    "T": "trailing_tail",
}


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input: {INPUT}")

    table = Table.read(INPUT, format="ascii.ecsv")

    # ---------------------------------------------------------
    # 1. Validate expected catalogue structure
    # ---------------------------------------------------------

    refs = set(str(x) for x in table["Ref"])
    classes = set(str(x) for x in table["Class"])
    grades = set(str(x) for x in table["Grade"])

    if refs != EXPECTED_REFS:
        raise RuntimeError(
            f"Unexpected references. Found {refs}, expected {EXPECTED_REFS}"
        )

    if not classes.issubset(VALID_CLASSES):
        raise RuntimeError(
            f"Unexpected membership classes: {classes}"
        )

    if grades != {"S"}:
        raise RuntimeError(
            f"Unexpected catalogue grades: {grades}"
        )

    # ---------------------------------------------------------
    # 2. Explicit within-catalogue duplicate audit
    # ---------------------------------------------------------

    duplicate_report = {}

    for ref in sorted(EXPECTED_REFS):
        subset = table[table["Ref"] == ref]

        ids = np.asarray(subset["GaiaDR3"], dtype=np.int64)

        unique_ids, counts = np.unique(ids, return_counts=True)

        duplicated = unique_ids[counts > 1]

        duplicate_report[ref] = {
            "rows": len(subset),
            "unique_ids": len(unique_ids),
            "duplicate_ids": len(duplicated),
        }

        if len(duplicated) != 0:
            raise RuntimeError(
                f"{ref} contains {len(duplicated)} duplicated Gaia source IDs."
            )

    # ---------------------------------------------------------
    # 3. Build canonical Jadhav provenance long table
    # ---------------------------------------------------------

    out = Table()

    out["gaia_dr3_source_id"] = np.asarray(
        table["GaiaDR3"],
        dtype=np.int64,
    )

    out["cluster_name"] = [
        "Stock 2"
        for _ in table
    ]

    out["catalogue_id"] = [
        str(x)
        for x in table["Ref"]
    ]

    out["catalogue_grade"] = [
        str(x)
        for x in table["Grade"]
    ]

    out["jadhav_class"] = [
        str(x)
        for x in table["Class"]
    ]

    out["jadhav_class_label"] = [
        CLASS_LABELS[str(x)]
        for x in table["Class"]
    ]

    out["classification_provenance"] = [
        "Jadhav2025_master_catalogue"
        for _ in table
    ]

    out["ra_deg"] = table["RA_ICRS"]
    out["dec_deg"] = table["DE_ICRS"]
    out["distance_bj_pc"] = table["rmedgeo"]

    out["pmra_masyr"] = table["pmRA"]
    out["pmdec_masyr"] = table["pmDE"]
    out["rv_kms"] = table["RV"]

    out["gmag"] = table["Gmag"]
    out["bpmag"] = table["BPmag"]
    out["rpmag"] = table["RPmag"]

    out.write(
        LONG_ECSV,
        format="ascii.ecsv",
        overwrite=True,
    )

    out.write(
        LONG_CSV,
        format="ascii.csv",
        overwrite=True,
    )

    # ---------------------------------------------------------
    # 4. Catalogue summary
    # ---------------------------------------------------------

    summary = Table(
        names=[
            "catalogue_id",
            "n_rows",
            "n_unique_sources",
            "n_cluster",
            "n_leading",
            "n_trailing",
            "grade",
        ],
        dtype=[
            "U20",
            "i8",
            "i8",
            "i8",
            "i8",
            "i8",
            "U1",
        ],
    )

    ref_maps = {}

    for ref in sorted(EXPECTED_REFS):
        subset = out[out["catalogue_id"] == ref]

        class_counts = Counter(
            str(x)
            for x in subset["jadhav_class"]
        )

        ids = set(
            int(x)
            for x in subset["gaia_dr3_source_id"]
        )

        ref_maps[ref] = {
            int(row["gaia_dr3_source_id"]): str(row["jadhav_class"])
            for row in subset
        }

        summary.add_row(
            [
                ref,
                len(subset),
                len(ids),
                class_counts.get("C", 0),
                class_counts.get("L", 0),
                class_counts.get("T", 0),
                "S",
            ]
        )

    summary.write(
        SUMMARY_CSV,
        format="ascii.csv",
        overwrite=True,
    )

    # ---------------------------------------------------------
    # 5. Catalogue overlap
    # ---------------------------------------------------------

    kos_ids = set(ref_maps["Kos2024"])
    risbud_ids = set(ref_maps["Risbud2025"])

    shared = kos_ids & risbud_ids
    union = kos_ids | risbud_ids

    kos_only = kos_ids - risbud_ids
    risbud_only = risbud_ids - kos_ids

    overlap = Table(
        rows=[
            (
                len(kos_ids),
                len(risbud_ids),
                len(shared),
                len(kos_only),
                len(risbud_only),
                len(union),
                len(shared) / len(union),
                len(shared) / len(kos_ids),
                len(shared) / len(risbud_ids),
            )
        ],
        names=[
            "n_kos2024",
            "n_risbud2025",
            "n_shared",
            "n_kos_only",
            "n_risbud_only",
            "n_union",
            "jaccard_similarity",
            "shared_fraction_kos",
            "shared_fraction_risbud",
        ],
    )

    overlap.write(
        OVERLAP_CSV,
        format="ascii.csv",
        overwrite=True,
    )

    # ---------------------------------------------------------
    # 6. Shared-source class agreement matrix
    # ---------------------------------------------------------

    matrix_counts = Counter()

    n_class_agree = 0

    for source_id in shared:
        kos_class = ref_maps["Kos2024"][source_id]
        risbud_class = ref_maps["Risbud2025"][source_id]

        matrix_counts[(kos_class, risbud_class)] += 1

        if kos_class == risbud_class:
            n_class_agree += 1

    matrix = Table(
        names=[
            "kos2024_class",
            "risbud2025_class",
            "n_sources",
        ],
        dtype=[
            "U1",
            "U1",
            "i8",
        ],
    )

    for kos_class in ["C", "L", "T"]:
        for risbud_class in ["C", "L", "T"]:
            matrix.add_row(
                [
                    kos_class,
                    risbud_class,
                    matrix_counts.get(
                        (kos_class, risbud_class),
                        0,
                    ),
                ]
            )

    matrix.write(
        CLASS_MATRIX_CSV,
        format="ascii.csv",
        overwrite=True,
    )

    class_agreement_fraction = (
        n_class_agree / len(shared)
        if shared
        else np.nan
    )

    # ---------------------------------------------------------
    # 7. Human-readable audit
    # ---------------------------------------------------------

    lines = [
        "Project 01 — Jadhav provenance reconstruction",
        "=" * 54,
        "",
        "Within-catalogue duplicate audit:",
    ]

    for ref in sorted(duplicate_report):
        x = duplicate_report[ref]

        lines.append(
            f"  {ref}: rows={x['rows']}, "
            f"unique={x['unique_ids']}, "
            f"duplicate_ids={x['duplicate_ids']}"
        )

    lines.extend(
        [
            "",
            "Catalogue overlap:",
            f"  Kos2024: {len(kos_ids)}",
            f"  Risbud2025: {len(risbud_ids)}",
            f"  shared: {len(shared)}",
            f"  Kos-only: {len(kos_only)}",
            f"  Risbud-only: {len(risbud_only)}",
            f"  union: {len(union)}",
            f"  Jaccard similarity: {len(shared) / len(union):.4f}",
            f"  shared / Kos2024: {len(shared) / len(kos_ids):.4f}",
            f"  shared / Risbud2025: {len(shared) / len(risbud_ids):.4f}",
            "",
            "Shared-source C/L/T agreement:",
            f"  shared sources: {len(shared)}",
            f"  same C/L/T class: {n_class_agree}",
            f"  class agreement fraction: {class_agreement_fraction:.4f}",
            "",
            "Class matrix (Kos2024 -> Risbud2025):",
        ]
    )

    for kos_class in ["C", "L", "T"]:
        values = []

        for risbud_class in ["C", "L", "T"]:
            values.append(
                f"{risbud_class}="
                f"{matrix_counts.get((kos_class, risbud_class), 0)}"
            )

        lines.append(
            f"  {kos_class}: " + ", ".join(values)
        )

    lines.extend(
        [
            "",
            "Canonical provenance output:",
            f"  {LONG_ECSV.relative_to(ROOT)}",
            f"  {LONG_CSV.relative_to(ROOT)}",
            "",
            "Important:",
            "  jadhav_class is treated as a homogenized literature",
            "  classification from the Jadhav2025 master catalogue.",
            "  It is not yet labelled as the original-source class.",
        ]
    )

    AUDIT_TXT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(lines))


if __name__ == "__main__":
    main()

