from pathlib import Path
from collections import Counter

import numpy as np
from astropy.table import Table


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "data"
    / "raw"
    / "literature"
    / "jadhav2025"
    / "jadhav2025_sources.ecsv"
)

INTERIM_DIR = ROOT / "data" / "interim" / "project01"
RESULTS_DIR = ROOT / "results" / "project01"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_ECSV = INTERIM_DIR / "jadhav2025_stock2.ecsv"
OUTPUT_CSV = INTERIM_DIR / "jadhav2025_stock2.csv"
AUDIT_OUTPUT = RESULTS_DIR / "jadhav2025_stock2_audit.txt"


def masked_count(column):
    """Count masked values safely."""
    if hasattr(column, "mask"):
        return int(np.sum(column.mask))
    return 0


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input catalogue not found: {INPUT}")

    table = Table.read(INPUT, format="ascii.ecsv")

    # ---------------------------------------------------------
    # 1. Discover how Stock 2 is named in the catalogue
    # ---------------------------------------------------------

    cluster_names = sorted(set(str(x) for x in table["Cluster"]))

    stock_like = [
        name
        for name in cluster_names
        if "stock" in name.lower()
    ]

    if not stock_like:
        raise RuntimeError(
            "No cluster names containing 'stock' were found."
        )

    print("Cluster names containing 'stock':")
    for name in stock_like:
        print(f"  - {name}")

    # Require an unambiguous Stock 2 match.
    candidates = [
        name
        for name in stock_like
        if name.lower().replace("_", "").replace(" ", "") == "stock2"
    ]

    if len(candidates) != 1:
        raise RuntimeError(
            f"Could not identify Stock 2 uniquely. Candidates: {candidates}"
        )

    stock2_name = candidates[0]

    # ---------------------------------------------------------
    # 2. Extract Stock 2
    # ---------------------------------------------------------

    stock2 = table[table["Cluster"] == stock2_name]

    if len(stock2) == 0:
        raise RuntimeError("Stock 2 extraction returned zero rows.")

    # ---------------------------------------------------------
    # 3. Basic catalogue audit
    # ---------------------------------------------------------

    refs = Counter(str(x) for x in stock2["Ref"])
    grades = Counter(str(x) for x in stock2["Grade"])
    classes = Counter(str(x) for x in stock2["Class"])

    source_ids = np.asarray(stock2["GaiaDR3"], dtype=np.int64)

    unique_source_ids, source_counts = np.unique(
        source_ids,
        return_counts=True,
    )

    duplicate_ids = unique_source_ids[source_counts > 1]
    duplicate_rows = int(np.sum(source_counts[source_counts > 1]))

    # ---------------------------------------------------------
    # 4. Per-reference Class counts
    # ---------------------------------------------------------

    ref_class_counts = {}

    for ref in sorted(refs):
        subset = stock2[stock2["Ref"] == ref]
        ref_class_counts[ref] = Counter(
            str(x) for x in subset["Class"]
        )

    # ---------------------------------------------------------
    # 5. Gaia-ID overlap between references
    # ---------------------------------------------------------

    ref_source_sets = {
        ref: set(
            int(x)
            for x in stock2[stock2["Ref"] == ref]["GaiaDR3"]
        )
        for ref in sorted(refs)
    }

    overlap_lines = []

    ref_names = sorted(ref_source_sets)

    for i, ref_a in enumerate(ref_names):
        for ref_b in ref_names[i + 1:]:
            set_a = ref_source_sets[ref_a]
            set_b = ref_source_sets[ref_b]

            overlap_lines.append(
                (
                    ref_a,
                    ref_b,
                    len(set_a & set_b),
                    len(set_a | set_b),
                )
            )

    # ---------------------------------------------------------
    # 6. Write extracted Stock 2 table
    # ---------------------------------------------------------

    stock2.write(
        OUTPUT_ECSV,
        format="ascii.ecsv",
        overwrite=True,
    )

    stock2.write(
        OUTPUT_CSV,
        format="ascii.csv",
        overwrite=True,
    )

    # ---------------------------------------------------------
    # 7. Human-readable audit
    # ---------------------------------------------------------

    lines = [
        "Project 01 — Jadhav et al. (2025) Stock 2 audit",
        "=" * 58,
        "",
        f"Input catalogue rows: {len(table)}",
        f"Detected Stock 2 label: {stock2_name}",
        f"Stock 2 catalogue rows: {len(stock2)}",
        f"Unique Gaia DR3 sources: {len(unique_source_ids)}",
        "",
        "References:",
    ]

    for ref, count in sorted(refs.items()):
        lines.append(f"  {ref}: {count}")

    lines.extend(
        [
            "",
            "Grades:",
        ]
    )

    for grade, count in sorted(grades.items()):
        lines.append(f"  {grade}: {count}")

    lines.extend(
        [
            "",
            "Classes:",
        ]
    )

    for cls, count in sorted(classes.items()):
        lines.append(f"  {cls}: {count}")

    lines.extend(
        [
            "",
            "Class counts by reference:",
        ]
    )

    for ref in sorted(ref_class_counts):
        counts = ref_class_counts[ref]
        formatted = ", ".join(
            f"{cls}={counts[cls]}"
            for cls in sorted(counts)
        )
        lines.append(f"  {ref}: {formatted}")

    lines.extend(
        [
            "",
            "Gaia DR3 source-ID audit:",
            f"  unique IDs: {len(unique_source_ids)}",
            f"  IDs appearing more than once: {len(duplicate_ids)}",
            f"  rows associated with repeated IDs: {duplicate_rows}",
            "",
            "Pairwise Gaia-source overlaps:",
        ]
    )

    if overlap_lines:
        for ref_a, ref_b, intersection, union in overlap_lines:
            lines.append(
                f"  {ref_a} vs {ref_b}: "
                f"intersection={intersection}, union={union}"
            )
    else:
        lines.append("  Only one reference present.")

    lines.extend(
        [
            "",
            "Missing-value audit:",
            f"  RV masked: {masked_count(stock2['RV'])}",
            f"  BPmag masked: {masked_count(stock2['BPmag'])}",
            f"  RPmag masked: {masked_count(stock2['RPmag'])}",
            "",
            "Outputs:",
            f"  {OUTPUT_ECSV.relative_to(ROOT)}",
            f"  {OUTPUT_CSV.relative_to(ROOT)}",
        ]
    )

    AUDIT_OUTPUT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("\n".join(lines))
    print()
    print(
        f"Wrote audit to: "
        f"{AUDIT_OUTPUT.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
