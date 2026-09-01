from pathlib import Path

from astroquery.vizier import Vizier


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = ROOT / "data/raw/literature/kos2024"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUTPUT_DIR / "kos2024_stock2.ecsv"

CATALOGUE = "J/A+A/691/A28/members"


def main():
    vizier = Vizier(
        columns=["*"],
        column_filters={
            "Cluster": "Stock_2",
        },
        row_limit=-1,
    )

    result = vizier.get_catalogs(CATALOGUE)

    if len(result) == 0:
        raise RuntimeError(
            "VizieR query returned no tables for Stock_2."
        )

    table = result[0]

    if len(table) == 0:
        raise RuntimeError(
            "Stock_2 query returned zero rows."
        )

    print(f"Downloaded rows: {len(table)}")
    print(f"Columns: {len(table.colnames)}")

    print("\nColumn names:")
    for name in table.colnames:
        print(f"  {name}")

    print("\nCluster values:")
    print(sorted(set(str(x) for x in table["Cluster"])))

    table.write(
        OUTPUT,
        format="ascii.ecsv",
        overwrite=True,
    )

    print()
    print(f"Wrote: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
