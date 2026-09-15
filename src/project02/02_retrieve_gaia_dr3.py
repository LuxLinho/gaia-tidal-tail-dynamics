from pathlib import Path

from astropy.table import Table
from astroquery.gaia import Gaia


# ============================================================
# Paths
# ============================================================

QUERY_PATH = Path(
    "data/interim/project02/gaia_dr3_source_query.ecsv"
)

RAW_DIR = Path("data/raw/gaia/project02")
RESULTS_DIR = Path("results/project02")

OUTPUT_ECSV = RAW_DIR / "gaia_dr3_enrichment_raw.ecsv"
OUTPUT_CSV = RAW_DIR / "gaia_dr3_enrichment_raw.csv"

QUERY_TEXT_PATH = RESULTS_DIR / "gaia_dr3_retrieval_query.adql"
RETRIEVAL_REPORT_PATH = RESULTS_DIR / "gaia_dr3_retrieval.txt"


# ============================================================
# Expected Project 02A invariant
# ============================================================

EXPECTED_SOURCE_COUNT = 1456
UPLOAD_TABLE_NAME = "stock2_source_ids"


# ============================================================
# Gaia DR3 query
#
# Important:
# - Exact source_id join only.
# - No coordinate crossmatch.
# - No quality cuts.
# - No membership cuts.
# - Raw Gaia measurements are preserved.
# ============================================================

ADQL_QUERY = """
SELECT
    u.gaia_dr3_source_id AS input_source_id,

    g.source_id,
    g.ra,
    g.dec,
    g.ref_epoch,

    g.parallax,
    g.parallax_error,

    g.pmra,
    g.pmra_error,

    g.pmdec,
    g.pmdec_error,

    g.ra_error,
    g.dec_error,

    g.ra_dec_corr,
    g.ra_parallax_corr,
    g.ra_pmra_corr,
    g.ra_pmdec_corr,
    g.dec_parallax_corr,
    g.dec_pmra_corr,
    g.dec_pmdec_corr,
    g.parallax_pmra_corr,
    g.parallax_pmdec_corr,
    g.pmra_pmdec_corr,

    g.astrometric_params_solved,
    g.astrometric_n_good_obs_al,
    g.astrometric_n_bad_obs_al,
    g.astrometric_gof_al,
    g.astrometric_excess_noise,
    g.astrometric_excess_noise_sig,
    g.visibility_periods_used,
    g.ruwe,
    g.ipd_frac_multi_peak,
    g.ipd_gof_harmonic_amplitude,
    g.duplicated_source,

    g.phot_g_mean_mag,
    g.phot_bp_mean_mag,
    g.phot_rp_mean_mag,

    g.phot_g_mean_flux_over_error,
    g.phot_bp_mean_flux_over_error,
    g.phot_rp_mean_flux_over_error,

    g.bp_rp,
    g.bp_g,
    g.g_rp,

    g.phot_bp_rp_excess_factor,

    g.radial_velocity,
    g.radial_velocity_error,

    g.rv_method_used,
    g.rv_nb_transits,
    g.rv_expected_sig_to_noise,
    g.rv_renormalised_gof,
    g.rv_chisq_pvalue,
    g.rv_time_duration,
    g.rv_amplitude_robust

FROM tap_upload.stock2_source_ids AS u

LEFT OUTER JOIN gaiadr3.gaia_source AS g
    ON u.gaia_dr3_source_id = g.source_id
"""


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------

    if not QUERY_PATH.exists():
        raise FileNotFoundError(
            "Project 02A query manifest not found: "
            f"{QUERY_PATH}"
        )

    query_table = Table.read(QUERY_PATH)

    required_column = "gaia_dr3_source_id"

    if required_column not in query_table.colnames:
        raise KeyError(
            f"Required column '{required_column}' is missing "
            f"from {QUERY_PATH}"
        )

    n_input = len(query_table)

    if n_input != EXPECTED_SOURCE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_SOURCE_COUNT} source IDs, "
            f"but found {n_input}. Retrieval aborted."
        )

    # --------------------------------------------------------
    # Save exact ADQL used for reproducibility
    # --------------------------------------------------------

    QUERY_TEXT_PATH.write_text(
        ADQL_QUERY.strip() + "\n",
        encoding="utf-8",
    )

    print(
        "Project 02A — Gaia DR3 exact-ID retrieval"
    )
    print("=" * 52)
    print()
    print(f"Input query manifest: {QUERY_PATH}")
    print(f"Input source IDs:      {n_input}")
    print()
    print("Retrieval strategy:")
    print("  source_id exact join")
    print("  Gaia table: gaiadr3.gaia_source")
    print("  coordinate matching: NO")
    print("  quality cuts:        NO")
    print("  membership cuts:     NO")
    print()
    print("Submitting Gaia TAP query...")

    # --------------------------------------------------------
    # Gaia TAP retrieval
    #
    # The local table is uploaded temporarily to TAP_UPLOAD,
    # then exact-joined against gaiadr3.gaia_source.
    # --------------------------------------------------------

    job = Gaia.launch_job_async(
        query=ADQL_QUERY,
        upload_resource=query_table,
        upload_table_name=UPLOAD_TABLE_NAME,
        verbose=True,
    )

    results = job.get_results()

    # --------------------------------------------------------
    # Basic retrieval-level statistics
    #
    # This is NOT the formal identity audit.
    # Formal identity validation belongs to script 03.
    # --------------------------------------------------------

    n_returned = len(results)

    if "source_id" not in results.colnames:
        raise RuntimeError(
            "Gaia result does not contain source_id."
        )

    source_id_col = results["source_id"]

    if hasattr(source_id_col, "mask"):
        mask = source_id_col.mask

        try:
            n_unresolved = int(mask.sum())
        except AttributeError:
            n_unresolved = int(bool(mask))
    else:
        n_unresolved = 0

    n_resolved = n_returned - n_unresolved

    # --------------------------------------------------------
    # Save untouched retrieval result
    # --------------------------------------------------------

    results.write(
        OUTPUT_ECSV,
        format="ascii.ecsv",
        overwrite=True,
    )

    results.write(
        OUTPUT_CSV,
        format="ascii.csv",
        overwrite=True,
    )

    # --------------------------------------------------------
    # Retrieval report
    # --------------------------------------------------------

    lines = [
        "Project 02A — Gaia DR3 retrieval",
        "=" * 46,
        "",
        f"Input source IDs:          {n_input}",
        f"Returned rows:             {n_returned}",
        f"Rows with Gaia source_id:  {n_resolved}",
        f"Rows without Gaia match:   {n_unresolved}",
        "",
        "Retrieval method:",
        "  exact Gaia DR3 source_id LEFT OUTER JOIN",
        "",
        "No quality cuts applied.",
        "No membership cuts applied.",
        "No coordinate crossmatch applied.",
        "",
        f"Gaia TAP job ID:           {job.jobid}",
        "",
        "Outputs:",
        f"  {OUTPUT_ECSV}",
        f"  {OUTPUT_CSV}",
        f"  {QUERY_TEXT_PATH}",
    ]

    report = "\n".join(lines)

    RETRIEVAL_REPORT_PATH.write_text(
        report + "\n",
        encoding="utf-8",
    )

    print()
    print(report)
    print()
    print(
        "RETRIEVAL COMPLETE — formal 1:1 identity "
        "validation is deferred to script 03."
    )


if __name__ == "__main__":
    main()
