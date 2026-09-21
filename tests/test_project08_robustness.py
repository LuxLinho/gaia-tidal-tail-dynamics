from pathlib import Path
import json
import hashlib

import numpy as np
from astropy.table import Table

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data/interim/project08"
RESULTS = ROOT / "results/project08"

MODELS = {
    "MWPotential2014",
    "McMillan17",
    "Irrgang13I",
    "Cautun20",
}


def read(path):
    return Table.read(path, format="ascii.ecsv")


def test_canonical_population_unchanged():
    master = read(
        ROOT
        / "data/processed/stock2_literature_master.ecsv"
    )

    assert len(master) == 1456

    classes = np.asarray(
        master["homogenized_class"],
        dtype=str,
    )

    assert np.sum(classes == "C") == 940
    assert np.sum(classes == "L") == 184
    assert np.sum(classes == "T") == 332


def test_potential_set():
    t = read(
        DATA / "stock2_potential_models.ecsv"
    )

    assert set(
        np.asarray(t["model"], dtype=str)
    ) == MODELS

    assert np.all(
        np.asarray(t["axisymmetric"], dtype=bool)
    )


def test_same_present_physical_position():
    for model in MODELS:
        t = read(
            DATA / f"stock2_orbit_{model}.ecsv"
        )

        idx = np.argmin(
            np.abs(
                np.asarray(
                    t["time_myr"],
                    dtype=float,
                )
            )
        )

        assert abs(
            float(t["time_myr"][idx])
        ) < 1e-12

        assert np.isclose(
            float(t["R_kpc"][idx]),
            8.383461176641,
            atol=1e-9,
        )


def test_all_orbits_have_4001_rows_and_finite_values():
    cols = [
        "R_kpc",
        "z_kpc",
        "vR_kms",
        "vT_kms",
        "vz_kms",
    ]

    for model in MODELS:
        t = read(
            DATA / f"stock2_orbit_{model}.ecsv"
        )

        assert len(t) == 4001

        times = np.asarray(
            t["time_myr"],
            dtype=float,
        )

        assert np.isclose(
            times[0],
            -500.0,
        )

        assert np.isclose(
            times[-1],
            500.0,
        )

        assert np.sum(
            np.isclose(
                times,
                0.0,
                atol=1e-12,
            )
        ) == 1

        for col in cols:
            assert np.all(
                np.isfinite(
                    np.asarray(
                        t[col],
                        dtype=float,
                    )
                )
            )


def test_orbit_summary_physical():
    t = read(
        DATA
        / "stock2_orbit_summary_by_potential.ecsv"
    )

    assert len(t) == 4

    assert np.all(
        np.asarray(
            t["rperi_kpc"],
            dtype=float,
        )
        <
        np.asarray(
            t["rap_kpc"],
            dtype=float,
        )
    )

    e = np.asarray(
        t["eccentricity"],
        dtype=float,
    )

    assert np.all(
        (e >= 0)
        & (e < 1)
    )


def test_project05_baseline_reproduced():
    t = read(
        DATA
        / "stock2_orbit_summary_by_potential.ecsv"
    )

    row = t[
        t["model"] == "MWPotential2014"
    ][0]

    assert np.isclose(
        float(row["rperi_kpc"]),
        7.969363343,
        atol=2e-8,
    )

    assert np.isclose(
        float(row["rap_kpc"]),
        9.676913611,
        atol=2e-8,
    )

    assert np.isclose(
        float(row["eccentricity"]),
        0.096765469,
        atol=2e-9,
    )

    assert np.isclose(
        float(row["zmax_kpc"]),
        0.103990988,
        atol=2e-9,
    )


def test_project06_baseline_jacobi_reproduced():
    t = read(
        DATA
        / "stock2_jacobi_summary_by_potential.ecsv"
    )

    row = t[
        t["model"] == "MWPotential2014"
    ][0]

    assert np.isclose(
        float(row["rj_now_pc"]),
        22.506883,
        atol=5e-5,
    )


def test_jacobi_positive_and_finite():
    t = read(
        DATA
        / "stock2_jacobi_robustness.ecsv"
    )

    D = np.asarray(
        t["denominator_kms2_kpc2"],
        dtype=float,
    )

    rj = np.asarray(
        t["rj_pc"],
        dtype=float,
    )

    assert np.all(
        np.isfinite(D)
    )

    assert np.all(
        D > 0
    )

    assert np.all(
        np.isfinite(rj)
    )

    assert np.all(
        rj > 0
    )


def test_project07_tidal_baseline_reproduced():
    t = read(
        DATA
        / "stock2_tidal_tensor_robustness.ecsv"
    )

    baseline = t[
        t["model"] == "MWPotential2014"
    ]

    vals = np.asarray(
        baseline["eigenvalue"],
        dtype=float,
    )

    expected = np.array(
        [
            826.9652069664706,
            -681.9860687262053,
            -4978.998355112044,
        ]
    )

    assert np.allclose(
        vals,
        expected,
        rtol=0,
        atol=1e-6,
    )


def test_tidal_vectors_orthonormal():
    t = read(
        DATA
        / "stock2_tidal_tensor_robustness.ecsv"
    )

    for model in MODELS:
        x = t[
            t["model"] == model
        ]

        V = np.column_stack(
            [
                np.asarray(
                    x["e_radial"],
                    dtype=float,
                ),
                np.asarray(
                    x["e_prograde"],
                    dtype=float,
                ),
                np.asarray(
                    x["e_vertical"],
                    dtype=float,
                ),
            ]
        )

        assert np.allclose(
            V @ V.T,
            np.eye(3),
            atol=1e-10,
        )


def test_alignment_angles_are_acute():
    t = read(
        DATA
        / "stock2_alignment_robustness.ecsv"
    )

    a = np.asarray(
        t["angle_to_stretching_deg"],
        dtype=float,
    )

    assert np.all(
        (a >= 0)
        & (a <= 90)
    )


def test_key_alignment_result_survives():
    t = read(
        DATA
        / "stock2_alignment_robustness.ecsv"
    )

    intermediate = np.asarray(
        t[
            t["population"] == "intermediate"
        ]["angle_to_stretching_deg"],
        dtype=float,
    )

    outer = np.asarray(
        t[
            t["population"] == "outer"
        ]["angle_to_stretching_deg"],
        dtype=float,
    )

    assert np.all(
        intermediate < 10
    )

    assert np.all(
        outer > 30
    )


def test_required_outputs_exist():
    required = [
        RESULTS / "project08_summary.txt",
        RESULTS / "project08_metadata.json",
        RESULTS / "verification.md",
        ROOT
        / "docs/project08_galactic_potential_robustness.md",
        RESULTS
        / "orbit_robustness/R_vs_time.png",
        RESULTS
        / "orbit_robustness/z_vs_time.png",
        RESULTS
        / "orbit_robustness/xy_orbit_comparison.png",
        RESULTS
        / "jacobi_robustness/jacobi_radius_vs_time.png",
        RESULTS
        / "jacobi_robustness/present_day_jacobi_radius.png",
        RESULTS
        / "tidal_geometry/tidal_eigenvalue_comparison.png",
        RESULTS
        / "tidal_geometry/morphology_stretching_alignment.png",
    ]

    for path in required:
        assert path.exists(), str(path)


def test_metadata_conclusions():
    meta = json.loads(
        (
            RESULTS
            / "project08_metadata.json"
        ).read_text()
    )

    conclusions = meta["conclusions"]

    assert conclusions[
        "low_eccentricity_disk_orbit"
    ]

    assert conclusions[
        "vertically_confined_near_disk"
    ]

    assert conclusions[
        "jacobi_scale_remains_tens_of_pc"
    ]

    assert conclusions[
        "intermediate_close_to_stretching_axis"
    ]

    assert conclusions[
        "outer_rotated_from_stretching_axis"
    ]
