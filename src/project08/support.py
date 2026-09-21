from pathlib import Path
import json
import hashlib
import warnings

import numpy as np
from astropy import units as u
from astropy.table import Table
from galpy import potential
from galpy.orbit import Orbit

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data/interim/project08"
RESULTS = ROOT / "results/project08"

REFERENCE = ROOT / "data/interim/project05/stock2_cluster_reference.ecsv"
BASELINE_ORBIT = ROOT / "data/interim/project05/stock2_galactic_orbit.ecsv"

RO_BASELINE = 8.0
VO_BASELINE = 220.0

TMIN_MYR = -500.0
TMAX_MYR = 500.0
DT_MYR = 0.25

MODELS = {
    "MWPotential2014": {
        "potential": lambda: potential.MWPotential2014,
        "baseline": True,
        "ro": 8.0,
        "vo": 220.0,
        "provenance": "galpy built-in MWPotential2014",
    },
    "McMillan17": {
        "potential": lambda: potential.mwpotentials.McMillan17,
        "baseline": False,
        "ro": 8.21,
        "vo": 233.1,
        "provenance": "galpy mwpotentials.McMillan17",
    },
    "Irrgang13I": {
        "potential": lambda: potential.mwpotentials.Irrgang13I,
        "baseline": False,
        "ro": 8.4,
        "vo": 242.0,
        "provenance": "galpy mwpotentials.Irrgang13I",
    },
    "Cautun20": {
        "potential": lambda: potential.mwpotentials.Cautun20,
        "baseline": False,
        "ro": 8.122,
        "vo": 229.0,
        "provenance": "galpy mwpotentials.Cautun20",
    },
}


def read(path):
    return Table.read(path, format="ascii.ecsv")


def save(table, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write(path, format="ascii.ecsv", overwrite=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def model_object(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MODELS[name]["potential"]()


def frozen_reference_state():
    t = read(REFERENCE)
    if len(t) != 1:
        raise ValueError("Expected one-row Project 05 cluster reference")

    row = t[0]

    xA = float(row["x_gc_kpc"])
    yA = float(row["y_gc_kpc"])
    zA = float(row["z_gc_kpc"])

    vxA = float(row["vx_gc_kms"])
    vyA = float(row["vy_gc_kms"])
    vzA = float(row["vz_gc_kms"])

    # Frozen Project 05 Astropy -> galpy convention:
    # x_g = -x_A, y_g = y_A, z_g = z_A
    # vx_g = -vx_A, vy_g = vy_A, vz_g = vz_A
    xg = -xA
    yg = yA
    zg = zA

    vxg = -vxA
    vyg = vyA
    vzg = vzA

    R = np.hypot(xg, yg)
    phi = np.arctan2(yg, xg)

    vR = (xg * vxg + yg * vyg) / R
    vT = (xg * vyg - yg * vxg) / R

    return {
        "x_kpc": xg,
        "y_kpc": yg,
        "z_kpc": zg,
        "vx_kms": vxg,
        "vy_kms": vyg,
        "vz_kms": vzg,
        "R_kpc": R,
        "phi_rad": phi,
        "vR_kms": vR,
        "vT_kms": vT,
    }


def orbit_for_model(name):
    cfg = MODELS[name]
    pot = model_object(name)
    state = frozen_reference_state()

    ro = cfg["ro"]
    vo = cfg["vo"]

    orb = Orbit(
        [
            state["R_kpc"] / ro,
            state["vR_kms"] / vo,
            state["vT_kms"] / vo,
            state["z_kpc"] / ro,
            state["vz_kms"] / vo,
            state["phi_rad"],
        ],
        ro=ro,
        vo=vo,
    )

    return orb, pot, state


def time_grid_myr():
    return np.arange(TMIN_MYR, TMAX_MYR + DT_MYR / 2, DT_MYR)


def galpy_time_unit_myr(ro, vo):
    # ro / vo converted to Myr
    return (ro * u.kpc / (vo * u.km / u.s)).to_value(u.Myr)


def integrate_symmetric(name):
    cfg = MODELS[name]
    ro = cfg["ro"]
    vo = cfg["vo"]

    full_t = time_grid_myr()
    neg_myr = full_t[full_t <= 0]
    pos_myr = full_t[full_t >= 0]

    unit_myr = galpy_time_unit_myr(ro, vo)

    orb0, pot, state = orbit_for_model(name)

    # Integrate independently from exact t=0 in both directions.
    pos = Orbit(orb0.vxvv.copy(), ro=ro, vo=vo)
    neg = Orbit(orb0.vxvv.copy(), ro=ro, vo=vo)

    pos_t = pos_myr / unit_myr
    neg_t = neg_myr[::-1] / unit_myr

    pos.integrate(pos_t, pot, method="dop853_c")
    neg.integrate(neg_t, pot, method="dop853_c")

    rows = []

    def extract(o, tt_nat, tt_myr):
        tt_myr = np.asarray(tt_myr, dtype=float).reshape(-1)

        R = np.asarray(
            o.R(tt_nat, use_physical=True),
            dtype=float,
        ).reshape(-1)

        vR = np.asarray(
            o.vR(tt_nat, use_physical=True),
            dtype=float,
        ).reshape(-1)

        vT = np.asarray(
            o.vT(tt_nat, use_physical=True),
            dtype=float,
        ).reshape(-1)

        z = np.asarray(
            o.z(tt_nat, use_physical=True),
            dtype=float,
        ).reshape(-1)

        vz = np.asarray(
            o.vz(tt_nat, use_physical=True),
            dtype=float,
        ).reshape(-1)

        phi = np.asarray(
            o.phi(tt_nat),
            dtype=float,
        ).reshape(-1)

        lengths = {
            len(tt_myr),
            len(R),
            len(vR),
            len(vT),
            len(z),
            len(vz),
            len(phi),
        }

        if len(lengths) != 1:
            raise RuntimeError(
                f"Orbit extraction length mismatch: "
                f"t={len(tt_myr)}, R={len(R)}, vR={len(vR)}, "
                f"vT={len(vT)}, z={len(z)}, vz={len(vz)}, phi={len(phi)}"
            )

        x = R * np.cos(phi)
        y = R * np.sin(phi)

        vx = vR * np.cos(phi) - vT * np.sin(phi)
        vy = vR * np.sin(phi) + vT * np.cos(phi)

        for i in range(len(tt_myr)):
            rows.append(
                (
                    float(tt_myr[i]),
                    float(x[i]),
                    float(y[i]),
                    float(z[i]),
                    float(R[i]),
                    float(phi[i]),
                    float(vx[i]),
                    float(vy[i]),
                    float(vz[i]),
                    float(vR[i]),
                    float(vT[i]),
                )
            )

    extract(neg, neg_t, neg_myr[::-1])
    extract(pos, pos_t[1:], pos_myr[1:])

    rows.sort(key=lambda r: r[0])

    names = [
        "time_myr",
        "x_galpy_kpc",
        "y_galpy_kpc",
        "z_kpc",
        "R_kpc",
        "phi_rad",
        "vx_galpy_kms",
        "vy_galpy_kms",
        "vz_kms",
        "vR_kms",
        "vT_kms",
    ]

    table = Table(rows=rows, names=names)

    table["time_myr"].unit = u.Myr
    for q in ["x_galpy_kpc", "y_galpy_kpc", "z_kpc", "R_kpc"]:
        table[q].unit = u.kpc
    for q in ["vx_galpy_kms", "vy_galpy_kms", "vz_kms", "vR_kms", "vT_kms"]:
        table[q].unit = u.km / u.s
    table["phi_rad"].unit = u.rad

    table.meta["model"] = name
    table.meta["ro_kpc"] = ro
    table.meta["vo_kms"] = vo
    table.meta["integration"] = {
        "tmin_myr": TMIN_MYR,
        "tmax_myr": TMAX_MYR,
        "dt_myr": DT_MYR,
        "method": "dop853_c",
        "symmetric_from_t0": True,
    }
    table.meta["initial_state"] = state

    return table


def local_force_quantities(name, R_kpc, z_kpc):
    cfg = MODELS[name]
    pot = model_object(name)

    if name == "MWPotential2014":
        ro = cfg["ro"]
        vo = cfg["vo"]
        Rn = R_kpc / ro
        zn = z_kpc / ro

        FR = (
            potential.evaluateRforces(pot, Rn, zn)
            * vo**2 / ro
        )
        FZ = (
            potential.evaluatezforces(pot, Rn, zn)
            * vo**2 / ro
        )
        vc = potential.vcirc(pot, Rn) * vo

    else:
        Rq = R_kpc * u.kpc
        zq = z_kpc * u.kpc

        FRq = potential.evaluateRforces(
            pot, Rq, zq, use_physical=True
        )
        FZq = potential.evaluatezforces(
            pot, Rq, zq, use_physical=True
        )
        vcq = potential.vcirc(
            pot, Rq, use_physical=True
        )

        if hasattr(FRq, "unit"):
            FR = FRq.to_value((u.km/u.s)**2/u.kpc)
            FZ = FZq.to_value((u.km/u.s)**2/u.kpc)
        else:
            FR = (
                FRq * u.km/u.s/u.Myr
            ).to_value((u.km/u.s)**2/u.kpc)
            FZ = (
                FZq * u.km/u.s/u.Myr
            ).to_value((u.km/u.s)**2/u.kpc)

        if hasattr(vcq, "unit"):
            vc = vcq.to_value(u.km/u.s)
        else:
            vc = float(vcq)

    Omega = vc / R_kpc

    return {
        "F_R_kms2_kpc": float(FR),
        "F_z_kms2_kpc": float(FZ),
        "vcirc_kms": float(vc),
        "Omega_kms_kpc": float(Omega),
    }


def extrema_summary(table):
    R = np.asarray(table["R_kpc"], dtype=float)
    z = np.asarray(table["z_kpc"], dtype=float)

    rperi = float(np.min(R))
    rap = float(np.max(R))
    ecc = (rap - rperi) / (rap + rperi)
    zmax = float(np.max(np.abs(z)))

    return {
        "rperi_kpc": rperi,
        "rap_kpc": rap,
        "eccentricity": float(ecc),
        "zmax_kpc": zmax,
    }


G_KPC_KMS2_MSUN = 4.30091e-6
CLUSTER_MASS_MSUN = 4000.0


def circular_tidal_quantities(name, R_kpc):
    cfg = MODELS[name]
    pot = model_object(name)

    ro = float(cfg["ro"])
    vo = float(cfg["vo"])

    R_arr = np.asarray(R_kpc, dtype=float)
    Rn = R_arr / ro

    vc_nat = np.asarray(
        potential.vcirc(
            pot,
            Rn,
            use_physical=False,
        ),
        dtype=float,
    )

    kappa_nat = np.asarray(
        potential.epifreq(
            pot,
            Rn,
            use_physical=False,
        ),
        dtype=float,
    )

    vc_kms = vc_nat * vo
    Omega_kms_kpc = vc_kms / R_arr
    kappa_kms_kpc = kappa_nat * vo / ro

    denominator = 4.0 * Omega_kms_kpc**2 - kappa_kms_kpc**2

    if np.any(~np.isfinite(denominator)):
        raise RuntimeError(f"{name}: non-finite Jacobi denominator")

    if np.any(denominator <= 0):
        bad = np.asarray(R_arr)[np.asarray(denominator <= 0)]
        raise RuntimeError(
            f"{name}: non-positive Jacobi denominator at R={bad}"
        )

    rj_kpc = (
        G_KPC_KMS2_MSUN
        * CLUSTER_MASS_MSUN
        / denominator
    ) ** (1.0 / 3.0)

    return {
        "vcirc_kms": vc_kms,
        "Omega_kms_kpc": Omega_kms_kpc,
        "kappa_kms_kpc": kappa_kms_kpc,
        "denominator_kms2_kpc2": denominator,
        "rj_kpc": rj_kpc,
    }


def tidal_tensor_local(name, R_kpc, z_kpc):
    cfg = MODELS[name]
    pot = model_object(name)

    ro = float(cfg["ro"])
    vo = float(cfg["vo"])

    Rn = float(R_kpc) / ro
    zn = float(z_kpc) / ro

    phi_RR = float(
        potential.evaluateR2derivs(
            pot,
            Rn,
            zn,
            use_physical=False,
        )
    ) * vo**2 / ro**2

    phi_ZZ = float(
        potential.evaluatez2derivs(
            pot,
            Rn,
            zn,
            use_physical=False,
        )
    ) * vo**2 / ro**2

    phi_RZ = float(
        potential.evaluateRzderivs(
            pot,
            Rn,
            zn,
            use_physical=False,
        )
    ) * vo**2 / ro**2

    F_R = float(
        potential.evaluateRforces(
            pot,
            Rn,
            zn,
            use_physical=False,
        )
    ) * vo**2 / ro

    phi_R = -F_R
    phi_phiphi_orthonormal = phi_R / float(R_kpc)

    T = -np.array(
        [
            [phi_RR, 0.0, phi_RZ],
            [0.0, phi_phiphi_orthonormal, 0.0],
            [phi_RZ, 0.0, phi_ZZ],
        ],
        dtype=float,
    )

    if not np.all(np.isfinite(T)):
        raise RuntimeError(f"{name}: non-finite tidal tensor")

    if not np.allclose(T, T.T, rtol=0.0, atol=1e-10):
        raise RuntimeError(f"{name}: tidal tensor is not symmetric")

    evals, evecs = np.linalg.eigh(T)

    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    return T, evals, evecs


def acute_axis_angle_deg(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    c = np.clip(np.abs(np.dot(a, b)), 0.0, 1.0)
    return float(np.degrees(np.arccos(c)))
