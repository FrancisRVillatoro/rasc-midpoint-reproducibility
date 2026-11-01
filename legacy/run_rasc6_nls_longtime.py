#!/usr/bin/env python3
"""Final long-time NLS check for RASC6.

This is a self-contained diagnostic for the proposed RASC paper.  It uses the
frozen F3 ASC implementation read-only and does not modify it.

Family:
    T(h) = sigma / (h^2 r_phi),   sigma=1,
where r_phi=505/768 is the group-phase O(h^2) rate for the eta=c=1 bright
soliton.  Thus the leading midpoint phase drift is O(1) at every h.

Reconstructions at t=T:
  MP2  = U
  ASC4 = U + h^2 P
  ASC6 = U + h^2 P + h^4 R
  RASC4 = G(h^2 a2)[U + h^2 W2]
  RASC6 = G(h^2 a2+h^4 a4)[U + h^2 W2+h^4 W4]

The fully discrete second cumulant is
    C4 = R - A2 P + 1/2 A2^2 U,
with A2 = a2_x d_x + a2_phi J, followed by a second tangent/transverse split
C4 = A4 U + W4.  Translation and U(1) phase commute and are applied exactly
by Fourier multipliers / rotations.
"""
import argparse
import csv
import hashlib
import json
import math
import pathlib
import platform
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "src"
sys.path.insert(0, str(SRC))

from pdes import nls, _spectral as S  # noqa: E402
from integrators import midpoint as MP  # noqa: E402
from integrators import asc  # noqa: E402

TOL = 1e-15
RATE_PHI = 505.0 / 768.0
HS = (0.05, 0.04, 0.032)
N_DEFAULT = 768
SIGMA_DEFAULT = 1.0


def provenance():
    kpath = SRC / "kernel" / "formulas.py"
    actual = hashlib.sha256(kpath.read_bytes()).hexdigest()
    expected = (SRC / "KERNEL_SHA256.txt").read_text().split()[0]
    if actual != expected:
        raise RuntimeError(f"FROZEN KERNEL MISMATCH: {actual} != {expected}")
    return {
        "kernel_sha256": actual,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "host": platform.node(),
    }


def wrap(z, L):
    return (z + L / 2.0) % L - L / 2.0


def state_phys(pde, t):
    eta, c, x0 = pde.eta, pde.c, pde.p["x0"]
    x, L = pde.grid.x, pde.grid.L
    xi = wrap(x - x0 - c * t, L)
    th = 0.5 * c * x + (eta**2 - 0.25 * c**2) * t
    env = eta / np.cosh(eta * xi)
    return np.stack([env * np.cos(th), env * np.sin(th)])


def J(V):
    return np.stack([-V[1], V[0]])


def dxop(pde, V):
    return pde.grid.project(1j * pde.grid.k[None, :] * V)


def apply_Aop(pde, V, a):
    return a[0] * dxop(pde, V) + a[1] * J(V)


def generators(pde, U):
    return [dxop(pde, U), J(U)]


def split(pde, U, V):
    """L2-orthogonal tangent/transverse split on {U_x, J U}."""
    gs = generators(pde, U)
    with S.no_count():
        B = np.array([pde.grid.to_phys(g).ravel() for g in gs])
        v = pde.grid.to_phys(V).ravel()
    dx = pde.grid.dx
    gram = dx * B @ B.T
    rhs = dx * B @ v
    a = np.linalg.solve(gram, rhs)
    Vp = V - a[0] * gs[0] - a[1] * gs[1]
    cond = float(np.linalg.cond(gram))
    return a, Vp, cond


def group(pde, Y, delta, theta):
    """G(delta,theta)Y=e^{i theta}Y(x+delta), exact on spectral state."""
    Z = Y * np.exp(1j * pde.grid.k[None, :] * delta)
    Z = pde.grid.project(Z)
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([c * Z[0] - s * Z[1], s * Z[0] + c * Z[1]])


def rel_sup_error(pde, Y, t):
    with S.no_count():
        ref = state_phys(pde, t)
        yy = pde.grid.to_phys(Y)
        scale = float(np.max(np.abs(ref)))
        return float(np.max(np.abs(yy - ref))) / scale


def l2norm(pde, V):
    with S.no_count():
        v = pde.grid.to_phys(V)
    return float(np.sqrt(pde.grid.dx * np.sum(v * v)))


def run_case(h, N=N_DEFAULT, sigma=SIGMA_DEFAULT, outdir=None):
    prov = provenance()
    pde = nls.make("bright", N=N)
    y0 = pde.grid.to_spec(state_phys(pde, 0.0))
    T_target = sigma / (h * h * RATE_PHI)
    nsteps = int(round(T_target / h))
    T = nsteps * h

    y = y0.copy()
    P = np.zeros_like(y)
    R = np.zeros_like(y)
    log = MP.Stats()
    t0 = time.perf_counter()
    for n in range(nsteps):
        y = MP.step(y, h, pde, tol=TOL, log=log)
        P, R, _ = asc.sweep_step(pde, log.stage, h, P, R, use_H2=True)
        if (n + 1) % max(1, nsteps // 10) == 0:
            print(f"h={h:g}: {(n+1)}/{nsteps} ({100*(n+1)/nsteps:.0f}%)", flush=True)

    a2, W2, cond2 = split(pde, y, P)
    A2P = apply_Aop(pde, P, a2)
    A2sqU = apply_Aop(pde, apply_Aop(pde, y, a2), a2)
    C4 = R - A2P + 0.5 * A2sqU
    a4, W4, cond4 = split(pde, y, C4)

    h2 = h * h
    h4 = h2 * h2
    ASC4 = y + h2 * P
    ASC6 = ASC4 + h4 * R
    RASC4 = group(pde, y + h2 * W2, h2 * a2[0], h2 * a2[1])
    RASC6 = group(
        pde,
        y + h2 * W2 + h4 * W4,
        h2 * a2[0] + h4 * a4[0],
        h2 * a2[1] + h4 * a4[1],
    )

    row = {
        "h": h,
        "h2": h2,
        "h4": h4,
        "sigma": sigma,
        "rate_phi": RATE_PHI,
        "T_target": T_target,
        "T": T,
        "nsteps": nsteps,
        "N": N,
        "elapsed_s": time.perf_counter() - t0,
        "a2_x": float(a2[0]),
        "a2_phi": float(a2[1]),
        "a4_x": float(a4[0]),
        "a4_phi": float(a4[1]),
        "angle2_x": float(h2 * a2[0]),
        "angle2_phi": float(h2 * a2[1]),
        "angle4_x": float(h4 * a4[0]),
        "angle4_phi": float(h4 * a4[1]),
        "cond_tangent_P": cond2,
        "cond_tangent_C4": cond4,
        "norm_h2W2": h2 * l2norm(pde, W2),
        "norm_h4W4": h4 * l2norm(pde, W4),
        "err_MP2": rel_sup_error(pde, y, T),
        "err_ASC4": rel_sup_error(pde, ASC4, T),
        "err_ASC6": rel_sup_error(pde, ASC6, T),
        "err_RASC4": rel_sup_error(pde, RASC4, T),
        "err_RASC6": rel_sup_error(pde, RASC6, T),
        **prov,
    }

    if outdir is None:
        outdir = HERE / "results"
    else:
        outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"rasc6_nls_long_h{h:.4g}.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(json.dumps(row, indent=2), flush=True)
    print(f"wrote {path}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, help="0,1,2 -> h=0.05,0.04,0.032")
    ap.add_argument("--h", type=float)
    ap.add_argument("--N", type=int, default=N_DEFAULT)
    ap.add_argument("--sigma", type=float, default=SIGMA_DEFAULT)
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        # cheap algebra/implementation check, not part of the final data
        h = 0.025
        pde = nls.make("bright", N=min(a.N, 384))
        y0 = pde.grid.to_spec(state_phys(pde, 0.0))
        y = y0.copy(); P = np.zeros_like(y); R = np.zeros_like(y); log = MP.Stats()
        for _ in range(20):
            y = MP.step(y, h, pde, tol=TOL, log=log)
            P, R, _ = asc.sweep_step(pde, log.stage, h, P, R, use_H2=True)
        a2, W2, _ = split(pde, y, P)
        C4 = R - apply_Aop(pde, P, a2) + 0.5 * apply_Aop(pde, apply_Aop(pde, y, a2), a2)
        a4, W4, _ = split(pde, y, C4)
        print("SMOKE_PASS", "a2=", a2, "a4=", a4,
              "h2W2=", h*h*l2norm(pde,W2), "h4W4=", h**4*l2norm(pde,W4))
        return
    if a.h is not None:
        h = a.h
    elif a.task is not None:
        if not (0 <= a.task < len(HS)):
            raise SystemExit("task must be 0,1,2")
        h = HS[a.task]
    else:
        raise SystemExit("give --task or --h")
    run_case(h, a.N, a.sigma, a.out)


if __name__ == "__main__":
    main()
