"""Renormalized ASC4 reconstruction versus additive ASC4 on the secular family.

Additive ASC4 is U + h^2 P.  Along a relative equilibrium the corrector P
grows linearly in its tangent (angle) directions, and U + h^2 P is only the
linearization of the group action that the true error actually is; once the
accumulated angle h^2 a(t) is of order one, the linearization fails and ASC4
stops converging.  The renormalized reconstruction keeps the corrector but
applies its tangent part as the exact group action,

    u_ren = G(h^2 beta) (U + h^2 P_perp),
    P = beta_x d_x U + beta_phi i U + P_perp     (L2-orthogonal split),

with G(delta, eps) = exp(i eps) * (translation by -delta), a Fourier
multiplier and a phase rotation, both of modulus one.  The generators are
applied to the NUMERICAL state, so nothing here uses the exact solution.

A second variant needs no corrector at all: MP2 renormalized a priori by the
angle rates predicted from G2 (predict.py), u = G(h^2 rate t) U.

The test family is T(h) = sigma / (h^2 * rate_phi) with sigma = 1: the phase
error accumulated by MP2 is one radian at every h, so MP2 and additive ASC4
cannot converge along it, while a reconstruction that is uniformly valid in
the angle must.  A short fixed-T run checks that the renormalization does not
cost ASC4 its fourth order.
"""
import argparse
import csv
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np                                               # noqa: E402
import projection as PR                                          # noqa: E402
from pdes import _spectral as S                                  # noqa: E402
from integrators import midpoint as MP                           # noqa: E402

RES = HERE / 'results'
# NLS eta=1, c=1: rates in the generator basis {d_x U, i U}, from predict.py
# (measured basis: phase 515/768, position -5/192; d_x0 u = -d_x u + i c/2 u)
RATE_X = 5.0 / 192.0
RATE_PHI = 505.0 / 768.0
# KdV kappa=1 on the periodic domain L=64 of the benchmark: coefficient of
# d_x u in the scaled error, from the Riesz projection of predict.py (the
# whole-line value -704/45 differs by the periodic mass-mode contribution)
RATE_X_KDV = -15.736200716846


def group(pde, y, delta, eps):
    """exp(i eps) * y(x + delta), exact, on the spectral state."""
    z = y * np.exp(1j * pde.grid.k[None, :] * delta)
    z = pde.grid.project(z)
    if pde.grid.ncomp == 1:
        return z
    c, s = np.cos(eps), np.sin(eps)
    return np.stack([c * z[0] - s * z[1], s * z[0] + c * z[1]])


def split(pde, U, P):
    """Orthogonal split of P onto the generators at the numerical state U."""
    ik = 1j * pde.grid.k[None, :]
    gens = [pde.grid.project(ik * U)]
    if pde.grid.ncomp == 2:
        gens.append(np.stack([-U[1], U[0]]))
    with S.no_count():
        B = np.array([pde.grid.to_phys(g).ravel() for g in gens])
        v = pde.grid.to_phys(P).ravel()
    dx = pde.grid.dx
    beta = np.linalg.solve(dx * B @ B.T, dx * B @ v)
    Pp = P - sum(b * g for b, g in zip(beta, gens))
    if len(beta) == 1:
        beta = np.array([beta[0], 0.0])
    return beta, Pp


def errors(pde, ex, recon):
    with S.no_count():
        ref = ex
        scale = float(np.max(np.abs(ref)))
        return {k: float(np.max(np.abs(pde.grid.to_phys(v) - ref))) / scale
                for k, v in recon.items()}


def run(h, T, nout, tag, name='nls'):
    pde, y0, state, modes, gens, _ = PR.spec(name)
    rx, rphi = (RATE_X, RATE_PHI) if name == 'nls' else (RATE_X_KDV, 0.0)
    n = int(round(T / h))
    every = max(1, n // nout)
    y, P, log = y0.copy(), np.zeros_like(y0), MP.Stats()
    rows, t0 = [], time.perf_counter()
    for k in range(1, n + 1):
        y = MP.step(y, h, pde, tol=PR.TOL, log=log)
        P = PR.p_step(pde, log.stage, h, P)
        if k % every and k != n:
            continue
        t = k * h
        beta, Pp = split(pde, y, P)
        recon = {
            'MP2': y,
            'ASC4_additive': y + h * h * P,
            'ASC4_renormalized': group(pde, y + h * h * Pp,
                                       h * h * beta[0], h * h * beta[1]),
            'MP2_renormalized_apriori': group(pde, y, h * h * rx * t,
                                              h * h * rphi * t),
        }
        e = errors(pde, state(t), recon)
        rows.append(dict(tag=tag, h=f"{h:.6e}", t=f"{t:.6f}",
                         phase_angle=f"{h * h * beta[1]:.6e}",
                         position_shift=f"{h * h * beta[0]:.6e}",
                         **{f"err_{k}": f"{v:.6e}" for k, v in e.items()}))
    RES.mkdir(exist_ok=True)
    path = RES / f"renorm_{tag}{'' if name == 'nls' else '_' + name}_h{h:.4g}.csv"
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    last = rows[-1]
    print(f"{tag} h={h:g} T={T:.1f} ({n} steps, {time.perf_counter() - t0:.0f} s): "
          + "  ".join(f"{k[4:]} {last[k]}" for k in last if k.startswith('err_')),
          flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', action='store_true')
    ap.add_argument('--fixed', action='store_true')
    ap.add_argument('--sigma', type=float, default=1.0)
    ap.add_argument('--pde', default='nls', choices=['nls', 'kdv'])
    ap.add_argument('--h', type=float, help='single step size (one array task)')
    a = ap.parse_args()
    if a.fixed:
        for h in (0.04, 0.02, 0.01):
            run(h, 4.0, 8, 'fixedT')
    rate = RATE_PHI if a.pde == 'nls' else abs(RATE_X_KDV)
    hs = [a.h] if a.h else ((0.05, 0.04, 0.032) if a.pde == 'nls'
                            else (0.004, 0.0032, 0.00256))
    if a.family:
        for h in hs:
            run(h, a.sigma / (h * h * rate), 40, 'family', a.pde)
