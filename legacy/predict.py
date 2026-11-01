"""Predict the secular drift rates of the projection experiment from G2 alone.

MP2 follows the modified equation U_t = f(U) + h^2 G2(U) + O(h^4), so the
scaled error e = (u - U)/h^2 obeys, to leading order,
    e_t = A(u(t)) e - G2(u(t)),   e(0) = 0,
the equation the ASC corrector P discretizes.  Nothing below uses the
numerical solution or the reference: only f, A, B and G2 of the frozen kernel,
evaluated on the exact family.

Relative equilibria (KdV, NLS).  In the frame that co-moves with speed c and,
for NLS, co-rotates with frequency Omega, the soliton phi is stationary and
    e~_t = Lop e~ - g,   Lop = A(phi) + c d_xi - Omega J,   g = G2(phi),
both constant.  Lop is nilpotent on its generalized kernel X0, so
    e~(t) = -t Pi0 g - (t^2/2) Lop Pi0 g - ... + bounded,
and the predicted linear rate is -Pi0 g, with no t^2 term iff Lop Pi0 g = 0.
Pi0 is the Riesz projector (1/2 pi i) oint (z - Lop)^{-1} dz around 0, computed
on the dense matrix of the discrete Lop by trapezoidal quadrature; it assumes
nothing about the Jordan structure, the dimension of X0 or any Hamiltonian
form.  The rates are then read in the SAME orthogonal mode basis used by the
measurement, so predicted and measured numbers are directly comparable.

Periodic orbit (sine-Gordon breather).  With the canonical symplectic form
w(a,b) = <a, J b>, J(a_u,a_v) = (a_v,-a_u), preserved by the linearized flow,
and the variational solutions s1 = d_theta Y, s2 = t d_theta Y + d_omega Y,
    d/dt w(s_j, e) = -w(s_j, G2(Y(t))),
which gives, with F1 = w(d_theta Y, G2), F2 = w(d_omega Y, G2),
Omega12 = w(d_theta Y, d_omega Y) and I1(u) = int_0^u F1,
    phase rate     = (<F2> - <I1>) / Omega12,
    frequency rate = -<F1> / Omega12,
averages over one period, evaluated spectrally.  No monodromy matrix is
needed.
"""
import csv
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import projection as PR                                          # noqa: E402
import kernel.formulas as K                                      # noqa: E402
from pdes import _spectral as S                                  # noqa: E402

RES = HERE / 'results'


# ----------------------------------------------------------- real coordinates
def maps(pde):
    nc, Nk = pde.grid.ncomp, pde.grid.Nk
    nper = 2 * Nk - 3

    def to_real(y):
        out = np.empty(nc * nper)
        for c in range(nc):
            o = c * nper
            out[o] = y[c, 0].real
            out[o + 1:o + nper:2] = y[c, 1:Nk - 1].real
            out[o + 2:o + nper:2] = y[c, 1:Nk - 1].imag
        return out

    def from_real(v):
        y = np.zeros((nc, Nk), dtype=complex)
        for c in range(nc):
            o = c * nper
            y[c, 0] = v[o]
            y[c, 1:Nk - 1] = v[o + 1:o + nper:2] + 1j * v[o + 2:o + nper:2]
        return y

    return to_real, from_real, nc * nper


def ortho_coef(v, basis, dx):
    B = np.array([b.ravel() for b in basis])
    return np.linalg.solve(dx * B @ B.T, dx * B @ v.ravel())


def measured_rates(files, keys):
    """Slopes over t >= T/4 at each h, and the h -> 0 Richardson limit."""
    out = {}
    for k in keys:
        rates = []
        for f, h in files:
            r = list(csv.DictReader(open(RES / f)))
            t = np.array([float(x['t']) for x in r])
            c = np.array([float(x['e_coef_' + k]) for x in r])
            m = t >= t[-1] / 4
            rates.append((h, float(np.polyfit(t[m], c[m], 1)[0])))
        rates.sort()
        (h1, r1), (h2, r2) = rates[0], rates[1]
        out[k] = (r1 + (r1 - r2) / ((h2 / h1) ** 2 - 1), rates)
    return out


# --------------------------------------------------------- relative equilibria
def predict_relative_equilibrium(name, K_quad=(16, 32), internals=False):
    pde, y0, state, modes, gens, T = PR.spec(name)
    ik = 1j * pde.grid.k[None, :]
    if name == 'kdv':
        c, Om = 4.0 * pde.p['kappa'] ** 2, 0.0
    else:
        c, Om = pde.c, pde.eta ** 2 + 0.25 * pde.c ** 2

    def Jop(y):
        return np.stack([-y[1], y[0]]) if pde.grid.ncomp == 2 else 0.0 * y

    def Lop(y):
        return pde.applyA(y0, y) + c * pde.grid.project(ik * y) - Om * Jop(y)

    # the soliton must be a steady state of the frame equation
    with S.no_count():
        steady = pde.f(y0) + c * pde.grid.project(ik * y0) - Om * Jop(y0)
        steady_rel = (np.max(np.abs(pde.grid.to_phys(steady)))
                      / np.max(np.abs(pde.grid.to_phys(pde.f(y0)))))

    to_real, from_real, dim = maps(pde)
    t0 = time.perf_counter()
    M = np.empty((dim, dim))
    E = np.zeros(dim)
    for j in range(dim):
        E[j] = 1.0
        M[:, j] = to_real(Lop(from_real(E)))
        E[j] = 0.0
    lam = np.linalg.eigvals(M)
    mag = np.sort(np.abs(lam))
    gap = float(mag[mag > 1e-3][0])
    rho = gap / 3.0
    n_in = int(np.sum(np.abs(lam) < rho))

    g = to_real(K.G2(y0, pde.f, pde.applyA, pde.applyB))
    Pi0g = {}
    for Kq in K_quad:
        acc = np.zeros(dim)
        I = np.eye(dim)
        for k in range(Kq // 2):
            z = rho * np.exp(2j * np.pi * (k + 0.5) / Kq)
            acc += 2.0 * np.real(z * np.linalg.solve(z * I - M, g))
        Pi0g[Kq] = acc / Kq
    quad_err = (np.linalg.norm(Pi0g[K_quad[1]] - Pi0g[K_quad[0]])
                / np.linalg.norm(Pi0g[K_quad[1]]))
    p = Pi0g[K_quad[1]]
    t2_ratio = np.linalg.norm(M @ p) / np.linalg.norm(p)

    with S.no_count():
        rate_phys = pde.grid.to_phys(from_real(-p))
        g_phys = pde.grid.to_phys(from_real(g))
    md = modes(0.0)
    coef = ortho_coef(rate_phys, list(md.values()), pde.grid.dx)
    frac_kernel = np.linalg.norm(p) / np.linalg.norm(g)
    diag = dict(steady_rel=steady_rel, gap=gap, rho=rho, eig_inside=n_in,
                quad_rel_diff=quad_err, t2_over_t=t2_ratio,
                Pi0g_over_g=frac_kernel, build_s=time.perf_counter() - t0,
                g_sup=float(np.max(np.abs(g_phys))))
    if internals:
        return dict(zip(md.keys(), coef)), diag, dict(
            pde=pde, M=M, g=g, p=p, lam=lam, rho=rho, from_real=from_real,
            modes=md)
    return dict(zip(md.keys(), coef)), diag


# ------------------------------------------------------------ periodic orbit
def predict_breather(omega=0.5, Ns=512):
    pde, y0, state, modes, gens, T = PR.spec('sg_breather', omega)
    tau = 2 * np.pi / omega
    dx = pde.grid.dx

    def w(a, b):                     # <a, J b>,  J(b_u, b_v) = (b_v, -b_u)
        return float(dx * np.sum(a[0] * b[1] - a[1] * b[0]))

    F1, F2, O12 = np.empty(Ns), np.empty(Ns), np.empty(Ns)
    t0 = time.perf_counter()
    for j in range(Ns):
        s = j * tau / Ns
        md = modes(s)
        with S.no_count():
            y = pde.grid.to_spec(state(s))
            G2 = pde.grid.to_phys(K.G2(y, pde.f, pde.applyA, pde.applyB))
        F1[j] = w(md['phase'], G2)
        F2[j] = w(md['frequency'], G2)
        O12[j] = w(md['phase'], md['frequency'])
    def rate_from(F1, F2, O12):
        n = len(F1)
        F1h = np.fft.fft(F1) / n
        m = np.fft.fftfreq(n, d=1.0 / n)
        nz = m != 0
        I1_mean = F1h[0].real * tau / 2 - np.real(
            np.sum(F1h[nz] / (1j * m[nz] * omega)))
        Om = float(np.mean(O12))
        return (F2.mean() - I1_mean) / Om, -F1.mean() / Om, Om

    ph, fr, Om12 = rate_from(F1, F2, O12)
    ph_half, _, _ = rate_from(F1[::2], F2[::2], O12[::2])
    rates = {'phase': ph, 'frequency': fr}
    diag = dict(Omega12=Om12, Omega12_variation=float(np.ptp(O12) / abs(Om12)),
                F1_mean_over_rms=float(abs(F1.mean()) / np.sqrt(np.mean(F1 ** 2))),
                sample_convergence=float(abs(ph_half - ph) / abs(ph)),
                build_s=time.perf_counter() - t0)
    return rates, diag


def main():
    rows = []
    cases = [
        ('kdv', lambda: predict_relative_equilibrium('kdv'),
         [('proj_kdv_h2e-03.csv', 0.002), ('proj_kdv_h1e-03.csv', 0.001)]),
        ('nls', lambda: predict_relative_equilibrium('nls'),
         [('proj_nls_h8e-03.csv', 0.008), ('proj_nls_h2e-02.csv', 0.016)]),
        ('sg_breather_w0.5', lambda: predict_breather(0.5),
         [('proj_sg_breather_w0.5_h1e-02.csv', 0.01),
          ('proj_sg_breather_w0.5_h2e-02.csv', 0.02)]),
    ]
    for name, fn, files in cases:
        pred, diag = fn()
        meas = measured_rates(files, list(pred.keys()))
        print(f"--- {name}")
        print("    diagnostics: " + ", ".join(
            f"{k}={v:.3e}" if isinstance(v, float) else f"{k}={v}"
            for k, v in diag.items()))
        for k, pv in pred.items():
            mlim, per_h = meas[k]
            rel = abs(pv - mlim) / abs(mlim) if abs(mlim) > 1e-6 else np.nan
            print(f"    {k:12s} predicted {pv:+.7e}   measured (h->0) {mlim:+.7e}   "
                  f"rel. diff {rel:.2e}   per h: "
                  + ", ".join(f"{h:g}: {r:+.6e}" for h, r in per_h))
            rows.append(dict(benchmark=name, mode=k,
                             predicted_rate=f"{pv:.8e}",
                             measured_rate_h0=f"{mlim:.8e}",
                             rel_diff=f"{rel:.3e}",
                             measured_per_h="; ".join(f"{h:g}:{r:.8e}"
                                                      for h, r in per_h),
                             **{f"diag_{a}": (f"{b:.4e}" if isinstance(b, float)
                                              else b) for a, b in diag.items()}))
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(RES / 'prediction_vs_measurement.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=keys, restval='')
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {RES / 'prediction_vs_measurement.csv'}")


if __name__ == '__main__':
    main()
