"""PDE module 'nls' (phase F2). Normalization FIXED by CLAUDE.md Sec. P2.

    i u_t + u_xx + 2 |u|^2 u = 0,   u = P + i Q,   periodic.

Real two-component form (the form prescribed by the contract):

    P_t = -Q_xx - 2 (P^2 + Q^2) Q,
    Q_t = +P_xx + 2 (P^2 + Q^2) P,

so the linear symbol is the traceless rotation generator

    L(k) = [[0, k^2], [-k^2, 0]].

Benchmark: bright soliton with eta = 1, c = 1,

    u = eta sech(eta (x - x0 - c t)) exp( i ( c x / 2 + (eta^2 - c^2/4) t ) ),

verified by sympy substitution in exact/verify_nls.py before first use.
Periodicity of the carrier requires c L / 2 in 2 pi Z, hence L = 4 pi m / c;
the default domain is L = 24 pi (m = 6), which also puts the sech tail at
the boundary below 1e-16 for the default time window.

Mass M = int |u|^2 dx = dx sum (P^2 + Q^2) is a quadratic invariant of the
Fourier-Galerkin truncation (the cubic term is fully dealiased with
factor-2 padding, so the discrete nonlinearity is the exact Galerkin
projection); the implicit midpoint rule therefore conserves it to round-off,
which gate G2 checks.

The cubic nonlinearity makes applyA, applyB and applyC exact and applyD
identically zero.  FFT budget: 4 transforms of length M = 2N per f, 6 per
applyA, 8 per applyB, 8 per applyC.
"""
import numpy as np

from . import _spectral as S

params = {
    'bright': dict(eta=1.0, c=1.0, m_period=6, N=768, x0=-2.0, T=4.0),
}


def L_symbol(k):
    """Linear symbol per Fourier mode: [[0, k^2], [-k^2, 0]]."""
    k2 = np.asarray(k) ** 2
    z = np.zeros_like(k2)
    return np.array([[z, k2], [-k2, z]], dtype=complex)


class NLS(S.SpectralPDE):
    name = 'nls'
    ncomp = 2
    lawson_allowed = True

    def __init__(self, config='bright', N=None, **over):
        p = dict(params[config])
        p.update(over)
        if N is not None:
            p['N'] = N
        self.config, self.p = config, p
        self.eta, self.c = p['eta'], p['c']
        L = 4.0 * np.pi * p['m_period'] / self.c
        self.grid = S.Grid(-0.5 * L, 0.5 * L, p['N'], 2)
        self.M = S.pad_size(p['N'], p.get('pad', 2.0))
        self._set_linear(L_symbol(self.grid.k))

    def L_symbol(self, k=None):
        if k is None:
            return self.Lmat.copy()
        return L_symbol(k)

    # ----------------------------------------------------------- nonlinearity
    def _phys(self, y):
        U = self.grid.to_phys(y, self.M)
        return U[0], U[1]

    def f(self, y):
        P, Q = self._phys(y)
        rho = P * P + Q * Q
        return self.apply_L(y) + self.grid.to_spec(
            np.stack([-2.0 * rho * Q, 2.0 * rho * P]))

    def applyA(self, y, v):
        P, Q = self._phys(y)
        Vp, Vq = self._phys(v)
        P2, Q2, PQ = P * P, Q * Q, P * Q
        np_ = -2.0 * (2.0 * PQ * Vp + (P2 + 3.0 * Q2) * Vq)
        nq_ = 2.0 * ((3.0 * P2 + Q2) * Vp + 2.0 * PQ * Vq)
        return self.apply_L(v) + self.grid.to_spec(np.stack([np_, nq_]))

    def applyB(self, y, u, v):
        P, Q = self._phys(y)
        Up, Uq = self._phys(u)
        Vp, Vq = self._phys(v)
        cross = Up * Vq + Uq * Vp
        np_ = -4.0 * Q * Up * Vp - 4.0 * P * cross - 12.0 * Q * Uq * Vq
        nq_ = 12.0 * P * Up * Vp + 4.0 * Q * cross + 4.0 * P * Uq * Vq
        return self.grid.to_spec(np.stack([np_, nq_]))

    def applyC(self, y, u, v, w):
        Up, Uq = self._phys(u)
        Vp, Vq = self._phys(v)
        Wp, Wq = self._phys(w)
        np_ = -4.0 * (Up * Vp * Wq + Up * Vq * Wp + Uq * Vp * Wp) \
              - 12.0 * Uq * Vq * Wq
        nq_ = 12.0 * Up * Vp * Wp \
              + 4.0 * (Up * Vq * Wq + Uq * Vp * Wq + Uq * Vq * Wp)
        return self.grid.to_spec(np.stack([np_, nq_]))

    def applyD(self, y, u, v, w, x):
        return np.zeros_like(y)

    # ------------------------------------------------------------ benchmarks
    def exact_soliton(self, t, phys=False):
        eta, c, x0 = self.eta, self.c, self.p['x0']
        x = self.grid.x
        env = eta / np.cosh(eta * (x - x0 - c * t))
        ph = 0.5 * c * x + (eta ** 2 - 0.25 * c ** 2) * t
        U = np.stack([env * np.cos(ph), env * np.sin(ph)])
        return U if phys else self.grid.to_spec(U)

    # ------------------------------------------------------------- invariants
    def mass(self, y):
        with S.no_count():
            U = self.grid.to_phys(y)
            return float(self.grid.dx * np.sum(U[0] ** 2 + U[1] ** 2))

    def momentum(self, y):
        """Im int conj(u) u_x dx, the second quadratic invariant."""
        with S.no_count():
            ik = 1j * self.grid.k
            dy = self.grid.project(np.stack([ik * y[0], ik * y[1]]))
            U = self.grid.to_phys(y)
            dU = self.grid.to_phys(dy)
            return float(self.grid.dx * np.sum(U[0] * dU[1] - U[1] * dU[0]))


def make(config='bright', N=None, **over):
    return NLS(config, N, **over)
