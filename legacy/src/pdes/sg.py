"""PDE module 'sg' (phase F2). Normalization FIXED by CLAUDE.md Sec. P1.

    u_tt - u_xx + sin u = 0,   periodic,

written as the first-order system (u, v = u_t),

    u_t = v,        v_t = u_xx - sin u,

so the linear symbol is the traceless wave generator

    L(k) = [[0, 1], [-k^2, 0]].

Benchmarks (CLAUDE.md P1):
  kink-antikink   u = 4 arctan( sinh(gamma c (t - tc)) / (c cosh(gamma x)) ),
                  gamma = 1/sqrt(1-c^2), c = 0.5, tc = 4, T = 8,
                  x in [-24, 24), N = 768 baseline;
  breather        u = 4 arctan( (sqrt(1-w^2)/w) sin(w t) / cosh(sqrt(1-w^2) x) ),
                  w = 0.1, x in [-40, 40), N = 768, reserved for the
                  rho-criterion study of phase F3.
Both are verified by sympy substitution in exact/verify_sg.py before first
use.  Neither is exactly periodic; the periodization error is reported with
the spectral-tail diagnostics.

The nonlinearity is a pointwise (Nemytskii) operator, so no padded product
applies and, per the contract, sine-Gordon relies instead on the documented
spectral-tail gate.  With g(U) = -sin U the derivatives are the pointwise
multiplications by g' = -cos U, g'' = sin U, g''' = cos U, g'''' = -sin U,
all evaluated on the same N-point grid as f, so applyA...applyD are the
exact derivatives of the discrete field.

FFT budget: 2 transforms of length N per f, 3 per applyA, 4 per applyB,
5 per applyC, 6 per applyD.
"""
import numpy as np

from . import _spectral as S

# CHANGES_REQUESTED.md CR-02 and CR-03, approved by the PI (APROBADO FRV):
# the kink-antikink resolution goes from N=256 to N=512 (CR-02) and then
# to N=768 (CR-05, required by the corrected running-maximum tail
# instrument, which the two-endpoint instrument had masked); the breather
# moves from [-20,20) with N=128 to [-40,40) with N=768.  Everything else in
# Sec. P1 (c, t_c, T, the kink-antikink domain, omega) is as written in
# CLAUDE.md.  N=256 is kept only as a deliberately under-resolved diagnostic
# in the N-uniformity study of phase F3, never as a gate configuration.
params = {
    'kink_antikink': dict(c=0.5, tc=4.0, a=-24.0, b=24.0, N=768, T=8.0),
    'breather': dict(omega=0.1, a=-40.0, b=40.0, N=768, T=8.0),
}


def L_symbol(k):
    """Linear symbol per Fourier mode: [[0, 1], [-k^2, 0]]."""
    k2 = np.asarray(k) ** 2
    z = np.zeros_like(k2)
    o = np.ones_like(k2)
    return np.array([[z, o], [-k2, z]], dtype=complex)


class SineGordon(S.SpectralPDE):
    name = 'sg'
    ncomp = 2
    lawson_allowed = True         # optional per CLAUDE.md; unused in F2

    def __init__(self, config='kink_antikink', N=None, **over):
        p = dict(params[config])
        p.update(over)
        if N is not None:
            p['N'] = N
        self.config, self.p = config, p
        self.grid = S.Grid(p['a'], p['b'], p['N'], 2)
        self.M = p['N']                       # no padding (see module docstring)
        self._set_linear(L_symbol(self.grid.k))

    def L_symbol(self, k=None):
        if k is None:
            return self.Lmat.copy()
        return L_symbol(k)

    # ----------------------------------------------------------- nonlinearity
    def _u(self, y):
        return self.grid.to_phys(y[0:1], self.M)[0]

    def _pack(self, g):
        z = np.zeros_like(g)
        return self.grid.to_spec(np.stack([z, g]))

    def f(self, y):
        U = self._u(y)
        return self.apply_L(y) + self._pack(-np.sin(U))

    def applyA(self, y, v):
        U, Vu = self._u(y), self._u(v)
        return self.apply_L(v) + self._pack(-np.cos(U) * Vu)

    def applyB(self, y, u, v):
        U, Uu, Vu = self._u(y), self._u(u), self._u(v)
        return self._pack(np.sin(U) * Uu * Vu)

    def applyC(self, y, u, v, w):
        U, Uu, Vu, Wu = self._u(y), self._u(u), self._u(v), self._u(w)
        return self._pack(np.cos(U) * Uu * Vu * Wu)

    def applyD(self, y, u, v, w, x):
        U = self._u(y)
        Uu, Vu, Wu, Xu = self._u(u), self._u(v), self._u(w), self._u(x)
        return self._pack(-np.sin(U) * Uu * Vu * Wu * Xu)

    # ------------------------------------------------------------ benchmarks
    def exact_kink_antikink(self, t, phys=False):
        p = self.p
        c, tc = p['c'], p['tc']
        g = 1.0 / np.sqrt(1.0 - c * c)
        x = self.grid.x
        s, ch = np.sinh(g * c * (t - tc)), np.cosh(g * x)
        arg = s / (c * ch)
        U = 4.0 * np.arctan(arg)
        V = 4.0 * (g * np.cosh(g * c * (t - tc)) / ch) / (1.0 + arg ** 2)
        Y = np.stack([U, V])
        return Y if phys else self.grid.to_spec(Y)

    def exact_breather(self, t, phys=False):
        w = self.p['omega']
        r = np.sqrt(1.0 - w * w)
        x = self.grid.x
        ch = np.cosh(r * x)
        arg = (r / w) * np.sin(w * t) / ch
        U = 4.0 * np.arctan(arg)
        V = 4.0 * (r * np.cos(w * t) / ch) / (1.0 + arg ** 2)
        Y = np.stack([U, V])
        return Y if phys else self.grid.to_spec(Y)

    # ------------------------------------------------------------- invariants
    def energy(self, y):
        """H = int ( v^2/2 + u_x^2/2 + 1 - cos u ) dx."""
        with S.no_count():
            ik = 1j * self.grid.k
            ux = self.grid.to_phys(self.grid.project(ik * y[0:1]))[0]
            Y = self.grid.to_phys(y)
            U, V = Y[0], Y[1]
            return float(self.grid.dx * np.sum(0.5 * V ** 2 + 0.5 * ux ** 2
                                               + 1.0 - np.cos(U)))


def make(config='kink_antikink', N=None, **over):
    return SineGordon(config, N, **over)
