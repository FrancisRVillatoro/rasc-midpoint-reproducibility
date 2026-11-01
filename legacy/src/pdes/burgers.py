"""PDE module 'burgers' (phase F2). Normalization FIXED by CLAUDE.md Sec. P4.

    u_t + u u_x = nu u_xx,   nu = 0.05,   x in [0, 2 pi).

Exact Cole-Hopf reference (CLAUDE.md P4)

    u = -2 nu theta_x / theta,   theta = 2 + exp(-nu t) cos x,

which is exactly 2 pi-periodic and analytic (theta >= 1 > 0), so there is
no periodization error for this benchmark.

The Lawson variant is EXCLUDED for this PDE by the contract: the linear
symbol is real and negative, so the Lawson midpoint would evaluate the
nonlinearity through exp(-h L/2) = exp(+ h nu k^2 / 2), an amplification
that grows like exp(h nu k_max^2 / 2) and destroys the conditioning of the
corrector solves at exactly the resolutions where the exponential variant
would be attractive for the dispersive equations.  No Lawson results are
produced for Burgers; this paragraph is the required explanation.

State: single component, spectral.  Discrete field

    f(y) = -nu k^2 y - (1/2) i k D2(y, y),

with D2 the 3/2-padded product; applyA and applyB are exact, applyC and
applyD vanish.  FFT budget: 2 transforms of length M = 3N/2 per f.
"""
import numpy as np

from . import _spectral as S

params = {
    'cole_hopf': dict(nu=0.05, a=0.0, b=2.0 * np.pi, N=128, T=2.0),
}


def L_symbol(k, nu=0.05):
    """Linear symbol of nu u_xx per Fourier mode: -nu k^2."""
    return -nu * np.asarray(k) ** 2 + 0j


class Burgers(S.SpectralPDE):
    name = 'burgers'
    ncomp = 1
    lawson_allowed = False        # CLAUDE.md P4

    def __init__(self, config='cole_hopf', N=None, **over):
        p = dict(params[config])
        p.update(over)
        if N is not None:
            p['N'] = N
        self.config, self.p = config, p
        self.nu = p['nu']
        self.grid = S.Grid(p['a'], p['b'], p['N'], 1)
        self.M = S.pad_size(p['N'], p.get('pad', 1.5))
        self._set_linear(L_symbol(self.grid.k, self.nu)[None, None, :])
        self.ik = (1j * self.grid.k)[None, :]

    def L_symbol(self, k=None):
        if k is None:
            return self.Lmat.copy()
        return L_symbol(k, self.nu)

    # ----------------------------------------------------------- nonlinearity
    def _D2(self, a, b):
        A = self.grid.to_phys(a, self.M)
        B = self.grid.to_phys(b, self.M)
        return self.grid.to_spec(A * B)

    def f(self, y):
        return self.apply_L(y) - 0.5 * self.ik * self._D2(y, y)

    def applyA(self, y, v):
        return self.apply_L(v) - self.ik * self._D2(y, v)

    def applyB(self, y, u, v):
        return -self.ik * self._D2(u, v)

    def applyC(self, y, u, v, w):
        return np.zeros_like(y)

    def applyD(self, y, u, v, w, x):
        return np.zeros_like(y)

    # ------------------------------------------------------------ benchmarks
    def exact_cole_hopf(self, t, phys=False):
        nu, x = self.nu, self.grid.x
        e = np.exp(-nu * t)
        theta = 2.0 + e * np.cos(x)
        U = 2.0 * nu * e * np.sin(x) / theta
        return U[None, :] if phys else self.grid.to_spec(U[None, :])


def make(config='cole_hopf', N=None, **over):
    return Burgers(config, N, **over)
