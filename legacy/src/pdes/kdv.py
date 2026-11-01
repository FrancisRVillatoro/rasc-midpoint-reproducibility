"""PDE module 'kdv' (phase F2). Normalizations FIXED by CLAUDE.md Sec. P3.

    u_t + alpha u u_x + beta u_xxx = 0,  periodic, Fourier pseudospectral.

Two configurations, both prescribed by the contract:
  'soliton'  alpha = 6, beta = 1        (u_t + 6 u u_x + u_xxx = 0)
             benchmarks: 1-soliton, 2-soliton interaction.
  'zk'       alpha = 1, beta = delta^2  (Zabusky-Kruskal stress test,
             u0 = cos(pi x), x in [0,2), delta in {0.022, 0.011, 0.0055});
             it carries no closed-form solution and is used from phase F3
             onwards with a Gauss s=3 reference.

State: single component, spectral (see pdes/_spectral.py for the exact
convention).  The discrete field is

    f(y) = i beta k^3 y - (alpha/2) i k D2(y, y),

where D2 is the 3/2-padded (dealiased) product.  Because D2 is an exact
symmetric bilinear form on the discrete state space, applyA and applyB are
the EXACT first and second derivatives of the discrete f, and the third and
fourth derivatives vanish identically.  That exactness is what the frozen
kernel requires; it is verified numerically in campaigns/f2_derivatives.py.

FFT budget: 2 transforms of length M = 3N/2 per evaluation of f, 4 per
applyA and 4 per applyB.
"""
import numpy as np

from . import _spectral as S

params = {
    # x0 = -2 centres the one-soliton trajectory on mid-domain over t in
    # [0, T] and is an exact integer number of grid points from the previous
    # anchoring (24 at N=768, 48 at N=1536); CR-1 item 5.
    'soliton': dict(alpha=6.0, beta=1.0, a=-32.0, b=32.0, N=768,
                    kappa=1.0, x0=-2.0, T=1.0,
                    kappa1=1.0, kappa2=2.0, x01=0.0, x02=0.0,
                    t2_start=-0.5, T2=1.0),
    'zk': dict(alpha=1.0, beta=0.022 ** 2, a=0.0, b=2.0, N=512, T=1.0),
}


def L_symbol(k, beta=1.0):
    """Linear symbol of -beta u_xxx per Fourier mode: i beta k^3."""
    return 1j * beta * np.asarray(k) ** 3


class KdV(S.SpectralPDE):
    name = 'kdv'
    ncomp = 1
    lawson_allowed = True

    def __init__(self, config='soliton', N=None, **over):
        p = dict(params[config])
        p.update(over)
        if N is not None:
            p['N'] = N
        self.config, self.p = config, p
        self.alpha, self.beta = p['alpha'], p['beta']
        self.grid = S.Grid(p['a'], p['b'], p['N'], 1)
        self.M = S.pad_size(p['N'], p.get('pad', 1.5))
        self._set_linear(L_symbol(self.grid.k, self.beta)[None, None, :])
        self.ik = (1j * self.grid.k)[None, :]

    def L_symbol(self, k=None):
        if k is None:
            return self.Lmat.copy()
        return L_symbol(k, self.beta)

    # ----------------------------------------------------------- nonlinearity
    def _D2(self, a, b):
        """3/2-padded (dealiased) product of two spectral rows."""
        A = self.grid.to_phys(a, self.M)
        B = self.grid.to_phys(b, self.M)
        return self.grid.to_spec(A * B)

    def f(self, y):
        return self.apply_L(y) - 0.5 * self.alpha * self.ik * self._D2(y, y)

    def applyA(self, y, v):
        return self.apply_L(v) - self.alpha * self.ik * self._D2(y, v)

    def applyB(self, y, u, v):
        return -self.alpha * self.ik * self._D2(u, v)

    def applyC(self, y, u, v, w):
        return np.zeros_like(y)

    def applyD(self, y, u, v, w, x):
        return np.zeros_like(y)

    # ------------------------------------------------------------ benchmarks
    def exact_soliton(self, t, phys=False):
        """u = (12 beta/alpha) kappa^2 sech^2(kappa (x - c t - x0)),
        c = 4 beta kappa^2."""
        p = self.p
        kap, x0 = p['kappa'], p['x0']
        c = 4.0 * self.beta * kap ** 2
        amp = 12.0 * self.beta / self.alpha * kap ** 2
        xi = self._wrap(self.grid.x - c * t - x0)
        U = amp / np.cosh(kap * xi) ** 2
        return U[None, :] if phys else self.grid.to_spec(U[None, :])

    def exact_two_soliton(self, t, phys=False):
        """Two-soliton in tau-function form; alpha = 6, beta = 1 only."""
        if not (self.alpha == 6.0 and self.beta == 1.0):
            raise ValueError("two-soliton is defined for alpha=6, beta=1")
        p = self.p
        k1, k2 = p['kappa1'], p['kappa2']
        x = self._wrap(self.grid.x)
        th1 = 2 * k1 * (x - p['x01']) - 8 * k1 ** 3 * t
        th2 = 2 * k2 * (x - p['x02']) - 8 * k2 ** 3 * t
        A = ((k1 - k2) / (k1 + k2)) ** 2
        e1, e2 = np.exp(th1), np.exp(th2)
        e12 = e1 * e2
        c1, c2, c12 = 2 * k1, 2 * k2, 2 * (k1 + k2)
        tau = 1.0 + e1 + e2 + A * e12
        taux = c1 * e1 + c2 * e2 + A * c12 * e12
        tauxx = c1 ** 2 * e1 + c2 ** 2 * e2 + A * c12 ** 2 * e12
        U = 2.0 * (tauxx / tau - (taux / tau) ** 2)
        return U[None, :] if phys else self.grid.to_spec(U[None, :])

    def initial_zk(self, phys=False):
        """u0 = cos(pi x) on [0,2) (Zabusky-Kruskal stress test)."""
        U = np.cos(np.pi * self.grid.x)
        return U[None, :] if phys else self.grid.to_spec(U[None, :])

    def _wrap(self, xi):
        """Map a displacement into the fundamental period [-L/2, L/2).

        The argument is a DISPLACEMENT from the soliton centre, so the
        domain offset a must not enter: subtracting it, as an earlier
        version did, translated every sampled benchmark by exactly L/2 and
        left the soliton peaks sitting on the domain edge instead of
        mid-domain.  Because L/2 is an integer number of grid points on
        every configuration used here, that translation left all errors and
        orders unchanged, which is why it survived gate G2; what it did
        corrupt is the edge diagnostic, which was reading the peak rather
        than the tail.  The equivariance is verified in
        campaigns/f3_kdv_recentering.py (CHANGES_REQUESTED CR-1 item 5).
        """
        L = self.grid.L
        return (xi + L / 2) % L - L / 2


def make(config='soliton', N=None, **over):
    return KdV(config, N, **over)
