"""Shared Fourier pseudospectral infrastructure for the four PDE modules.

Not a PDE module: it holds the grid object, the FFT-counted transforms and
the padded (dealiased) product routines used by sg.py, nls.py, kdv.py and
burgers.py.  Adding this helper does not change the closed scope of
CLAUDE.md; it only factors code that would otherwise be repeated verbatim
in the four modules (logged in reports/phase2_report.md).

Conventions
-----------
State.  A state is a complex array of shape ``(ncomp, Nk)`` holding the
TRUE Fourier coefficients c_m of each real component field,

    u(x) = sum_m c_m exp(i k_m (x - a)),   k_m = 2 pi m / L,

with ``Nk = N//2 + 1`` (rfft layout).  With this normalization zero
padding and truncation are exact, with no rescaling.

Real subspace.  Hermitian-symmetric rfft arrays form a REAL vector space
of dimension N per component; the multilinear forms below are the
complex-linear extensions of the corresponding real multilinear forms,
so all kernel operations stay inside that subspace.  Two coordinates need
an explicit projection: the imaginary part of mode 0 (unused by irfft) and
the Nyquist mode N/2, which odd-order derivative symbols would otherwise
map outside the subspace.  ``Grid.project`` sets Im c_0 = 0 and c_{N/2} = 0
and is applied by every forward transform, so the discrete state space is
exactly R^{N-1} per component and every discrete field below is a smooth
map on it.  Since ``L_symbol`` also annihilates the Nyquist mode, the
projection commutes with all operators used here and therefore does not
interfere with the derivative structure required by kernel/formulas.py.

FFT accounting.  Every one-dimensional real transform of one component
counts as one FFT, regardless of length; the length is reported
separately.  Diagnostics must be wrapped in ``no_count()`` so that the
cost metric measures the integrator only.
"""
import numpy as np

_STATE = {'fft': 0, 'enabled': True}


def reset_fft_counter():
    _STATE['fft'] = 0


def fft_count():
    return _STATE['fft']


class no_count:
    """Context manager: do not charge transforms to the cost metric."""

    def __enter__(self):
        self._prev = _STATE['enabled']
        _STATE['enabled'] = False
        return self

    def __exit__(self, *exc):
        _STATE['enabled'] = self._prev
        return False


def _charge(n):
    if _STATE['enabled']:
        _STATE['fft'] += int(n)


class Grid:
    """Uniform periodic grid on [a, b) with N points and ncomp components."""

    def __init__(self, a, b, N, ncomp):
        if N % 2 != 0:
            raise ValueError("N must be even")
        self.a, self.b, self.N, self.ncomp = float(a), float(b), int(N), int(ncomp)
        self.L = float(b) - float(a)
        self.Nk = N // 2 + 1
        self.dx = self.L / N
        self.x = self.a + self.L * np.arange(N) / N
        self.m = np.arange(self.Nk)
        self.k = 2.0 * np.pi * self.m / self.L

    # ---------------------------------------------------------------- states
    def zeros(self):
        return np.zeros((self.ncomp, self.Nk), dtype=complex)

    def project(self, y):
        """Enforce membership of the real subspace (Im c_0 = 0, c_Nyq = 0)."""
        y = np.array(y, dtype=complex, copy=True)
        y[..., 0] = y[..., 0].real
        y[..., -1] = 0.0
        return y

    # ------------------------------------------------------------ transforms
    def to_phys(self, y, M=None):
        """Coefficients -> physical values on M points (M >= N, zero padded)."""
        M = self.N if M is None else int(M)
        Mk = M // 2 + 1
        y = np.atleast_2d(y)
        if M == self.N:
            yy = y
        else:
            yy = np.zeros(y.shape[:-1] + (Mk,), dtype=complex)
            yy[..., :self.Nk] = y
        out = np.fft.irfft(yy, n=M, axis=-1) * M
        _charge(np.prod(y.shape[:-1]))
        return out

    def to_spec(self, U):
        """Physical values on M points -> truncated, projected coefficients."""
        U = np.atleast_2d(U)
        M = U.shape[-1]
        c = np.fft.rfft(U, axis=-1) / M
        _charge(np.prod(U.shape[:-1]))
        if M != self.N:
            c = c[..., :self.Nk]
        return self.project(c)

    # ------------------------------------------------------------------ norms
    def sup(self, y):
        """Sup norm of the physical field (not charged to the cost metric)."""
        with no_count():
            return float(np.max(np.abs(self.to_phys(y))))

    def l2(self, y):
        with no_count():
            U = self.to_phys(y)
            return float(np.sqrt(self.dx * np.sum(U * U)))

    def spectral_tail(self, y, frac=2.0 / 3.0):
        """max |c_m| over the top (1-frac) of the modes, absolute and relative."""
        c = np.abs(np.atleast_2d(y))
        m0 = int(np.ceil(frac * (self.Nk - 1)))
        top = float(c[..., m0:].max())
        peak = float(c.max())
        last = float(c[..., -2].max())      # highest retained mode (Nyq is 0)
        return top, top / peak if peak > 0 else np.nan, last


# --------------------------------------------------------------- padding sizes
def pad_size(N, factor):
    """Smallest even, FFT-friendly M >= factor*N."""
    M = int(np.ceil(factor * N))
    if M % 2:
        M += 1
    while max(_factors(M)) > 7:
        M += 2
    return M


def _factors(n):
    fs, d = [], 2
    while d * d <= n:
        while n % d == 0:
            fs.append(d)
            n //= d
        d += 1
    if n > 1:
        fs.append(n)
    return fs or [1]


# ------------------------------------------------------------------ base class
class SpectralPDE:
    """Common linear-part machinery: L y, (I-(h/2)L)^{-1} and exp(tL).

    Subclasses set ``self.grid`` and ``self.Lmat`` of shape
    ``(ncomp, ncomp, Nk)`` (the mode-diagonal symbol), and implement the
    nonlinear part together with ``f``, ``applyA``, ``applyB``, ``applyC``
    and ``applyD``.  The Nyquist column of ``Lmat`` is zeroed so that the
    linear operator preserves the real subspace defined in ``Grid.project``.
    """

    lawson_allowed = True

    def _set_linear(self, Lmat):
        Lmat = np.array(Lmat, dtype=complex)
        Lmat[..., -1] = 0.0
        self.Lmat = Lmat
        self._pc_cache = {}
        self._ex_cache = {}

    def L_symbol(self, k=None):
        """Linear symbol; full (ncomp,ncomp,Nk) array when k is omitted."""
        if k is None:
            return self.Lmat.copy()
        idx = np.searchsorted(self.grid.k, np.atleast_1d(k))
        return self.Lmat[..., idx]

    def apply_L(self, y):
        return np.einsum('ijm,jm->im', self.Lmat, y)

    # --------------------------------------------------- mode-diagonal solves
    def precond(self, r, h):
        """Return (I - (h/2) L)^{-1} r, the mode-diagonal preconditioner."""
        return self._solve_shift(r, -0.5 * h)

    def _solve_shift(self, r, s):
        """Solve (I + s L) x = r per mode."""
        key = ('s', float(s))
        inv = self._pc_cache.get(key)
        if inv is None:
            nc = self.grid.ncomp
            Amat = np.eye(nc)[:, :, None] + s * self.Lmat
            inv = _inv_stack(Amat)
            self._pc_cache[key] = inv
        return np.einsum('ijm,jm->im', inv, r)

    # ------------------------------------------------- nonlinear part helpers
    def f_nl(self, y):
        """N(y) = f(y) - L y."""
        return self.f(y) - self.apply_L(y)

    def A_nl(self, y, v):
        """N'(y) v = f'(y) v - L v."""
        return self.applyA(y, v) - self.apply_L(v)

    def expL(self, y, t):
        """Exact exp(t L) y (mode-diagonal, closed form)."""
        key = float(t)
        E = self._ex_cache.get(key)
        if E is None:
            E = _expm_stack(self.Lmat, t)
            self._ex_cache[key] = E
        return np.einsum('ijm,jm->im', E, y)


def _inv_stack(Amat):
    """Inverse of a stack of 1x1 or 2x2 matrices, shape (n,n,Nk)."""
    n = Amat.shape[0]
    if n == 1:
        return (1.0 / Amat[0, 0])[None, None, :]
    if n == 2:
        a, b, c, d = Amat[0, 0], Amat[0, 1], Amat[1, 0], Amat[1, 1]
        det = a * d - b * c
        return np.array([[d / det, -b / det], [-c / det, a / det]])
    raise NotImplementedError("only 1 or 2 components")


def _expm_stack(Lmat, t):
    """exp(t L) for a stack of 1x1 or traceless 2x2 symbols."""
    n = Lmat.shape[0]
    if n == 1:
        return np.exp(t * Lmat[0, 0])[None, None, :]
    if n != 2:
        raise NotImplementedError("only 1 or 2 components")
    tr = Lmat[0, 0] + Lmat[1, 1]
    if np.max(np.abs(tr)) > 1e-14 * max(1.0, np.max(np.abs(Lmat))):
        raise NotImplementedError("closed form assumes a traceless symbol")
    det = Lmat[0, 0] * Lmat[1, 1] - Lmat[0, 1] * Lmat[1, 0]
    # M^2 = -det I  =>  exp(tM) = cos(w t) I + sin(w t)/w M with w = sqrt(det)
    w = np.sqrt(det.astype(complex))
    small = np.abs(w) < 1e-14
    cs = np.cos(w * t)
    sn = np.where(small, t, np.sin(w * t) / np.where(small, 1.0, w))
    I = np.eye(2)[:, :, None]
    return cs[None, None, :] * I + sn[None, None, :] * Lmat
