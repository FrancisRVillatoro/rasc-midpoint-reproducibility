"""Integrator module 'midpoint' (phase F2): MP2, the implicit midpoint rule.

Stage form, exactly Eq. (2.1) of docs/T1_theorem.tex:

    Y_m = U^n + (h/2) f(Y_m),      U^{n+1} = 2 Y_m - U^n,

so that Y_m = (U^n + U^{n+1})/2.  The stage is the primary unknown rather
than U^{n+1} because the theorem, the corrector sweeps of phase F3 and the
error estimator are all written at Y_m: the Jacobian of the stage equation,

    J = I - (h/2) A(Y_m),

is the matrix the P and R sweeps have to solve with at every step, so
solving for the stage makes the terminal Newton operator of the base step
literally the operator the sweeps reuse (Remark on postprocessing and reuse
of the linear algebra in T1).  ``step`` therefore optionally returns the
stage through the Stats object.

Linear algebra.  With the mode-diagonal preconditioner M = I - (h/2) L
prescribed by CLAUDE.md, left preconditioning gives

    M^{-1} J = I - (h/2) M^{-1} N'(Y_m),

an identity-plus-non-stiff-perturbation system: the stiff part of the
Jacobian is inverted exactly and only the nonlinearity is left to the
Krylov solver.  GMRES uses the real inner product induced by the
Hermitian-symmetric state space of pdes/_spectral.py, so every iterate
stays in that space exactly.  The Newton iteration is damped by
backtracking on the residual norm; with the step sizes of the F2 campaign
the full step is always accepted, but the safeguard is kept because the
Lawson track requires it (docs/T3_lawson.md, Sec. 1) and both tracks share
this solver.

Stopping criterion (the "Newton residual" of gate G2): sup norm over the
physical grid of the stage residual G(Y) = Y - U^n - (h/2) f(Y), relative
to the sup norm of the physical stage,

    ||G(Y)||_inf / max(||Y||_inf, 1e-30) < tol,   tol = 1e-13 by default.

Iteration counts, GMRES counts, residuals and FFT counts are accumulated in
the optional Stats object.  No adaptive time stepping is used anywhere: the
damping acts on the Newton correction, never on h.
"""
import time

import numpy as np

from pdes import _spectral as S

MAXNEWTON = 20
MAXGMRES = 40
MAXBACKTRACK = 8
LAMBDA_MIN = 1.0 / 64.0
STAGNATION_FACTOR = 0.25
INNER_RTOL = 1e-10


class Stats:
    """Accumulator for solver diagnostics over a run."""

    def __init__(self):
        self.steps = 0
        self.newton_its = 0
        self.gmres_its = 0
        self.newton_max = 0
        self.backtracks = 0
        self.max_residual = 0.0
        self.converged = True
        self.wall = 0.0
        self.fft0 = S.fft_count()
        self.stage = None

    def record(self, its, gits, back, res):
        self.steps += 1
        self.newton_its += its
        self.gmres_its += gits
        self.backtracks += back
        self.newton_max = max(self.newton_max, its)
        self.max_residual = max(self.max_residual, res)

    @property
    def ffts(self):
        return S.fft_count() - self.fft0

    def asdict(self, prefix=''):
        n = max(self.steps, 1)
        return {prefix + 'steps': self.steps,
                prefix + 'newton_its_per_step': self.newton_its / n,
                prefix + 'newton_its_max': self.newton_max,
                prefix + 'gmres_its_per_step': self.gmres_its / n,
                prefix + 'backtracks': self.backtracks,
                prefix + 'newton_residual_max': self.max_residual,
                prefix + 'newton_converged': int(self.converged),
                prefix + 'ffts': self.ffts,
                prefix + 'wall': self.wall}


# ------------------------------------------------------------- linear algebra
def dot(a, b):
    return float(np.sum((a.conjugate() * b).real))


def norm(a):
    return float(np.sqrt(np.sum((a.conjugate() * a).real)))


def gmres(apply_op, b, rtol=INNER_RTOL, maxit=MAXGMRES):
    """Preconditioned GMRES with the real inner product of the state."""
    beta = norm(b)
    if beta == 0.0:
        return np.zeros_like(b), 0
    V = [b / beta]
    H = np.zeros((maxit + 1, maxit))
    yv = np.zeros(1)
    for j in range(maxit):
        w = apply_op(V[j])
        for i in range(j + 1):
            H[i, j] = dot(V[i], w)
            w = w - H[i, j] * V[i]
        H[j + 1, j] = norm(w)
        e1 = np.zeros(j + 2)
        e1[0] = beta
        yv, *_ = np.linalg.lstsq(H[:j + 2, :j + 1], e1, rcond=None)
        res = float(np.linalg.norm(H[:j + 2, :j + 1] @ yv - e1))
        if res <= rtol * beta or H[j + 1, j] <= 1e-300:
            return sum(yv[i] * V[i] for i in range(j + 1)), j + 1
        V.append(w / H[j + 1, j])
    return sum(yv[i] * V[i] for i in range(maxit)), maxit


def rel_residual(pde, G, Y):
    with S.no_count():
        num = float(np.max(np.abs(pde.grid.to_phys(G))))
        den = max(float(np.max(np.abs(pde.grid.to_phys(Y)))), 1e-30)
    return num / den


def damped_newton(pde, residual, jac_op, precond, Y, tol):
    """Damped Newton on a stage equation.

    ``tol > 0`` stops at the prescribed relative residual, which is the
    contract criterion.  ``tol <= 0`` instead iterates to STAGNATION, i.e.
    until the residual stops falling by at least a factor of four, which
    drives the algebraic error of the stage solve down to the round-off
    limit of the arithmetic; it is used only to build the reference orbits
    that measure the algebraic floor left by the prescribed tolerance, never
    in a gate run.

    Returns (Y, newton_its, gmres_its, backtracks, residual).
    """
    r = residual(Y)
    res = rel_residual(pde, r, Y)
    its = gits = backs = 0
    prev = np.inf
    for _ in range(MAXNEWTON):
        if tol > 0.0:
            if res < tol:
                break
        elif res == 0.0 or res > STAGNATION_FACTOR * prev:
            break
        prev = res
        d, ng = gmres(jac_op(Y), -precond(r))
        gits += ng
        lam = 1.0
        while True:
            Yt = Y + lam * d
            rt = residual(Yt)
            rest = rel_residual(pde, rt, Yt)
            if rest < res or lam <= LAMBDA_MIN:
                break
            lam *= 0.5
            backs += 1
        Y, r, res = Yt, rt, rest
        its += 1
    return Y, its, gits, backs, res


def solve_stage_jacobian(pde, ym, h, rhs):
    """Solve (I - (h/2) A(ym)) x = rhs with the mode-diagonal preconditioner.

    This is the operator that terminates the Newton iteration of the stage
    equation, and the corrector sweeps of phase F3 solve with exactly it,
    at exactly the same stage value, so the linear algebra of the base step
    is reused rather than rebuilt (docs/T1_theorem.tex, remark on
    postprocessing).  Returns (x, gmres_iterations).
    """
    def op(v):
        return v - 0.5 * h * pde.precond(pde.A_nl(ym, v), h)

    return gmres(op, pde.precond(rhs, h))


# --------------------------------------------------------------------- MP2
def step(y, h, pde, tol=1e-13, log=None):
    """One implicit midpoint step; returns U_{n+1}."""
    half = 0.5 * h

    def residual(Y):
        return Y - y - half * pde.f(Y)

    def jac_op(Y):
        def op(v):
            return v - half * pde.precond(pde.A_nl(Y, v), h)
        return op

    Y0 = y + half * pde.f(y)                       # explicit half Euler
    Y, its, gits, backs, res = damped_newton(
        pde, residual, jac_op, lambda r: pde.precond(r, h), Y0, tol)
    if log is not None:
        log.record(its, gits, backs, res)
        log.stage = Y
        if res >= tol:
            log.converged = False
    return 2.0 * Y - y


def integrate(pde, y0, h, nsteps, tol=1e-13, log=None, monitor=None):
    """Fixed-step MP2 integration; returns the final state."""
    y = np.array(y0, dtype=complex, copy=True)
    t0 = time.perf_counter()
    for n in range(nsteps):
        y = step(y, h, pde, tol=tol, log=log)
        if monitor is not None:
            monitor(n + 1, (n + 1) * h, y)
    if log is not None:
        log.wall += time.perf_counter() - t0
    return y
