"""Integrator module 'asc' (phase F3): the asymptotic successive correction
hierarchy on top of the direct implicit midpoint rule.

Reconstructions, as fixed by CLAUDE.md:

    MP2  = U^n,
    ASC4 = U^n + h^2 P^n,
    ASC6 = U^n + h^2 P^n + h^4 R^n.

The two correctors are advanced by SEQUENTIAL SWEEPS, one per base step, in
the same loop as the base method and reading only the stage value it already
produced.  With Y_m the stage of the base step and A_m = A(Y_m),

    (I - (h/2) A_m) P^{n+1} = (I + (h/2) A_m) P^n - h G2(Y_m),
    (I - (h/2) A_m) R^{n+1} = (I + (h/2) A_m) R^n + h S(Y_m, P^{n+1/2}),
    S = (1/2) B(P, P) - G4 - H2,     P^{n+1/2} = (P^n + P^{n+1})/2,

which is the implicit midpoint rule applied to the corrector equations
P' = A P - G2 and R' = A R + S, frozen at the stage.  Every coefficient
comes from kernel/formulas.py and nothing is reimplemented here: G2, G4 and
the full source S through ``R_source``.  The ABLATION required by gate G3
removes the H2 term from S and nothing else, so it is built from
``G4`` plus the PDE's own B; it is the only place where the source is
assembled outside the kernel, exactly as in kernel/test_kernel.py.

Linear algebra.  The sweep matrix I - (h/2) A(Y_m) is the operator that
terminates the Newton iteration of the base stage equation, at the same
stage value, so the sweeps solve with it through
``midpoint.solve_stage_jacobian`` rather than rebuilding anything.  This is
the reuse the theorem note relies on when it counts the cost of the
hierarchy as one extra linear solve per corrector per step.

Why H2 is not optional in the mathematics.  The first corrector is itself
discretized, so the midpoint sweep for P commits its own modified-equation
defect at order h^2; H2 is that defect, and feeding it into the source of R
is what makes the third reconstruction order six rather than order four.
Dropping it must therefore drop ASC6 to ASC4, which is the sharpness
statement gate G3 checks.
"""
import time

import numpy as np

# CHANGES_REQUESTED CR-1 item 6: from phase F3 onward the stage tolerance
# is 1e-15 relative.  The F2 campaigns stand as shipped at 1e-13, and every
# floor table must carry the two-tolerance control run whose difference is
# the algebraic contribution.
TOL_F3 = 1e-15

import kernel.formulas as K
from integrators import midpoint as MP
from pdes import _spectral as S


class Stats(MP.Stats):
    """Base-method diagnostics plus the sweep counters."""

    def __init__(self):
        super().__init__()
        self.sweep_gmres = 0
        self.sweeps = 0

    def record_sweep(self, gits):
        self.sweeps += 1
        self.sweep_gmres += gits


def _ops(pde):
    return dict(f=pde.f, A=pde.applyA, B=pde.applyB, C=pde.applyC,
                D=pde.applyD)


def sweep_step(pde, ym, h, P, R, use_H2=True):
    """Advance both correctors across one base step whose stage was ym."""
    o = _ops(pde)
    half = 0.5 * h

    def apply_plus(v):
        return v + half * pde.applyA(ym, v)

    rhs_p = apply_plus(P) - h * K.G2(ym, o['f'], o['A'], o['B'])
    P1, g1 = MP.solve_stage_jacobian(pde, ym, h, rhs_p)
    Pm = 0.5 * (P + P1)
    if use_H2:
        src = K.R_source(ym, Pm, **o)
    else:
        # ablation: the same source with the H2 term removed and nothing else
        src = 0.5 * pde.applyB(ym, Pm, Pm) - K.G4(ym, **o)
    R1, g2 = MP.solve_stage_jacobian(pde, ym, h, apply_plus(R) + h * src)
    return P1, R1, g1 + g2


def integrate(pde, y0, h, nsteps, tol=TOL_F3, use_H2=True, log=None,
              monitor=None):
    """Run MP2 and both sweeps; return the three reconstructions.

    Returns a dict with keys 'MP2', 'ASC4', 'ASC6', 'P' and 'R'.
    """
    y = np.array(y0, dtype=complex, copy=True)
    P = np.zeros_like(y)
    R = np.zeros_like(y)
    base = log if log is not None else MP.Stats()
    t0 = time.perf_counter()
    for n in range(nsteps):
        y = MP.step(y, h, pde, tol=tol, log=base)
        ym = base.stage
        P, R, g = sweep_step(pde, ym, h, P, R, use_H2=use_H2)
        if isinstance(base, Stats):
            base.record_sweep(g)
        if monitor is not None:
            monitor(n + 1, (n + 1) * h, y, P, R)
    if log is not None:
        log.wall += time.perf_counter() - t0
    h2, h4 = h ** 2, h ** 4
    return {'MP2': y, 'ASC4': y + h2 * P, 'ASC6': y + h2 * P + h4 * R,
            'P': P, 'R': R}
