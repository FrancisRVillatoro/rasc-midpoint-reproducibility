"""Integrator module 'lawson' (phase F2): the Lawson-midpoint BASE scheme.

Implemented in the stage form specified by the PI note docs/T3_lawson.md,
Sec. 1, which anchors the Lawson gauge at t_n on every step so that no
exponential with |s| > h is ever formed:

    stage:   Y_m = e^{hL/2} U^n + (h/2) N(Y_m),
    update:  U^{n+1} = 2 e^{hL/2} Y_m - e^{hL} U^n,

with N = f - L the nonlinear part.  Eliminating Y_m recovers the form in
which the scheme is usually written,

    U^{n+1} = e^{hL} U^n + h e^{hL/2} N( (e^{hL/2}U^n + e^{-hL/2}U^{n+1})/2 ),

i.e. the implicit midpoint rule applied to the Lawson-transformed system
v' = e^{-sL} N(e^{sL} v); autonomization of that nonautonomous system is
exact for the midpoint rule (Corollary 2.8 of docs/T1_theorem.tex), so the
stage is simply evaluated at t_{n+1/2}.  The stage form is preferred for
the same reason as in midpoint.py: the stage Jacobian is

    J = I - (h/2) N'(Y_m),

with no L in it, so the linear part never enters the corrector solves and
no preconditioner is needed (T3, Sec. 1).  Newton is DAMPED by
backtracking, which that note makes mandatory: undamped Newton diverges
for large h k_max ||u||.

Scope note (CLAUDE.md P4): Burgers is EXCLUDED, because its linear symbol
is real and negative and e^{-hL/2} = exp(+h nu k^2/2) amplifies; the module
refuses to run on a PDE whose ``lawson_allowed`` flag is False.

Phase scope: this module provides the BASE scheme only.  ASC on top of
Lawson is phase F5, which stays BLOCKED until the header of
docs/T3_lawson.md carries the line "APROBADO FRV" (as of this phase it
does not).
"""
import time

import numpy as np

from integrators.midpoint import Stats, damped_newton


def step(y, h, pde, tol=1e-13, log=None):
    """One Lawson-midpoint step; returns U_{n+1}."""
    if not getattr(pde, 'lawson_allowed', True):
        raise ValueError(f"Lawson variant excluded for PDE '{pde.name}' "
                         "by CLAUDE.md Sec. P")
    half = 0.5 * h
    Ehu = pde.expL(y, half)

    def residual(Y):
        return Y - Ehu - half * pde.f_nl(Y)

    def jac_op(Y):
        def op(v):
            return v - half * pde.A_nl(Y, v)
        return op

    Y0 = Ehu + half * pde.f_nl(Ehu)
    Y, its, gits, backs, res = damped_newton(
        pde, residual, jac_op, lambda r: r, Y0, tol)
    if log is not None:
        log.record(its, gits, backs, res)
        log.stage = Y
        if res >= tol:
            log.converged = False
    return 2.0 * pde.expL(Y, half) - pde.expL(Ehu, half)


def integrate(pde, y0, h, nsteps, tol=1e-13, log=None, monitor=None):
    y = np.array(y0, dtype=complex, copy=True)
    t0 = time.perf_counter()
    for n in range(nsteps):
        y = step(y, h, pde, tol=tol, log=log)
        if monitor is not None:
            monitor(n + 1, (n + 1) * h, y)
    if log is not None:
        log.wall += time.perf_counter() - t0
    return y
