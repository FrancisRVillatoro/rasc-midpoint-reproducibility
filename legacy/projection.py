"""Projection experiment for the secularity paper (not part of the F-phase
contract; it imports the frozen repository read-only and writes nothing
inside it).

Question.  Over long times the base error of MP2 and the ASC corrector P grow.
For a benchmark that is a relative equilibrium or a periodic orbit, how much
of that growth lies along the tangent space of the solution family (position,
phase, velocity, amplitude, frequency), which a time-dependent phase or
translation could absorb without being expanded, and how much is transversal
deformation, which it could not?

Quantities, recorded at output times t_j along one long run:
  e(t) = (u_exact(t) - U(t)) / h^2    scaled base error of MP2
  P(t)                                the first ASC corrector (ASC4 = U + h^2 P)
Each is split with the L2-orthogonal projector onto the tangent space T(t)
spanned by the parameter derivatives of the exact family at time t,
  v = Pi v + (I - Pi) v,
and the coefficients of Pi v on each mode, ||Pi v|| and ||(I - Pi) v|| are
written out.  The individual coefficients depend on the choice of complement,
but whether ||(I - Pi) v|| stays bounded does not: for any other projector Pi'
onto the same T, (I - Pi') v = (I - Pi')(I - Pi) v, and ||I - Pi'|| is
constant in t because the modes are translates or rotations of fixed shapes.

Tangent modes (derivatives at fixed co-moving coordinate, so that none of them
carries an explicit factor t):
  kdv   u = 2k^2 sech^2(k xi), xi = x - 4k^2 t - x0      translation, amplitude
  nls   u = eta sech(eta xi) e^{i Theta}                 phase, translation,
        xi = x - x0 - c t, Theta = c x/2 + (eta^2-c^2/4) t  amplitude, boost
  sg    breather Y(theta, omega), theta = omega t        phase, frequency,
                                                          translation (control)
At the final time the split is repeated with a basis built from the NUMERICAL
state only, i.e. the symmetry generators applied to U (translation d_x U,
phase iU, boost i xi_U U, scaling, time translation f(U)); if the transversal
norm agrees, the tangent part can be extracted from quantities the method
already has, without the exact solution.

Stage tolerance 1e-15, as fixed for corrector work in CR-1 item 6.  The P sweep
is the one of integrators/asc.py, verified bit for bit by --verify.
"""
import argparse
import csv
import pathlib
import sys
import time

REPO = pathlib.Path('/home/claude/repo')
sys.path.insert(0, str(REPO))

import numpy as np                                               # noqa: E402
import kernel.formulas as K                                      # noqa: E402
from pdes import _spectral as S                                  # noqa: E402
from pdes import kdv, nls, sg                                    # noqa: E402
from integrators import midpoint as MP                           # noqa: E402
from integrators import asc                                      # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / 'results'
TOL = 1e-15


def wrap(z, L):
    return (z + L / 2) % L - L / 2


def ddx(pde, V):
    """Spectral x-derivative of a physical stacked array (not charged)."""
    with S.no_count():
        c = pde.grid.to_spec(V)
        c = pde.grid.project(1j * pde.grid.k[None, :] * c)
        return pde.grid.to_phys(c)


def centroid(pde, U):
    """Circular mean of |U|^2 on the periodic domain."""
    w = np.sum(U ** 2, axis=0)
    L, a = pde.grid.L, pde.grid.a
    z = np.sum(w * np.exp(2j * np.pi * (pde.grid.x - a) / L))
    return a + L * (np.angle(z) / (2 * np.pi) % 1.0)


# ------------------------------------------------------------------ benchmarks
def spec(name, omega=None):
    if name == 'kdv':
        pde = kdv.make('soliton')
        kap, x0 = pde.p['kappa'], pde.p['x0']
        c = 4.0 * kap ** 2

        def state(t):
            return pde.exact_soliton(t, phys=True)

        def modes(t):
            xi = pde._wrap(pde.grid.x - c * t - x0)
            s, th = 1.0 / np.cosh(kap * xi), np.tanh(kap * xi)
            return {'translation': (-4 * kap ** 3 * s ** 2 * th)[None, :],
                    'amplitude': (4 * kap * s ** 2
                                  - 4 * kap ** 2 * xi * s ** 2 * th)[None, :]}

        def generators(U):
            Ux = ddx(pde, U)
            xi = wrap(pde.grid.x - centroid(pde, U), pde.grid.L)
            return [Ux, 2 * U + xi[None, :] * Ux]

        return pde, pde.exact_soliton(0.0), state, modes, generators, 32.0

    if name == 'nls':
        pde = nls.make('bright')
        eta, c, x0 = pde.eta, pde.c, pde.p['x0']
        x, L = pde.grid.x, pde.grid.L

        def cplx(t):
            xi = wrap(x - x0 - c * t, L)
            th = 0.5 * c * x + (eta ** 2 - 0.25 * c ** 2) * t
            return xi, eta / np.cosh(eta * xi), np.exp(1j * th)

        def pack(w):
            return np.stack([w.real, w.imag])

        def state(t):
            xi, env, car = cplx(t)
            return pack(env * car)

        def modes(t):
            xi, env, car = cplx(t)
            s, th = 1.0 / np.cosh(eta * xi), np.tanh(eta * xi)
            u = env * car
            return {'phase': pack(1j * u),
                    'translation': pack(eta ** 2 * s * th * car),
                    'amplitude': pack(s * (1 - eta * xi * th) * car),
                    'boost': pack(0.5j * xi * u)}

        def generators(U):
            Ux = ddx(pde, U)
            iU = np.stack([-U[1], U[0]])
            xi = wrap(x - centroid(pde, U), L)[None, :]
            return [Ux, iU, xi * iU, U + xi * Ux]

        return pde, pde.grid.to_spec(state(0.0)), state, modes, generators, 80.0

    if name == 'sg_breather':
        # omega other than the contract value 0.1 is a DIFFERENT benchmark,
        # chosen for temporal resolvability; the breather formula was
        # verified by sympy for symbolic omega before this use.
        pde = sg.make('breather') if omega is None else sg.make('breather',
                                                                  omega=omega)
        om0 = pde.p['omega']
        x = pde.grid.x

        def Y(theta, om):
            r = np.sqrt(1.0 - om * om)
            ch = np.cosh(r * x)
            s = (r / om) * np.sin(theta) / ch
            Ut = 4.0 * (r / om) * np.cos(theta) / ch / (1.0 + s * s)
            return np.stack([4.0 * np.arctan(s), om * Ut])

        def rich(g, d):
            return (4 * (g(d) - g(-d)) / (2 * d) - (g(2 * d) - g(-2 * d)) / (4 * d)) / 3

        def state(t):
            return Y(om0 * t, om0)

        def modes(t):
            th = om0 * t
            return {'phase': rich(lambda d: Y(th + d, om0), 1e-3),
                    'frequency': rich(lambda d: Y(th, om0 + d), 1e-5),
                    'translation': ddx(pde, Y(th, om0))}

        def generators(U):
            with S.no_count():
                fU = pde.grid.to_phys(pde.f(pde.grid.to_spec(U)))
            return [fU, ddx(pde, U)]

        period = 2 * np.pi / om0
        return (pde, pde.exact_breather(0.0), state, modes, generators,
                8 * period)
    raise ValueError(name)


# ------------------------------------------------------------------- algebra
def decompose(v, basis, dx):
    B = np.array([b.ravel() for b in basis])
    coef = np.linalg.solve(dx * B @ B.T, dx * B @ v.ravel())
    tan = coef @ B
    nrm = lambda a: float(np.sqrt(dx * np.sum(a * a)))           # noqa: E731
    return coef, nrm(v.ravel()), nrm(tan), nrm(v.ravel() - tan)


def p_step(pde, ym, h, P):
    """The P update of integrators/asc.py, verbatim."""
    rhs = P + 0.5 * h * pde.applyA(ym, P) - h * K.G2(ym, pde.f, pde.applyA,
                                                     pde.applyB)
    P1, _ = MP.solve_stage_jacobian(pde, ym, h, rhs)
    return P1


def verify(name, nsteps=20):
    pde, y0, *_ = spec(name)
    h = 1e-3 if name == 'kdv' else 1e-2
    ref = asc.integrate(pde, y0, h, nsteps, tol=TOL)
    y, P, log = y0.copy(), np.zeros_like(y0), MP.Stats()
    for _ in range(nsteps):
        y = MP.step(y, h, pde, tol=TOL, log=log)
        P = p_step(pde, log.stage, h, P)
    d = float(np.max(np.abs(P - ref['P']))) / float(np.max(np.abs(ref['P'])))
    print(f"verify {name}: max |P - P_asc| / max |P_asc| = {d:.2e}")
    return d


def run(name, h, nout, omega=None):
    pde, y0, state, modes, generators, T = spec(name, omega)
    nsteps = int(round(T / h))
    every = max(1, nsteps // nout)
    dx = pde.grid.dx
    y, P, log = y0.copy(), np.zeros_like(y0), MP.Stats()
    rows, t0 = [], time.perf_counter()
    for n in range(1, nsteps + 1):
        y = MP.step(y, h, pde, tol=TOL, log=log)
        P = p_step(pde, log.stage, h, P)
        if n % every and n != nsteps:
            continue
        t = n * h
        with S.no_count():
            U = pde.grid.to_phys(y)
            Pp = pde.grid.to_phys(P)
        ex = state(t)
        e = (ex - U) / h ** 2
        md = modes(t)
        ce, ne, te, re = decompose(e, list(md.values()), dx)
        cp, npn, tp, rp = decompose(Pp, list(md.values()), dx)
        row = dict(benchmark=name, h=f"{h:.6e}", N=pde.grid.N, t=f"{t:.6f}",
                   err_mp2_sup=f"{float(np.max(np.abs(ex - U))):.6e}",
                   e_norm=f"{ne:.6e}", e_tangent=f"{te:.6e}",
                   e_transversal=f"{re:.6e}",
                   P_norm=f"{npn:.6e}", P_tangent=f"{tp:.6e}",
                   P_transversal=f"{rp:.6e}",
                   e_minus_P_rel=f"{np.sqrt(dx * np.sum((e - Pp) ** 2)) / ne:.3e}")
        for k, name_k in enumerate(md):
            row[f'e_coef_{name_k}'] = f"{ce[k]:.6e}"
            row[f'P_coef_{name_k}'] = f"{cp[k]:.6e}"
        rows.append(row)
    # method-internal basis at the final time
    gen = generators(U)
    if name == 'sg_breather':
        gen = gen + [modes(t)['frequency']]
    _, _, _, re_g = decompose(e, gen, dx)
    _, _, _, rp_g = decompose(Pp, gen, dx)
    rows[-1]['e_transversal_Ubasis'] = f"{re_g:.6e}"
    rows[-1]['P_transversal_Ubasis'] = f"{rp_g:.6e}"
    OUT.mkdir(exist_ok=True)
    fields = list(rows[-1].keys())
    tag = '' if omega is None else f"_w{omega:g}"
    path = OUT / f"proj_{name}{tag}_h{h:.0e}.csv"
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval='')
        w.writeheader()
        w.writerows(rows)
    print(f"{name} h={h:g}: {nsteps} steps, {time.perf_counter() - t0:.0f} s, "
          f"final |e|={rows[-1]['e_norm']} tan={rows[-1]['e_tangent']} "
          f"trans={rows[-1]['e_transversal']} (U-basis {re_g:.3e})  -> {path.name}",
          flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('bench', choices=['kdv', 'nls', 'sg_breather'])
    ap.add_argument('--h', type=float)
    ap.add_argument('--nout', type=int, default=48)
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--omega', type=float)
    a = ap.parse_args()
    if a.verify:
        sys.exit(0 if verify(a.bench) < 1e-12 else 1)
    run(a.bench, a.h, a.nout, a.omega)
