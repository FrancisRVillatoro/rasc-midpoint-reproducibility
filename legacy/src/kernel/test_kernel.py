"""Gate G1 runner: validation of the frozen kernel (see CLAUDE.md).

Tests, on the prescribed generic nonlinear 2D ODE:
  T1  dG2, ddG2 and H2 operator formulas vs direct symbolic differentiation.
  T2  one-step defect of midpoint vs the exact flow of f + h^2 G2
      (ratios -> 32) and of f + h^2 G2 + h^4 G4 (ratios -> 128).
  T3  sequential fully discrete ASC scheme: orders 2 / 4 / 6, and order 4
      when H2 is ablated.

Run:  python kernel/test_kernel.py     (exit code 0 iff all gates pass)
"""
import sys
import pathlib
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from kernel import formulas as K

RNG = np.random.default_rng(20260821)

# ------------------------------------------------- prescribed 2D test field
y1, y2, p1, p2 = sp.symbols('y1 y2 p1 p2')
Y, P = [y1, y2], [p1, p2]
F = [sp.sin(y1) + sp.Rational(1, 2) * y2 + sp.Rational(1, 5) * y2**2,
     -y1 + sp.Rational(3, 10) * sp.sin(y2) + sp.Rational(1, 7) * y1 * y2]
n = 2

A_s = [[sp.diff(F[i], Y[j]) for j in range(n)] for i in range(n)]
B_s = [[[sp.diff(F[i], Y[j], Y[k]) for k in range(n)] for j in range(n)]
       for i in range(n)]
C_s = [[[[sp.diff(F[i], Y[j], Y[k], Y[l]) for l in range(n)]
         for k in range(n)] for j in range(n)] for i in range(n)]
D_s = [[[[[sp.diff(F[i], Y[j], Y[k], Y[l], Y[m]) for m in range(n)]
          for l in range(n)] for k in range(n)] for j in range(n)]
       for i in range(n)]

f_n = sp.lambdify(Y, F, 'numpy')
A_n = sp.lambdify(Y, A_s, 'numpy')
B_n = sp.lambdify(Y, B_s, 'numpy')
C_n = sp.lambdify(Y, C_s, 'numpy')
D_n = sp.lambdify(Y, D_s, 'numpy')

fv = lambda y: np.array(f_n(*y), float)
Av = lambda y, v: np.array(A_n(*y), float) @ v
Bv = lambda y, u, v: np.einsum('ijk,j,k->i', np.array(B_n(*y), float), u, v)
Cv = lambda y, u, v, w: np.einsum('ijkl,j,k,l->i',
                                  np.array(C_n(*y), float), u, v, w)
Dv = lambda y, u, v, w, x: np.einsum('ijklm,j,k,l,m->i',
                                     np.array(D_n(*y), float), u, v, w, x)

OPS = dict(f=fv, A=Av, B=Bv, C=Cv, D=Dv)


def rel(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)


# ------------------------------------------------------------------ T1
def test_T1_directional_derivatives():
    """Operator dG2 / ddG2 / H2 vs symbolic ground truth."""
    aA = lambda v: [sum(A_s[i][j] * v[j] for j in range(n)) for i in range(n)]
    aB = lambda x, z: [sum(B_s[i][j][k] * x[j] * z[k]
                           for j in range(n) for k in range(n))
                       for i in range(n)]
    G2sym = [sp.Rational(1, 12) * aA(aA(F))[i]
             - sp.Rational(1, 24) * aB(F, F)[i] for i in range(n)]
    w1, w2, u1s, u2s = sp.symbols('w1 w2 u1s u2s')
    W, U = [w1, w2], [u1s, u2s]
    dG2sym = [sum(sp.diff(G2sym[i], Y[j]) * W[j] for j in range(n))
              for i in range(n)]
    ddG2sym = [sum(sp.diff(G2sym[i], Y[j], Y[k]) * U[j] * W[k]
                   for j in range(n) for k in range(n)) for i in range(n)]
    dG2_n = sp.lambdify(Y + W, dG2sym, 'numpy')
    ddG2_n = sp.lambdify(Y + U + W, ddG2sym, 'numpy')

    # symbolic H2: P-component of G2 of the augmented triangular field
    Faug = list(F) + [sum(A_s[i][j] * P[j] for j in range(n)) - G2sym[i]
                      for i in range(n)]
    Z = Y + P
    Aaug = [[sp.diff(Faug[i], Z[j]) for j in range(4)] for i in range(4)]
    aAg = lambda v: [sum(Aaug[i][j] * v[j] for j in range(4)) for i in range(4)]
    Baug = [[[sp.diff(Faug[i], Z[j], Z[k]) for k in range(4)]
             for j in range(4)] for i in range(4)]
    aBg = lambda x, z: [sum(Baug[i][j][k] * x[j] * z[k]
                            for j in range(4) for k in range(4))
                        for i in range(4)]
    G2aug = [sp.Rational(1, 12) * aAg(aAg(Faug))[i]
             - sp.Rational(1, 24) * aBg(Faug, Faug)[i] for i in range(4)]
    H2sym_n = sp.lambdify(Y + P, [G2aug[2], G2aug[3]], 'numpy')

    worst = 0.0
    for _ in range(5):
        y = RNG.uniform(-0.8, 0.8, 2)
        w = RNG.uniform(-1, 1, 2)
        u = RNG.uniform(-1, 1, 2)
        p = RNG.uniform(-1, 1, 2)
        e1 = rel(K.dG2(y, w, **{k: OPS[k] for k in 'fABC'}),
                 np.array(dG2_n(*y, *w), float))
        e2 = rel(K.ddG2(y, u, w, **OPS),
                 np.array(ddG2_n(*y, *u, *w), float))
        e3 = rel(K.H2(y, p, **OPS), np.array(H2sym_n(*y, *p), float))
        worst = max(worst, e1, e2, e3)
    print(f"  T1 dG2/ddG2/H2 vs symbolic: worst rel err = {worst:.2e}")
    assert worst < 1e-10
    return worst


# ------------------------------------------------------------------ T2
def midpoint_step(y, h):
    z = y + h * fv(y)
    for _ in range(60):
        r = z - y - h * fv(0.5 * (y + z))
        J = np.eye(2) - 0.5 * h * np.array(A_n(*(0.5 * (y + z))), float)
        dz = np.linalg.solve(J, r)
        z = z - dz
        if np.linalg.norm(dz) < 1e-15:
            break
    return z


def flow(rhs, y0, T):
    s = solve_ivp(lambda t, y: rhs(y), [0.0, T], y0,
                  method='DOP853', rtol=1e-13, atol=1e-15)
    return s.y[:, -1]


def test_T2_defect_ratios():
    y0 = np.array([0.4, -0.25])
    g2r = lambda y: fv(y)
    d2, d4 = [], []
    hs = [0.4, 0.2, 0.1, 0.05]
    for h in hs:
        ym = midpoint_step(y0, h)
        f2 = flow(lambda y: fv(y) + h**2 * K.G2(y, fv, Av, Bv), y0, h)
        f4 = flow(lambda y: fv(y) + h**2 * K.G2(y, fv, Av, Bv)
                  + h**4 * K.G4(y, **OPS), y0, h)
        d2.append(np.linalg.norm(ym - f2))
        d4.append(np.linalg.norm(ym - f4))
    r2 = [d2[i - 1] / d2[i] for i in range(1, len(hs))]
    r4 = [d4[i - 1] / d4[i] for i in range(1, len(hs))]
    print(f"  T2 defect ratios vs f+h^2G2      : "
          + "  ".join(f"{r:6.2f}" for r in r2) + "   (gate [30,35])")
    print(f"  T2 defect ratios vs f+h^2G2+h^4G4: "
          + "  ".join(f"{r:6.2f}" for r in r4) + "   (gate [115,140])")
    assert all(30.0 <= r <= 35.0 for r in r2)
    assert all(115.0 <= r <= 140.0 for r in r4)
    return r2, r4


# ------------------------------------------------------------------ T3
def asc_run(h, T, use_H2=True):
    N = int(round(T / h))
    U = np.array([0.4, -0.25])
    Pn = np.zeros(2)
    Rn = np.zeros(2)
    I = np.eye(2)
    for _ in range(N):
        Un = U.copy()
        U = midpoint_step(Un, h)
        Um = 0.5 * (Un + U)
        Am = np.array(A_n(*Um), float)
        Kp = I - 0.5 * h * Am
        Pn1 = np.linalg.solve(Kp, (I + 0.5 * h * Am) @ Pn
                              - h * K.G2(Um, fv, Av, Bv))
        Pm = 0.5 * (Pn + Pn1)
        src = 0.5 * Bv(Um, Pm, Pm) - K.G4(Um, **OPS)
        if use_H2:
            src = src - K.H2(Um, Pm, **OPS)
        Rn1 = np.linalg.solve(Kp, (I + 0.5 * h * Am) @ Rn + h * src)
        Pn, Rn = Pn1, Rn1
    return U, U + h**2 * Pn, U + h**2 * Pn + h**4 * Rn


def test_T3_sequential_orders():
    T = 1.0
    ref = flow(fv, np.array([0.4, -0.25]), T)
    hs = [0.2, 0.1, 0.05, 0.025]
    E = []
    En = []
    for h in hs:
        u2, u4, u6 = asc_run(h, T, True)
        _, _, u6n = asc_run(h, T, False)
        E.append([np.linalg.norm(x - ref) for x in (u2, u4, u6)])
        En.append(np.linalg.norm(u6n - ref))
    E = np.array(E)
    En = np.array(En)
    ordp = lambda e, i: np.log2(e[i - 1] / e[i])
    p2 = ordp(E[:, 0], 3)
    p4 = ordp(E[:, 1], 3)
    p6a, p6b = ordp(E[:, 2], 1), ordp(E[:, 2], 2)
    pab = [ordp(En, 2), ordp(En, 3)]
    print(f"  T3 orders: p2={p2:.3f} [1.97,2.03]  p4={p4:.3f} [3.90,4.10]  "
          f"p6={p6a:.3f},{p6b:.3f} [5.85,6.15]")
    print(f"  T3 ablation (no H2): p={pab[0]:.3f},{pab[1]:.3f} [3.90,4.10]")
    assert 1.97 <= p2 <= 2.03 and 3.90 <= p4 <= 4.10
    assert 5.85 <= p6a <= 6.15 and 5.85 <= p6b <= 6.15
    assert all(3.90 <= p <= 4.10 for p in pab)
    return p2, p4, (p6a, p6b), pab


if __name__ == '__main__':
    print("Gate G1 (kernel validation)")
    ok = True
    for t in (test_T1_directional_derivatives,
              test_T2_defect_ratios,
              test_T3_sequential_orders):
        try:
            t()
        except AssertionError:
            ok = False
            print(f"  FAILED: {t.__name__}")
    print("GATE G1:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
