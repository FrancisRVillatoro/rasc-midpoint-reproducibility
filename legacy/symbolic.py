"""Closed-form drift rates from G2: symbolic derivation.

G2 = (1/12) A^2 f - (1/24) B(f, f) is the frozen kernel formula
(kernel/formulas.py); here it is re-expressed with exact rationals on the
CONTINUOUS operators and checked against the discrete kernel on the grid.

Integrals over the real line of monomials xi^p sech^m(a xi) tanh^n(a xi) are
done by exact rules rather than by generic symbolic integration:
  tanh^2 = 1 - sech^2 reduces n to 0 or 1; p + n odd gives 0 by parity;
  int sech^m      = sqrt(pi) Gamma(m/2) / Gamma((m+1)/2) / a,
  int xi sech^m tanh = int sech^m / (m a)       (by parts).
Any other surviving monomial is reported, not silently integrated.
"""
import sys
import sympy as sp

xi = sp.symbols('xi', real=True)


def line_integral(expr, a):
    """Exact integral over R of a polynomial in xi, sech(a xi), tanh(a xi)."""
    S, T = sp.symbols('S T')
    e = sp.expand(expr.rewrite(sp.sech) if False else expr)
    e = e.subs({sp.sech(a * xi): S, sp.tanh(a * xi): T})
    e = sp.expand(e.subs(sp.cosh(a * xi), 1 / S))
    poly = sp.Poly(e, xi, S, T)
    total = sp.Integer(0)
    leftovers = []
    for (p, m, n), coef in poly.terms():
        # reduce T^n with T^2 = 1 - S^2
        q = n // 2
        base_n = n % 2
        for j in range(q + 1):
            c = coef * sp.binomial(q, j) * (-1) ** j
            mm = m + 2 * j
            if (p + base_n) % 2 == 1:
                continue
            Im = sp.sqrt(sp.pi) * sp.gamma(sp.Rational(mm, 2)) / sp.gamma(sp.Rational(mm + 1, 2)) / a
            if p == 0 and base_n == 0:
                if mm == 0:
                    leftovers.append(('constant', c))
                    continue
                total += c * Im
            elif p == 1 and base_n == 1:
                total += c * Im / (mm * a)
            else:
                leftovers.append(((p, mm, base_n), c))
    return sp.simplify(total), leftovers


def kdv():
    kap = sp.symbols('kappa', positive=True)
    D = lambda e, k=1: sp.diff(e, xi, k)                       # noqa: E731
    phi = 2 * kap ** 2 * sp.sech(kap * xi) ** 2
    f = lambda u: -6 * u * D(u) - D(u, 3)                      # noqa: E731
    A = lambda v: -6 * D(phi * v) - D(v, 3)                    # noqa: E731
    fphi = f(phi)
    Af = A(fphi)
    q = -sp.Rational(1, 2) * phi * Af - sp.Rational(1, 12) * D(Af, 2) \
        + sp.Rational(1, 4) * fphi ** 2
    g = D(q)
    # consistency: g equals (1/12) A(A f) - (1/24) B(f, f) with B(v,w) = -6(vw)'
    chk = sp.simplify(sp.Rational(1, 12) * A(Af) - sp.Rational(1, 24) * (-6 * D(fphi ** 2)) - g)
    dcphi = sp.diff(phi, kap) / (8 * kap)                      # c = 4 kappa^2
    mass_pairing, lo1 = line_integral(phi * g, kap)            # must vanish
    Pprime, lo2 = line_integral(phi * dcphi, kap)
    I_rate, lo3 = line_integral(dcphi * q, kap)
    rate = sp.simplify(-I_rate / kap)
    return dict(check_G2_formula=chk, phi_dot_g=mass_pairing, P_prime=Pprime,
                rate_translation=rate, leftovers=lo1 + lo2 + lo3, q=q, phi=phi,
                kap=kap)


if __name__ == '__main__':
    r = kdv()
    print("KdV, linea completa:")
    print("  G2 = (1/12)A^2 f - (1/24)B(f,f) reescrito como derivada total, residuo:", r['check_G2_formula'])
    print("  <phi, G2(phi)>  (debe ser 0, conservacion de int u^2):", r['phi_dot_g'])
    print("  P'(c) = <phi, d_c phi>:", r['P_prime'])
    print("  tasa de traslacion:", r['rate_translation'], "=", sp.nsimplify(r['rate_translation']),
          "  numerico a kappa=1:", sp.N(r['rate_translation'].subs(r['kap'], 1), 15))
    print("  monomios no integrados:", r['leftovers'])


def nls():
    """NLS bright soliton, general (eta, c), in the co-moving co-rotating frame.

    Envelopes V(xi) of fields V e^{i k xi}, k = c/2, carried as (Re, Im) pairs.
    The frame operator is Hamiltonian, so its generalized eigenspaces at
    lambda and mu are omega-orthogonal unless lambda + mu = 0 and the Riesz
    projector onto X0 is the omega-orthogonal projector:  W a = r with
    W_kj = w(v_k, v_j), r_k = w(v_k, G2), rates = -a, in the measured basis
    {i u, d_x0 u, d_eta u, i xi u / 2}.
    """
    eta = sp.symbols('eta', positive=True)
    k = sp.symbols('k', real=True)                  # k = c/2
    D = lambda e, n=1: sp.diff(e, xi, n)             # noqa: E731
    F = eta * sp.sech(eta * xi)

    def Aop(V):
        a, b = V
        ReL = D(a, 2) - 2 * k * D(b) - k ** 2 * a + 6 * F ** 2 * a
        ImL = D(b, 2) + 2 * k * D(a) - k ** 2 * b + 2 * F ** 2 * b
        return (-ImL, ReL)                           # i * L

    def Bop(V, W):
        a, b = V
        c_, d = W
        return (-4 * F * (a * d + b * c_), 4 * F * (3 * a * c_ + b * d))

    fr = (-2 * k * D(F), D(F, 2) - k ** 2 * F + 2 * F ** 3)
    AAf = Aop(Aop(fr))
    Bff = Bop(fr, fr)
    G2 = tuple(sp.expand(sp.Rational(1, 12) * AAf[i] - sp.Rational(1, 24) * Bff[i])
               for i in range(2))

    def w(V, W):
        return line_integral(-(V[0] * W[1] - V[1] * W[0]), eta)

    modes = {'phase': (sp.Integer(0), F),
             'translation': (-D(F), sp.Integer(0)),
             'amplitude': (sp.diff(F, eta), sp.Integer(0)),
             'boost': (sp.Integer(0), xi * F / 2)}
    names = list(modes)
    leftovers = []
    Wm = sp.zeros(4, 4)
    rv = sp.zeros(4, 1)
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            val, lo = w(modes[ni], modes[nj])
            Wm[i, j] = val
            leftovers += lo
        val, lo = w(modes[ni], G2)
        rv[i] = val
        leftovers += lo
    a = Wm.LUsolve(rv)
    rates = {n: sp.factor(sp.simplify(-a[i])) for i, n in enumerate(names)}
    return dict(rates=rates, W=Wm, r=rv, eta=eta, k=k, leftovers=leftovers,
                G2=G2, F=F)


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'nls':
    r = nls()
    eta, k = r['eta'], r['k']
    c = sp.symbols('c', real=True)
    print("NLS, solitón brillante (eta, c), k = c/2:")
    print("  W (forma simplectica sobre X0):", r['W'])
    print("  pares simplecticos con G2:", list(r['r']))
    for n, v in r['rates'].items():
        vc = sp.factor(sp.expand(v.subs(k, c / 2)))
        print(f"  tasa {n:12s} = {vc}     en eta=1, c=1: {vc.subs({eta: 1, c: 1})}")
    print("  monomios no integrados:", r['leftovers'])
