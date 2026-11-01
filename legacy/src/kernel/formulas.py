"""FROZEN mathematical kernel of the ASC-midpoint-PDE project.

DO NOT EDIT. Any change requires a CHANGES_REQUESTED.md entry approved by
the PI with the literal string "APROBADO FRV" (see CLAUDE.md). The SHA-256
of this file must be printed in every phase report.

All formulas are written in operator form. The caller supplies callables
    f(y)            -> vector field
    A(y, v)         -> f'(y) v                    (Jacobian-vector)
    B(y, u, v)      -> f''(y)[u, v]
    C(y, u, v, w)   -> f'''(y)[u, v, w]
    D(y, u, v, w, x)-> f''''(y)[u, v, w, x]
acting on whatever state representation the module uses (ODE vectors or
spectral PDE states). Nonautonomous fields are handled by autonomization
(append tau' = 1), which is EXACT for the implicit midpoint rule.

Provenance. G2 and G4 are the h^2 and h^4 coefficients of the modified
vector field of the implicit midpoint rule; G4 was verified against the
midpoint one-step map by a B-series-free numerical test (defect ratios
2^5 and 2^7) and coincides, after autonomization, with Eq. (7) of
Villatoro & Ramos, Appl. Math. Comput. 103 (1999) 241-285, up to a single
global sign erratum documented in docs/T2_errata_partV.md. H2 is the
midpoint modified-equation defect of the discretized first corrector,
i.e. the P-component of G2 applied to the triangular augmented field
F(U,P) = (f(U), A(U)P - G2(U)); it is the compact vectorial form of the
mechanism of Eqs. (126)-(127) of the same 1999 paper.
"""


# ----------------------------------------------------------------- G2, G4
def G2(y, f, A, B):
    """G2 = (1/12) A^2 f - (1/24) B(f, f)."""
    fy = f(y)
    return (1.0 / 12.0) * A(y, A(y, fy)) - (1.0 / 24.0) * B(y, fy, fy)


def G4(y, f, A, B, C, D):
    """Nine-term h^4 coefficient of the midpoint modified field.

    G4 = (1/80)   A^4 f          + (1/480) A^2 B(f,f)
       - (1/240)  A B(Af, f)     - (1/80)  B(A^2 f, f)
       - (1/480)  B(B(f,f), f)   - (7/1440) A C(f,f,f)
       + (1/160)  B(Af, Af)      + (1/480) C(Af, f, f)
       + (7/5760) D(f,f,f,f)
    """
    fy = f(y)
    Af = A(y, fy)
    A2f = A(y, Af)
    A3f = A(y, A2f)
    A4f = A(y, A3f)
    Bff = B(y, fy, fy)
    return ((1.0 / 80.0) * A4f
            + (1.0 / 480.0) * A(y, A(y, Bff))
            - (1.0 / 240.0) * A(y, B(y, Af, fy))
            - (1.0 / 80.0) * B(y, A2f, fy)
            - (1.0 / 480.0) * B(y, Bff, fy)
            - (7.0 / 1440.0) * A(y, C(y, fy, fy, fy))
            + (1.0 / 160.0) * B(y, Af, Af)
            + (1.0 / 480.0) * C(y, Af, fy, fy)
            + (7.0 / 5760.0) * D(y, fy, fy, fy, fy))


# --------------------------------------------- directional derivatives of G2
def dG2(y, w, f, A, B, C):
    """G2'(y)[w], derived by differentiating G2 along w:

    d[A^2 f][w] = B(w, Af) + A B(w, f) + A^3 w
    d[B(f,f)][w] = C(w, f, f) + 2 B(Aw, f)

    G2'[w] = (1/12)[B(w,Af) + A B(w,f) + A^3 w]
           - (1/24)[C(w,f,f) + 2 B(Aw,f)]
    """
    fy = f(y)
    Af = A(y, fy)
    return ((1.0 / 12.0) * (B(y, w, Af) + A(y, B(y, w, fy))
                            + A(y, A(y, A(y, w))))
            - (1.0 / 24.0) * (C(y, w, fy, fy) + 2.0 * B(y, A(y, w), fy)))


def ddG2(y, u, v, f, A, B, C, D):
    """G2''(y)[u, v], symmetric second directional derivative:

    (1/12)[ C(u,v,Af) + B(u,B(v,f)) + B(v,B(u,f)) + B(u,A^2 v) + B(v,A^2 u)
            + A C(u,v,f) + A B(u,Av) + A B(v,Au) + A^2 B(u,v) ]
    - (1/24)[ D(u,v,f,f) + 2 C(u,Av,f) + 2 C(v,Au,f)
              + 2 B(B(u,v),f) + 2 B(Au,Av) ]

    Validated in kernel/test_kernel.py against direct symbolic
    differentiation of G2 on a generic nonlinear 2D field.
    """
    fy = f(y)
    Af = A(y, fy)
    Au = A(y, u)
    Av = A(y, v)
    part12 = (C(y, u, v, Af)
              + B(y, u, B(y, v, fy)) + B(y, v, B(y, u, fy))
              + B(y, u, A(y, Av)) + B(y, v, A(y, Au))
              + A(y, C(y, u, v, fy))
              + A(y, B(y, u, Av)) + A(y, B(y, v, Au))
              + A(y, A(y, B(y, u, v))))
    part24 = (D(y, u, v, fy, fy)
              + 2.0 * C(y, u, Av, fy) + 2.0 * C(y, v, Au, fy)
              + 2.0 * B(y, B(y, u, v), fy) + 2.0 * B(y, Au, Av))
    return (1.0 / 12.0) * part12 - (1.0 / 24.0) * part24


# ------------------------------------------------------- corrector defect H2
def L_field(y, p, f, A, B):
    """First-corrector field L(U, P) = A(U) P - G2(U)."""
    return A(y, p) - G2(y, f, A, B)


def H2(y, p, f, A, B, C, D):
    """Midpoint modified-equation defect of the discretized first corrector.

    With L = A P - G2 and M = B(f, P) + A L - G2'[f],

    H2 = (1/12){ B(Af, P) + A M - G2'[Af] }
       - (1/24){ C(f, f, P) + 2 B(f, L) - G2''[f, f] }

    equal to the P-component of G2 of the triangular augmented field
    (f(U), L(U,P)).
    """
    fy = f(y)
    Af = A(y, fy)
    Lv = L_field(y, p, f, A, B)
    Mv = B(y, fy, p) + A(y, Lv) - dG2(y, fy, f, A, B, C)
    return ((1.0 / 12.0) * (B(y, Af, p) + A(y, Mv)
                            - dG2(y, Af, f, A, B, C))
            - (1.0 / 24.0) * (C(y, fy, fy, p) + 2.0 * B(y, fy, Lv)
                              - ddG2(y, fy, fy, f, A, B, C, D)))


# ------------------------------------------------------------- R-field source
def R_source(y, p, f, A, B, C, D):
    """Right-hand-side inhomogeneity of the second corrector:
    (1/2) B(P, P) - G4(U) - H2(U, P).   (The A(U)R term is added by the
    linear midpoint sweep in the integrator.)"""
    return (0.5 * B(y, p, p) - G4(y, f, A, B, C, D)
            - H2(y, p, f, A, B, C, D))
