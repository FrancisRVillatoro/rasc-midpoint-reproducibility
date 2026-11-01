#!/usr/bin/env python3
"""Exact KdV line-branch residual and invariant normalization (R1).

The branch fixes the quadratic KdV invariant at first variation, but
not the independent linear mass. This distinction is intentional.
"""
import sympy as s

y, S, T, kap = s.symbols('y S T kappa', real=True, positive=True)
k=s.symbols('kappa', positive=True)
# For polynomial identities S=sech(y), T=tanh(y), T^2 = 1-S^2.
def red(q):
    pp=s.Poly(s.expand(q),T)
    return s.expand(pp.rem(s.Poly(T*T+S*S-1,T)).as_expr())
def dy(q):
    return red(s.diff(q,y)-S*T*s.diff(q,S)+(1-T*T)*s.diff(q,T))
def dyy(q): return dy(dy(q))

A=s.Rational(-1072,45); B=s.Integer(24); D=s.Rational(832,45)
RHO=s.Rational(-704,45)
phi=2*k*k*S*S
chi=k**8*(A*S*S+B*S**4+D*y*S*S*T)
c=4*k*k
Hchi=k*k*red(-dyy(chi)+(4-12*S*S)*chi)
phi_x=k*dy(phi)
phi_xx=k*k*dyy(phi)
rhs=c**3*phi_xx/12-s.Rational(3,4)*c*c*phi_x**2-RHO*k**8*phi
assert s.simplify(red(Hchi-rhs))==0, 'KdV modified stationary residual not zero'

# Exact integrals: int S^2=2, int S^4=4/3, int S^6=16/15,
# int y S^2 T=1, int y S^4 T=1/3.
linear_mass=k**7*(2*A+s.Rational(4,3)*B+D)
quadratic_variation=2*k**9*(s.Rational(4,3)*A+s.Rational(16,15)*B+s.Rational(1,3)*D)
assert s.simplify(linear_mass-s.Rational(128,45)*k**7)==0
assert s.simplify(quadratic_variation)==0
print('KdV stationary residual:',s.simplify(red(Hchi-rhs)))
print('first variation of quadratic invariant:',quadratic_variation)
print('first variation of linear mass:',linear_mass)
print('rate from quadratic-invariant normalized localized branch:',RHO*k**8)
print('PASS KdV normalization')
