#!/usr/bin/env python3
"""Independent exact stationary-midpoint family for focusing NLS, c=0.

No modified-field, ASC, or PDE time-integration code is used.
"""
import sympy as sp
import mpmath as mp

x=sp.symbols('x')
a3,a5,a7=sp.symbols('a3 a5 a7')
zz=x+a3*x**3+a5*x**5+a7*x**7
expr=sp.series(zz*(1+zz*zz)**2-x,x,0,9).removeO()
s=sp.solve(sp.Poly(expr,x).all_coeffs(),[a3,a5,a7],dict=True)[0]
T=sp.series(2*sp.atan(zz.subs(s)),x,0,8).removeO()
h,eta=sp.symbols('h eta',positive=True)
frequency=sp.series(T.subs(x,h*eta**2/2)/h,h,0,6).removeO().expand()
assert sp.simplify(frequency-(eta**2-sp.Rational(7,12)*h**2*eta**6+sp.Rational(33,40)*h**4*eta**10))==0

mp.mp.dps=70
for hs in ['0.1','0.05','0.025','0.0125']:
    hh=mp.mpf(hs); et=mp.mpf(1)
    root=mp.findroot(lambda z:z*(1+z*z)**2-hh*et*et/2, (hh/3,hh/2))
    theta=2*mp.atan(root); ct=mp.cos(theta/2)
    b=et*ct*ct; aa=et*ct
    mass=2*aa*aa/b
    def profile(xx):return aa/mp.cosh(b*xx)
    max_res=mp.mpf(0)
    for xx in [mp.mpf(0),mp.mpf('.5'),mp.mpf(2),mp.mpf(4)]:
        u0=profile(xx);u1=mp.exp(1j*theta)*u0
        u_mid=(u0+u1)/2
        u_mid_xx=(1+mp.exp(1j*theta))/2*mp.diff(profile,xx,2)
        residual=(u1-u0)/hh-1j*(u_mid_xx+2*abs(u_mid)**2*u_mid)
        max_res=max(max_res,abs(residual))
    assert abs(mass-2)<mp.mpf('1e-60')
    assert max_res<mp.mpf('1e-60')
    print('h=',hs,'mass=',mp.nstr(mass,12),'max_midpoint_residual=',mp.nstr(max_res,5))
print('omega_h =',frequency)
print('r_phi4(c=0) = -33*eta**10/40')
print('PASS exact NLS midpoint c=0')
