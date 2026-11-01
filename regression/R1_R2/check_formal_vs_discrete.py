#!/usr/bin/env python3
"""R2 fixed-time check: actual discrete midpoint P_h converges to formal e2.
This test deliberately makes no long-time uniformity inference.
"""
import cmath,math

def sample(h,T=2.0):
    n=round(T/h); assert abs(n*h-T)<1e-11
    z=1j*h
    r=(1+z/2)/(1-z/2)
    D=1-z*z/4
    alpha=-z**3/(12*D)
    U=r**n
    P=U*n*alpha/(h*h)
    e2=-T*(1j)**3/12*cmath.exp(1j*T)
    return abs(P-e2)

hs=[.2,.1,.05,.025]
errors=[sample(h) for h in hs]
print('h, ||P_h-e2|| on T=2')
for h,e in zip(hs,errors):print('%.5g %.12e'%(h,e))
orders=[math.log(errors[j]/errors[j+1],2) for j in range(len(errors)-1)]
print('orders:',orders)
assert min(orders)>1.9
print('R2 FORMAL/DISCRETE PASS (fixed T only)')
