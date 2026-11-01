#!/usr/bin/env python3
"""R2 countercheck: quadratic nonlinear remainder + bounded linear products
need not imply O(h^2) on arbitrary slow intervals. This is not a test of RASC.
"""
import numpy as np
from scipy.integrate import solve_ivp

def ivp(h):
    eps=h*h
    stop=2.0/eps
    def rhs(t,y):return [eps*eps+y[0]*y[0]*(1-y[0])]
    out=solve_ivp(rhs,(0.0,stop),[0.0],method='DOP853',rtol=2e-12,atol=2e-14)
    if not out.success:raise RuntimeError(out.message)
    return out.y[0,-1]

vals=[]
for h in (.1,.05,.025):
    v=ivp(h);vals.append(v)
    print('h=%.4f, T=%.2f, nonlinear response=%.12f'%(h,2/h**2,v))
assert vals[-1]>.99 and vals[0]>.97
print('R2 COUNTEREXAMPLE PASS: linear-product control alone is insufficient for all slow times')
