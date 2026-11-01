#!/usr/bin/env python3
"""R2 audit: continuous shift equivariance for Fourier Galerkin, not collocation.

Finite N periodic cubic NLS (2*pi domain), no dealiasing for collocation.
The example uses high Fourier modes whose cubic products alias to k=0.
This is a scope regression, not a PDE solver validation.
"""
import numpy as np
N=24
x=2*np.pi*np.arange(N)/N
k=np.fft.fftfreq(N,d=1/N).astype(int)
a=.17
u=.2*np.exp(1j*8*x)+.15*np.exp(-1j*8*x)+.08*np.exp(1j*3*x)

def shift(u,s):return np.fft.ifft(np.fft.fft(u)*np.exp(1j*k*s))
def dxx(u):return np.fft.ifft(-k*k*np.fft.fft(u))
def fc(u):return 1j*(dxx(u)+2*np.abs(u)**2*u)

def fg(u):
    aa=np.fft.fft(u)/N
    coeff={int(kv):aa[i] for i,kv in enumerate(k)}
    out=np.zeros(N,dtype=complex)
    for i,kv in enumerate(k):
        v=0j
        for p,ap in coeff.items():
            for q,aq in coeff.items():
                r=int(kv)-p+q
                if r in coeff:
                    v+=ap*np.conj(aq)*coeff[r]
        out[i]=1j*(-kv*kv*aa[i]+2*v)
    return np.fft.ifft(N*out)

for fn,name in [(fc,'collocation'),(fg,'Galerkin')]:
    defect=np.linalg.norm(fn(shift(u,a))-shift(fn(u),a))
    print('%s translation-equivariance defect: %.12e'%(name,defect))
    if name=='collocation': assert defect>1e-3
    else: assert defect<2e-12
for fn,name in [(fc,'collocation'),(fg,'Galerkin')]:
    delta=np.linalg.norm(fn(np.exp(1j*a)*u)-np.exp(1j*a)*fn(u))
    print('%s U1-equivariance defect: %.12e'%(name,delta))
    assert delta<3e-12
print('R2 SYMMETRY-SCOPE PASS')
