"""Exact algebra and independent numerical controls for the second audit."""
from pathlib import Path
import json, math
import sympy as s
import mpmath as mp
mp.mp.dps=70
z=s.symbols('z');D=1-z*z/4;r=(1+z/2)/(1-z/2)
a=-z**3/(12*D);beta=z**5/(120*D)+z**6/(288*(1-z/2)**2*(1+z/2));g=z**5/(120*D)+z**7/(576*D**2)
assert s.simplify(beta-a*a/2-g)==0
assert s.series(s.log(r)+a-z,z,0,6).removeO()==-z**5/120
assert s.series(s.log(r)+a+g-z,z,0,8).removeO()==17*z**7/20160
out={'linear_identity':'PASS','linear_series_R4':str(s.series(s.log(r)+a,z,0,8)),'linear_series_R6':str(s.series(s.log(r)+a+g,z,0,10))}
x=s.symbols('x',positive=True);t=s.symbols('t');cs=s.symbols('a3 a5 a7')
zeta=x+cs[0]*x**3+cs[1]*x**5+cs[2]*x**7
coeff=s.Poly(s.series(zeta*(1+zeta*zeta)**2-x,x,0,9).removeO(),x)
sol=s.solve(coeff.all_coeffs(),cs,dict=True)[0]
phase=s.series(2*s.atan(zeta.subs(sol)),x,0,9)
out['c0_zeta_in_s']=str(zeta.subs(sol));out['c0_theta_in_s']=str(phase)
rows=[]
for hh in ('0.1','0.05','0.025','0.0125'):
 h=mp.mpf(hh);eta=mp.mpf(1)
 zz=mp.findroot(lambda u:u*(1+u*u)**2-h*eta**2/2,(h/3,h/2))
 th=2*mp.atan(zz);co=mp.cos(th/2);b=eta*co*co
 profile=lambda xx: eta*co/mp.cosh(b*xx)
 residuals=[]
 for xx in (mp.mpf(0),mp.mpf('.5'),mp.mpf(2),mp.mpf(4)):
  u0=profile(xx);u1=mp.exp(1j*th)*u0;m=(u0+u1)/2
  mxx=(1+mp.exp(1j*th))/2*mp.diff(profile,xx,2)
  rr=(u1-u0)/h-1j*(mxx+2*abs(m)**2*m)
  residuals.append(abs(rr))
 om=th/h
 rows.append({'h':hh,'mass':mp.nstr(2*(eta*co)**2/b,25),'max_midpoint_residual':mp.nstr(max(residuals),12),'omega_h':mp.nstr(om,35),'h4_frequency_coefficient':mp.nstr((om-1+mp.mpf(7)*h*h/12)/h**4,35)})
out['c0_exact_family']=rows
# SG: check closed expression and direct phase/space quadrature.
w=s.symbols('omega',positive=True);rho=s.sqrt(1-w*w)
Kbar=-(2*rho**5-5*rho**3+3*rho+w*s.acos(w))/3
K0=-4*rho*(4-w*w)/9
rate=s.simplify((-s.diff(Kbar,w)-(K0-Kbar)/w)/(-16/rho))
target=(6*w**6-4*w**4+5*w*w-4)/(36*w)
assert s.simplify(rate-target)==0
out['sg_algebra']='PASS'
# Direct quadrature of explicit utt. Normalized phase integral can be done
# first using high-order Gauss points; no simulation or time fitting.
import numpy as np
from numpy.polynomial.legendre import leggauss
qnodes,qweights=leggauss(300);ts=math.pi*(qnodes+1);wt=qweights/2
xx=40*qnodes;wx=40*qweights
qout=[]
for om in (.3,.5,.8):
 rr=math.sqrt(1-om*om);aa=(rr/om)/np.cosh(rr*xx)
 sn=np.sin(ts[:,None]);cn=np.cos(ts[:,None]);a2=aa[None,:]**2
 uthetatheta=-4*aa[None,:]*sn*(1+2*a2-a2*sn**2)/(1+a2*sn**2)**2
 val=float(wt @ (om**4*(uthetatheta**2) @ wx))
 pred=4*(2*rr**5-5*rr**3+3*rr+om*math.acos(om))
 qout.append({'omega':om,'direct_utt2_average':val,'formula':pred,'relative_error':abs(val-pred)/abs(pred)})
out['sg_quadrature']=qout
# Bounded nonlinear recurrence: homogeneous products equal 1, defect h^5,
# quadratic remainder h*rho^2*(1-rho). Nothing here is a RASC integration.
rec=[]
for hh in (.1,.05,.025):
 nn=round(2/hh**3);v=0.
 for j in range(nn):v+=hh**5+hh*v*v*(1-v)
 rec.append({'h':hh,'n':nn,'tau':nn*hh**3,'rho':v,'rho_over_h2':v/hh**2})
out['bounded_recurrence_counterexample']=rec
p=Path(__file__).resolve().parent/'exact_controls.json';p.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
