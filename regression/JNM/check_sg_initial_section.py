"""Independent SG phase-section control; no time-domain integrator.
The autonomous midpoint modified Hamiltonian is H+h^2 K2. Starting
on the unperturbed breather at phase theta0 selects an averaged action
shift (K2(theta0,omega)-Kbar2(omega))/omega. Thus the IVP corrective
frequency is -(Kbar2'(omega)+(K2(theta0)-Kbar2)/omega)/I'(omega).
"""
import json
import sympy as s
import mpmath as mp
mp.mp.dps=45
w=s.symbols('omega',positive=True)
r=s.sqrt(1-w*w)
Kbar=-(2*r**5-5*r**3+3*r+w*s.acos(w))/3
K0=-4*r*(4-w*w)/9
Khalf=-s.Rational(2,3)*(w**4*r+w**6*s.atanh(r))
Ip=-16/r
rate0=s.simplify(-(s.diff(Kbar,w)+(K0-Kbar)/w)/Ip)
target=(6*w**6-4*w**4+5*w*w-4)/(36*w)
assert s.simplify(rate0-target)==0
ratehalf=s.simplify(-(s.diff(Kbar,w)+(Khalf-Kbar)/w)/Ip)
targethalf=w**3*(-w**2*r*s.atanh(r)+5*w**2-3)/24
assert s.simplify(ratehalf-targethalf)==0
# At theta=pi/2, v=utx=0 and integrating utt^2 by z=tanh(rho*x)
# gives 16*omega^4*rho+16*omega^6*atanh(rho).
def K_quad(om,theta):
 rho=mp.sqrt(1-om*om)
 def density(x):
  aa=(rho/om)/mp.cosh(rho*x)
  sn,cn=mp.sin(theta),mp.cos(theta)
  den=1+aa*aa*sn*sn
  ut=4*aa*om*cn/den
  utt=-4*aa*om*om*sn*(1+2*aa*aa-aa*aa*sn*sn)/(den*den)
  ax=-rho*aa*mp.tanh(rho*x)
  utx=4*om*cn*ax*(1-aa*aa*sn*sn)/(den*den)
  cosu=mp.cos(4*mp.atan(aa*sn))
  return -(utt*utt+utx*utx+cosu*ut*ut)/24
 return mp.quad(density,[-mp.inf,0,mp.inf])
rows=[]
for om in (mp.mpf('.3'),mp.mpf('.5'),mp.mpf('.8')):
 rho=mp.sqrt(1-om*om)
 pred0=-4*rho*(4-om*om)/9
 predhalf=-mp.mpf(2)/3*(om**4*rho+om**6*mp.atanh(rho))
 quad0,quadhalf=K_quad(om,0),K_quad(om,mp.pi/2)
 assert abs(quad0-pred0)<mp.mpf('1e-35')
 assert abs(quadhalf-predhalf)<mp.mpf('1e-35')
 r0=mp.mpf(str(s.N(rate0.subs(w,s.Rational(str(om))),40)))
 rh=r0+rho*(predhalf-pred0)/(16*om)
 rows.append(dict(omega=str(om),K2_section0=str(quad0),K2_section_pi2=str(quadhalf),rate_section0=str(r0),rate_section_pi2=str(rh)))
print(json.dumps({'general_rate':'-(Kbar2_prime+(K2(theta0)-Kbar2)/omega)/I_prime','quadrature_checks':'PASS','rate0_symbolic':str(rate0),'rate_pi2_symbolic':str(ratehalf),'rows':rows},indent=2))
