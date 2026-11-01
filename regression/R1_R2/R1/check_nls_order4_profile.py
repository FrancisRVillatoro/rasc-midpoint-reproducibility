"""Independent polynomial residual check at eta=c=1; not a time integration."""
import sympy as s
x,S,T=s.symbols('x S T',real=True)
e=s.Integer(1);k=s.Rational(1,2);F=e*S

def red(v):
 return s.Poly(s.expand(v),T).rem(s.Poly(T**2-(1-S**2),T)).as_expr().expand()
def dx(v,n=1):
 for j in range(n):v=red(s.diff(v,x)-e*S*T*s.diff(v,S)+e*(1-T*T)*s.diff(v,T))
 return v

def add(*vs):return tuple(red(sum(v[i] for v in vs)) for i in range(2))
def mul(q,v):return tuple(red(q*vi) for vi in v)
def A(v):
 a,b=v
 return (-red(dx(b,2)+2*k*dx(a)-k*k*b+2*F*F*b),red(dx(a,2)-2*k*dx(b)-k*k*a+6*F*F*a))
def B(v,w):
 a,b=v;c,d=w
 return (-red(4*F*(a*d+b*c)),red(4*F*(3*a*c+b*d)))
def C(v,w,u):
 a,b=v;c,d=w;p,q=u
 return (-red(4*(a*c*q+a*d*p+b*c*p)+12*b*d*q),red(12*a*c*p+4*(a*d*q+b*c*q+b*d*p)))
f=(-2*k*dx(F),red(dx(F,2)-k*k*F+2*F**3))
G2=add(mul(s.Rational(1,12),A(A(f))),mul(-s.Rational(1,24),B(f,f)))
Af=A(f);A2=A(Af);A4=A(A(A2));Bff=B(f,f)
G4=add(mul(s.Rational(1,80),A4),mul(s.Rational(1,480),A(A(Bff))),mul(-s.Rational(1,240),A(B(Af,f))),mul(-s.Rational(1,80),B(A2,f)),mul(-s.Rational(1,480),B(Bff,f)),mul(-s.Rational(7,1440),A(C(f,f,f))),mul(s.Rational(1,160),B(Af,Af)),mul(s.Rational(1,480),C(Af,f,f)))
a=e*(-3*e**2+k**2)*(e**2+9*k**2)/24*S+e**3*k**2*S**3+e**2*(3*e**4+10*e**2*k**2-9*k**4)/12*x*S*T
b=e**2*k*(-e**2+3*k**2)/2*S*T+e**3*k*(e**2-3*k**2)/3*x*S
chi=(a,b)
dG2=add(mul(s.Rational(1,12),add(B(chi,Af),A(B(chi,f)),A(A(A(chi))))),mul(-s.Rational(1,24),add(C(chi,f,f),mul(2,B(A(chi),f)))))
v2=s.Rational(5,192); om2=-s.Rational(505,768)
Q=add(G4,dG2,mul(s.Rational(1,2),B(chi,chi)),mul(v2,(dx(a)-k*b,dx(b)+k*a)),mul(-om2,(-b,a)))
print('Q coefficients count:',*[len(s.Poly(vi,x,S,T).terms()) for vi in Q],flush=True)
print('Q=',Q,flush=True)
def int0(m):return s.sqrt(s.pi)*s.gamma(s.Rational(m,2))/s.gamma(s.Rational(m+1,2))/e
def int2(m):return s.simplify(int0(m)*s.polygamma(1,s.Rational(m,2))/(2*e**2))
def integ(v):
 total=0
 for (p,m,n),co in s.Poly(red(v),x,S,T).terms():
  if (p+n)%2:continue
  if (p,n)==(0,0):vv=int0(m)
  elif (p,n)==(2,0):vv=int2(m)
  elif (p,n)==(1,1):vv=int0(m)/(m*e)
  elif (p,n)==(3,1):vv=3*int2(m)/(m*e)
  else:raise ValueError((p,m,n))
  total+=co*vv
 return s.factor(s.simplify(s.expand_func(total)))
qM=integ(a*a+b*b);qP=k*qM+integ(a*dx(b)-b*dx(a))
abasis=[S**j for j in (1,3,5,7,9)]+[x*S**j*T for j in (1,3,5,7)]+[x*x*S**j for j in (1,3,5)]+[x**3*S*T]
bbasis=[S**j*T for j in (1,3,5,7)]+[x*S**j for j in (1,3,5,7)]+[x*x*S**j*T for j in (1,3,5)]+[x**3*S]
ac=s.symbols('a:'+str(len(abasis)));bc=s.symbols('b:'+str(len(bbasis)));rx,rp=s.symbols('rx rp')
a4=sum(c*v for c,v in zip(ac,abasis));b4=sum(c*v for c,v in zip(bc,bbasis))
resR=red(-(dx(b4,2)-e*e*b4+2*F*F*b4)+Q[0]+rx*dx(F))
resI=red((dx(a4,2)-e*e*a4+6*F*F*a4)+Q[1]+(k*rx+rp)*F)
eqs=s.Poly(resR,x,S,T).coeffs()+s.Poly(resI,x,S,T).coeffs()
mvar=integ(2*F*a4)
pvar=2*k*integ(F*a4)+integ(F*dx(b4)-b4*dx(F))
for label,mextra,pextra in [('linearized leaf',0,0),('exact mass/momentum leaf',qM,qP)]:
 sol=s.linsolve(eqs+[mvar+mextra,pvar+pextra],list(ac)+list(bc)+[rx,rp])
 print(label, 'nonempty=',bool(sol),flush=True)
 assert len(sol)==1
 vals=next(iter(sol));d=dict(zip(list(ac)+list(bc)+[rx,rp],vals))
 print('rx=',s.simplify(d[rx]),'rp=',s.simplify(d[rp]),flush=True)
 assert s.simplify(resR.subs(d)) == 0
 assert s.simplify(resI.subs(d)) == 0
 if label == 'exact mass/momentum leaf':
  assert s.simplify((mvar+qM).subs(d)) == 0
  assert s.simplify((pvar+qP).subs(d)) == 0
  assert s.simplify(d[rx]+s.Rational(17747,184320)) == 0
  assert s.simplify(d[rp]+s.Rational(717481,737280)) == 0
 print('residual R=',s.simplify(resR.subs(d)),'I=',s.simplify(resI.subs(d)),flush=True)
 print('M order4=',s.simplify((mvar+qM).subs(d)),'P order4=',s.simplify((pvar+qP).subs(d)),flush=True)
 print('chi4 real=',s.simplify(a4.subs(d)),flush=True)
 print('chi4 imag=',s.simplify(b4.subs(d)),flush=True)
