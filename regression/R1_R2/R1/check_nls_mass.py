import sympy as s
x,S,T=s.symbols('x S T',real=True)
e=s.symbols('eta',positive=True); k,c=s.symbols('k c',real=True)
def red(v):
 return s.Poly(s.expand(v),T).rem(s.Poly(T**2-(1-S**2),T)).as_expr().expand()
def dx(v):return red(s.diff(v,x)-e*S*T*s.diff(v,S)+e*(1-T*T)*s.diff(v,T))
def int0(m):return s.sqrt(s.pi)*s.gamma(s.Rational(m,2))/s.gamma(s.Rational(m+1,2))/e
def int2(m):return s.simplify(int0(m)*s.polygamma(1,s.Rational(m,2))/(2*e**2))
def integ(v):
 total=0
 for (p,m,n),co in s.Poly(red(v),x,S,T).terms():
  if (p+n)%2: continue
  if (p,n)==(0,0):val=int0(m)
  elif (p,n)==(2,0):val=int2(m)
  elif (p,n)==(1,1):val=int0(m)/(m*e)
  elif (p,n)==(3,1):val=3*int2(m)/(m*e)
  else: raise ValueError((p,m,n))
  total+=co*val
 return s.factor(s.simplify(s.expand_func(total)))
a=e*(-3*e**2+k**2)*(e**2+9*k**2)/24*S+e**3*k**2*S**3+e**2*(3*e**4+10*e**2*k**2-9*k**4)/12*x*S*T
b=e**2*k*(-e**2+3*k**2)/2*S*T+e**3*k*(e**2-3*k**2)/3*x*S
qM=integ(a*a+b*b); qP=s.factor(k*qM+integ(a*dx(b)-b*dx(a)))
print('First-order mass variation:',integ(2*e*S*a))
print('First-order momentum variation:',s.factor(2*k*integ(e*S*a)+integ(e*S*dx(b)-b*dx(e*S))))
print('qM=',qM)
print('qP=',qP)
oldx=c/s.Integer(331776)*(81*c**8+(648-216*s.pi**2)*c**6*e**2+(1248*s.pi**2-5760)*c**4*e**4-(16512+128*s.pi**2)*c**2*e**6-(2304+1536*s.pi**2)*e**8)
oldphi=-1/s.Integer(6635520)*(729*c**10+(18630-1350*s.pi**2)*c**8*e**2+(22560*s.pi**2-256320)*c**6*e**4+(340800-40000*s.pi**2)*c**4*e**6+(679680+53760*s.pi**2)*c**2*e**8+(5543424+23040*s.pi**2)*e**10)
deta=-qM/2
dc=(k*qM-qP)/e
dphi=-2*e*deta-k*dc
newx=s.factor(oldx+dc.subs(k,c/2));newphi=s.factor(oldphi+dphi.subs(k,c/2))
print('delta rx4=',s.factor(dc.subs(k,c/2)))
print('delta rphi4=',s.factor(dphi.subs(k,c/2)))
print('candidate constrained rx4=',newx)
print('candidate constrained rphi4=',newphi)
for label,v in [('qM at 1,1',qM.subs({e:1,k:s.Rational(1,2)})),('qP at 1,1',qP.subs({e:1,k:s.Rational(1,2)})),('old x',oldx.subs({e:1,c:1})),('old phi',oldphi.subs({e:1,c:1})),('new x',newx.subs({e:1,c:1})),('new phi',newphi.subs({e:1,c:1})),('old phi c0',oldphi.subs({c:0})),('new phi c0',newphi.subs({c:0}))]:print(label,s.simplify(v),s.N(v,20))
