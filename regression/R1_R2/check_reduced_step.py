#!/usr/bin/env python3
"""R2 regression: exact reduced-step identity and fifth-order local defect.

An equivariant nonlinear U(1) example f(u)=i (1+0.8 |u|^2) u.
The midpoint stage is solved to floating-point tolerance. The physical
nonlinear flow is exact, and a2 is of size h**(-2), to test cancellation
of the large tangent part without assuming bounded P.
"""
import numpy as np
import sympy as sp
from scipy.optimize import root

x,y=sp.symbols('x y', real=True)
J=sp.Matrix([[0,-1],[1,0]])
JN=np.array([[0.,-1.],[1.,0.]])
v=sp.Matrix([x,y])
f=(1+sp.Rational(4,5)*(x*x+y*y))*J*v
A=f.jacobian([x,y])
B=lambda u,w:sp.Matrix([sum(sp.diff(f[i],vv,vw)*u[j]*w[k] for j,vv in enumerate((x,y)) for k,vw in enumerate((x,y))) for i in range(2)])
G2=A*A*f/sp.Integer(12)-B(f,f)/sp.Integer(24)
f_fn=sp.lambdify((x,y),f,'numpy')
A_fn=sp.lambdify((x,y),A,'numpy')
G2_fn=sp.lambdify((x,y),G2,'numpy')

def f_eval(u):return np.asarray(f_fn(*u),float).ravel()
def A_eval(u):return np.asarray(A_fn(*u),float)
def G2_eval(u):return np.asarray(G2_fn(*u),float).ravel()
def phase(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s],[s,c]])
def flow(u,h):return phase((1+.8*np.dot(u,u))*h)@u

def run(h):
    eps=h*h
    U=np.array([.87,.27]); W=.17*U
    a=1.0/eps+.3
    P=a*(JN@U)+W
    def residual(un):return un-U-h*f_eval((un+U)/2)
    sol=root(residual,flow(U,h),tol=1e-12)
    if not sol.success and np.linalg.norm(residual(sol.x))>2e-13:raise RuntimeError(sol.message)
    Un=sol.x; um=(U+Un)/2; Am=A_eval(um)
    Bmat=np.eye(2)-h/2*Am
    dmid=np.linalg.solve(Bmat,np.eye(2)+h/2*Am)
    q=-h*np.linalg.solve(Bmat,G2_eval(um))
    Pn=dmid@P+q
    def project(u,p):return np.dot(JN@u,p)/np.dot(u,u)
    an=project(Un,Pn)
    wn=Pn-an*(JN@Un)
    a_start=project(U,P)
    w_start=P-a_start*(JN@U)
    assert abs(a_start-a)<1e-10
    delta=eps*(an-a_start)
    V=U+eps*w_start
    Vn=Un+eps*wn
    expected=phase(-delta)@flow(V,h)
    defect=np.linalg.norm(Vn-expected)
    firstorder=np.linalg.norm(Un+eps*q-flow(U,h))
    tangent_identity=np.linalg.norm(dmid@(JN@U)-JN@Un)
    decomposition=np.linalg.norm(Vn-(Un+eps*(dmid@w_start+q)-delta*(JN@Un)))
    return defect,firstorder,tangent_identity,decomposition,(an-a_start)/h,eps*a_start

def main():
    print('h, local_defect, local_defect/h^5, ASC4_step_error, tangent_covariance, splitting_residual, delta_a/h, h2*a')
    rows=[]
    for h in (.2,.1,.05,.025):
        out=run(h)
        rows.append((h,*out));print('%.5g %.10e %.10e %.10e %.2e %.2e %.9f %.9f' %(h,out[0],out[0]/h**5,*out[1:]))
        assert out[2]<4e-12 and out[3]<4e-12
        assert out[4]<10 and .9<out[5]<1.1
    errs=[q[1] for q in rows]
    orders=[np.log2(errs[i]/errs[i+1]) for i in range(len(errs)-1)]
    print('defect orders:',orders)
    assert min(orders[-2:])>4.6, orders
    assert max(q[1]/q[0]**5 for q in rows)<5.0
    print('R2 REDUCED-STEP PASS')
if __name__=='__main__':main()
