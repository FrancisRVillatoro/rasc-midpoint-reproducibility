#!/usr/bin/env python3
"""Independent DOP853 reference for the *same* NLS Galerkin ODE.
Integration is done in its co-moving/co-rotating frame, not against a fitted
trajectory. Two max-step/tolerance settings assess self-consistency only.
"""
from pathlib import Path
import argparse,sys,json,platform,time,hashlib
import numpy as np
from scipy.integrate import solve_ivp
import scipy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'legacy'))
import run_rasc6_nls_longtime as R

def main(a):
    out=ROOT/a.out;out.mkdir(parents=True,exist_ok=False)
    p=R.nls.make('bright',N=a.N);u0=p.grid.to_spec(R.state_phys(p,0));shape=u0.shape
    om=p.eta**2+p.c**2/4
    def rhs(t,z):
        u=z.reshape(shape)
        return (p.f(u)+p.c*R.dxop(p,u)-om*R.J(u)).ravel()
    tt={0.}
    for h in (.05,.04,.032):
        n=int(round((1/(h*h*R.RATE_PHI))/h));every=max(1,n//40)
        indices=set(range(every,n+1,every))|{n}|{int(round(q*n)) for q in [.25,.5,.75]}
        tt.update(j*h for j in indices if j*h<=a.T)
    tt.add(a.T);tt=np.array(sorted(tt))
    t0=time.perf_counter()
    print('REFERENCE_START',a.N,a.T,a.max_step,a.rtol,a.atol,flush=True)
    sol=solve_ivp(rhs,(0,a.T),u0.ravel(),method='DOP853',rtol=a.rtol,atol=a.atol,max_step=a.max_step,t_eval=tt)
    if not sol.success or not np.all(np.isfinite(sol.y)):raise RuntimeError(sol.message)
    frame=sol.y.T.reshape((-1,)+shape)
    ref=np.array([R.group(p,z,-p.c*t,om*t) for t,z in zip(tt,frame)])
    np.savez_compressed(out/'reference_states.npz',times=tt,U=ref,frame=frame,U0=u0,N=a.N,L=p.grid.L,k=p.grid.k,x=p.grid.x)
    diff=[]
    for t,u in zip(tt,ref):
        v=p.grid.to_phys(u);ex=R.state_phys(p,t);e=v-ex
        diff.append({'t':float(t),'l2_relative':float(np.linalg.norm(e)/np.linalg.norm(ex)),
            'cinf_relative':float(np.max(np.sqrt(np.sum(e*e,axis=0)))/np.max(np.sqrt(np.sum(ex*ex,axis=0))))})
    summary={'status':'PASS','method':'DOP853 on exact co-moving Galerkin ODE','N':a.N,'M':p.M,'T':a.T,
      'rtol':a.rtol,'atol':a.atol,'max_step':a.max_step,'nfev':sol.nfev,'elapsed_s':time.perf_counter()-t0,
      'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
      'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'max_reference_vs_analytic_l2_relative':max(r['l2_relative'] for r in diff),
      'max_reference_vs_analytic_cinf_relative':max(r['cinf_relative'] for r in diff),
      'comparison_samples':diff,
      'qualification':'Successful adaptive integration and paired-reference agreement are not rigorous global error bounds.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('REFERENCE_CONTROL_PASS',json.dumps({k:v for k,v in summary.items() if k!='comparison_samples'}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--N',type=int,default=384);p.add_argument('--T',type=float,default=1485.152)
    p.add_argument('--max-step',type=float,default=.02);p.add_argument('--rtol',type=float,default=2e-11);p.add_argument('--atol',type=float,default=2e-14);p.add_argument('--out',required=True)
    main(p.parse_args())
