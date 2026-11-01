#!/usr/bin/env python3
"""Repeat one archived first-corrector pilot, retaining stage/linear certificates.
The original diagnostic bases and sampling are unchanged. This is a new run.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,platform,sys,time
from pathlib import Path
import numpy as np
from certify_nls_run import ROOT, Certificates, write_csv, sha
sys.path.insert(0,str(ROOT/'legacy'))
import projection as PR
MP=PR.MP

def run(a):
    out=ROOT/a.out;out.mkdir(parents=True,exist_ok=False)
    pde,y0,state,modes,generators,T=PR.spec(a.bench,a.omega)
    nsteps=int(round(T/a.h));every=max(1,nsteps//a.nout)
    y=y0.copy();P=np.zeros_like(y);log=MP.Stats(); cert=Certificates();cert.install()
    rows=[];cs=[];us=[];ps=[];refs=[];ts=[];t0=time.perf_counter()
    meta={'run_kind':'NEW_R3_CERTIFIED_PILOT','benchmark':a.bench,'omega':a.omega,'N':pde.grid.N,
          'M':pde.M,'L':pde.grid.L,'h':a.h,'T':nsteps*a.h,'nsteps':nsteps,'nout_requested':a.nout,
          'stage_tol':PR.TOL,'certificate_contract':'same as certify_nls_run.py',
          'sources':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'legacy/src').rglob('*.py'))+[ROOT/'legacy/projection.py',Path(__file__),ROOT/'scripts/certify_nls_run.py']},
          'python':platform.python_version(),'numpy':np.__version__,'status':'RUNNING'}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    try:
      for n in range(1,nsteps+1):
        cert.reset();old=y.copy()
        y=MP.step(y,a.h,pde,tol=PR.TOL,log=log)
        rr=MP.rel_residual(pde,log.stage-old-.5*a.h*pde.f(log.stage),log.stage)
        if not log.converged or not np.isfinite(rr) or rr>=PR.TOL:raise RuntimeError(f'Stage failure {n} {rr}')
        P=PR.p_step(pde,log.stage,a.h,P)
        cs.append([n,rr,cert.gmres_count,cert.gmres_iterations,cert.gmres_max_iter,
                   cert.gmres_max_ratio,cert.sweep_max_ratio,cert.sweep_max_rel])
        if n%every and n!=nsteps:continue
        t=n*a.h;U=pde.grid.to_phys(y);Pp=pde.grid.to_phys(P);ex=state(t);e=(ex-U)/a.h**2;md=modes(t)
        ce,ne,te,re=PR.decompose(e,list(md.values()),pde.grid.dx)
        cp,npn,tp,rp=PR.decompose(Pp,list(md.values()),pde.grid.dx)
        row={'benchmark':a.bench,'h':a.h,'N':pde.grid.N,'t':t,'err_mp2_sup':float(np.max(np.abs(ex-U))),
          'e_norm':ne,'e_tangent':te,'e_transversal':re,'P_norm':npn,'P_tangent':tp,'P_transversal':rp,
          'e_minus_P_rel':float(np.sqrt(pde.grid.dx*np.sum((e-Pp)**2))/ne)}
        for k,key in enumerate(md):row['e_coef_'+key]=ce[k];row['P_coef_'+key]=cp[k]
        rows.append(row);us.append(y.copy());ps.append(P.copy());refs.append(ex);ts.append(t)
        if n%max(every,(nsteps//10//every)*every)==0:print(a.bench,n,nsteps,'elapsed',time.perf_counter()-t0,flush=True)
      cs=np.asarray(cs,float)
      write_csv(out/'projection.csv',rows)
      np.savez_compressed(out/'sampled_states.npz',times=ts,U=us,P=ps,reference=refs,h=a.h,N=pde.grid.N,
                          x=pde.grid.x,k=pde.grid.k,L=pde.grid.L,U0=y0)
      np.savez_compressed(out/'all_step_certificates.npz',values=cs,columns=np.array(['step','stage_residual',
           'gmres_calls','gmres_iterations','gmres_max_iter','krylov_certificate_ratio','sweep_certificate_ratio','sweep_relative_residual']))
      summary={'status':'PASS','certified_steps':nsteps,'certified_correction_solves':nsteps,
        'certified_gmres_calls':int(np.sum(cs[:,2])),'stage_residual_max':float(np.max(cs[:,1])),
        'krylov_certificate_ratio_max':float(np.max(cs[:,5])),
        'sweep_certificate_ratio_max':float(np.max(cs[:,6])),
        'sweep_relative_residual_max':float(np.max(cs[:,7])), 'elapsed_s':time.perf_counter()-t0,
        'final':rows[-1]}
      (out/'certificate_summary.json').write_text(json.dumps(summary,indent=2)+'\n');meta['status']='PASS'
      print('R3_PROJECTION_PASS',json.dumps(summary),flush=True)
    except Exception as ex:
      meta['status']='FAIL';meta['exception']=repr(ex);raise
    finally:
      cert.uninstall();(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('bench',choices=['nls','kdv','sg_breather']);p.add_argument('--h',type=float,required=True)
    p.add_argument('--omega',type=float);p.add_argument('--nout',type=int,default=48);p.add_argument('--out',required=True)
    run(p.parse_args())
