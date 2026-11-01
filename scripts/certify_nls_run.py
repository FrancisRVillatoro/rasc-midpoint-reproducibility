#!/usr/bin/env python3
"""R3: newly certified repetitions of the archived RASC NLS driver.

The PDE, midpoint, kernel and ASC source files in legacy/src are unmodified.
Wrappers inspect the returned linear iterates and independently recompute
stage/correction residuals; no certificate is inferred for a historical run.
Each directory is append-protected. No network access or pandas is required.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, platform, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'legacy'))
import run_rasc6_nls_longtime as R
import diagnose_rasc6_W4 as D
MP=R.MP

def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def norm(v):return float(np.sqrt(np.sum(np.abs(v)**2)))

class Certificates:
    def __init__(self,inner_rtol=None):
        self.inner_rtol=inner_rtol
        self.gmres_original=MP.gmres
        self.solve_original=MP.solve_stage_jacobian
        self.reset();self.all_rows=[]
    def reset(self):
        self.gmres_count=0;self.gmres_max_iter=0;self.gmres_max_ratio=0.
        self.sweep_count=0;self.sweep_max_ratio=0.;self.sweep_max_rel=0.
        self.gmres_iterations=0;self.sweep_max_abs=0.
    def install(self):
        def checked_gmres(op,b,rtol=MP.INNER_RTOL,maxit=MP.MAXGMRES):
            rr=rtol if self.inner_rtol is None else self.inner_rtol
            x,it=self.gmres_original(op,b,rtol=rr,maxit=maxit)
            r=norm(b-op(x));scale=max(1e-8*norm(b),1e-28)
            rat=r/scale
            self.gmres_count+=1;self.gmres_iterations+=it
            self.gmres_max_iter=max(it,self.gmres_max_iter)
            self.gmres_max_ratio=max(rat,self.gmres_max_ratio)
            if not np.isfinite(rat) or rat>1:raise RuntimeError(f'Krylov residual certificate failed: {rat}')
            return x,it
        def checked_solve(pde,ym,h,b):
            x,it=self.solve_original(pde,ym,h,b)
            res=b-x+.5*h*pde.applyA(ym,x)
            nr=norm(res);nb=norm(b);scale=max(1e-8*nb,1e-14)
            rat=nr/scale
            self.sweep_count+=1;self.sweep_max_ratio=max(rat,self.sweep_max_ratio)
            self.sweep_max_rel=max(nr/max(nb,1e-300),self.sweep_max_rel)
            self.sweep_max_abs=max(nr,self.sweep_max_abs)
            if not np.isfinite(rat) or rat>1:raise RuntimeError(f'Correction residual certificate failed: {rat}')
            return x,it
        MP.gmres=checked_gmres;MP.solve_stage_jacobian=checked_solve
    def uninstall(self):MP.gmres=self.gmres_original;MP.solve_stage_jacobian=self.solve_original

def reconstructions(pde,U,P,RR,h):
    a2,W2,c2=R.split(pde,U,P)
    C4=RR-R.apply_Aop(pde,P,a2)+.5*R.apply_Aop(pde,R.apply_Aop(pde,U,a2),a2)
    a4,W4,c4=R.split(pde,U,C4)
    states={'MP2':U,'ASC4':U+h*h*P,'ASC6':U+h*h*P+h**4*RR,
        'RASC4':R.group(pde,U+h*h*W2,*(h*h*a2)),
        'RASC6':R.group(pde,U+h*h*W2+h**4*W4,*(h*h*a2+h**4*a4))}
    return states,a2,a4,W2,W4,c2,c4

def errors(pde,Y,ref):
    yp=pde.grid.to_phys(Y);r=ref[0]+1j*ref[1];z=yp[0]+1j*yp[1]
    return {'comp':float(np.max(np.abs(yp-ref))/np.max(np.abs(ref))),
       'cinf':float(np.max(np.abs(z-r))/np.max(np.abs(r))),
       'l2':float(np.linalg.norm(z-r)/np.linalg.norm(r))}

def run(args):
    out=ROOT/args.out
    out.mkdir(parents=True,exist_ok=False)
    pde=R.nls.make('bright',N=args.N)
    h=args.h
    Ttarget=args.T if args.T is not None else args.sigma/(h*h*R.RATE_PHI)
    steps=int(round(Ttarget/h)); T=steps*h
    if steps<4:raise ValueError('Need at least 4 steps')
    checkpoint_steps=sorted(set(max(1,int(round(q*steps))) for q in (.25,.5,.75,1.)))
    every=max(1,steps//args.nout)
    sample_steps=set(range(every,steps+1,every))|set(checkpoint_steps)|{steps}
    source_files=sorted((ROOT/'legacy/src').rglob('*.py'))+[ROOT/'legacy/run_rasc6_nls_longtime.py',ROOT/'legacy/diagnose_rasc6_W4.py',Path(__file__)]
    meta={'run_kind':'NEW_R3_CERTIFIED_REPETITION','N':args.N,'padded_M':pde.M,'L':pde.grid.L,
      'h':h,'T_target':Ttarget,'T':T,'steps':steps,'sigma_requested':args.sigma,
      'actual_slow_phase':R.RATE_PHI*h*h*T,'checkpoint_steps':checkpoint_steps,
      'requested_stage_tolerance':R.TOL,'inner_rtol':args.inner_rtol or MP.INNER_RTOL,
      'stage_contract':'relative physical component sup residual < 1e-15',
      'linear_contract':'unpreconditioned coefficient Euclidean residual <= max(1e-8*||rhs||,1e-14)',
      'krylov_contract':'recomputed left-preconditioned residual <= max(1e-8*||rhs_pre||,1e-28)',
      'legacy_kernel':R.provenance(), 'python':platform.python_version(),'numpy':np.__version__,
      'platform':platform.platform(),'threads':{s:os.environ.get(s) for s in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},
      'sources':{str(p.relative_to(ROOT)):sha(p) for p in source_files},
      'status':'RUNNING'}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    U=pde.grid.to_spec(R.state_phys(pde,0));P=np.zeros_like(U);RR=np.zeros_like(U)
    U0=U.copy(); M0=pde.mass(U); P0=pde.momentum(U)
    certificates=Certificates(args.inner_rtol);certificates.install();log=MP.Stats()
    rows=[];diag=[];history_U=[];history_P=[];history_R=[];history_ref=[];history_stage=[]
    step_rows=[];t0=time.perf_counter()
    try:
      for j in range(1,steps+1):
        certificates.reset();Uold=U.copy();newtonold=log.newton_its
        U=MP.step(U,h,pde,tol=R.TOL,log=log)
        ym=log.stage
        rr=MP.rel_residual(pde,ym-Uold-.5*h*pde.f(ym),ym)
        if not log.converged or not np.isfinite(rr) or rr>=R.TOL:
            raise RuntimeError(f'Stage certificate failed at step {j}: {rr}')
        P,RR,_=R.asc.sweep_step(pde,ym,h,P,RR,use_H2=True)
        if certificates.sweep_count!=2:raise RuntimeError('Expected two correction solves')
        step_rows.append([j,rr,certificates.gmres_count,certificates.gmres_iterations,
            certificates.gmres_max_iter,certificates.gmres_max_ratio,certificates.sweep_max_ratio,
            certificates.sweep_max_rel,certificates.sweep_max_abs,log.newton_its-newtonold])
        if j in sample_steps:
            t=j*h;ref=R.state_phys(pde,t)
            states,a2,a4,W2,W4,c2,c4=reconstructions(pde,U,P,RR,h)
            states['PROC']=R.group(pde,U,h*h*(5/192)*t,h*h*R.RATE_PHI*t)
            row={'step':j,'t':t,'h':h,'N':args.N,
                 'angle2_x':h*h*a2[0],'angle2_phi':h*h*a2[1],
                 'angle4_x':h**4*a4[0],'angle4_phi':h**4*a4[1],
                 'W2_l2':R.l2norm(pde,W2),'h2W2_l2':h*h*R.l2norm(pde,W2),
                 'W4_l2':R.l2norm(pde,W4),'h4W4_l2':h**4*R.l2norm(pde,W4),
                 'mass_U_drift':pde.mass(U)-M0,'momentum_U_drift':pde.momentum(U)-P0,
                 'cond2':c2,'cond4':c4}
            for method,Y in states.items():
                for k,v in errors(pde,Y,ref).items():row[f'err_{method}_{k}']=v
            rows.append(row);history_U.append(U.copy());history_P.append(P.copy());history_R.append(RR.copy())
            history_ref.append(ref);history_stage.append(ym.copy())
        if j in checkpoint_steps:
            dr,arrays=D.analyze_snapshot(pde,U,P,RR,h,j*h);diag.append(dr)
            np.savez_compressed(out/f'w4_snapshot_t{j*h:.6f}.npz',t=j*h,h=h,N=args.N,k=pde.grid.k,
                                x=pde.grid.x,L=pde.grid.L,**arrays)
            print(f'CHECKPOINT {j}/{steps} t={j*h:g} h4W4={dr["norm_h4W4"]:.12g}',flush=True)
        elif j%max(1,steps//10)==0:
            print(f'{j}/{steps} {100*j/steps:.0f}% elapsed={time.perf_counter()-t0:.1f}s',flush=True)
      write_csv(out/'history.csv',rows);write_csv(out/'w4_diagnostic.csv',diag)
      columns=['step','stage_residual','gmres_calls','gmres_iterations','gmres_max_iter','krylov_certificate_ratio',
               'sweep_certificate_ratio','sweep_relative_residual','sweep_absolute_residual','newton_iterations']
      certarr=np.asarray(step_rows,float)
      np.savez_compressed(out/'all_step_certificates.npz',columns=np.array(columns),values=certarr)
      np.savez_compressed(out/'sampled_states.npz',times=np.array([r['t'] for r in rows]),U=np.array(history_U),
          P=np.array(history_P),R=np.array(history_R),reference=np.array(history_ref),stage=np.array(history_stage),
          h=h,N=args.N,k=pde.grid.k,x=pde.grid.x,L=pde.grid.L,U0=U0)
      summary={'status':'PASS','certified_steps':steps,'certified_correction_solves':2*steps,
        'certified_gmres_calls':int(np.sum(certarr[:,2])), 'stage_residual_max':float(np.max(certarr[:,1])),
        'krylov_certificate_ratio_max':float(np.max(certarr[:,5])),
        'sweep_certificate_ratio_max':float(np.max(certarr[:,6])),
        'sweep_relative_residual_max':float(np.max(certarr[:,7])),
        'gmres_max_iterations':int(np.max(certarr[:,4])),
        'max_sampled_mass_drift':max(abs(r['mass_U_drift']) for r in rows),
        'max_sampled_momentum_drift':max(abs(r['momentum_U_drift']) for r in rows),
        'final':rows[-1], 'max_sampled_errors':{k:max(r[k] for r in rows) for k in rows[-1] if k.startswith('err_')},
        'elapsed_s':time.perf_counter()-t0}
      (out/'certificate_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
      meta['status']='PASS';meta['elapsed_s']=summary['elapsed_s']
      print('R3_CERTIFIED_RUN_PASS',json.dumps(summary),flush=True)
    except Exception as exc:
      meta['status']='FAIL';meta['exception']=repr(exc)
      np.savez_compressed(out/'partial_certificates_NOT_ACCEPTED.npz',values=np.asarray(step_rows,float))
      raise
    finally:
      certificates.uninstall();(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--N',type=int,default=384);ap.add_argument('--h',type=float,default=.05)
    ap.add_argument('--T',type=float);ap.add_argument('--sigma',type=float,default=1.)
    ap.add_argument('--inner-rtol',type=float);ap.add_argument('--nout',type=int,default=40)
    ap.add_argument('--out',required=True)
    run(ap.parse_args())
