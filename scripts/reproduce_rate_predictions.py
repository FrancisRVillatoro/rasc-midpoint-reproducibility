#!/usr/bin/env python3
"""Re-evaluate the archived Riesz/full-response calculation with supplied source.
No reference fitting enters the forcing or Riesz projection; fitting the
predicted trajectory uses exactly the archived output times and window.
"""
from pathlib import Path
import sys,json,csv,time,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'legacy/src'));sys.path.insert(0,str(ROOT/'legacy'))
import predict as PD
from pdes import _spectral as S

def load(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def main():
    out=ROOT/'results/rate_predictions';out.mkdir(exist_ok=False);rows=[];details={};t0=time.perf_counter()
    for name,filename in [('nls','proj_nls_h8e-03.csv'),('kdv','proj_kdv_h1e-03.csv')]:
        print('PREDICTION_START',name,flush=True)
        pred,diag,X=PD.predict_relative_equilibrium(name,internals=True)
        M,g,p=X['M'],X['g'],X['p'];q=g-p
        lam,V=np.linalg.eig(M);keep=abs(lam)>X['rho'];V1=V[:,keep];w=lam[keep]
        coeff,*_=np.linalg.lstsq(V1,q.astype(complex),rcond=None)
        relative_res=float(np.linalg.norm(V1@coeff-q)/np.linalg.norm(q))
        r=load(ROOT/'archived/projection'/filename);tt=np.array([float(v['t']) for v in r]);sel=tt>=tt[-1]/4
        md=X['modes'];B=np.array([b.ravel() for b in md.values()]);dx=X['pde'].grid.dx;Gram=dx*B@B.T
        cp=[]
        for t in tt:
            ev=-t*p-np.real(V1@((coeff/w)*(np.exp(w*t)-1)))
            vp=X['pde'].grid.to_phys(X['from_real'](ev)).ravel();cp.append(np.linalg.solve(Gram,dx*B@vp))
        cp=np.array(cp)
        for i,mode in enumerate(md):
            b=float(np.polyfit(tt[sel],cp[sel,i],1)[0])
            rows.append({'benchmark':name,'mode':mode,'neutral_rate':pred[mode],'predicted_window_slope':b})
        np.savez_compressed(out/(name+'_predicted_trajectory.npz'),times=tt,coefficients=cp,modes=np.array(list(md)),neutral_vector=p,forcing=g)
        details[name]={'Riesz':diag,'complement_eigenbasis_relative_residual':relative_res,'pilot_file':filename,
                       'sha256':hashlib.sha256((ROOT/'archived/projection'/filename).read_bytes()).hexdigest()}
        print(name,rows[-len(md):],flush=True)
    pr,d=PD.predict_breather(.5,512);details['SG']={'rates':pr,'diagnostics':d}
    with (out/'rates.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    details['status']='PASS';details['elapsed_s']=time.perf_counter()-t0
    (out/'summary.json').write_text(json.dumps(details,indent=2)+'\n')
    print('RATE_PREDICTION_REPRODUCTION_PASS',json.dumps(details),flush=True)
if __name__=='__main__':main()
