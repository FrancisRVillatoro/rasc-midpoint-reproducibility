#!/usr/bin/env python3
"""Source-specific symmetry, invariant and sampled-profile controls for R3.
These are diagnostics, not a nonlinear-stability theorem or global error bound.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'legacy'))
import run_rasc6_nls_longtime as R
import projection as PR
from pdes import nls,kdv,sg

def derivative(pde,V):return pde.grid.project(1j*pde.grid.k[None,:]*V)
def group(pde,V,a):return pde.grid.project(V*np.exp(1j*pde.grid.k[None,:]*a))
def norm(pde,V):return pde.grid.l2(V)
def defect(pde,y,a,nonlinear=False):
    f=pde.f_nl if nonlinear else pde.f
    base=f(y)
    return norm(pde,f(group(pde,y,a))-group(pde,base,a))/max(norm(pde,base),1e-300)

def profile_nls(pde,t):
    physical=R.state_phys(pde,t);phi=pde.grid.to_spec(physical)
    xi=R.wrap(pde.grid.x-pde.p['x0']-pde.c*t,pde.grid.L)
    dt=pde.c*pde.eta*np.tanh(pde.eta*xi)[None,:]*physical
    dt+=(pde.eta**2-pde.c**2/4)*np.stack([-physical[1],physical[0]])
    r=pde.grid.to_spec(dt)-pde.f(phi)
    return {'t':float(t),'absolute_residual_l2':norm(pde,r),
      'relative_residual_l2':norm(pde,r)/max(norm(pde,pde.grid.to_spec(dt)),1e-300),
      'projection_discrepancy_cinf':float(np.max(np.sqrt(np.sum((physical-pde.grid.to_phys(phi))**2,axis=0))))}

def main():
    rng=np.random.default_rng(9312026);checks=[]
    for name,make in [('NLS',lambda pad:nls.make('bright',N=64,pad=pad)),('KdV',lambda pad:kdv.make('soliton',N=64,pad=pad))]:
      p0=make(2. if name=='NLS' else 1.5)
      y0=p0.grid.project(rng.normal(size=(p0.grid.ncomp,p0.grid.Nk))+1j*rng.normal(size=(p0.grid.ncomp,p0.grid.Nk)))
      y0=y0/norm(p0,y0)*3
      for pad in ([2.,1.] if name=='NLS' else [1.5,1.]):
        p=make(pad);y=y0.copy()
        ph=p.grid.to_phys(y);f=p.f(y);fp=p.grid.to_phys(f)
        dquad=2*p.grid.dx*np.sum(ph*fp)
        den=2*norm(p,y)*norm(p,f)
        row={'test_state_sha256':hashlib.sha256(y.tobytes()).hexdigest(),'pde':name,'N':p.grid.N,'M':p.M,'padding':pad,
             'nonlinear_translation_defect':defect(p,y,.371,True),
             'full_translation_defect':defect(p,y,.371),
             'normalized_quadratic_invariant_derivative':float(abs(dquad)/max(den,1e-300))}
        if name=='NLS':
            dy=p.grid.to_phys(derivative(p,y));df=p.grid.to_phys(derivative(p,f))
            dp=p.grid.dx*np.sum(fp[0]*dy[1]+ph[0]*df[1]-fp[1]*dy[0]-ph[1]*df[0])
            row['normalized_momentum_derivative']=float(abs(dp)/max(norm(p,f)*norm(p,derivative(p,y))*2,1e-300))
            row['phase_defect']=norm(p,p.f(R.group(p,y,0,.731))-R.group(p,p.f(y),0,.731))/norm(p,p.f(y))
        else:row['mass_derivative']=float(p.grid.dx*np.sum(fp))
        checks.append(row)
    profile={}
    for N in [256,384,768]:
      p=nls.make('bright',N=N)
      # moving centers across the periodic box, independent of the long-time endpoints
      profile[str(N)]=[profile_nls(p,t) for t in np.linspace(0,p.grid.L/p.c,33)]
    # actual first-pilot code defaults and sampled orbit residual for KdV/SG
    pilot_controls=[]
    for name,om in [('kdv',None),('sg_breather',.5)]:
      p,y0,state,modes,gens,T=PR.spec(name,om)
      if name=='kdv':
        xi=p._wrap(p.grid.x-p.p['x0']);kap=p.p['kappa'];c=4*kap**2
        up=-4*kap**3/np.cosh(kap*xi)**2*np.tanh(kap*xi)
        dt=p.grid.to_spec((-c*up)[None,:]);r=dt-p.f(y0)
      else:
        # time derivative of the analytic two-component breather is tested
        # by centered Richardson differences; this is not an exact residual certificate.
        delta=1e-3;t=1.0
        dtp=(4*(state(t+delta)-state(t-delta))/(2*delta)-(state(t+2*delta)-state(t-2*delta))/(4*delta))/3
        y0=p.grid.to_spec(state(t));dt=p.grid.to_spec(dtp);r=dt-p.f(y0)
      pilot_controls.append({'case':name,'N':p.grid.N,'M':p.M,'L':p.grid.L,'T_default':T,
         'residual_l2':norm(p,r),'residual_relative_l2':norm(p,r)/norm(p,dt),
         'derivative_kind':'analytic' if name=='kdv' else 'Richardson finite difference dt=1e-3'})
    out={'random_seed':9312026,'pairing':'identical spectral state for padded and unpadded realization of each PDE','random_state_symmetry_and_invariant_checks':checks,'sampled_analytic_NLS_profile_controls':profile,
         'pilot_profile_controls':pilot_controls,
         'interpretation':'NLS cubic 2N and KdV quadratic 3N/2 padding are Galerkin for retained modes with Nyquist removed; SG sine is not a polynomial Galerkin product. Sampled residuals do not imply a global reference-error bound.'}
    (ROOT/'results/spatial_model_controls.json').write_text(json.dumps(out,indent=2)+'\n')
    for r in checks:print(r)
    for n,r in profile.items():print('PROFILE',n,'max rel',max(x['relative_residual_l2'] for x in r))
    print(pilot_controls)
    assert all(r['nonlinear_translation_defect']<1e-11 for r in checks if r['padding']>1)
    print('SPATIAL_MODEL_CONTROLS_PASS')
if __name__=='__main__':main()
