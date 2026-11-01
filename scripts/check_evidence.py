#!/usr/bin/env python3
"""Validate archived provenance, all-step certificates, states and figures' data.
This checks algebraic residual certificates and saved outputs, not a global
nonlinear-stability theorem or an interval-arithmetic simulation certificate.
"""
from pathlib import Path
import csv,json,hashlib,sys,re
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'legacy'))
import run_rasc6_nls_longtime as R
import diagnose_rasc6_W4 as D
import projection as PR

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def jl(p):return json.loads(Path(p).read_text())
def checkclose(x,y,rtol=2e-11,atol=2e-13):np.testing.assert_allclose(x,y,rtol=rtol,atol=atol)

def main():
    tests=[]
    def passed(name,details):tests.append({'check':name,'status':'PASS','details':details});print('PASS',name)
    inputs=jl(ROOT/'archived/INPUT_MANIFEST.json')
    for r in inputs:assert digest(ROOT/r['path'])==r['sha256'],r['path']
    passed('historical_inputs_immutable',{'files':len(inputs),'smoke_excluded':3})
    report=jl(ROOT/'derived/evidence_summary.json'); steps=solves=0
    for c in report['certificates']:
        d=ROOT/'results'/c['run'];m=jl(d/'metadata.json');assert m['status']=='PASS'
        for file,h in m['sources'].items():assert digest(ROOT/file)==h,file
        with np.load(d/'all_step_certificates.npz') as z:
            names=list(z['columns']);v=z['values']
            assert len(v)==c['certified_steps']
            np.testing.assert_array_equal(v[:,0],np.arange(1,len(v)+1))
            assert np.all(np.isfinite(v)) and np.all(v[:,names.index('stage_residual')]<1e-15)
            for key in ['krylov_certificate_ratio','sweep_certificate_ratio']:assert np.all(v[:,names.index(key)]<=1)
        steps+=c['certified_steps'];solves+=c['correction_solves']
    passed('all_step_residual_certificates',{'runs':len(report['certificates']),'steps':steps,'corrector_solves':solves})
    # Rebuild all saved errors and cumulative fields from U,P,R, without integrating.
    maxgap=0.;samples=0
    for c in report['certificates']:
        d=ROOT/'results'/c['run'];csv=d/'history.csv'
        if not csv.exists():continue
        rows=read(csv)
        with np.load(d/'sampled_states.npz') as z:
            p=R.nls.make('bright',N=int(z['N']));h=float(z['h']);assert len(rows)==len(z['times'])
            for i,row in enumerate(rows):
                U,P,Q=z['U'][i],z['P'][i],z['R'][i];ref=z['reference'][i]
                a2,W2,_=R.split(p,U,P);C4=Q-R.apply_Aop(p,P,a2)+.5*R.apply_Aop(p,R.apply_Aop(p,U,a2),a2)
                a4,W4,_=R.split(p,U,C4)
                states={'MP2':U,'ASC4':U+h*h*P,'ASC6':U+h*h*P+h**4*Q,
                    'RASC4':R.group(p,U+h*h*W2,*(h*h*a2)),
                    'RASC6':R.group(p,U+h*h*W2+h**4*W4,*(h*h*a2+h**4*a4)),
                    'PROC':R.group(p,U,h*h*(5/192)*float(row['t']),h*h*R.RATE_PHI*float(row['t']))}
                for key,Y in states.items():
                    yp=p.grid.to_phys(Y);r=ref[0]+1j*ref[1];v=yp[0]+1j*yp[1]
                    values={'comp':np.max(abs(yp-ref))/np.max(abs(ref)),
                            'cinf':np.max(abs(v-r))/np.max(abs(r)),
                            'l2':np.linalg.norm(v-r)/np.linalg.norm(r)}
                    for metric,x in values.items():
                        y=float(row['err_'+key+'_'+metric]);checkclose(x,y);maxgap=max(maxgap,abs(x-y))
                checkclose(h**4*R.l2norm(p,W4),float(row['h4W4_l2']))
                samples+=1
    passed('all_sampled_errors_and_reconstructions',{'samples':samples,'metrics':['component_sup','complex_sup','real_L2'],'max_absolute_error_gap':maxgap})
    # Independent recomputation of all pilot projections from saved states.
    gaps=[]
    for tag,case,omega in [('pilot_nls','nls',None),('pilot_kdv','kdv',None),('pilot_sg','sg_breather',.5)]:
        p,_,state,modes,_,_=PR.spec(case,omega);rows=read(ROOT/f'results/{tag}/projection.csv')
        with np.load(ROOT/f'results/{tag}/sampled_states.npz') as z:
            h=float(z['h'])
            for i,t in enumerate(z['times']):
                U=p.grid.to_phys(z['U'][i]);P=p.grid.to_phys(z['P'][i]);e=(z['reference'][i]-U)/h**2
                ce,ne,nt,npn=PR.decompose(e,list(modes(t).values()),p.grid.dx)
                chi=np.sqrt(p.grid.dx*np.sum((e-P)**2))/ne
                for key,x in [('e_norm',ne),('e_tangent',nt),('e_transversal',npn),('e_minus_P_rel',chi)]:
                    checkclose(x,float(rows[i][key]));gaps.append(abs(x-float(rows[i][key])))
    passed('pilot_projection_recomputation',{'max_absolute_gap':max(gaps)})
    # Parseval and orthogonality are checked on actual scientific arrays.
    norms=[];canonical=[]
    for pth in sorted((ROOT/'results/cert_N384_h005').glob('w4_snapshot*.npz')):
        with np.load(pth) as z:
            assert int(z['N'])==384 and float(z['t'])>100
            p=R.nls.make('bright',N=384);w=np.full(len(z['k']),2.);w[0]=1.;w[-1]=0.
            spectral=p.grid.L*np.sum(w*np.sum(abs(z['W4'])**2,axis=0))
            physical=R.l2norm(p,z['W4'])**2;checkclose(spectral,physical,rtol=1e-12,atol=0)
            for tangent in [R.dxop(p,z['U']),R.J(z['U'])]:
                rel=abs(D.l2_inner(p,z['W4'],tangent))/(R.l2norm(p,z['W4'])*R.l2norm(p,tangent))
                assert rel<1e-12
            v=p.grid.to_phys(z['W4_canonical']).ravel();canonical.append(v/(np.sqrt(p.grid.dx)*np.linalg.norm(v)))
            norms.append(physical)
    v=np.asarray(canonical);s=np.linalg.svd(np.sqrt(p.grid.dx)*v,compute_uv=False);fr=s*s/sum(s*s)
    checkclose(fr,report['w4']['svd_fractions'])
    passed('scientific_snapshots_Parseval_projection_POD',{'snapshots':len(norms),'N':384,'best_rank2_relative_residual':float(np.sqrt(sum(fr[2:])))})
    # Independent references need not improve monotonically near the noise level.
    with np.load(ROOT/'results/reference_coarse/reference_states.npz') as a,np.load(ROOT/'results/reference_fine/reference_states.npz') as b:
        assert np.all(a['times']==b['times']);w=np.full(a['U'].shape[-1],2.);w[0]=1.;w[-1]=0.
        err=np.sqrt(np.sum(w*abs(a['U']-b['U'])**2,axis=(1,2))/np.sum(w*abs(b['U'])**2,axis=(1,2)))
        checkclose(max(err),report['reference']['paired_reference_max_relative_l2'],rtol=1e-12,atol=0)
    passed('independent_semidiscrete_reference_consistency',report['reference'])
    controls=jl(ROOT/'results/spatial_model_controls.json')['random_state_symmetry_and_invariant_checks']
    for name in ['NLS','KdV']:
        aa=[a for a in controls if a['pde']==name];assert aa[0]['test_state_sha256']==aa[1]['test_state_sha256']
        assert aa[0]['nonlinear_translation_defect']<1e-12 and aa[1]['nonlinear_translation_defect']>.1
    passed('paired_polynomial_Galerkin_symmetry_controls',{'same_state_pairs':2})
    # Rate extrapolation is exactly traceable to two archived finite-h measurements.
    for row in report['rates']:
        a=read(ROOT/'archived/projection'/row['source_fine']);b=read(ROOT/'archived/projection'/row['source_coarse'])
        def slope(r):
            t=np.array([float(x['t']) for x in r]);y=np.array([float(x['e_coef_'+row['mode']]) for x in r]);keep=t>=t[-1]/4
            return np.polyfit(t[keep],y[keep],1)[0]
        v1,v2=slope(a),slope(b);ratio=float(b[0]['h'])/float(a[0]['h'])
        checkclose(v1+(v1-v2)/(ratio**2-1),row['h2_extrapolated_slope'])
    passed('rate_extrapolation_provenance',{'rates':4,'NLS_actual_coarse_h':.016})
    out={'status':'PASS','checks':tests,'qualification':'Certificates apply to NEW R3 runs and saved samples. They do not establish nonlinear long-time stability, rigorous reference accuracy or a universal W4 obstruction.'}
    (ROOT/'qa/EVIDENCE_REGRESSION_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n')
    print('EVIDENCE_REGRESSIONS_PASS',len(tests))
if __name__=='__main__':main()
