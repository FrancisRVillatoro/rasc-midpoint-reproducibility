#!/usr/bin/env python3
"""Execute the complete recorded campaign in a NEW directory.
Never overwrite the shipped results. No network, installation or polling.
Default execution is sequential, single-threaded. This repeats PDE runs.
"""
from pathlib import Path
import argparse,os,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--destination',required=True,
      help='New directory for a complete self-contained repetition; must not exist.')
    a=ap.parse_args();dest=Path(a.destination).expanduser().resolve()
    subprocess.run([sys.executable,str(ROOT/'verify_release.py')],check=True)
    if dest.exists():raise SystemExit(f'Refusing to overwrite {dest}')
    if ROOT in dest.parents:raise SystemExit('Choose a destination outside the archived bundle.')
    dest.mkdir(parents=True)
    for name in ['legacy','archived','scripts','regression']:
        shutil.copytree(ROOT/name,dest/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['results','qa','tables','derived','panels']:(dest/name).mkdir()
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MPLBACKEND='Agg')
    def run(name,args):
        print('RUN',name,flush=True)
        with (dest/'qa'/f'{name}.log').open('w') as log:
            p=subprocess.run([sys.executable,*args],cwd=dest,env=env,stdout=log,stderr=subprocess.STDOUT)
        if p.returncode:raise SystemExit(f'{name} failed; inspect {dest}/qa/{name}.log')
    configs=[('cert_N384_h005',384,.05,None,None),('cert_N384_h004',384,.04,None,None),
      ('cert_N384_h0032',384,.032,None,None),('cert_N768_h005',768,.05,None,None),
      ('cert_N384_h005_inner13',384,.05,None,1e-13),
      ('fixed_N768_h004',768,.04,4,None),('fixed_N768_h002',768,.02,4,None),
      ('fixed_N768_h001',768,.01,4,None),('time_N384_h0025',384,.025,20,None),
      ('time_N384_h00125',384,.0125,20,None)]
    for name,n,h,t,inner in configs:
        args=['scripts/certify_nls_run.py','--N',str(n),'--h',str(h),'--out',f'results/{name}']
        if t is not None:args+=['--T',str(t)]
        if inner is not None:args+=['--inner-rtol',str(inner)]
        run(name,args)
    for case,name,h,omega,nout in [('nls','pilot_nls',.008,None,48),('kdv','pilot_kdv',.001,None,48),('sg_breather','pilot_sg',.01,.5,256)]:
        args=['scripts/certify_projection_run.py',case,'--h',str(h),'--nout',str(nout),'--out',f'results/{name}']
        if omega is not None:args+=['--omega',str(omega)]
        run(name,args)
    run('spatial_model_controls',['scripts/check_spatial_model.py'])
    run('rate_predictions',['scripts/reproduce_rate_predictions.py'])
    for name,step,rtol,atol in [('reference_coarse',.02,2e-11,2e-14),('reference_fine',.01,2e-13,2e-16)]:
        run(name,['scripts/reference_control.py','--max-step',str(step),'--rtol',str(rtol),'--atol',str(atol),'--out',f'results/{name}'])
    for name,script in [('evidence_build','build_evidence.py'),('figures','make_figures.py'),('evidence_regressions','check_evidence.py')]:run(name,['scripts/'+script])
    run('R2_regressions',['regression/R1_R2/run_r2_regressions.py'])
    run('R1_regressions',['regression/R1_R2/R1/run_r1_regressions.py'])
    run('additional_controls',['scripts/check_additional_controls.py'])
    print('COMPLETE_NUMERICAL_REPETITION_PASS',dest)
if __name__=='__main__':main()
