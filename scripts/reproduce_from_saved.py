#!/usr/bin/env python3
"""Rebuild evidence in a new scratch copy; never overwrite the sealed package."""
from __future__ import annotations
import argparse,os,shutil,subprocess,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main()->None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--destination',required=True,help='New, non-existent directory outside the artifact.')
    a=ap.parse_args();dest=Path(a.destination).expanduser().resolve()
    if dest.exists() or dest==ROOT or ROOT in dest.parents:
        raise SystemExit('STOP: destination must be new and outside the source tree')
    subprocess.run([sys.executable,str(ROOT/'verify_release.py')],check=True)
    dest.mkdir(parents=True)
    for name in ['legacy','archived','results','scripts','regression']:
        shutil.copytree(ROOT/name,dest/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['qa','derived','tables','panels']:(dest/name).mkdir(exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
      NUMEXPR_NUM_THREADS='1',MPLBACKEND='Agg',PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH']=os.pathsep.join([str(dest/'legacy/src'),str(dest/'legacy')])
    commands=[('evidence','scripts/build_evidence.py'),('figures','scripts/make_figures.py'),
      ('evidence_checks','scripts/check_evidence.py'),
      ('R2','regression/R1_R2/run_r2_regressions.py'),
      ('R1','regression/R1_R2/R1/run_r1_regressions.py'),
      ('additional_controls','scripts/check_additional_controls.py')]
    report=[]
    for name,script in commands:
        print('RUN',name,flush=True)
        with (dest/'qa'/f'{name}.log').open('w') as log:
            p=subprocess.run([sys.executable,script],cwd=dest,env=env,stdout=log,stderr=subprocess.STDOUT)
        report.append({'check':name,'returncode':p.returncode})
        (dest/'qa/REBUILD_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
        if p.returncode:raise SystemExit(f'STOP: {name}; inspect {dest}/qa/{name}.log')
    print('SAVED_DATA_REBUILD_PASS',dest)
if __name__=='__main__':main()
