#!/usr/bin/env python3
"""Run all R2 scope and algebra checks without modifying scientific source."""
from pathlib import Path
import subprocess,sys,json,hashlib
base=Path(__file__).resolve().parent
names=['check_neutral_closure.py','check_formal_vs_discrete.py',
       'check_reduced_step.py','check_nonlinear_stability_counterexample.py',
       'check_translation_equivariance.py']
report=[]
for name in names:
    p=base/name
    proc=subprocess.run([sys.executable,str(p)],capture_output=True,text=True)
    (base/(name.replace('.py','.log'))).write_text(proc.stdout+proc.stderr)
    report.append({'test':name,'passed':proc.returncode==0,
                   'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                   'log':name.replace('.py','.log'),'returncode':proc.returncode})
    print(('PASS' if proc.returncode==0 else 'FAIL'),name)
(base/'R2_REGRESSION_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
if not all(it['passed'] for it in report):raise SystemExit('R2 regressions failed')
