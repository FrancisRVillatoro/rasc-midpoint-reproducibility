#!/usr/bin/env python3
"""R1 mathematical regression runner. All checks are local and CPU only."""
from pathlib import Path
import subprocess,sys,hashlib,json,re,datetime
here=Path(__file__).resolve().parent
names=[
 'check_nls_mass.py',
 'check_nls_order4_profile.py',
 'check_nls_c0_midpoint.py',
 'check_kdv_normalization.py',
]
results=[]
for name in names:
    p=here/name
    r=subprocess.run([sys.executable,str(p)],cwd=str(here),text=True,capture_output=True)
    (here/(p.stem+'.log')).write_text(r.stdout+r.stderr)
    row=dict(script=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),returncode=r.returncode,
             log=p.stem+'.log')
    results.append(row)
    print(f'{name}: {"PASS" if r.returncode == 0 else "FAIL"}')
    if r.returncode:
        print(r.stdout[-1800:]);print(r.stderr[-1800:]);break
if len(results)==len(names) and all(x['returncode']==0 for x in results):
    extra=(here/'check_nls_order4_profile.log').read_text()
    assert 'exact mass/momentum leaf nonempty= True' in extra
    assert 'rx= -17747/184320 rp= -717481/737280' in extra
    assert 'M order4= 0 P order4= 0' in extra
    mass=(here/'check_nls_mass.log').read_text()
    assert 'candidate constrained rx4=' in mass
    assert 'candidate constrained rphi4=' in mass
    assert 'new phi c0 -33*eta**10/40' in mass
    print('R1 constraints and corrected rates: PASS')
    status='PASS'
else:status='FAIL'
record={'status':status,'scripts':results,
        'scope':'continuous-spatial modified-profile R1, not long-time PDE solver validation'}
(here/'R1_REGRESSION_RESULTS.json').write_text(json.dumps(record,indent=2)+'\n')
sys.exit(0 if status=='PASS' else 2)
