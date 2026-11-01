#!/usr/bin/env python3
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
tasks=[
 ('fixed_N768_h004',['--N','768','--h','.04','--T','4','--nout','8']),
 ('fixed_N768_h002',['--N','768','--h','.02','--T','4','--nout','8']),
 ('fixed_N768_h001',['--N','768','--h','.01','--T','4','--nout','8']),
 ('time_N384_h0025',['--N','384','--h','.025','--T','20','--nout','40']),
 ('time_N384_h00125',['--N','384','--h','.0125','--T','20','--nout','40']),
]
for name,args in tasks:
    with (ROOT/'qa'/f'{name}.log').open('w') as f:
        p=subprocess.run([sys.executable,str(ROOT/'scripts/certify_nls_run.py')]+args+['--out',f'results/{name}'],stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:raise SystemExit(f'FAILED {name} code {p.returncode}')
    print('DONE',name,flush=True)
