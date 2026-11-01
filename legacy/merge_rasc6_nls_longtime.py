#!/usr/bin/env python3
import csv, glob, math, pathlib, sys

HERE=pathlib.Path(__file__).resolve().parent
indir=pathlib.Path(sys.argv[1]) if len(sys.argv)>1 else HERE/'results'
paths=sorted(indir.glob('rasc6_nls_long_h*.csv'))
if len(paths)!=3:
    raise SystemExit(f'expected 3 result CSVs, found {len(paths)}: {[p.name for p in paths]}')
rows=[]
for p in paths:
    with p.open() as f: rows.append(next(csv.DictReader(f)))
rows.sort(key=lambda r: float(r['h']), reverse=True)
for r in rows:
    for k,v in list(r.items()):
        try: r[k]=float(v)
        except (TypeError,ValueError): pass

for method in ['ASC6','RASC4','RASC6']:
    prev=None
    for r in rows:
        if prev is None:
            r['p_'+method]=''
        else:
            r['p_'+method]=math.log(prev['err_'+method]/r['err_'+method])/math.log(prev['h']/r['h'])
        prev=r
for r in rows:
    r['RASC4_over_h2']=r['err_RASC4']/r['h2']
    r['RASC6_over_h4']=r['err_RASC6']/r['h4']

fields=['h','T','nsteps','err_MP2','err_ASC4','err_ASC6','err_RASC4','err_RASC6',
        'p_ASC6','p_RASC4','p_RASC6','RASC4_over_h2','RASC6_over_h4',
        'angle2_x','angle2_phi','angle4_x','angle4_phi','norm_h2W2','norm_h4W4',
        'cond_tangent_P','cond_tangent_C4','elapsed_s','N','kernel_sha256','numpy','python','host']
out=HERE/'objective_final_nls_rasc6_longtime.csv'
with out.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:r.get(k,'') for k in fields} for r in rows])

print('h        T         ASC6          RASC4         RASC6          pR4    pR6    R4/h^2       R6/h^4')
for r in rows:
    def ff(x): return '-' if x=='' else f'{x:.3f}'
    print(f"{r['h']:<8.3g} {r['T']:<9.2f} {r['err_ASC6']:<13.5e} {r['err_RASC4']:<13.5e} {r['err_RASC6']:<13.5e} {ff(r['p_RASC4']):>6} {ff(r['p_RASC6']):>6} {r['RASC4_over_h2']:<12.5e} {r['RASC6_over_h4']:<12.5e}")

# Deliberately broad finite-grid gates: the conclusion is scaling, not a fitted universal constant.
p4=[r['p_RASC4'] for r in rows if r['p_RASC4']!='']
p6=[r['p_RASC6'] for r in rows if r['p_RASC6']!='']
asc6=[r['err_ASC6'] for r in rows]
pass_r4=(min(p4)>1.2 and max(p4)<2.8)
pass_r6=(min(p6)>2.8 and max(p6)<5.3)
pass_add=(min(asc6)>1e-3 and max(asc6)/min(asc6)<3.0)
print('GATE_RASC4_H2=', 'PASS' if pass_r4 else 'CHECK')
print('GATE_RASC6_H4=', 'PASS' if pass_r6 else 'CHECK')
print('GATE_ASC6_NONUNIFORM=', 'PASS' if pass_add else 'CHECK')
print('FINAL=', 'PASS' if (pass_r4 and pass_r6 and pass_add) else 'REVIEW')
print('wrote',out)
