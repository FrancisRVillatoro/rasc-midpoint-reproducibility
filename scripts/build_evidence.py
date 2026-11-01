#!/usr/bin/env python3
"""Build every R3 numerical table and plot input from identified source files.
No empirical fit is a confidence interval; no solver check is a global error bound.
"""
from __future__ import annotations
import csv,json,sys,hashlib,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'legacy'))
import run_rasc6_nls_longtime as R
import diagnose_rasc6_W4 as D
OUT=ROOT/'derived';OUT.mkdir(exist_ok=True)
TABLE=ROOT/'tables';TABLE.mkdir(exist_ok=True)

def csvrows(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def col(rows,key):return np.asarray([float(r[key]) for r in rows])
def writecsv(p,rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def jload(p):return json.loads(Path(p).read_text())
def sample_slope(rows,key,lo=.25):
    t=col(rows,'t');v=col(rows,key);m=t>=lo*t[-1]
    return float(np.polyfit(t[m],v[m],1)[0])
def exponent(rows,key,period=None):
    t=col(rows,'t');v=col(rows,key)
    if period is not None:
        bins=np.floor(t/period+1e-9).astype(int);inds=[]
        for b in np.unique(bins):
            ii=np.flatnonzero(bins==b)
            if len(ii)>3:inds.append(ii[np.argmax(v[ii])])
        t=t[inds];v=v[inds]
    mask=(t>=t[-1]/4)&(v>0)
    return float(np.polyfit(np.log(t[mask]),np.log(v[mask]),1)[0])
def sci(x,n=3):
    if not math.isfinite(float(x)):return '--'
    if x==0:return '$0$'
    a,b=f'{float(x):.{n}e}'.split('e')
    return '$'+a+'\\times10^{'+str(int(b))+'}$'
def table(path,caption,label,cols,head,rows,foot=''):
    text='\\begin{table}[tbp]\n\\centering\n\\small\n\\caption{'+caption+'}\n\\label{'+label+'}\n'
    text+='\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{'+cols+'}\n\\toprule\n'+head+' \\\\\n\\midrule\n'
    text+='\n'.join(' & '.join(r)+' \\\\' for r in rows)+'\n\\bottomrule\n\\end{tabular}\n'
    if foot:text+='\\par\\smallskip\\begin{minipage}{.98\\textwidth}\\footnotesize '+foot+'\\end{minipage}\n'
    text+='\\end{table}\n';(TABLE/path).write_text(text)

def main():
    report={}
    # Archived pilot series retain their original decimal precision.
    cases=[('NLS','nls','proj_nls_h8e-03.csv','proj_nls_h2e-02.csv',None),
           ('KdV','kdv','proj_kdv_h1e-03.csv','proj_kdv_h2e-03.csv',None),
           ('SG','sg','proj_sg_breather_w0.5_h1e-02.csv','proj_sg_breather_w0.5_h2e-02.csv',4*np.pi)]
    ps=[];rates=[];certs=[];certnames=[]
    for title,name,fine,coarse,period in cases:
        old=csvrows(ROOT/'archived/projection'/fine);oldcoarse=csvrows(ROOT/'archived/projection'/coarse)
        fresh=csvrows(ROOT/'results'/('pilot_'+name)/'projection.csv')
        cert=jload(ROOT/'results'/('pilot_'+name)/'certificate_summary.json');assert cert['status']=='PASS'
        certnames.append('pilot_'+name)
        np.testing.assert_allclose(col(old,'t'),col(fresh,'t'),rtol=0,atol=1e-8)
        final=fresh[-1]
        row={'case':title,'h':float(final['h']),'N':int(final['N']),'T':float(final['t']),
          'rho_family_final':float(final['e_transversal'])/float(final['e_norm']),
          'chi_P_final':float(final['e_minus_P_rel']),
          'tangent_exponent':exponent(fresh,'e_tangent',period),
          'transverse_exponent':exponent(fresh,'e_transversal',period),
          'P_transverse_exponent':exponent(fresh,'P_transversal',period),
          'max_relative_difference_archived_e_norm':float(np.max(np.abs(col(old,'e_norm')-col(fresh,'e_norm'))/col(fresh,'e_norm'))),
          'max_relative_difference_archived_transverse_norm':float(np.max(np.abs(col(old,'e_transversal')-col(fresh,'e_transversal'))/col(fresh,'e_transversal')))}
        ps.append(row)
        for mode in ['phase','translation'] if name=='nls' else ['translation'] if name=='kdv' else ['phase']:
            h1=float(old[0]['h']);h2=float(oldcoarse[0]['h'])
            b1=sample_slope(old,'e_coef_'+mode);b2=sample_slope(oldcoarse,'e_coef_'+mode)
            extrap=b1+(b1-b2)/((h2/h1)**2-1)
            rr={'case':title,'mode':mode,'h_fine':h1,'h_coarse':h2,
              'archived_e_slope':b1,'archived_P_slope':sample_slope(old,'P_coef_'+mode),
              'new_e_slope':sample_slope(fresh,'e_coef_'+mode),'new_P_slope':sample_slope(fresh,'P_coef_'+mode),
              'h2_extrapolated_slope':extrap,'source_fine':fine,'source_coarse':coarse}
            rates.append(rr)
    closure=csvrows(ROOT/'archived/projection/closure.csv')
    for rr in rates:
        key='nls' if rr['case']=='NLS' else 'kdv' if rr['case']=='KdV' else None
        match=next((v for v in closure if v['benchmark']==key and v['mode']==rr['mode']),None)
        rr['neutral_coefficient']=float(match['rate_Pi0g']) if match else -31/192
        rr['same_window_prediction']=float(match['fitted_slope_predicted_trajectory']) if match else float('nan')
        rr['relative_extrapolated_fit_gap']=abs(rr['h2_extrapolated_slope']-rr['same_window_prediction'])/abs(rr['h2_extrapolated_slope']) if match else float('nan')
    writecsv(OUT/'projection_summary.csv',ps);writecsv(OUT/'rates_audit.csv',rates)
    report['projection']=ps;report['rates']=rates
    newrates=csvrows(ROOT/'results/rate_predictions/rates.csv')
    reproduced=[]
    for r in rates:
        key={'NLS':'nls','KdV':'kdv'}.get(r['case'])
        if key is None:continue
        v=next(v for v in newrates if v['benchmark']==key and v['mode']==r['mode'])
        reproduced.append({'case':r['case'],'mode':r['mode'],
            'neutral_absolute_difference':abs(float(v['neutral_rate'])-r['neutral_coefficient']),
            'window_slope_absolute_difference':abs(float(v['predicted_window_slope'])-r['same_window_prediction'])})
    report['rate_prediction_reproduction']=reproduced
    report['rate_prediction_diagnostics']=jload(ROOT/'results/rate_predictions/summary.json')
    sgold=csvrows(ROOT/'archived/projection/proj_sg_breather_w0.5_h1e-02.csv')
    report['SG_window_sensitivity']=[{'lower_fraction':q,'slope':sample_slope(sgold,'e_coef_phase',q)} for q in [0,.25,.5,.75]]
    table('projection_summary.tex','Certified repetitions of the finest first-corrector pilots. Norms are the real discrete $L^2$ norm; the residual is relative to the full diagnostic family, not only the symmetry group.','tab:projection-summary','lrrrrrr',
          'Problem & $h$ & $T_h$ & $p_{\\parallel}$ & $p_{\\perp}$ & $\\rho_{\\rm fam}(T_h)$ & $\\chi_P(T_h)$',
          [[x['case'],f"{x['h']:g}",f"{x['T']:.2f}",f"{x['tangent_exponent']:.3f}",f"{x['transverse_exponent']:.3f}",sci(x['rho_family_final'],2),sci(x['chi_P_final'],2)] for x in ps],
          'All three pilots use $N=768$. Fits are over the last three quarters of the sampled interval; SG norm slopes use per-period maxima. A fitted slope is a finite-window diagnostic, not a uniform boundedness statement.')
    table('rates.tex','Audited rate comparison. The finite-step columns and the two-step $h^2$ extrapolation are regenerated from the archived pilot CSVs; the full-response slopes are from the separately archived linearized-response calculation.','tab:G2-rates','llrrrr',
          'Problem & Mode & Neutral & Response fit & Measured & Extrapolated',
          [[x['case'],x['mode'],f"{x['neutral_coefficient']:.8f}",('--' if not math.isfinite(x['same_window_prediction']) else f"{x['same_window_prediction']:.8f}"),f"{x['archived_e_slope']:.8f}",f"{x['h2_extrapolated_slope']:.8f}"] for x in rates],
          'Fine/coarse steps are $(0.008,0.016)$ for NLS, $(0.001,0.002)$ for KdV and $(0.01,0.02)$ for SG. NLS rates use the parameter basis; KdV uses the periodic $L=64$ projection. The SG entry has no full Floquet-response prediction. Extrapolations are not confidence intervals.')
    # New fixed-time and secular cases, in three explicitly different metrics.
    fixed=[];long=[];all_histories={}
    for kind,hs,tags in [('fixed',[.04,.02,.01],['fixed_N768_h004','fixed_N768_h002','fixed_N768_h001']),
                         ('long',[.05,.04,.032],['cert_N384_h005','cert_N384_h004','cert_N384_h0032'])]:
        for h,tag in zip(hs,tags):
            c=jload(ROOT/'results'/tag/'certificate_summary.json');assert c['status']=='PASS';certnames.append(tag)
            r=csvrows(ROOT/'results'/tag/'history.csv');all_histories[tag]=r
            last={k:float(v) for k,v in r[-1].items()}
            last['run']=tag;last['max_sampled_RASC4_l2']=max(col(r,'err_RASC4_l2'))
            last['samples']=len(r);last['certified_steps']=c['certified_steps']
            (fixed if kind=='fixed' else long).append(last)
    def orders(series,key):return [float(np.log(a[key]/b[key])/np.log(a['h']/b['h'])) for a,b in zip(series[:-1],series[1:])]
    report['fixed']=fixed;report['long']=long
    report['orders']={k:{m:orders(v,m) for m in ['err_ASC4_l2','err_RASC4_l2','err_RASC4_cinf','err_RASC4_comp','err_PROC_l2']} for k,v in [('fixed',fixed),('long',long)]}
    writecsv(OUT/'fixed_time.csv',fixed);writecsv(OUT/'long_family.csv',long)
    table('fixed_time.tex','Certified fixed-time NLS repetitions at $N=768$, $T=4$. The main error is the relative complex $L^2$ error; the last column also records the component-sup metric used in the historical bank.','tab:rasc4-fixed','rrrrr',
          '$h$ & $E_2(\\ASC4)$ & $E_2(\\RASC4)$ & $E_2(\\mathrm{proc})$ & $E_{\\rm comp}(\\RASC4)$',
          [[f"{r['h']:g}",sci(r['err_ASC4_l2']),sci(r['err_RASC4_l2']),sci(r['err_PROC_l2']),sci(r['err_RASC4_comp'])] for r in fixed])
    table('long_family.tex','Certified R3 NLS repetitions at $N=384$ on the secular family. Errors here are relative $L^2$ errors against the sampled analytic reference. The last column is a maximum over the stored output times, not over all time steps.','tab:rasc4-long','rrrrrrr',
          '$h$ & $T_h$ & $E_2(\\mathrm{MP2})$ & $E_2(\\ASC4)$ & $E_2(\\RASC4)$ & $E_2(\\mathrm{proc})$ & $E_{2,\\max}^{\\rm R4}$',
          [[f"{r['h']:g}",f"{r['t']:.3f}",f"{r['err_MP2_l2']:.4f}",f"{r['err_ASC4_l2']:.4f}",sci(r['err_RASC4_l2'],2),sci(r['err_PROC_l2'],2),sci(r['max_sampled_RASC4_l2'],2)] for r in long])
    table('w4_family.tex','Second-corrector diagnostics in the same certified NLS runs as Table~\\ref{tab:rasc4-long}. No reference alignment is used to compute reconstruction errors.','tab:w4-long-family','rrrrr',
          '$h$ & $E_2(\\ASC6)$ & $E_2(\\RASC6)$ & $E_\\infty^{\\C}(\\RASC6)$ & $h^4\\norm{W_4}_N$',
          [[f"{r['h']:g}",f"{r['err_ASC6_l2']:.5f}",f"{r['err_RASC6_l2']:.5f}",f"{r['err_RASC6_cinf']:.5f}",f"{r['h4W4_l2']:.6f}"] for r in long])
    # Recompute W4 figures from newly stored scientific snapshots, not smoke data.
    diag=csvrows(ROOT/'results/cert_N384_h005/w4_diagnostic.csv');pde=R.nls.make('bright',N=384)
    snapshots=sorted((ROOT/'results/cert_N384_h005').glob('w4_snapshot_*.npz'),key=lambda p:float(np.load(p)['t']))
    can=[];specrows=[];raw_recomputed=[]
    for p in snapshots:
        with np.load(p) as a:
            assert int(a['N'])==384 and float(a['h'])==.05
            row,arr=D.analyze_snapshot(pde,a['U'],a['P'],a['R'],float(a['h']),float(a['t']))
            np.testing.assert_allclose(arr['W4'],a['W4'],rtol=1e-12,atol=1e-10)
            vp=pde.grid.to_phys(a['W4_canonical']).ravel();vp=vp/(np.sqrt(pde.grid.dx)*np.linalg.norm(vp));can.append(vp)
            e=np.sum(np.abs(a['W4'])**2,axis=0);w=np.full_like(e,2.);w[0]=1.;w[-1]=0.;e*=w
            assert abs(pde.grid.L*np.sum(e)-row['norm_W4']**2)/row['norm_W4']**2<1e-12
            for k,en in zip(a['k'],e/np.sum(e)):specrows.append({'t':row['t'],'k':float(k),'fraction':float(en)})
            raw_recomputed.append(row)
    V=np.array(can);gram=pde.grid.dx*V@V.T;ss=np.linalg.svd(np.sqrt(pde.grid.dx)*V,compute_uv=False);fr=ss**2/np.sum(ss**2)
    writecsv(OUT/'w4_spectra.csv',specrows);writecsv(OUT/'w4_checkpoints.csv',raw_recomputed)
    writecsv(OUT/'w4_pod.csv',[{'mode':i+1,'L2_singular_value':float(s),'squared_fraction':float(f)} for i,(s,f) in enumerate(zip(ss,fr))])
    np.savetxt(OUT/'w4_gram.csv',gram,delimiter=',',header='L2-normalized canonical snapshot Gram matrix',comments='# ')
    olddiag=csvrows(ROOT/'archived/integrated/w4_n384_checkpoints.csv')
    w4={'growth_exponent':exponent(raw_recomputed,'norm_W4'),'residual_growth_exponent':exponent(raw_recomputed,'norm_h4_res_action_W2'),
        'final':raw_recomputed[-1],'svd_fractions':fr.tolist(),'best_rank2_relative_residual':float(np.sqrt(np.sum(fr[2:]))),
        'max_offdiagonal_gram':float(np.max(np.abs(gram-np.eye(4)))),
        'relative_norm_difference_vs_archived_checkpoints':(abs(col(diag,'norm_h4W4')-col(olddiag,'norm_h4W4'))/col(olddiag,'norm_h4W4')).tolist(),
        'hk_peak_squared':float(.05*raw_recomputed[-1]['k_peak']**2)}
    report['w4']=w4
    # Resolution and inner-tolerance controls compare complete Fourier arrays.
    comparisons=[]
    for tag in ['cert_N768_h005','cert_N384_h005_inner13']:
        certnames.append(tag)
        for base in snapshots:
            with np.load(base) as a:
                t=float(a['t'])
                other=ROOT/'results'/tag/f'w4_snapshot_t{t:.6f}.npz'
                with np.load(other) as b:
                    fine=R.nls.make('bright',N=int(b['N']))
                    for key in ['U','W2','W4']:
                        aa=np.zeros_like(b[key]);aa[:,:a[key].shape[-1]]=a[key]
                        delta=R.l2norm(fine,aa-b[key]);den=R.l2norm(fine,b[key])
                        comparisons.append({'control':tag,'t':t,'field':key,'relative_field_difference':delta/den,
                            'relative_norm_difference':abs(R.l2norm(fine,aa)-den)/den})
    writecsv(OUT/'field_controls.csv',comparisons);report['field_controls']=comparisons
    # Reference checks and new norms against an independently integrated semidiscrete flow.
    refs={}
    for tag in ['coarse','fine']:
        refs[tag]=jload(ROOT/'results'/f'reference_{tag}'/'summary.json');assert refs[tag]['status']=='PASS'
    with np.load(ROOT/'results/reference_coarse/reference_states.npz') as ac,np.load(ROOT/'results/reference_fine/reference_states.npz') as af:
        np.testing.assert_array_equal(ac['times'],af['times'])
        w=np.full(ac['U'].shape[-1],2.);w[0]=1.;w[-1]=0.
        diff=np.sqrt(np.sum(w*np.abs(ac['U']-af['U'])**2,axis=(1,2))/np.sum(w*np.abs(af['U'])**2,axis=(1,2)))
        refcomparison={'paired_reference_max_relative_l2':float(diff.max()),
           'coarse_vs_analytic_max_l2':refs['coarse']['max_reference_vs_analytic_l2_relative'],
           'fine_vs_analytic_max_l2':refs['fine']['max_reference_vs_analytic_l2_relative'],
           'fine_vs_analytic_max_cinf':refs['fine']['max_reference_vs_analytic_cinf_relative']}
        referr=[]
        for series in long:
            tag=series['run']
            with np.load(ROOT/'results'/tag/'sampled_states.npz') as a:
                for i,t in enumerate(a['times']):
                    j=int(np.argmin(abs(af['times']-t)));assert abs(float(af['times'][j])-t)<1e-8
                    ref=pde.grid.to_phys(af['U'][j]);U,P,RR=a['U'][i],a['P'][i],a['R'][i];h=float(a['h'])
                    a2,W2,_=R.split(pde,U,P);C4=RR-R.apply_Aop(pde,P,a2)+.5*R.apply_Aop(pde,R.apply_Aop(pde,U,a2),a2);a4,W4,_=R.split(pde,U,C4)
                    states={'MP2':U,'ASC4':U+h*h*P,'ASC6':U+h*h*P+h**4*RR,
                        'RASC4':R.group(pde,U+h*h*W2,*(h*h*a2)),
                        'RASC6':R.group(pde,U+h*h*W2+h**4*W4,*(h*h*a2+h**4*a4))}
                    row={'h':h,'t':float(t),'N':384}
                    hh=all_histories[tag];ii=int(np.argmin(abs(col(hh,'t')-t)))
                    assert abs(float(hh[ii]['t'])-t)<1e-8
                    for method,Y in states.items():row['err_'+method+'_semidiscrete_l2']=float(np.linalg.norm(pde.grid.to_phys(Y)-ref)/np.linalg.norm(ref))
                    row['RASC4_absolute_error_change']=abs(row['err_RASC4_semidiscrete_l2']-float(hh[ii]['err_RASC4_l2']))
                    referr.append(row)
        writecsv(OUT/'errors_vs_semidiscrete_reference.csv',referr)
    refcomparison['max_RASC4_absolute_error_change']=max(r['RASC4_absolute_error_change'] for r in referr)
    report['reference']=refcomparison
    # Supplemental shorter-h controls.
    short=[]
    for tag in ['time_N384_h0025','time_N384_h00125']:
        certnames.append(tag);c=jload(ROOT/'results'/tag/'certificate_summary.json');assert c['status']=='PASS'
        short.append({'run':tag,**c['final']})
    report['short_time_controls']=short
    # Every run has independent arrays: check counts/status, summarize maxima, retain raw arrays.
    certrows=[]
    for tag in certnames:
        c=jload(ROOT/'results'/tag/'certificate_summary.json');assert c['status']=='PASS'
        with np.load(ROOT/'results'/tag/'all_step_certificates.npz') as a:
            vv=a['values'];cc=list(a['columns']);assert len(vv)==c['certified_steps']
            assert np.all(np.isfinite(vv))
            assert np.max(vv[:,cc.index('stage_residual')])<1e-15
            assert np.max(vv[:,cc.index('krylov_certificate_ratio')])<=1
            assert np.max(vv[:,cc.index('sweep_certificate_ratio')])<=1
        certrows.append({'run':tag,'certified_steps':c['certified_steps'],'correction_solves':c['certified_correction_solves'],
           'stage_max':c['stage_residual_max'],'krylov_ratio_max':c['krylov_certificate_ratio_max'],
           'sweep_ratio_max':c['sweep_certificate_ratio_max'],'sweep_relative_residual_max':c['sweep_relative_residual_max']})
    report['certificates']=certrows;report['total_certified_steps']=sum(c['certified_steps'] for c in certrows)
    report['total_certified_correction_solves']=sum(c['correction_solves'] for c in certrows)
    report['total_certified_runs']=len(certrows)
    writecsv(OUT/'solver_certificate_summary.csv',certrows)
    table('solver_certificates.tex','Independent certificates for the three main secular runs. Each step contains one nonlinear base stage and two corrector solves; residuals are recomputed from the returned states.','tab:solver-certificates','rrrrr',
          '$h$ & Steps & Correction solves & Max. stage residual & Max. correction ratio',
          [[f"{h:g}",str(r['certified_steps']),str(r['correction_solves']),sci(r['stage_max'],2),sci(r['sweep_ratio_max'],2)]
           for h,r in zip([.05,.04,.032],[next(c for c in certrows if c['run']==tag) for tag in ['cert_N384_h005','cert_N384_h004','cert_N384_h0032']])],
          'A correction ratio below one means $\\|b-Ax\\|_{\\ell^2}\\le\\max(10^{-8}\\|b\\|_{\\ell^2},10^{-14})$. This is an algebraic acceptance certificate, not a global discretization-error bound.')
    # Check that no smoke file contributes to primary evidence.
    report['excluded_inputs']=['archived/EXCLUDED_smoke_N128/*: N=128, T=2; never used in scientific plots or fits']
    def clean_json(x):
        if isinstance(x,dict):return {k:clean_json(v) for k,v in x.items()}
        if isinstance(x,list):return [clean_json(v) for v in x]
        if isinstance(x,float) and not math.isfinite(x):return None
        return x
    (OUT/'evidence_summary.json').write_text(json.dumps(clean_json(report),indent=2,allow_nan=False)+'\n')
    print('R3_EVIDENCE_BUILD_PASS')
    print(json.dumps({k:report[k] for k in ['total_certified_steps','orders','reference']},indent=2))
if __name__=='__main__':main()
