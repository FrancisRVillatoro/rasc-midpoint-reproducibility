#!/usr/bin/env python3
"""Regenerate the four R3 editorial figures from archived or newly audited data.

Each numerical panel is an independent Matplotlib figure (no subplots).
PyMuPDF only arranges vector panels on a page; it does not change the data.
No custom colors or style sheet are used. Requires numpy, matplotlib, pymupdf.
"""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fitz
ROOT=Path(__file__).resolve().parents[1]
PANELS=ROOT/'panels';PANELS.mkdir(exist_ok=True)
D=ROOT/'derived'
plt.rcParams.update({'font.size':11,'axes.labelsize':11,'legend.fontsize':8.7,
    'xtick.labelsize':10,'ytick.labelsize':10,'pdf.fonttype':42,
    'savefig.bbox':'tight'})

def read(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def col(r,k):return np.array([float(v[k]) for v in r])
def single(size=(4.5,3.45)):
    fig=plt.figure(figsize=size);return fig,fig.add_subplot(111)
def save(fig,name):
    fig.tight_layout(pad=.7)
    p=PANELS/f'{name}.pdf';fig.savefig(p);fig.savefig(PANELS/f'{name}.png',dpi=160)
    plt.close(fig);return p

def compose(stem,panels,layout):
    # Dimensions in PDF points. Each slot preserves the source aspect ratio.
    if layout=='single':
        doc=fitz.open(panels[0]);out=doc
    else:
        width=660.;gap=12.;h=252.
        if layout=='double':
            boxes=[fitz.Rect(0,0,(width-gap)/2,h),fitz.Rect((width+gap)/2,0,width,h)]
            height=h
        else:
            # Two upper panels; wider rank panel below to keep lettering readable.
            boxes=[fitz.Rect(0,0,(width-gap)/2,h),fitz.Rect((width+gap)/2,0,width,h),
                   fitz.Rect(width*.24,h+12,width*.76,2*h-7)]
            height=2*h-7
        out=fitz.open();page=out.new_page(width=width,height=height)
        for p,box in zip(panels,boxes):
            with fitz.open(p) as inp:page.show_pdf_page(box,inp,0,keep_proportion=True)
    out.set_metadata({'title':stem,'author':'Francisco R. Villatoro','subject':'RASC R3 numerical evidence'})
    out.save(ROOT/f'{stem}.pdf',garbage=4,deflate=True)
    out[0].get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(ROOT/f'{stem}.png')
    out.close()

# 1. Archived exact scalar calculation, not a PDE rerun.
a=[r for r in read(ROOT/'archived/integrated/objective4_linear.csv')
   if r['test']=='secular_family' and r['H2']=='full']
a.sort(key=lambda r:float(r['h']));h=col(a,'h')
fig,ax=single((5.8,3.75))
for m,mk in zip(['MP2','ASC4','ASC6','RASC4','RASC6'],['o','s','^','D','v']):
    ax.loglog(h,col(a,'err_'+m),marker=mk,markersize=4,label=m)
ax.loglog(h,.1*h**2,'--',label=r'$h^2/10$')
ax.loglog(h,(17/1680)*h**4,':',label=r'$17h^4/1680$')
ax.set_xlabel(r'Step $h$');ax.set_ylabel('Endpoint absolute error')
ax.set_title(r'Scalar oscillator: $T_h=12h^{-2}$',fontsize=11)
ax.grid(True,which='both',alpha=.2);ax.legend(ncol=2,loc='best')
compose('fig01_linear_secular',[save(fig,'fig01')],'single')

# 2. New certified repeats; family tangent and independently computed P.
pp=[]
for metric,lab,name in [('rho',r'$\|(I-\Pi_{\rm fam})e_h\|_N/\|e_h\|_N$','fig02a'),
                       ('chi',r'$\|e_h-P_h\|_N/\|e_h\|_N$','fig02b')]:
    fig,ax=single()
    for case,label in [('nls','NLS'),('kdv','KdV'),('sg',r'SG, $\omega=0.5$')]:
        r=read(ROOT/f'results/pilot_{case}/projection.csv');t=col(r,'t')
        v=col(r,'e_transversal')/col(r,'e_norm') if metric=='rho' else col(r,'e_minus_P_rel')
        ax.semilogy(t/t[-1],v,label=label)
    ax.set_xlabel(r'Normalized time $t/T_h$');ax.set_ylabel(lab)
    ax.set_xlim(0,1);ax.set_title('A. Family-transverse error' if metric=='rho' else 'B. First-corrector agreement',fontsize=11)
    ax.grid(True,which='both',alpha=.2);ax.legend(loc='best')
    pp.append(save(fig,name))
compose('fig02_first_order_projection',pp,'double')

# 3. New certified long NLS family in the phase-invariant complex L2 norm.
r=read(D/'long_family.csv');r.sort(key=lambda r:float(r['h']));h=col(r,'h')
fig,ax=single()
for m,lab,mk in [('MP2','MP2','o'),('ASC4','ASC4','s'),('RASC4','RASC4','D'),('PROC','A-priori MP2','^')]:
    ax.loglog(h,col(r,'err_'+m+'_l2'),marker=mk,markersize=5,label=lab)
ax.loglog(h,col(r,'err_RASC4_l2')[0]*(h/h[0])**2,'--',label=r'$h^2$ guide')
ax.set_xlabel(r'Step $h$');ax.set_ylabel(r'Endpoint relative error $E_2$')
ax.set_xticks(h,labels=[f'{v:g}' for v in h]);ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_title(r'A. $N=384$, $T_h\simeq(r_\phi^{(2)}h^2)^{-1}$',fontsize=10.4)
ax.grid(True,which='both',alpha=.2);ax.legend(loc='best',ncol=1)
p1=save(fig,'fig03a')
r=read(ROOT/'results/cert_N384_h004/history.csv');t=col(r,'t')
fig,ax=single()
for m,lab in [('MP2','MP2'),('ASC4','ASC4'),('RASC4','RASC4'),('PROC','A-priori MP2')]:
    ax.semilogy(t/t[-1],col(r,'err_'+m+'_l2'),label=lab)
ax.set_xlabel(r'Normalized time $t/T_h$');ax.set_ylabel(r'Sampled relative error $E_2$')
ax.set_xlim(0,1);ax.set_title(r'B. $h=0.04$, $T_h=950.48$',fontsize=11)
ax.grid(True,which='both',alpha=.2);ax.legend(loc='best')
p2=save(fig,'fig03b');compose('fig03_rasc4_long_family',[p1,p2],'double')

# 4. All plotted information comes from new N=384 scientific snapshots.
r=read(D/'w4_checkpoints.csv');t=col(r,'t');y=col(r,'norm_h4W4')
fig,ax=single()
ax.loglog(t,y,'o-',label=r'$h^4\|W_4\|_N$',markersize=6)
ax.loglog(t,col(r,'norm_h4_res_action_W2'),'x',label='After trial-space projection',markersize=8)
ax.loglog(t,y[0]*(t/t[0])**2,'--',label=r'$t^2$ guide')
p=float(np.polyfit(np.log(t),np.log(y),1)[0])
ax.text(.04,.94,f'Fit: p = {p:.6f}',transform=ax.transAxes,va='top',fontsize=9)
ax.set_xlabel(r'Time $t$');ax.set_ylabel('Scaled transverse norm')
ax.set_title('A. Full and residual norms',fontsize=11)
ax.grid(True,which='both',alpha=.2);ax.legend(loc='lower right',fontsize=8)
p1=save(fig,'fig04a')
sp=read(D/'w4_spectra.csv');fig,ax=single()
for tm in sorted(set(col(sp,'t'))):
    a=[v for v in sp if float(v['t'])==tm];k=col(a,'k');v=col(a,'fraction')
    sel=v>0
    ax.semilogy(k[sel],v[sel],label=f't = {tm:g}')
ax.axvline(16,linestyle='--',linewidth=1,label=r'$k_{\rm Nyq}=16$')
ax.set_xlim(0,16.3);ax.set_ylim(1e-12,.3)
ax.set_xlabel(r'Wavenumber $|k|$');ax.set_ylabel('Fraction of squared norm per mode')
ax.set_title('B. Snapshot spectra',fontsize=11);ax.legend(loc='lower right',fontsize=8)
ax.grid(True,which='both',alpha=.2);p2=save(fig,'fig04b')
r=read(D/'w4_pod.csv');fig,ax=single((4.5,3.15));xx=col(r,'mode');v=col(r,'squared_fraction')
ax.bar(xx,v,width=.65)
for x,yy in zip(xx,v):ax.text(x,yy+.009,f'{100*yy:.2f}%',ha='center',va='bottom',fontsize=10)
ax.set_xticks(xx);ax.set_ylim(0,.34)
ax.set_xlabel('POD mode');ax.set_ylabel('Squared singular-value fraction')
ax.set_title('C. Four normalized aligned snapshots',fontsize=10.6)
ax.grid(axis='y',alpha=.2);p3=save(fig,'fig04c')
compose('fig04_w4_obstruction',[p1,p2,p3],'three')
print('R3_FIGURES_PASS: four vector PDFs and previews; eight independent numerical panels.')
