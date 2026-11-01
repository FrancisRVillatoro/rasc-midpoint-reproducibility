Focused W4 diagnostic for the RASC6 NLS long-time experiment.

Place diagnose_rasc6_W4.py and diagnose_rasc6_W4.sbatch in the existing
~/fscratch/rasc6_nls_longtime_check directory.  They import the already
verified run_rasc6_nls_longtime.py and src/ tree read-only.

The scientific run is N=384, h=0.05, sigma=1.  It samples four times and
reports:
  - W4 growth exponent;
  - projection of W4 onto profile/action modes d_eta and d_c after removal
    of phase/translation;
  - additional fraction explained by the first transverse corrector W2;
  - spectral fractions |k|<=2, 2<|k|<=6, |k|>6 and top 10% of the cutoff;
  - physical core fraction |x-xc|<=6;
  - pairwise cosines and SVD/POD fractions of canonicalized W4 snapshots.

The run writes results_W4diag/ and does not modify the frozen source tree.
