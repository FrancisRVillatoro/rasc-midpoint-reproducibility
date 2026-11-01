# RASC midpoint — reproducibility artifact 1.0.0

**Study:** Renormalized asymptotic successive corrections for implicit midpoint:
secular modulation and long-time accuracy. **Author:** Francisco R. Villatoro,
Universidad de Málaga, ORCID 0000-0003-4314-6213.

This is the scientific code/data package, not the manuscript submission source.
It is prepared for a new GitHub/Zenodo release; no remote publication or DOI is
claimed. Its version is **1.0.0** in VERSION, CITATION.cff and .zenodo.json.
It must not replace the distinct, immutable ASC midpoint v1.0.1 artifact.

## What is supported

RASC4 applies a finite symmetry action inferred from the first fully discrete
ASC corrector and retains its transverse part additively. The artifact checks
fixed-time consistency and the mathematical components of a **conditional**
long-time estimate. The uniform nonlinear stability hypotheses of that estimate
are not established by the PDE experiments. The large NLS W4 field and its
approximately quadratic norm fit describe the recorded finite window, not a
universal asymptotic obstruction. Algebraic solver certificates are not rigorous
bounds on the global discretization error or the reference solution.

## Verify without dependencies or network

From the extracted root:

```bash
python3 verify_release.py
```

This stdlib-only check validates every listed SHA-256, safe paths, version
agreement and the basic public metadata. It does not execute a simulation.
`SHA256SUMS.txt` seals all payload files; the manifest itself is not recursively
hashed. New Git metadata and Python bytecode are not part of the payload.

## Rebuild the numerical evidence from saved states

Use Python 3.13 with the versions recorded in `requirements-tested.txt`, or
an independently validated compatible environment. These commands do not install
packages or access the network; no installation is attempted on Picasso.

```bash
python3 scripts/reproduce_from_saved.py --destination ../rasc_saved_rebuild
```

The destination **must not exist** and must be outside the sealed source tree.
This copies the inputs to a scratch directory, rebuilds derived CSVs and numerical
LaTeX table fragments, recreates the four vector PDF figures, recomputes all eight
saved-evidence checks, and executes the five R2 and four R1 regression scripts.
It also runs the exact scalar-cumulant, stationary NLS and SG controls and the
generic nonlinear kernel gate, including the H2 ablation.
The JNM SG control independently checks the dependence of the corrective
frequency on the initial phase section; the reported closed rate uses theta0=0.
Outputs and logs are written only in that new directory. PDF creation uses
matplotlib and PyMuPDF; LaTeX is not required for these figures. PDF metadata can
change between rebuilds, so equivalence is checked at the numerical/visual level,
not by requiring newly rendered PDFs to have identical bytes.

The main scientific entry points are:

| Entry point | Purpose |
|---|---|
| `scripts/build_evidence.py` | Derive all numerical tables from recorded rows and states |
| `scripts/make_figures.py` | Generate four figures and their individual panel sources |
| `scripts/check_evidence.py` | Eight independent saved-state/provenance/certificate checks |
| `regression/R1_R2/run_r2_regressions.py` | Five theory/control regressions |
| `regression/R1_R2/R1/run_r1_regressions.py` | Four invariant-normalization/exact regressions |
| `scripts/check_additional_controls.py` | Exact scalar/SG controls, independent stationary NLS check and kernel G1/H2 ablation |
| `regression/JNM/check_sg_initial_section.py` | Symbolic SG initial-section formula and independent quadrature at theta0=0 and pi/2 |

The low-level checks may write result logs in their working copy. Run them through
the scratch-copy entry point rather than altering the sealed artifact.

## Repeat the integrations

```bash
python3 scripts/reproduce_all_numerics.py --destination ../rasc_new_campaign
```

This is a substantially longer calculation. It repeats the 13 PDE configurations,
the spatial/symmetry controls, rate predictions and two DOP853 references, then
builds the evidence and runs regressions. It runs sequentially and forces one
thread per numerical library. It refuses an existing output directory, does not
modify the saved campaign, uses no network and aborts on failed acceptance checks.
On a cluster use this driver inside an allocated batch job, not on a login node.
Exact wall times and binary-identical PDE results across platforms are not promised.

## Layout and provenance

- `results/`: newly residual-checked campaign, metadata, all-step certificates,
  sampled states, four scientific W4 checkpoints and reference integrations.
- `archived/`: historical CSVs, the original input manifest and a separately named
  excluded N=128 smoke-test directory. Smoke tests do not enter scientific figures.
- `legacy/`: byte-preserved scientific source, including the kernel and PDE code.
- `source_archives/`: byte-preserved original source delivery archives, not install targets.
- `derived/`: calculated summaries/CSV values used by the figures and tables.
- `regression/`: symbolic and numerical R1/R2 checks with recorded outputs.
- `PROVENANCE.json`, `PROVENANCE.md`: figure/table-to-input map and hashes.
- `qa/`: packaging and regression summaries.

Per-run `metadata.json` distinguishes new residual-checked computations from
historical observations and identifies exact producer files. Some archived
filenames are approximate: the NLS pilot named h2e-02 records h=0.016 in its rows.
Do not infer numerical parameters from filenames. Preserve the distinction
between a neutral rate, a same-window response fit, a finite-step measured slope
and a two-step extrapolation.

NLS and KdV use padded Galerkin polynomial products (2N and 3N/2 respectively,
Nyquist removed). SG evaluates its sine nonlinearity on the grid. Real L2 projection
norms, relative complex L2 errors, complex-sup errors and historical component-sup
errors are separately named. The physical Parseval norm and the coefficient-space
norm for linear solve acceptance are not silently identified.

## Licensing and citation

Funding: This work was supported by Universidad de Málaga, project
PPRO-B4-2026-004 (Plan Propio de Investigación, Transferencia y Divulgación
Científica).

Code is MIT; original data, documentation and figure outputs are CC BY 4.0.
See LICENSE, DATA_LICENSE and NOTICE.md. Citation metadata is in CITATION.cff;
Zenodo metadata is in .zenodo.json. The latter explicitly fixes version 1.0.0.
No self-DOI or uncreated repository URL is invented. After a real deposit, add the
assigned identifier in a subsequent metadata workflow without moving an existing
tag. The earlier ASC artifact DOI 10.5281/zenodo.23048130 is only provenance.

## Scope of verification

The recorded R3 bank contains 161,824 accepted base stages and 271,595 corrector
solves across 13 runs, plus independent references and separate controls. R4
rechecks saved evidence and regressions; it does **not** claim that those lengthy
integrations were repeated again during publication preparation. The raw inputs,
producer code and results are preserved byte for byte. No manuscript text or
manuscript PDF is distributed in this release.
