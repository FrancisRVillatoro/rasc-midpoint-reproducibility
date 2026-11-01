# JNM pre-release technical review — 29 October 2026

This technical review complements the manuscript peer review. It validates the
actual local release candidate; it does not assert remote publication or a DOI.

## Findings and changes

- The retrieved candidate already contained JNM A/B/C panel labels. All four
  vector figures match the submitted JNM figures exactly when rendered at 144 dpi.
  No additional figure relabelling was needed.
- The confirmed funding declaration, Universidad de Málaga project
  PPRO-B4-2026-004, was absent from public metadata. It is now in README,
  CITATION.cff and the Zenodo description. The package remains unpublished 1.0.0.
- Existing exact scalar/SG/stationary-NLS controls and the frozen kernel G1/H2
  gate were present but omitted from both public reproduction drivers. A new
  wrapper now runs them, checks finite values and applies a 1e-6 relative SG
  quadrature threshold. The raw checks are preserved unchanged.
- A new independent SG initial-section regression checks the manuscript's
  corrected scope: the reported closed rate selects theta0=0. Symbolic algebra
  and direct 45-digit spatial quadrature also verify the theta0=pi/2 section.
- The publication guard now rejects the known JNM manuscript and cover-letter
  filenames as well as the earlier R3/R4 names.

## Executed validation

A fresh, external scratch copy completed evidence reconstruction, four figure
regenerations, eight saved-state checks, five R2 regressions, four R1 regressions,
and all three additional control programs. Twelve derived files are byte-identical
to the candidate. Four figures render identically to both the archived outputs
and the JNM submission. The funding metadata parses as JSON/YAML.

Kernel evidence: symbolic dG2/ddG2/H2 relative discrepancy 1.31e-15; local defect
ratios approach 32 and 128; observed ASC orders 2.000, 4.000 and 5.993/6.001.
Removing H2 reduces the observed sixth-level order to 3.999/4.000.

The original archived inputs, recorded runs, complete states, solver certificates,
scientific sources and source archives remain byte-identical. The source archives
contain no TeX or PDF files. Detailed comparisons, environment and check results
are retained in qa/PRERELEASE_20261029.json; the additional-control logs accompany it.
Earlier QA reports retain their original historical scope and counts.

## Scope

No long PDE integrations were repeated during this pre-release check. The checks
validate recorded evidence and mathematical/kernel components. They do not prove
the conditional uniform nonlinear long-time stability assumptions, rigorous global
reference accuracy, or a universal asymptotic W4 obstruction. No manuscript or
cover letter belongs to this public scientific artifact.
