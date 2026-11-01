# R3 evidence map

All new runs are distinct from the historical measurements.

## Figure 1

Type: archived scalar algebra

Producer: `scripts/make_figures.py`

- `archived/integrated/objective4_linear.csv`

## Figure 2 and Table 1

Type: new certified pilot repetitions

Producer: `scripts/build_evidence.py; scripts/make_figures.py`

- `results/pilot_nls/projection.csv`
- `results/pilot_kdv/projection.csv`
- `results/pilot_sg/projection.csv`

## Table 2

Type: archived measurements with explicit reconstruction of extrapolation; new prediction control

Producer: `scripts/build_evidence.py`

- `archived/projection/closure.csv`
- `archived/projection/proj_nls_h8e-03.csv`
- `archived/projection/proj_nls_h2e-02.csv`
- `archived/projection/proj_kdv_h1e-03.csv`
- `archived/projection/proj_kdv_h2e-03.csv`
- `archived/projection/proj_sg_breather_w0.5_h1e-02.csv`
- `archived/projection/proj_sg_breather_w0.5_h2e-02.csv`
- `results/rate_predictions/rates.csv`

## Table 3

Type: new certified fixed-time repetitions

Producer: `scripts/build_evidence.py`

- `results/fixed_N768_h004/history.csv`
- `results/fixed_N768_h002/history.csv`
- `results/fixed_N768_h001/history.csv`

## Figure 3, Tables 4--6

Type: new certified secular family

Producer: `scripts/build_evidence.py; scripts/make_figures.py`

- `results/cert_N384_h005/history.csv`
- `results/cert_N384_h004/history.csv`
- `results/cert_N384_h0032/history.csv`
- `results/cert_N384_h005/all_step_certificates.npz`
- `results/cert_N384_h004/all_step_certificates.npz`
- `results/cert_N384_h0032/all_step_certificates.npz`

## Figure 4

Type: new scientific snapshots N384, not smoke

Producer: `scripts/build_evidence.py; scripts/make_figures.py`

- `results/cert_N384_h005/w4_snapshot_t152.100000.npz`
- `results/cert_N384_h005/w4_snapshot_t304.150000.npz`
- `results/cert_N384_h005/w4_snapshot_t456.200000.npz`
- `results/cert_N384_h005/w4_snapshot_t608.300000.npz`

## Spatial/reference controls

Type: new independent checks; not rigorous global bounds

Producer: `scripts/check_spatial_model.py; scripts/reference_control.py; scripts/build_evidence.py`

- `results/spatial_model_controls.json`
- `results/reference_coarse/reference_states.npz`
- `results/reference_fine/reference_states.npz`

## Tolerance/resolution field comparisons

Type: new complete-field comparisons

Producer: `scripts/build_evidence.py`

- `results/cert_N768_h005/w4_snapshot_t152.100000.npz`
- `results/cert_N768_h005/w4_snapshot_t304.150000.npz`
- `results/cert_N768_h005/w4_snapshot_t456.200000.npz`
- `results/cert_N768_h005/w4_snapshot_t608.300000.npz`
- `results/cert_N384_h005_inner13/w4_snapshot_t152.100000.npz`
- `results/cert_N384_h005_inner13/w4_snapshot_t304.150000.npz`
- `results/cert_N384_h005_inner13/w4_snapshot_t456.200000.npz`
- `results/cert_N384_h005_inner13/w4_snapshot_t608.300000.npz`

Input SHA-256 values are in PROVENANCE.json; the bundle manifest covers all files.
