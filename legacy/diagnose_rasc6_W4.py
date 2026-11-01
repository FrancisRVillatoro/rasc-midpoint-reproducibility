#!/usr/bin/env python3
"""Focused diagnostic of the secular transverse second cumulant W4.

Run from the existing rasc6_nls_longtime_check directory.  The script imports
that directory's frozen source tree and reruns one NLS bright-soliton case.
It does not modify the frozen source files.

Default scientific case: N=384, h=0.05, sigma=1, with
T=sigma/(h^2*(505/768)).  Diagnostics are sampled at 1/4, 1/2, 3/4, and 1
of the final time.

Outputs:
  w4_diagnostic.csv      checkpoint diagnostics
  w4_snapshot_*.npz      final/checkpoint spectral arrays
  w4_direction_gram.csv  pairwise cosines of canonicalized W4 snapshots
  w4_svd.csv             POD/SVD fractions of normalized W4 snapshots
  w4_summary.txt         compact interpretation data
"""
import argparse
import csv
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_rasc6_nls_longtime as R  # noqa: E402


def l2_inner(pde, A, B):
    with R.S.no_count():
        a = pde.grid.to_phys(A).ravel()
        b = pde.grid.to_phys(B).ravel()
    return float(pde.grid.dx * np.dot(a, b))


def project_basis(pde, V, basis):
    """L2 orthogonal projection of V onto span(basis)."""
    if len(basis) == 0:
        return np.empty(0), np.zeros_like(V), V.copy(), float('nan')
    with R.S.no_count():
        B = np.array([pde.grid.to_phys(g).ravel() for g in basis])
        v = pde.grid.to_phys(V).ravel()
    dx = pde.grid.dx
    gram = dx * (B @ B.T)
    rhs = dx * (B @ v)
    coeff, *_ = np.linalg.lstsq(gram, rhs, rcond=1e-12)
    proj = np.zeros_like(V)
    for c, g in zip(coeff, basis):
        proj += c * g
    res = V - proj
    return coeff, proj, res, float(np.linalg.cond(gram))


def center_from_state(pde, U):
    """Circular center of mass of |u|^2 on the periodic interval."""
    with R.S.no_count():
        up = pde.grid.to_phys(U)
    rho = up[0] ** 2 + up[1] ** 2
    a, L, x = pde.grid.a, pde.grid.L, pde.grid.x
    ang = 2.0 * np.pi * (x - a) / L
    z = np.sum(rho * np.exp(1j * ang))
    phi = np.mod(np.angle(z), 2.0 * np.pi)
    return float(a + L * phi / (2.0 * np.pi))


def family_shape_modes(pde, U):
    """Profile derivatives d_eta phi and d_c phi at fixed center/group phase.

    The formulas are evaluated using the numerical U itself, so they are
    covariant under the accumulated numerical phase.  The center is obtained
    from |U|^2.
    """
    eta = float(pde.eta)
    xc = center_from_state(pde, U)
    x = pde.grid.x
    xi = R.wrap(x - xc, pde.grid.L)
    with R.S.no_count():
        up = pde.grid.to_phys(U)
    fac_eta = 1.0 / eta - xi * np.tanh(eta * xi)
    deta_p = up * fac_eta[None, :]
    dc_p = 0.5 * xi[None, :] * np.stack([-up[1], up[0]])
    deta = pde.grid.to_spec(deta_p)
    dc = pde.grid.to_spec(dc_p)
    return deta, dc, xc


def spectral_fractions(pde, V):
    """Squared-L2 spectral fractions, with rfft multiplicity weights."""
    k = pde.grid.k
    e = np.sum(np.abs(V) ** 2, axis=0)
    w = np.full_like(k, 2.0, dtype=float)
    w[0] = 1.0
    if len(w) > 1:
        w[-1] = 0.0  # projected Nyquist is identically zero
    e = w * e
    total = float(np.sum(e))
    if total == 0.0:
        return dict(frac_k_le_2=np.nan, frac_2_6=np.nan,
                    frac_k_gt_6=np.nan, frac_top10=np.nan,
                    k_peak=np.nan)
    low = float(np.sum(e[k <= 2.0])) / total
    mid = float(np.sum(e[(k > 2.0) & (k <= 6.0)])) / total
    high = float(np.sum(e[k > 6.0])) / total
    top = float(np.sum(e[k >= 0.9 * np.max(k)])) / total
    kp = float(k[int(np.argmax(e))])
    return dict(frac_k_le_2=low, frac_2_6=mid,
                frac_k_gt_6=high, frac_top10=top, k_peak=kp)


def spatial_core_fraction(pde, V, xc, radius=6.0):
    with R.S.no_count():
        vp = pde.grid.to_phys(V)
    xi = R.wrap(pde.grid.x - xc, pde.grid.L)
    dens = np.sum(vp * vp, axis=0)
    tot = float(np.sum(dens))
    if tot == 0.0:
        return np.nan
    return float(np.sum(dens[np.abs(xi) <= radius]) / tot)


def analyze_snapshot(pde, U, P, RR, h, t):
    # The exact same first/second cumulant construction used in the final check.
    a2, W2, cond_tan = R.split(pde, U, P)
    A2P = R.apply_Aop(pde, P, a2)
    A2sqU = R.apply_Aop(pde, R.apply_Aop(pde, U, a2), a2)
    C4 = RR - A2P + 0.5 * A2sqU
    a4, W4, cond_c4 = R.split(pde, U, C4)

    # Action/profile directions after removal of phase and translation.
    deta, dc, xc = family_shape_modes(pde, U)
    tan_basis = R.generators(pde, U)
    _, _, deta_perp, _ = project_basis(pde, deta, tan_basis)
    _, _, dc_perp, _ = project_basis(pde, dc, tan_basis)

    coeff_ac, proj_ac, res_ac, cond_ac = project_basis(
        pde, W4, [deta_perp, dc_perp]
    )
    nW4 = R.l2norm(pde, W4)
    nres_ac = R.l2norm(pde, res_ac)
    frac_ac = 0.0 if nW4 == 0 else max(0.0, 1.0 - (nres_ac / nW4) ** 2)

    # Does the first-order transverse shape W2 itself explain W4?
    _, _, W2_perp_ac, _ = project_basis(pde, W2, [deta_perp, dc_perp])
    basis3 = [deta_perp, dc_perp]
    if R.l2norm(pde, W2_perp_ac) > 1e-13:
        basis3.append(W2_perp_ac)
    coeff_3, proj_3, res_3, cond_3 = project_basis(pde, W4, basis3)
    nres3 = R.l2norm(pde, res_3)
    frac3 = 0.0 if nW4 == 0 else max(0.0, 1.0 - (nres3 / nW4) ** 2)

    h2, h4 = h * h, h ** 4
    spec = spectral_fractions(pde, W4)
    spec_res = spectral_fractions(pde, res_3)

    # Canonicalize W4 to compare directions at different times.  First-order
    # RASC alignment plus removal of the physical soliton motion.
    theta_phys = -(pde.eta ** 2 + 0.25 * pde.c ** 2) * t
    delta_phys = pde.c * t
    W4_can = R.group(pde, W4,
                     delta_phys + h2 * a2[0],
                     theta_phys + h2 * a2[1])

    row = {
        't': float(t),
        'h': float(h),
        'N': int(pde.grid.N),
        'a2_x': float(a2[0]),
        'a2_phi': float(a2[1]),
        'h2a2_x': float(h2 * a2[0]),
        'h2a2_phi': float(h2 * a2[1]),
        'a4_x': float(a4[0]),
        'a4_phi': float(a4[1]),
        'h4a4_x': float(h4 * a4[0]),
        'h4a4_phi': float(h4 * a4[1]),
        'norm_W2': R.l2norm(pde, W2),
        'norm_h2W2': h2 * R.l2norm(pde, W2),
        'norm_W4': nW4,
        'norm_h4W4': h4 * nW4,
        'norm_h4_res_action': h4 * nres_ac,
        'norm_h4_res_action_W2': h4 * nres3,
        'frac_action_sq': frac_ac,
        'frac_action_plus_W2_sq': frac3,
        'coef_eta_W4': float(coeff_ac[0]) if len(coeff_ac) > 0 else np.nan,
        'coef_c_W4': float(coeff_ac[1]) if len(coeff_ac) > 1 else np.nan,
        'h4coef_eta_W4': float(h4 * coeff_ac[0]) if len(coeff_ac) > 0 else np.nan,
        'h4coef_c_W4': float(h4 * coeff_ac[1]) if len(coeff_ac) > 1 else np.nan,
        'cond_tangent': cond_tan,
        'cond_C4_tangent': cond_c4,
        'cond_action': cond_ac,
        'cond_action_plus_W2': cond_3,
        'center_x': xc,
        'frac_spatial_core6': spatial_core_fraction(pde, W4, xc, 6.0),
        **spec,
        'res_frac_k_le_2': spec_res['frac_k_le_2'],
        'res_frac_2_6': spec_res['frac_2_6'],
        'res_frac_k_gt_6': spec_res['frac_k_gt_6'],
        'res_frac_top10': spec_res['frac_top10'],
        'res_k_peak': spec_res['k_peak'],
    }
    arrays = dict(U=U, P=P, R=RR, C4=C4, W2=W2, W4=W4,
                  deta_perp=deta_perp, dc_perp=dc_perp,
                  proj_action=proj_ac, res_action=res_ac,
                  proj_action_W2=proj_3, res_action_W2=res_3,
                  W4_canonical=W4_can)
    return row, arrays


def run(N=384, h=0.05, sigma=1.0, out='results_W4diag', smoke=False):
    pde = R.nls.make('bright', N=N)
    y0 = pde.grid.to_spec(R.state_phys(pde, 0.0))
    if smoke:
        nsteps = 40
        T = nsteps * h
        targets = [10, 20, 30, 40]
    else:
        T_target = sigma / (h * h * R.RATE_PHI)
        nsteps = int(round(T_target / h))
        T = nsteps * h
        targets = sorted(set([max(1, int(round(q * nsteps))) for q in (0.25, 0.5, 0.75, 1.0)]))

    outdir = pathlib.Path(out)
    outdir.mkdir(parents=True, exist_ok=True)

    y = y0.copy()
    P = np.zeros_like(y)
    RR = np.zeros_like(y)
    log = R.MP.Stats()
    rows = []
    can_vectors = []
    times = []

    ti = set(targets)
    for n in range(nsteps):
        y = R.MP.step(y, h, pde, tol=R.TOL, log=log)
        P, RR, _ = R.asc.sweep_step(pde, log.stage, h, P, RR, use_H2=True)
        step = n + 1
        if step in ti:
            t = step * h
            row, arrays = analyze_snapshot(pde, y, P, RR, h, t)
            rows.append(row)
            times.append(t)
            with R.S.no_count():
                vp = pde.grid.to_phys(arrays['W4_canonical']).ravel()
            nv = math.sqrt(pde.grid.dx * float(np.dot(vp, vp)))
            can_vectors.append(vp / nv if nv > 0 else vp)
            np.savez_compressed(outdir / f'w4_snapshot_t{t:.6f}.npz',
                                t=t, h=h, N=N, k=pde.grid.k,
                                **arrays)
            print('CHECKPOINT', json.dumps(row, sort_keys=True), flush=True)
        elif step % max(1, nsteps // 10) == 0:
            print(f'{step}/{nsteps} ({100*step/nsteps:.0f}%)', flush=True)

    csv_path = outdir / 'w4_diagnostic.csv'
    with csv_path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    V = np.array(can_vectors)
    G = pde.grid.dx * (V @ V.T)
    # V vectors have unit L2 norm, so G is a cosine Gram matrix.
    with (outdir / 'w4_direction_gram.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['t'] + [f'{t:.12g}' for t in times])
        for t, rr in zip(times, G):
            w.writerow([f'{t:.12g}'] + [f'{x:.16e}' for x in rr])

    # SVD/POD of equally weighted normalized canonical snapshots.
    s = np.linalg.svd(V, compute_uv=False)
    frac = (s * s) / np.sum(s * s)
    with (outdir / 'w4_svd.csv').open('w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['mode', 'singular_value', 'fraction'])
        for i, (sv, fr) in enumerate(zip(s, frac), 1):
            w.writerow([i, f'{sv:.16e}', f'{fr:.16e}'])

    # Time-growth fits on checkpoint data.
    tt = np.array([r['t'] for r in rows])
    nw = np.array([r['norm_W4'] for r in rows])
    nr = np.array([r['norm_h4_res_action_W2'] / (h ** 4) for r in rows])
    pW = float(np.polyfit(np.log(tt), np.log(nw), 1)[0]) if np.all(nw > 0) else np.nan
    pR = float(np.polyfit(np.log(tt), np.log(nr), 1)[0]) if np.all(nr > 0) else np.nan
    last = rows[-1]
    summary = [
        f'N={N} h={h:.16g} T={T:.16g} nsteps={nsteps}',
        f'growth_exponent_W4={pW:.10f}',
        f'growth_exponent_residual_after_action_W2={pR:.10f}',
        f'final_norm_h4W4={last["norm_h4W4"]:.16e}',
        f'final_frac_action_sq={last["frac_action_sq"]:.16e}',
        f'final_frac_action_plus_W2_sq={last["frac_action_plus_W2_sq"]:.16e}',
        f'final_frac_k_le_2={last["frac_k_le_2"]:.16e}',
        f'final_frac_2_6={last["frac_2_6"]:.16e}',
        f'final_frac_k_gt_6={last["frac_k_gt_6"]:.16e}',
        f'final_frac_top10={last["frac_top10"]:.16e}',
        f'final_k_peak={last["k_peak"]:.16e}',
        f'final_res_frac_k_gt_6={last["res_frac_k_gt_6"]:.16e}',
        f'final_res_k_peak={last["res_k_peak"]:.16e}',
        'svd_fractions=' + ','.join(f'{x:.10e}' for x in frac),
    ]
    (outdir / 'w4_summary.txt').write_text('\n'.join(summary) + '\n')
    print('\n'.join(summary), flush=True)
    print(f'wrote {csv_path}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=384)
    ap.add_argument('--h', type=float, default=0.05)
    ap.add_argument('--sigma', type=float, default=1.0)
    ap.add_argument('--out', default='results_W4diag')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    run(a.N, a.h, a.sigma, a.out, a.smoke)


if __name__ == '__main__':
    main()
