"""Apples-to-apples closure of the prediction against the measurement.

The measured rate is a least-squares slope over t >= T/4 of a coefficient that
also carries the bounded X1 response.  The linear model predicts the WHOLE
trajectory, not only its slope,
    e~(t) = -t Pi0 g - (exp(Lop t) - I) Lop^D (I - Pi0) g,
with Lop^D the inverse of Lop on X1, evaluated in the eigenbasis of X1
(diagonalizable, spectrum on the imaginary axis).  Applying the SAME fit to
the predicted trajectory at the SAME output times isolates any real
disagreement from the bias of the fit itself.
"""
import csv, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import predict as PD
from pdes import _spectral as S
from fractions import Fraction

RES = HERE / 'results'

def load(f):
    r = list(csv.DictReader(open(RES / f)))
    return r, np.array([float(x['t']) for x in r])

def slope(t, c, T):
    m = t >= T / 4
    return float(np.polyfit(t[m], c[m], 1)[0])

rows = []
for name, finest in (('kdv', 'proj_kdv_h1e-03.csv'), ('nls', 'proj_nls_h8e-03.csv')):
    pred, diag, X = PD.predict_relative_equilibrium(name, internals=True)
    M, g, p, lam, rho = X['M'], X['g'], X['p'], X['lam'], X['rho']
    q1 = g - p
    w, V = np.linalg.eig(M)
    sel = np.abs(w) > rho
    V1, w1 = V[:, sel], w[sel]
    c, *_ = np.linalg.lstsq(V1, q1.astype(complex), rcond=None)
    lsq_res = np.linalg.norm(V1 @ c - q1) / np.linalg.norm(q1)
    cd = c / w1
    r, t = load(finest)
    T = t[-1]
    pde, dx, md = X['pde'], X['pde'].grid.dx, X['modes']
    B = np.array([b.ravel() for b in md.values()])
    G = dx * B @ B.T
    cpred = []
    for tt in t:
        et = -tt * p - np.real(V1 @ (cd * (np.exp(w1 * tt) - 1.0)))
        with S.no_count():
            ph = pde.grid.to_phys(X['from_real'](et))
        cpred.append(np.linalg.solve(G, dx * B @ ph.ravel()))
    cpred = np.array(cpred)
    print(f"--- {name}: residuo de q1 en la base de X1 = {lsq_res:.2e}")
    for i, k in enumerate(md.keys()):
        cm = np.array([float(x['e_coef_' + k]) for x in r])
        sm, sp = slope(t, cm, T), slope(t, cpred[:, i], T)
        traj = np.max(np.abs(cm - cpred[:, i])) / max(np.max(np.abs(cm)), 1e-300)
        print(f"    {k:12s} tasa -Pi0g {pred[k]:+.8e} | pendiente ajustada: prediccion completa {sp:+.8e}, "
              f"medida {sm:+.8e}, dif.rel {abs(sp-sm)/max(abs(sm),1e-12):.2e} | trayectoria max|dif|/max {traj:.2e}")
        rows.append(dict(benchmark=name, mode=k, rate_Pi0g=f"{pred[k]:.9e}",
                         fitted_slope_predicted_trajectory=f"{sp:.9e}",
                         fitted_slope_measured=f"{sm:.9e}",
                         rel_diff_fitted=f"{abs(sp-sm)/max(abs(sm),1e-12):.3e}",
                         trajectory_max_rel_diff=f"{traj:.3e}"))

# breather: sensitivity of the measured slope to the fit window
print("--- breather w=0.5: pendiente medida de la fase segun la ventana de ajuste (h=0.01)")
r, t = load('proj_sg_breather_w0.5_h1e-02.csv'); T = t[-1]
c = np.array([float(x['e_coef_phase']) for x in r])
for lo in (0.0, 0.25, 0.5, 0.75):
    m = t >= lo * T
    print(f"    t >= {lo:.2f} T : {np.polyfit(t[m], c[m], 1)[0]:+.8e}")

# rational identification of the predicted rates
print("--- identificacion racional de las tasas predichas (limit_denominator 10^4)")
pr = list(csv.DictReader(open(RES / 'prediction_vs_measurement.csv')))
for x in pr:
    v = float(x['predicted_rate'])
    if abs(v) < 1e-10:
        continue
    fr = Fraction(v).limit_denominator(10000)
    print(f"    {x['benchmark']:17s} {x['mode']:12s} {v:+.12e}  ~ {fr}  (resto {abs(v-float(fr)):.1e})")

with open(RES / 'closure.csv', 'w', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    wr.writeheader(); wr.writerows(rows)
