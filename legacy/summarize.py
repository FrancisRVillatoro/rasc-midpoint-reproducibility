"""Summarize the projection experiment from results/proj_*.csv.

Growth exponents are least-squares slopes of log(norm) against log(t) over
t >= T/4.  For the breathers the norms are modulated within each period,
because the phase mode is much larger at the zero crossings of u than at the
turning points, so their exponents are fitted on the per-period maxima.
Coefficient rates are least-squares slopes of the coefficient against t over
the same window; a coefficient is reported as bounded when its linear fit
leaves a relative residual above 5 per cent, i.e. when it is not a line.
"""
import csv
import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RES, FIG = HERE / 'results', HERE / 'figures'

RUNS = [('nls', 'proj_nls_h2e-02.csv', 0.016, None),
        ('nls', 'proj_nls_h8e-03.csv', 0.008, None),
        ('kdv', 'proj_kdv_h4e-03.csv', 0.004, None),
        ('kdv', 'proj_kdv_h2e-03.csv', 0.002, None),
        ('kdv', 'proj_kdv_h1e-03.csv', 0.001, None),
        ('sg_breather_w0.5', 'proj_sg_breather_w0.5_h2e-02.csv', 0.02, 0.5),
        ('sg_breather_w0.5', 'proj_sg_breather_w0.5_h1e-02.csv', 0.01, 0.5)]


def load(f):
    return list(csv.DictReader(open(RES / f)))


def col(r, k):
    return np.array([float(x[k]) for x in r])


def per_period_max(t, v, period):
    k = np.floor(t / period + 1e-9).astype(int)
    tt, vv = [], []
    for j in np.unique(k):
        m = k == j
        if m.sum() > 3:
            i = np.argmax(v[m])
            tt.append(t[m][i])
            vv.append(v[m][i])
    return np.array(tt), np.array(vv)


def exponent(t, v, period=None):
    if period is not None:
        t, v = per_period_max(t, v, period)
    m = t >= t[-1] / 4
    return float(np.polyfit(np.log(t[m]), np.log(v[m]), 1)[0])


def main():
    FIG.mkdir(exist_ok=True)
    out = []
    for bench, f, h, om in RUNS:
        r = load(f)
        t = col(r, 't')
        period = 2 * np.pi / om if om else None
        modes = [k[len('e_coef_'):] for k in r[0] if k.startswith('e_coef_')]
        row = dict(benchmark=bench, h=h, T=f"{t[-1]:.2f}",
                   e_norm_final=f"{float(r[-1]['e_norm']):.4e}",
                   tangent_exponent=f"{exponent(t, col(r, 'e_tangent'), period):.3f}",
                   transversal_exponent=f"{exponent(t, col(r, 'e_transversal'), period):.3f}",
                   transversal_fraction_final=f"{float(r[-1]['e_transversal']) / float(r[-1]['e_norm']):.2e}",
                   P_transversal_exponent=f"{exponent(t, col(r, 'P_transversal'), period):.3f}",
                   e_minus_P_rel_final=r[-1]['e_minus_P_rel'],
                   transversal_Ubasis_over_exact=f"{float(r[-1]['e_transversal_Ubasis']) / float(r[-1]['e_transversal']):.3f}")
        m = t >= t[-1] / 4
        for k in modes:
            c = col(r, 'e_coef_' + k)
            b, a = np.polyfit(t[m], c[m], 1)
            dev = np.max(np.abs(c[m] - (a + b * t[m]))) / max(np.max(np.abs(c[m])), 1e-300)
            row[f'{k}_rate'] = f"{b:+.5e}" if dev < 0.05 else 'bounded'
            row[f'{k}_maxabs'] = f"{np.max(np.abs(c)):.3e}"
        out.append(row)
    keys = []
    for row in out:
        keys += [k for k in row if k not in keys]
    with open(RES / 'summary.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval='')
        w.writeheader()
        w.writerows(out)
    for row in out:
        print(row)

    # figures: finest h of each benchmark, one file per panel, same format
    finest = {}
    for bench, f, h, om in RUNS:
        if bench not in finest or h < finest[bench][1]:
            finest[bench] = (f, h, om)
    for bench, (f, h, om) in finest.items():
        r = load(f)
        t = col(r, 't')
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
        ax.loglog(t, col(r, 'e_norm'), 'k-', lw=1.2, label=r'$\|e\|$')
        ax.loglog(t, col(r, 'e_tangent'), 'C0--', lw=1.2, label='tangent part')
        ax.loglog(t, col(r, 'e_transversal'), 'C3-', lw=1.2, label='transversal part')
        ax.loglog(t, col(r, 'P_transversal'), 'C3:', lw=1.2,
                  label=r'transversal part of $P$')
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'$L^2$ norm, scaled by $h^{-2}$')
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / f'norms_{bench}.png', dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.0, 3.6))
        for k in [k[len('e_coef_'):] for k in r[0] if k.startswith('e_coef_')]:
            ax.plot(t, col(r, 'e_coef_' + k), lw=1.2, label=k)
        ax.set_xlabel(r'$t$')
        ax.set_ylabel('tangent coefficient')
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / f'coefficients_{bench}.png', dpi=160)
        plt.close(fig)

    # the contract breather: error / h^2 at one period, to document why it
    # is outside the asymptotic regime at every step size tried
    print("\nomega = 0.1 breather, MP2 sup error / h^2 at t = one period:")
    per = 2 * np.pi / 0.1
    rows = []
    for f, h in (('proj_sg_breather_h1e-01.csv', 0.1),
                 ('proj_sg_breather_h5e-02.csv', 0.05),
                 ('proj_sg_breather_h3e-02.csv', 0.025),
                 ('proj_sg_breather_h1e-02.csv', 0.0125)):
        r = load(f)
        t, e = col(r, 't'), col(r, 'err_mp2_sup')
        i = np.argmin(abs(t - per))
        j = np.argmin(abs(t - per / 2))
        rows.append(dict(h=h, err_half_period=f"{e[j]:.3e}",
                         err_one_period=f"{e[i]:.3e}",
                         err_over_h2_one_period=f"{e[i] / h ** 2:.3e}"))
        print(f"  h={h}: err(T/2) {e[j]:.3e}  err(T) {e[i]:.3e}  err(T)/h^2 {e[i] / h ** 2:.3e}")
    with open(RES / 'summary_breather_w0.1.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
