# Secular drift rates of MP2 predicted from G2

Rates are d/dt of the coefficient of the scaled error e = (u - U)/h^2 on each
tangent mode; the physical drift of the numerical solution is h^2 times the
rate, with the opposite sign (U lags where the rate is positive).

| benchmark | mode | closed form | derivation | check |
|---|---|---|---|---|
| KdV soliton, whole line | translation (d_xi phi) | -704 kappa^8 / 45 | symbolic, biorthogonal projection | L -> infinity extrapolation of the Riesz projection: -15.64446 |
| KdV soliton, periodic L=64 | translation | -15.736200716846 (kappa=1) | Riesz projection, dense operator | measured -15.741563 (h->0); fitted full trajectory -15.7415576 |
| NLS bright soliton | phase (i u) | -(c^2+4 eta^2)(c^4+8 c^2 eta^2-112 eta^4)/768 | symbolic, symplectic projection | 515/768 at eta=c=1, numeric to 3.7e-15 |
| NLS bright soliton | position (d_x0 u) | c (c^2+4 eta^2)(3 c^2-4 eta^2)/192 | symbolic | -5/192 at eta=c=1, numeric to 1.1e-15 |
| NLS bright soliton | amplitude, boost | 0 | mass and momentum conserved by MP2 | |
| SG breather | phase (d_theta Y) | (6 w^6 - 4 w^4 + 5 w^2 - 4)/(36 w) | numeric Floquet-symplectic formula at 9 frequencies, exact through 5 rationals (overdetermined) | limit w -> 1 is 1/12, the linear-oscillator value |
| SG breather | frequency | 0 | energy not drifted by G2 | |

The KdV periodic-domain value differs from the whole-line one by the mass
mode, which exists only with periodic boundary conditions and couples to
translation through Galilean invariance; the fit a + b/L + c/L^2 over
L = 64, 96, 128, 192 has residual 7.6e-7.

The symbolic G2 agrees with the discrete frozen kernel to 3.4e-10 (KdV) and
5.9e-11 (NLS) on the lowest third of the spectrum; the full-spectrum
difference (4.4e-5, 9.3e-8) is round-off amplified by the 9 and 6 spatial
derivatives inside G2, confined to modes orthogonal to the soliton family.
