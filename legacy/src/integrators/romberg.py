"""Integrator module 'romberg'. Interfaces fixed in CLAUDE.md.

midpoint: step(y,h,pde,tol=1e-13) -> y1, preconditioned Newton, iteration
          counts logged.
asc     : sequential sweeps calling ONLY kernel.formulas for G2, G4, H2,
          R_source; reconstructions U, U+h^2 P, U+h^2 P+h^4 R.
lawson  : midpoint on the Lawson-transformed system (autonomized).
gauss   : s=2 and s=3 collocation (orders 4, 6).
yoshida : triple-jump compositions of midpoint (orders 4, 6).
romberg : extrapolation of midpoint (orders 4, 6) by rerunning at h, h/2,
          h/4.
etdrk4  : Cox-Matthews with Kassam-Trefethen contour phi-functions.
"""
raise NotImplementedError("phase F2/F4")
