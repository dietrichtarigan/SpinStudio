"""Physics check: simulated wavenumber vs. analytic Kalinikos-Slavin dispersion.

A line antenna excites plane spin waves along x in a YIG film (B0 along y,
Damon-Eshbach geometry). The wavenumber is measured from the spatial phase of
the steady-state DFT at the drive frequency and compared with theory.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spinstudio.config import Config, Grid, Source, Study  # noqa: E402
from spinstudio.solver import Model, kalinikos_slavin  # noqa: E402

print(f"{'f (GHz)':>8} {'k sim':>9} {'k theory':>9} {'error':>7}   (rad/um)")
for f in (3.6, 3.8, 4.0, 4.1):
    cfg = Config(
        grid=Grid(nx=600, ny=128, dx_nm=50, dy_nm=50, dz_nm=20),
        sources=[Source(x0=40, y0=0, x1=40, y1=127, component="z", f_GHz=f, amp_mT=0.1, ramp_ns=2)],
        study=Study(steps=6000, dt_ps=20, save_every=200, normalize=True),
    )
    cfg.boundary.width = 20
    res = Model(cfg, "cuda").simulate()
    # unwrap the phase along the center line, only where the wave has clearly arrived
    row = res.dft[2, :, 64]
    amp = np.abs(row)
    i0 = 45
    i1 = i0 + int(np.argmax(amp[i0:] < 0.3 * amp[i0:i0 + 20].max()))
    i1 = min(i1 if i1 > i0 + 10 else 500, 500)
    ph = np.unwrap(np.angle(row[i0:i1]))
    x = np.arange(i0, i1) * cfg.grid.dx_nm * 1e-9
    k_sim = abs(np.polyfit(x, ph, 1)[0])
    kk = np.linspace(1e4, 6e7, 200000)
    fk = kalinikos_slavin(cfg, kk, 0.0)
    k_th = kk[np.argmin(np.abs(fk - f * 1e9))]
    print(f"{f:8.2f} {k_sim*1e-6:9.3f} {k_th*1e-6:9.3f} {100*(k_sim/k_th-1):6.2f}%   fit over {i1-i0} cells")
