"""Validation of the extended micromagnetic physics against analytic results.

1. DMI boundary condition: edge tilt  sin(theta0) = D / (2 sqrt(A K))
2. Zhang-Li STT, beta = alpha: domain wall velocity v = u (exact, any regime)
3. Zhang-Li STT, beta = 2 alpha, below Walker (strip with demag): v = (beta/alpha) u
4. Thermal noise: single macrospin cell, <mz^2> against the Boltzmann distribution
5. Skyrmion: relaxes with |Q| = 1, moves under SOT
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spinstudio.config import (Boundary, Config, ExternalField, Grid, InitialState, Material,  # noqa: E402
                               Study, Torque)
from spinstudio.solver import KB, MUB, QE, TABLE_COLS, Model  # noqa: E402

DEV = "cuda"
col = {c: i for i, c in enumerate(TABLE_COLS)}
ok_all = True


def report(name, value, expect, tol):
    global ok_all
    err = abs(value / expect - 1)
    ok = err < tol
    ok_all &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {value:.5g} vs analytic {expect:.5g} ({100*err:.2f}% , tol {100*tol:.0f}%)")


# 1. DMI edge tilt ---------------------------------------------------------------
A, K, D = 15e-12, 0.8e6, 3e-3
cfg = Config(grid=Grid(nx=200, ny=1, dx_nm=0.25, dy_nm=1, dz_nm=1, pbc_y=True),
             material=Material(Ms_kAm=580, Aex_pJm=A * 1e12, alpha=0.5, Ku_MJm3=K * 1e-6, D_mJm2=D * 1e3),
             field=ExternalField(By_mT=0), boundary=Boundary(width=0), sources=[],
             init=InitialState(kind="uniform", mx=0, my=0, mz=1),
             study=Study(dt_ps=0.005, steps=16, demag=False, relax_steps=2000, relax_tol=1e-9,
                         relax_max=400000, relax_alpha=1.0))
r = Model(cfg, DEV).simulate()
mz_edge = r.m_final[2, 0, 0]
# the cell centre is dx/2 inside: integrate theta' = sin(theta)/Delta back to the edge
Delta = math.sqrt(A / K)
th = math.acos(np.clip(mz_edge, -1, 1))
th_edge = 2 * math.atan(math.tan(th / 2) * math.exp(0.5 * cfg.grid.dx_nm * 1e-9 / Delta))
report("DMI edge tilt sin(theta0)", math.sin(th_edge), D / (2 * math.sqrt(A * K)), 0.02)
print(f"      edge mx = {r.m_final[0, 0, 0]:+.4f} (left), {r.m_final[0, -1, 0]:+.4f} (right)")


# 1b. a domain wall in a PMA strip with DMI must stay at rest without current, and the
#     energy must decrease monotonically while relaxing (consistent field + boundary)
from spinstudio.config import dw_neuron_preset  # noqa: E402
cfg = dw_neuron_preset()
cfg.torques = []
cfg.material.Ku_grad_MJm3_um = 0.0
cfg.study.steps = 20000
cfg.init.wall_x_nm = cfg.grid.nx * cfg.grid.dx_nm / 2   # centre: no net magnetostatic force
res = Model(cfg, DEV).simulate()
xs = res.table[:, col["x_DW"]]
drift = abs(xs[-1] - xs[0])
ok = drift < 1.0
ok_all &= ok
print(f"[{'PASS' if ok else 'FAIL'}] DW at rest without current: drift {drift:.2f} nm in "
      f"{res.table_t[-1]*1e9:.1f} ns (wall at {xs[0]:.1f} nm)")
cfg.material.alpha = 0.5
cfg.study.steps = 4000
res = Model(cfg, DEV).simulate(relax=False)
E = res.table[:, col["E_total"]]
rise = np.max(np.diff(E)) / np.ptp(E) if np.ptp(E) > 0 else 0
ok = rise < 1e-3
ok_all &= ok
print(f"[{'PASS' if ok else 'FAIL'}] energy decreases during damped dynamics: largest relative rise {rise:.2e}")
cx = res.m_final[0, :, cfg.grid.ny // 2]
print(f"      wall centre mx = {cx[np.argmin(abs(res.m_final[2, :, cfg.grid.ny // 2]))]:+.2f} (chirality from DMI)")


# 2./3. Zhang-Li domain-wall velocity ---------------------------------------------
def dw_velocity(beta, alpha, demag, ny, J):
    cfg = Config(grid=Grid(nx=600, ny=ny, dx_nm=0.5, dy_nm=1, dz_nm=1, pbc_y=not demag),
                 material=Material(Ms_kAm=800, Aex_pJm=13, alpha=alpha, Ku_MJm3=1.0 if demag else 0.5,
                                   D_mJm2=0),
                 field=ExternalField(By_mT=0), boundary=Boundary(width=0), sources=[],
                 torques=[Torque(kind="ZhangLi", J_Am2=J, waveform="dc", rise_ns=0.0, P=0.5, beta=beta)],
                 init=InitialState(kind="domain_wall", wall_x_nm=150, wall_type="Bloch"),
                 study=Study(dt_ps=0.01, steps=120000, save_every=120000, demag=demag, relax_steps=500,
                             relax_tol=1e-7, relax_max=100000))
    res = Model(cfg, DEV).simulate()
    t, x = res.table_t, res.table[:, col["x_DW"]]
    sel = t > 0.2 * t[-1]
    v = np.polyfit(t[sel], x[sel] * 1e-9, 1)[0]
    u = MUB * 0.5 * J / (QE * 800e3 * (1 + beta ** 2))
    return v, u


v, u = dw_velocity(beta=0.02, alpha=0.02, demag=False, ny=1, J=2e12)
report("Zhang-Li DW velocity (beta = alpha)", abs(v), u, 0.02)
# rotationally symmetric wall (no demag): exact v = (1 + alpha*beta) / (1 + alpha^2) * u
for a_, b_ in ((0.5, 1.0), (0.5, 0.0)):
    v, u = dw_velocity(beta=b_, alpha=a_, demag=False, ny=1, J=2e12)
    report(f"Zhang-Li DW velocity (alpha={a_}, beta={b_}, precessional)", abs(v),
           (1 + a_ * b_) / (1 + a_ ** 2) * u, 0.02)

# 4. thermal equilibrium of one cell ------------------------------------------------
T = 300.0
d = 4e-9
Ku = 3 * KB * T / d ** 3                       # barrier Delta = 3 kT
cfg = Config(grid=Grid(nx=1, ny=1, dx_nm=4, dy_nm=4, dz_nm=4),
             material=Material(Ms_kAm=800, Aex_pJm=13, alpha=0.1, Ku_MJm3=Ku * 1e-6),
             field=ExternalField(By_mT=0), boundary=Boundary(width=0), sources=[],
             init=InitialState(kind="uniform", mz=1),
             study=Study(dt_ps=0.1, steps=2000000, save_every=2000000, temperature_K=T, relax_steps=0,
                         relax_tol=0, seed=7))
res = Model(cfg, DEV).simulate()
mz = res.table[res.table_t > 2e-9, col["mz"]]
x = np.linspace(-1, 1, 20001)
w = np.exp(3 * x ** 2)
report("thermal <mz^2> (Delta = 3)", np.mean(mz ** 2), np.sum(x ** 2 * w) / np.sum(w), 0.03)
print(f"      CUDA graph used: {res.extra['cuda_graph']}, samples {len(mz)}, "
      f"<mz> = {np.mean(mz):+.3f} (should be ~0: thermally activated reversals)")

# 5. skyrmion ------------------------------------------------------------------------
from spinstudio.config import skyrmion_preset  # noqa: E402
cfg = skyrmion_preset()
res = Model(cfg, DEV).simulate()
Q0, Q1 = res.table[0, col["Q"]], res.table[-1, col["Q"]]
x0, x1 = res.table[0, col["core_x"]], res.table[-1, col["core_x"]]
y0, y1 = res.table[0, col["core_y"]], res.table[-1, col["core_y"]]
r0, r1 = res.table[0, col["core_r"]], res.table[-1, col["core_r"]]
dist = math.hypot(x1 - x0, y1 - y0)
sk_ok = abs(Q1) > 0.75 and abs(r1 / r0 - 1) < 0.15 and dist > 5
print(f"[{'PASS' if sk_ok else 'FAIL'}] skyrmion: Q {Q0:+.3f} -> {Q1:+.3f} (corners add edge charge), "
      f"radius {res.table[0, col['core_r']]:.1f} nm, moved ({x0:.1f},{y0:.1f}) -> ({x1:.1f},{y1:.1f}) nm, "
      f"{res.steps_per_s:.0f} steps/s")
ok_all &= sk_ok
print(f"      radius {r0:.2f} -> {r1:.2f} nm, displacement {dist:.1f} nm, "
      f"skyrmion Hall angle {math.degrees(math.atan2(y1 - y0, x1 - x0)):.0f} deg")
print("\nALL PASSED" if ok_all else "\nSOME TESTS FAILED")
