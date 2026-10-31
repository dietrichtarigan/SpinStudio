"""Validation of the macrospin engine against analytic results."""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spinstudio.macrospin import (KB, MU0, BField, Device, Drive, MacroConfig, MStudy, Stack,  # noqa: E402
                                  device_info, run_study)

ok_all = True


def report(name, value, expect, tol):
    global ok_all
    err = abs(value / expect - 1)
    ok_all &= err < tol
    print(f"[{'PASS' if err < tol else 'FAIL'}] {name}: {value:.5g} vs analytic {expect:.5g} ({100*err:.2f}%, tol {100*tol:.0f}%)")


# 1. Boltzmann distribution for a PMA macrospin with Delta = 3
cfg = MacroConfig(device=Device(a_nm=20, b_nm=20, t_nm=1.2, Ms_kAm=1100, alpha=0.1, Ku_MJm3=0.0),
                  drive=Drive(J_Am2=0, waveform="dc"),
                  study=MStudy(kind="transient", dt_ps=0.5, duration_ns=200, n_devices=2048, temperature_K=300,
                               init="up", record_every=20))
info = device_info(cfg)
Keff_shape = info["Keff_MJm3"] * 1e6                       # shape part only (Ku = 0)
V = info["volume_nm3"] * 1e-27
cfg.device.Ku_MJm3 = (3 * KB * 300 / V - Keff_shape) * 1e-6
info = device_info(cfg)
t0 = time.time()
res = run_study(cfg)
ens_mz2 = None
from spinstudio.macrospin import MacroModel  # noqa: E402
mm = MacroModel(cfg)
r = mm.run(int(200e-9 / 0.5e-12), record_every=200, n_traces=1, keep_full=True)
mz = r["full"][r["full"].shape[0] // 4:, :, 2].cpu().numpy().ravel()
x = np.linspace(-1, 1, 20001)
w = np.exp(info["Delta"] * x ** 2)
report(f"Boltzmann <mz^2> (Delta = {info['Delta']:.2f})", np.mean(mz ** 2), np.sum(x ** 2 * w) / np.sum(w), 0.02)
print(f"      {r['rate']:.0f} steps/s for {mm.n} devices")

# 2. T = 0 STT instability threshold of a pMTJ
cfg = MacroConfig(device=Device(alpha=0.05),
                  drive=Drive(J_Am2=0, waveform="dc", t_start_ns=0, rise_ns=0),
                  study=MStudy(kind="sweep", sweep_param="J", n_devices=1, temperature_K=0, init="up",
                               dt_ps=1.0, duration_ns=60))
Jc0 = device_info(cfg)["Jc0_Am2"]
cfg.study.sweep_from, cfg.study.sweep_to, cfg.study.sweep_points = -0.9 * Jc0, -1.1 * Jc0, 41
res = run_study(cfg)
vals, p = np.array(res["values"]), np.array(res["p_switch"])
Jth = abs(vals[np.argmax(p > 0.5)]) if p.max() > 0.5 else float("nan")
report("STT critical current |Jc0| (T = 0)", Jth, Jc0, 0.03)

# 3. spin-torque oscillator, out-of-plane precession: f = gamma B_DL / (2 pi alpha)
cfg = MacroConfig(device=Device(a_nm=60, b_nm=60, t_nm=2, Ms_kAm=1000, alpha=0.01, Ku_MJm3=0, anis_axis="x"),
                  stack=Stack(torque="STT", P=0.5, px=0, py=0, pz=1),
                  drive=Drive(J_Am2=-1e11, waveform="dc", t_start_ns=0, rise_ns=0),
                  study=MStudy(kind="spectrum", sweep_param="J", sweep_from=-0.5e11, sweep_to=-1.0e11,
                               sweep_points=3, n_traces=1, n_devices=1, temperature_K=0, init="random",
                               dt_ps=0.5, duration_ns=40, readout="mx"))
res = run_study(cfg)
bdl = device_info(cfg)["B_DL_per_1e11_mT"] * 1e-3
for J, f in zip(res["values"], res["peak_GHz"]):
    report(f"STO frequency at J = {J:.2g} A/m2", f, 1.7595e11 * bdl * abs(J) / 1e11 / (2 * math.pi * 0.01) / 1e9, 0.03)
print("\nALL PASSED" if ok_all else "\nSOME TESTS FAILED")
