"""Neuromorphic application tests: p-bit networks and STO reservoir computing."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spinstudio.macrospin import run_pbit, run_reservoir, smtj_preset, sto_preset  # noqa: E402

ok_all = True


def check(name, ok, detail):
    global ok_all
    ok_all &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def log(t):
    print("      " + t)


valid = {"000", "010", "100", "111"}           # A B C with C = A AND B (bit order A,B,C)
for mode in ("behavioral", "physical"):
    cfg = smtj_preset()
    cfg.pbit.problem, cfg.pbit.mode, cfg.pbit.beta = "AND", mode, 2.0
    cfg.pbit.duration_ns = 400 if mode == "physical" else 2000
    cfg.pbit.replicas = 64
    t0 = time.time()
    r = run_pbit(cfg, log=log)
    p = dict(zip(r["states"], r["p_sim"]))
    pv = sum(p[s] for s in valid)
    pe = sum(dict(zip(r["states"], r["p_exact"]))[s] for s in valid)
    check(f"p-bit AND gate ({mode})", r["KL"] < 0.1 and pv > 0.95 * pe,
          f"P(valid) = {pv:.3f} (exact {pe:.3f}), KL = {r['KL']:.4f}, {time.time() - t0:.1f} s")

for mode in ("behavioral", "physical"):
    cfg = smtj_preset()
    cfg.pbit.problem, cfg.pbit.mode, cfg.pbit.n_nodes, cfg.pbit.beta = "maxcut", mode, 14, 1.5
    cfg.pbit.duration_ns = 200
    t0 = time.time()
    r = run_pbit(cfg, log=log)
    check(f"MaxCut n=14 ({mode})", r["cut_best"] == r["cut_optimum"],
          f"best cut {r['cut_best']:.0f} / optimum {r['cut_optimum']:.0f}, {time.time() - t0:.1f} s")

for task in ("narma2", "memory"):
    cfg = sto_preset()
    cfg.reservoir.task = task
    cfg.reservoir.n_samples = 1000
    t0 = time.time()
    r = run_reservoir(cfg, log=log)
    if task == "narma2":
        check("STO reservoir NARMA2", r["NMSE"] < r["NMSE_linear"],
              f"NMSE {r['NMSE']:.4f} vs linear baseline {r['NMSE_linear']:.4f}, {time.time() - t0:.1f} s")
    else:
        check("STO reservoir memory capacity", r["MC"] > 1.0, f"MC = {r['MC']:.2f}, {time.time() - t0:.1f} s")
print("\nALL PASSED" if ok_all else "\nSOME TESTS FAILED")
