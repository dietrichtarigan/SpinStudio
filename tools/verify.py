"""Compare SpinStudio against the original SpinTorch (accuracy + speed).

Run on a GPU node from ~/micromagnetics:
    .venv/bin/python tools/verify.py
"""
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "SpinTorch"))

import spintorch  # noqa: E402
from spinstudio.config import focus_preset  # noqa: E402
from spinstudio.solver import Model, focus_loss  # noqa: E402

dev = torch.device("cuda")
T = int(os.environ.get("STEPS", 600))


def original_model():
    nx = ny = 100
    geom = spintorch.WaveGeometryMs((nx, ny), (50e-9, 50e-9, 20e-9), 140e3, 60e-3)
    src = spintorch.WaveLineSource(10, 0, 10, ny - 1, dim=2)
    probes = [spintorch.WaveIntensityProbeDisk(nx - 15, int(ny * (p + 1) / 20), 2) for p in range(19)]
    model = spintorch.MMSolver(geom, 20e-12, [src], probes).to(dev)
    t = torch.arange(0, T * 20e-12, 20e-12, device=dev).unsqueeze(0).unsqueeze(2)[:, :T]
    X = 1e-3 * torch.sin(2 * np.pi * 4e9 * t)
    return model, X


def new_model(compile_mode, legacy=True, normalize=False, precision="float32", train=True):
    cfg = focus_preset()
    cfg.study.steps = T
    cfg.study.legacy_source = legacy
    cfg.study.normalize = normalize
    cfg.study.compile = compile_mode
    cfg.study.precision = precision
    cfg.study.relax_tol = 0          # SpinTorch relaxes a fixed number of steps
    return Model(cfg, "cuda", train=train)


def rel(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def timed(fn, n=1):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) / n


print(f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}, steps={T}")

# ---------------------------------------------------------------- original
om, X = original_model()
om.retain_history = True


def orig_epoch():
    om.zero_grad()
    u = om(X).sum(dim=1)
    loss = focus_loss(u[0], 9)
    loss.backward()
    return u[0].detach(), loss.item(), om.geom.rho.grad.detach().clone()


(u_o, L_o, g_o), t_orig = timed(orig_epoch)
m_last_o = om.m_history[-1][0].numpy()
print(f"\n[original] epoch time {t_orig:.2f} s, loss {L_o:.6f}")

# ---------------------------------------------------------------- new, eager, legacy source
nm = new_model("off")


def new_epoch(model):
    model.zero_grad()
    u, _, _ = model.train_forward()
    loss = focus_loss(u, 9)
    loss.backward()
    return u.detach(), loss.item(), model.rho.grad.detach().clone()


(u_n, L_n, g_n), t_new = timed(lambda: new_epoch(nm))
print(f"[new eager, legacy source] epoch time {t_new:.2f} s, loss {L_n:.6f}")
print(f"  probe outputs rel. error : {rel(u_n.cpu(), u_o.cpu()):.2e}")
print(f"  gradient rel. error      : {rel(g_n.cpu(), g_o.cpu()):.2e}")
cos = torch.nn.functional.cosine_similarity(g_n.flatten(), g_o.flatten(), dim=0).item()
print(f"  gradient cosine sim.     : {cos:.6f}")

# final magnetisation via simulate() (no grad, same physics)
nm2 = new_model("off", train=False)
res = nm2.simulate()
m_last_n = res.frames[-1].numpy() + res.m0
print(f"  final m rel. error (dm)  : {rel(m_last_n - res.m0, m_last_o - om.m0[0].cpu().numpy()):.2e}")

# ---------------------------------------------------------------- speed of compiled variants
for mode in ("default", "cudagraphs"):
    try:
        mm = new_model(mode, train=False)
        mm.simulate(steps=50)                       # warm-up / compile
        r, ts = timed(lambda: mm.simulate())
        dm_err = rel(r.frames[-1].numpy(), res.frames[-1].numpy())
        print(f"[simulate, compile={mode}] {ts:.2f} s for relax+{T} steps "
              f"({r.steps_per_s:.0f} steps/s in loop), dm rel. diff vs eager {dm_err:.1e}")
    except Exception as e:  # noqa: BLE001
        print(f"[simulate, compile={mode}] FAILED: {type(e).__name__}: {e}")

try:
    mt = new_model("default")
    new_epoch(mt)                                    # warm-up / compile
    (u_c, L_c, g_c), t_c = timed(lambda: new_epoch(mt))
    print(f"[train epoch, compiled] {t_c:.2f} s (original {t_orig:.2f} s -> x{t_orig/t_c:.1f}), "
          f"loss {L_c:.6f}")
except Exception as e:  # noqa: BLE001
    print(f"[train epoch, compiled] FAILED: {type(e).__name__}: {e}")

ms = new_model("off", train=False)
_, t_sim_eager = timed(lambda: ms.simulate())
print(f"[simulate, eager] {t_sim_eager:.2f} s")

# ---------------------------------------------------------------- effect of the source bug
fixed = new_model("off", legacy=False, normalize=True, train=False).simulate()
leg = res
print("\nSource accumulation bug:")
for name, r in (("legacy (SpinTorch)", leg), ("fixed", fixed)):
    print(f"  {name:20s} max|dm_z| at end = {np.abs(r.frames[-1][2].numpy()).max():.3e}, "
          f"target probe intensity = {(r.probe[:, 0, 9] ** 2).sum():.3e}")
