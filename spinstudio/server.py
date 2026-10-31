"""SpinStudio web server: FastAPI backend + single-page GUI.

Start on a GPU node:
    .venv/bin/python -m spinstudio.server --port 8765
then open http://localhost:8765 (through an SSH tunnel when remote).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import datetime as _dt
import io
import json
import math
import os
import threading
import time
import traceback
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import PRESETS, Config
from .macrospin import MACRO_PRESETS, MacroConfig, device_info, run_pbit, run_reservoir, run_study
from .solver import COMP, TABLE_COLS, TABLE_UNITS, Model, focus_loss, kalinikos_slavin, physics_report

HERE = Path(__file__).parent
RUNS = Path(os.environ.get("SPINSTUDIO_RUNS", HERE.parent / "runs"))
LIVE_MAX = 256          # live preview is downsampled to at most this many cells per axis

warnings.filterwarnings("ignore", message=".*complex operators.*")
warnings.filterwarnings("ignore", message=".*TensorFloat32.*")


@asynccontextmanager
async def lifespan(_app):
    S.loop = asyncio.get_running_loop()
    yield


app = FastAPI(title="SpinStudio", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


# ----------------------------------------------------------------------------
# encoding helpers
# ----------------------------------------------------------------------------
def enc(a) -> dict:
    """2D field (nx, ny) -> base64 float32 in display order (rows = y)."""
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 2:
        a = a.T
    a = np.ascontiguousarray(a)
    finite = a[np.isfinite(a)]
    return {"shape": list(a.shape), "b64": base64.b64encode(a.tobytes()).decode(),
            "min": float(finite.min()) if finite.size else 0.0,
            "max": float(finite.max()) if finite.size else 0.0}


def enc1(a) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()


def downsample(a, nmax=LIVE_MAX):
    s = max(1, math.ceil(max(a.shape[-2:]) / nmax))
    return a[..., ::s, ::s], s


# ----------------------------------------------------------------------------
# job state
# ----------------------------------------------------------------------------
class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.stop = False
        self.status = "idle"            # idle | running | done | error | stopped
        self.mode = None
        self.cfg: Optional[Config] = None
        self.result = None              # SimResult
        self.model: Optional[Model] = None
        self.train = {"loss": [], "outputs": [], "epoch_time": []}
        self.design = None              # last design map (nx, ny)
        self.rho = None
        self.log: list = []
        self.run_dir: Optional[Path] = None
        self.sockets: set = set()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.progress = {}
        self.macro_cfg: Optional[MacroConfig] = None
        self.macro_result = None


S = State()


def emit(kind: str, **data):
    msg = json.dumps({"type": kind, **data})
    if S.loop is None:
        return
    for ws in list(S.sockets):
        asyncio.run_coroutine_threadsafe(_send(ws, msg), S.loop)


async def _send(ws, msg):
    try:
        await ws.send_text(msg)
    except Exception:  # noqa: BLE001
        S.sockets.discard(ws)


def log(text: str, level: str = "info"):
    ts = _dt.datetime.now().strftime("%H:%M:%S")
    item = {"t": ts, "level": level, "text": text}
    S.log.append(item)
    S.log = S.log[-500:]
    emit("log", **item)


# ----------------------------------------------------------------------------
# workers
# ----------------------------------------------------------------------------
def _progress_cb(phase_label):
    def cb(p):
        if p.get("phase") == "relax":
            emit("progress", phase="relax", step=0, total=p["total"], label="Relaxing ground state")
            return
        live = p.get("live")
        msg = {"phase": phase_label, "step": p["step"], "total": p["total"],
               "rate": p.get("rate", 0), "label": f"{phase_label}: step {p['step']}/{p['total']}"}
        S.progress = msg
        if live is not None:
            arr, s = downsample(live.numpy())
            msg["live"] = [enc(arr[c]) for c in range(3)]
            msg["live_stride"] = s
            msg["t_ns"] = p["step"] * S.cfg.study.dt_ps * 1e-3
        emit("progress", **msg)
    return cb


def _new_run_dir(cfg: Config, mode: str) -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in cfg.name)[:40]
    d = RUNS / f"{stamp}_{mode}_{safe}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(cfg.model_dump_json(indent=2))
    return d


def _save_result(d: Path, res, extra=None):
    np.savez_compressed(
        d / "result.npz", frames=res.frames.numpy(), frame_steps=res.frame_steps, probe=res.probe,
        intensity=res.intensity, dft=res.dft if res.dft is not None else np.zeros(0),
        f_dft=res.f_dft, m0=res.m0, Msat=res.Msat, Bext=res.Bext, dt=res.dt,
        **(extra or {}))


def _finish_sim(res, model):
    S.result = res
    S.model = model
    info = (f"Simulation finished: {res.steps} steps in {res.wall_time:.2f} s "
            f"({res.steps_per_s:.0f} steps/s, {res.steps * res.dt * 1e9:.2f} ns simulated)")
    if res.extra.get("cuda_graph"):
        info += ", CUDA graph"
    log(info, "ok")
    rx = res.extra.get("relax")
    if rx and rx.get("residual") is not None:
        txt = f"Ground state: {rx['steps']} relaxation steps, residual |dm|/step = {rx['residual']:.1e}"
        log(txt, "info" if rx["converged"] else "warn")
        if not rx["converged"]:
            log("Ground state did not converge: increase 'Max. relaxation steps' or the relaxation damping.", "warn")
    if res.extra.get("graph_error"):
        log("CUDA graph capture failed, ran without it: " + res.extra["graph_error"], "warn")


def run_simulation(cfg: Config):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Building model {cfg.grid.nx}x{cfg.grid.ny} on {dev} (demag kernel)...")
    t0 = time.perf_counter()
    model = Model(cfg, dev)
    log(f"Model ready in {time.perf_counter() - t0:.2f} s. First run per grid compiles GPU kernels "
        f"(can take ~30-60 s).")
    res = model.simulate(progress=_progress_cb("Time stepping"), should_stop=lambda: S.stop)
    _finish_sim(res, model)
    S.design = res.Msat if cfg.training.design == "Ms" else None
    if S.run_dir:
        _save_result(S.run_dir, res)
        log(f"Saved to {S.run_dir}")
    S.status = "stopped" if res.extra.get("stopped") else "done"


def run_training(cfg: Config):
    if len(cfg.probes) == 0:
        raise ValueError("Training needs at least one probe")
    if not (0 <= cfg.training.target < len(cfg.probes)):
        raise ValueError(f"Target probe {cfg.training.target} does not exist "
                         f"(have {len(cfg.probes)} probes, index from 0)")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = cfg.training
    log(f"Building trainable model ({tr.design} design) on {dev}...")
    model = Model(cfg, dev, train=True)
    opt = torch.optim.Adam([model.rho], lr=tr.lr)
    S.train = {"loss": [], "outputs": [], "epoch_time": []}
    S.model = model
    for epoch in range(tr.epochs):
        if S.stop:
            break
        t0 = time.perf_counter()
        opt.zero_grad()

        def fprog(step, total, epoch=epoch):
            emit("progress", phase="train", step=step, total=total,
                 label=f"Epoch {epoch + 1}/{tr.epochs}: forward {step}/{total}",
                 epoch=epoch, epochs=tr.epochs)

        u, Msat, Bext = model.train_forward(progress=fprog)
        loss = focus_loss(u, tr.target, tr.loss)
        emit("progress", phase="train", step=1, total=1,
             label=f"Epoch {epoch + 1}/{tr.epochs}: backward", epoch=epoch, epochs=tr.epochs)
        loss.backward()
        opt.step()
        model.clamp_()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        L = float(loss.item())
        uu = u.detach().float().cpu().numpy()
        S.train["loss"].append(L)
        S.train["outputs"].append(uu.tolist())
        S.train["epoch_time"].append(dt)
        with torch.no_grad():
            S2 = model.materials()
            Msat2, Bext2 = S2["Msat"], S2["Bext"]
            if tr.design == "Ms":
                design = Msat2[0, 0].float().cpu().numpy() / 1e3      # kA/m
            else:
                u0 = torch.tensor(model.B0dir, dtype=Bext2.dtype, device=Bext2.device).view(1, 3, 1, 1)
                design = (Bext2 * u0).sum(1)[0].float().cpu().numpy() * 1e3   # mT along B0
        S.design = design
        S.rho = model.rho.detach().float().cpu().numpy()
        frac = float(uu[tr.target] / max(uu.sum(), 1e-30))
        log(f"Epoch {epoch + 1}/{tr.epochs}: loss {L:.5f}, target share {100 * frac:.1f}%, {dt:.2f} s")
        emit("epoch", epoch=epoch, loss=L, outputs=uu.tolist(), time=dt, target=tr.target,
             design=enc(design), design_unit="kA/m" if tr.design == "Ms" else "mT")
    if S.run_dir:
        np.save(S.run_dir / "rho.npy", S.rho)
        (S.run_dir / "training.json").write_text(json.dumps(S.train))
    if tr.simulate_after and not S.stop:
        log("Running a full recorded simulation of the trained design...")
        res = model.simulate(progress=_progress_cb("Trained design"), should_stop=lambda: S.stop,
                             warm=True)
        _finish_sim(res, model)
        if S.run_dir:
            _save_result(S.run_dir, res, {"rho": S.rho})
    S.status = "stopped" if S.stop else "done"


def run_macro(cfg: MacroConfig, mode: str):
    label = {"macro_study": "Device study", "macro_pbit": "p-bit network",
             "macro_reservoir": "Reservoir"}[mode]

    def prog(step, total, rate):
        emit("progress", phase="macro", step=step, total=total, rate=rate,
             label=f"{label}: {step}/{total}")

    t0 = time.perf_counter()
    if mode == "macro_study":
        res = run_study(cfg, progress=prog, should_stop=lambda: S.stop)
        msg = f"{cfg.study.kind} study finished: {res['n_devices']} devices, {res['steps_per_s']:.0f} steps/s"
    elif mode == "macro_pbit":
        res = run_pbit(cfg, progress=prog, should_stop=lambda: S.stop, log=log)
        msg = f"p-bit network finished: best energy {res['best_E']:.3f}"
        if "KL" in res:
            msg += f", KL(sim || Boltzmann) = {res['KL']:.4f}"
        if "cut_best" in res:
            msg += f", best cut {res['cut_best']:.0f}" + (f" / optimum {res['cut_optimum']:.0f}" if "cut_optimum" in res else "")
    else:
        res = run_reservoir(cfg, progress=prog, should_stop=lambda: S.stop, log=log)
        msg = f"Reservoir ({res['task']}) finished: {res['metric']}"
        if "NMSE_linear" in res:
            msg += f" (linear baseline NMSE {res['NMSE_linear']:.4f})"
    res["mode"] = mode
    res["config"] = json.loads(cfg.model_dump_json())
    res["elapsed"] = time.perf_counter() - t0
    S.macro_result = res
    log(msg + f" in {res['elapsed']:.1f} s", "ok")
    if S.run_dir:
        (S.run_dir / "result.json").write_text(json.dumps(res, default=float))
        log(f"Saved to {S.run_dir}")
    S.status = "stopped" if S.stop else "done"


def _worker(cfg, mode: str):
    try:
        S.run_dir = _new_run_dir(cfg, mode)
        if mode == "train":
            run_training(cfg)
        elif mode.startswith("macro_"):
            run_macro(cfg, mode)
        else:
            run_simulation(cfg)
    except Exception as e:  # noqa: BLE001
        S.status = "error"
        log(f"{type(e).__name__}: {e}", "error")
        log(traceback.format_exc(), "debug")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        emit("status", status=S.status, mode=mode, has_result=S.result is not None,
             has_training=len(S.train["loss"]) > 0, has_macro=S.macro_result is not None)


# ----------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/vendor/plotly.min.js")
def plotly_js():
    import plotly
    p = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    return FileResponse(p, media_type="application/javascript")


@app.get("/api/info")
def info():
    d = {"torch": torch.__version__, "cuda": torch.cuda.is_available(), "presets": list(PRESETS),
         "macro_presets": list(MACRO_PRESETS),
         "host": os.uname().nodename if hasattr(os, "uname") else "", "status": S.status}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        d.update(gpu=p.name, mem_total_gb=total / 2 ** 30, mem_free_gb=free / 2 ** 30)
    return d


@app.get("/api/preset/{name}")
def preset(name: str):
    if name not in PRESETS:
        raise HTTPException(404, "unknown preset")
    return json.loads(PRESETS[name]().model_dump_json())


@app.get("/api/schema")
def schema():
    return Config.model_json_schema()


def _geometry(cfg: Config) -> dict:
    m = Model(cfg, "cpu", build_kernel=False)
    S = m.materials()
    Msat, Bext = S["Msat"], S["Bext"]
    g = cfg.grid
    srcs = []
    for s in cfg.sources:
        srcs.append({"name": s.name, "x0": s.x0, "y0": s.y0, "x1": s.x1, "y1": s.y1,
                     "enabled": s.enabled, "component": s.component})
    return {
        "nx": g.nx, "ny": g.ny, "dx": g.dx_nm * 1e-3, "dy": g.dy_nm * 1e-3,   # um
        "Ms": enc(Msat[0, 0].detach().numpy() / 1e3),
        "B": enc(torch.linalg.vector_norm(Bext[0], dim=0).detach().numpy() * 1e3),
        "alpha": enc(np.log10(m.alpha[0, 0].numpy())),
        "sources": srcs,
        "probes": [{"x": p.x, "y": p.y, "r": p.r} for p in cfg.probes],
        "boundary": cfg.boundary.width,
        "design": [cfg.training.region_x0, cfg.training.region_y0,
                   cfg.training.region_x1, cfg.training.region_y1],
    }


def _dispersion(cfg: Config) -> dict:
    dx = min(cfg.grid.dx_nm, cfg.grid.dy_nm) * 1e-9
    k = np.linspace(0, math.pi / dx, 600)
    k[0] = 1e2
    out = {"k": (k * 1e-6).tolist(), "curves": []}
    for name, phi in (("k ∥ x", 0.0), ("k ∥ y", math.pi / 2)):
        f = kalinikos_slavin(cfg, k, phi) / 1e9
        out["curves"].append({"name": name, "f": [None if not np.isfinite(v) else float(v) for v in f]})
    out["sources"] = [s.f_GHz for s in cfg.sources if s.enabled and s.waveform != "sinc"]
    return out


@app.post("/api/check")
def check(cfg: Config):
    try:
        rep = physics_report(cfg)
        rep["geometry"] = _geometry(cfg)
        rep["dispersion"] = _dispersion(cfg)
        return json.loads(json.dumps(rep, default=float))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"{type(e).__name__}: {e}")


class RunRequest(BaseModel):
    config: Config
    mode: str = "simulate"


class MacroRunRequest(BaseModel):
    config: MacroConfig
    mode: str = "macro_study"


@app.post("/api/macro/run")
def macro_run(req: MacroRunRequest):
    if req.mode not in ("macro_study", "macro_pbit", "macro_reservoir"):
        raise HTTPException(400, "unknown mode")
    return _start(req.config, req.mode)


@app.post("/api/run")
def run(req: RunRequest):
    return _start(req.config, req.mode)


def _start(cfg, mode):
    with S.lock:
        if S.thread is not None and S.thread.is_alive():
            raise HTTPException(409, "A job is already running")
        S.stop = False
        S.status = "running"
        S.mode = mode
        if mode.startswith("macro_"):
            S.macro_cfg = cfg
        else:
            S.cfg = cfg
        if mode == "train":
            S.train = {"loss": [], "outputs": [], "epoch_time": []}
        what = {"train": "training", "simulate": "simulation", "macro_study": "device study",
                "macro_pbit": "p-bit network", "macro_reservoir": "reservoir computing"}.get(mode, mode)
        log(f"Starting {what}: {cfg.name}")
        emit("status", status="running", mode=mode)
        S.thread = threading.Thread(target=_worker, args=(cfg, mode), daemon=True)
        S.thread.start()
    return {"ok": True}


@app.post("/api/stop")
def stop():
    S.stop = True
    log("Stop requested", "warn")
    return {"ok": True}


@app.get("/api/status")
def status():
    return {"status": S.status, "mode": S.mode, "progress": S.progress, "log": S.log[-200:],
            "has_result": S.result is not None, "has_training": len(S.train["loss"]) > 0,
            "has_macro": S.macro_result is not None}


def _res():
    if S.result is None:
        raise HTTPException(404, "no result yet")
    return S.result


@app.get("/api/result/meta")
def result_meta():
    r = _res()
    cfg = S.cfg
    return {
        "nx": cfg.grid.nx, "ny": cfg.grid.ny, "dx": cfg.grid.dx_nm * 1e-3, "dy": cfg.grid.dy_nm * 1e-3,
        "dt_ns": r.dt * 1e9, "steps": r.steps, "save_every": r.save_every,
        "frames": len(r.frame_steps), "frame_t_ns": (r.frame_steps * r.dt * 1e9).tolist(),
        "n_probes": int(r.probe.shape[2]), "f_dft_GHz": r.f_dft / 1e9, "has_dft": r.dft is not None,
        "wall_time": r.wall_time, "steps_per_s": r.steps_per_s, "extra": r.extra,
        "record": r.record, "has_table": r.table is not None and len(r.table) > 1,
        "config": json.loads(cfg.model_dump_json()),
        "geometry": _geometry(cfg),
        "run_dir": str(S.run_dir) if S.run_dir else None,
    }


def _component(arr3, comp):
    if comp == "abs":
        return np.sqrt((arr3 ** 2).sum(0))
    return arr3[COMP[comp]]


@app.get("/api/result/frames")
def result_frames(start: int = 0, count: int = 1, comp: str = "z", stride: int = 1):
    r = _res()
    n = len(r.frame_steps)
    start = max(0, min(start, n - 1))
    idx = list(range(start, min(n, start + count * stride), stride))
    out = []
    for i in idx:
        out.append(enc(_component(r.frames[i].numpy(), comp)))
    return {"start": start, "idx": idx, "frames": out}


@app.get("/api/result/range")
def result_range(comp: str = "z"):
    """Global color range over all frames (robust 99.9% percentile)."""
    r = _res()
    a = r.frames.numpy()
    sub = a[:: max(1, len(a) // 50)]
    v = _component(np.moveaxis(sub, 1, 0), comp) if comp == "abs" else sub[:, COMP[comp]]
    lim = float(np.percentile(np.abs(v), 99.9)) if v.size else 1.0
    return {"lim": lim or 1.0}


@app.get("/api/result/map/{name}")
def result_map(name: str, comp: str = "z"):
    r = _res()
    if name == "intensity":
        a = _component(r.intensity, comp) if comp != "abs" else r.intensity.sum(0)
        return {"map": enc(a), "unit": "Σ dm²"}
    if name in ("dft_amp", "dft_phase", "dft_real"):
        if r.dft is None:
            raise HTTPException(404, "no frequency-domain data")
        c = r.dft[COMP[comp if comp != "abs" else "z"]]
        a = {"dft_amp": np.abs(c), "dft_phase": np.angle(c), "dft_real": c.real}[name]
        return {"map": enc(a), "f_GHz": r.f_dft / 1e9}
    if name == "Msat":
        return {"map": enc(r.Msat / 1e3), "unit": "kA/m"}
    if name == "B":
        return {"map": enc(np.linalg.norm(r.Bext, axis=0) * 1e3), "unit": "mT"}
    if name == "m0":
        return {"map": enc(r.m0[COMP[comp if comp != "abs" else "z"]]), "unit": ""}
    raise HTTPException(404, "unknown map")


@app.get("/api/result/probes")
def result_probes(comp: str = "x"):
    r = _res()
    p = r.probe                                      # (T, 3, Np)
    sig = np.sqrt((p ** 2).sum(1)) if comp == "abs" else p[:, COMP[comp]]
    T = sig.shape[0]
    t = (np.arange(1, T + 1) * r.dt * 1e9)
    # spectrum (Hann window) of each probe
    w = np.hanning(T)[:, None]
    spec = np.abs(np.fft.rfft((sig - sig.mean(0)) * w, axis=0)) * 2 / max(w.sum(), 1)
    f = np.fft.rfftfreq(T, r.dt) / 1e9
    stride = max(1, T // 4000)
    intensity = (p[:, COMP[S.cfg.study.probe_component]] ** 2).sum(0)
    return {"t": enc1(t[::stride]), "signals": [enc1(sig[::stride, i]) for i in range(sig.shape[1])],
            "f": enc1(f), "spectra": [enc1(spec[:, i]) for i in range(sig.shape[1])],
            "intensity": intensity.tolist(), "unit": "A/m (Σ over disk of Δm·Ms)",
            "positions": [{"x": q.x, "y": q.y, "r": q.r} for q in S.cfg.probes]}


@app.get("/api/result/kspace/{i}")
def result_kspace(i: int, comp: str = "z"):
    r = _res()
    cfg = S.cfg
    a = _component(r.frames[max(0, min(i, len(r.frame_steps) - 1))].numpy(), comp)
    nx, ny = a.shape
    win = np.outer(np.hanning(nx), np.hanning(ny))
    A = np.abs(np.fft.fftshift(np.fft.fft2((a - a.mean()) * win)))
    dx, dy = cfg.grid.dx_nm * 1e-9, cfg.grid.dy_nm * 1e-9
    kx = np.fft.fftshift(np.fft.fftfreq(nx, dx)) * 2 * np.pi * 1e-6     # rad/um
    ky = np.fft.fftshift(np.fft.fftfreq(ny, dy)) * 2 * np.pi * 1e-6
    out = {"map": enc(np.log10(A + A.max() * 1e-4)), "kx0": float(kx[0]), "dkx": float(kx[1] - kx[0]),
           "ky0": float(ky[0]), "dky": float(ky[1] - ky[0])}
    f = r.f_dft
    if f > 0:
        kmax = math.pi / min(dx, dy)
        kk = np.linspace(1e3, kmax, 3000)
        cx, cy = [], []
        for phi in np.linspace(0, 2 * np.pi, 361):
            fk = kalinikos_slavin(cfg, kk, phi)
            s = np.where(np.diff(np.sign(np.nan_to_num(fk, nan=-1) - f)))[0]
            if len(s):
                k0 = kk[s[0]] * 1e-6
                cx.append(k0 * math.cos(phi))
                cy.append(k0 * math.sin(phi))
            else:
                cx.append(None)
                cy.append(None)
        out["iso"] = {"kx": cx, "ky": cy, "f_GHz": f / 1e9}
    return out


@app.get("/api/result/linecut/{i}")
def result_linecut(i: int, comp: str = "z", axis: str = "x", index: int = -1):
    r = _res()
    a = _component(r.frames[max(0, min(i, len(r.frame_steps) - 1))].numpy(), comp)
    if axis == "x":
        j = a.shape[1] // 2 if index < 0 else min(index, a.shape[1] - 1)
        return {"v": a[:, j].tolist(), "d": S.cfg.grid.dx_nm * 1e-3, "index": j}
    j = a.shape[0] // 2 if index < 0 else min(index, a.shape[0] - 1)
    return {"v": a[j, :].tolist(), "d": S.cfg.grid.dy_nm * 1e-3, "index": j}


@app.get("/api/train/history")
def train_history():
    d = dict(S.train)
    d["target"] = S.cfg.training.target if S.cfg else 0
    d["design"] = enc(S.design) if S.design is not None else None
    return d


@app.get("/api/result/export.npz")
def export_npz():
    r = _res()
    buf = io.BytesIO()
    np.savez_compressed(buf, frames=r.frames.numpy(), frame_steps=r.frame_steps, probe=r.probe,
                        intensity=r.intensity, dft=r.dft if r.dft is not None else np.zeros(0),
                        f_dft=r.f_dft, m0=r.m0, Msat=r.Msat, Bext=r.Bext, dt=r.dt,
                        rho=S.rho if S.rho is not None else np.zeros(0),
                        config=np.array(S.cfg.model_dump_json()))
    name = (S.cfg.name or "result").replace(" ", "_")
    return Response(buf.getvalue(), media_type="application/octet-stream",
                    headers={"Content-Disposition": f'attachment; filename="{name}.npz"'})


@app.get("/api/runs")
def list_runs():
    if not RUNS.exists():
        return []
    out = []
    for d in sorted(RUNS.iterdir(), reverse=True)[:100]:
        if (d / "config.json").exists():
            out.append({"id": d.name, "has_result": (d / "result.npz").exists()})
    return out


@app.get("/api/runs/{rid}/config")
def run_config(rid: str):
    p = RUNS / Path(rid).name / "config.json"
    if not p.exists():
        raise HTTPException(404, "not found")
    return json.loads(p.read_text())


@app.get("/api/result/table")
def result_table():
    r = _res()
    if r.table is None:
        raise HTTPException(404, "no table")
    return {"cols": TABLE_COLS, "units": TABLE_UNITS, "t_ns": (r.table_t * 1e9).tolist(),
            "data": np.nan_to_num(r.table).T.tolist()}


@app.get("/api/macro/preset/{name}")
def macro_preset(name: str):
    if name not in MACRO_PRESETS:
        raise HTTPException(404, "unknown preset")
    return json.loads(MACRO_PRESETS[name]().model_dump_json())


@app.post("/api/macro/check")
def macro_check(cfg: MacroConfig):
    try:
        info = device_info(cfg)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"{type(e).__name__}: {e}")
    d, st = cfg.device, cfg.study
    rep = {"info": info, "warnings": [], "notes": []}
    Ms = d.Ms_kAm * 1e3
    B = (abs(info["Bk_eff_mT"]) * 1e-3 + 4e-7 * math.pi * Ms
         + math.hypot(cfg.field.Bx_mT, cfg.field.By_mT, cfg.field.Bz_mT) * 1e-3
         + abs(info["B_DL_per_1e11_mT"]) * 1e-3 * abs(cfg.drive.J_Am2) / 1e11)
    wdt = d.gamma * B * st.dt_ps * 1e-12
    rep["omega_dt"] = wdt
    lim = 0.3 if st.temperature_K > 0 else 1.0
    if wdt > lim:
        rep["warnings"].append(f"Time step too large (omega*dt = {wdt:.2f} > {lim}); "
                               f"use dt < {lim / (d.gamma * B) * 1e12:.2f} ps")
    if "Delta" in info:
        D = info["Delta"]
        if D < 1:
            rep["notes"].append(f"Thermal barrier Delta = {D:.2f} kT: superparamagnetic, fluctuates (p-bit regime)")
        elif D < 40:
            rep["notes"].append(f"Delta = {D:.1f} kT: retention ~ {info['tau_retention_s']:.2g} s (stochastic MTJ)")
        else:
            rep["notes"].append(f"Delta = {D:.0f} kT: non-volatile memory regime")
    steps = st.duration_ns * 1e-9 / (st.dt_ps * 1e-12)
    pts = st.sweep_points if st.kind != "transient" else 1
    rep["steps"] = steps
    rep["devices"] = (st.n_devices if st.kind != "spectrum" else st.n_traces) * pts
    return json.loads(json.dumps(rep, default=float))


@app.get("/api/macro/result")
def macro_result():
    if S.macro_result is None:
        raise HTTPException(404, "no result yet")
    return json.loads(json.dumps(S.macro_result, default=lambda o: None if o is None else float(o)))


@app.get("/api/macro/export.json")
def macro_export():
    if S.macro_result is None:
        raise HTTPException(404, "no result yet")
    body = json.dumps(S.macro_result, default=float, indent=1)
    return Response(body, media_type="application/json",
                    headers={"Content-Disposition": 'attachment; filename="macrospin_result.json"'})


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    S.sockets.add(websocket)
    try:
        await websocket.send_text(json.dumps({"type": "hello", "status": S.status,
                                              "has_result": S.result is not None}))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        S.sockets.discard(websocket)


def main():
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"SpinStudio on {os.uname().nodename if hasattr(os, 'uname') else ''} ({gpu}): "
          f"http://localhost:{a.port}  (Ctrl+C to stop)", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
