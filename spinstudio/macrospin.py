"""Batched macrospin engine for spintronic devices (MTJ, sMTJ / p-bits, STOs).

Thousands of independent free layers are integrated in parallel on the GPU
(stochastic LLGS with Slonczewski STT or spin-orbit torque). On top of the
device model:
  * transient / sweep studies: switching probability, write-error rate,
    p-bit transfer curve, oscillator spectra,
  * p-bit network (probabilistic computing, Ising / Boltzmann machines),
  * physical reservoir computing with spin-torque oscillators.
"""
from __future__ import annotations

import itertools
import math
import time
from typing import Callable, List, Literal, Optional

import numpy as np
import torch
from pydantic import BaseModel, Field

HBAR = 1.054571817e-34
QE = 1.602176634e-19
KB = 1.380649e-23
MU0 = 4e-7 * math.pi
AX = {"x": 0, "y": 1, "z": 2}
RD = {"mx": 0, "my": 1, "mz": 2}           # read-out component of the study


# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------
class Device(BaseModel):
    name: str = "pMTJ CoFeB/MgO"
    shape: Literal["ellipse", "rect"] = "ellipse"
    a_nm: float = Field(40.0, gt=0, description="length along x")
    b_nm: float = Field(40.0, gt=0, description="width along y")
    t_nm: float = Field(1.2, gt=0, description="free-layer thickness")
    Ms_kAm: float = Field(1100.0, gt=0)
    alpha: float = Field(0.01, ge=0)
    gamma: float = Field(1.7595e11, gt=0)
    Ku_MJm3: float = Field(1.0, description="uniaxial (interface) anisotropy")
    anis_axis: Literal["x", "y", "z"] = "z"
    demag: Literal["auto", "manual"] = "auto"
    Nx: float = 0.0
    Ny: float = 0.0
    Nz: float = 1.0


class Stack(BaseModel):
    torque: Literal["STT", "SOT"] = "STT"
    P: float = Field(0.6, description="spin polarisation (STT)")
    theta_SH: float = Field(0.3, description="spin Hall angle (SOT)")
    xi_FL: float = Field(0.0, description="field-like / damping-like ratio")
    px: float = 0.0          # STT: reference layer; SOT: spin polarisation sigma
    py: float = 0.0
    pz: float = 1.0
    rx: float = 0.0          # read-out reference layer (TMR)
    ry: float = 0.0
    rz: float = 1.0
    TMR_pct: float = 150.0
    RA_Ohm_um2: float = 5.0


class BField(BaseModel):
    Bx_mT: float = 0.0
    By_mT: float = 0.0
    Bz_mT: float = 0.0


class Drive(BaseModel):
    J_Am2: float = Field(-1e11, description="current density (sign sets the torque direction)")
    waveform: Literal["dc", "pulse", "pulse_train", "sine"] = "pulse"
    t_start_ns: float = 1.0
    width_ns: float = Field(5.0, gt=0)
    period_ns: float = Field(10.0, gt=0)
    n_pulses: int = Field(1, ge=1)
    f_GHz: float = 1.0
    rise_ns: float = Field(0.05, ge=0)


class MStudy(BaseModel):
    kind: Literal["transient", "sweep", "spectrum"] = "transient"
    dt_ps: float = Field(1.0, gt=0)
    duration_ns: float = Field(10.0, gt=0)
    n_devices: int = Field(512, ge=1, le=1_000_000)
    n_traces: int = Field(8, ge=1, le=64)
    temperature_K: float = Field(300.0, ge=0)
    seed: int = 1
    init: Literal["up", "down", "thermal", "random"] = "up"
    sweep_param: Literal["J", "Bz", "Bx", "width", "T"] = "J"
    sweep_from: float = -2e11
    sweep_to: float = 0.0
    sweep_points: int = Field(21, ge=1, le=401)
    sweep_log: bool = False
    record_every: int = Field(10, ge=1)
    readout: Literal["mx", "my", "mz", "R"] = "mz"


class PBitApp(BaseModel):
    problem: Literal["AND", "maxcut", "spin_glass"] = "AND"
    n_nodes: int = Field(12, ge=2, le=512)
    density: float = Field(0.5, gt=0, le=1)
    seed: int = 3
    beta: float = Field(1.0, gt=0, description="inverse temperature of the network")
    mode: Literal["physical", "behavioral"] = "physical"
    replicas: int = Field(64, ge=1, le=4096)
    duration_ns: float = Field(200.0, gt=0)
    sample_ns: float = Field(0.5, gt=0)
    tau_ns: float = Field(1.0, gt=0, description="behavioral p-bit update time")
    calib_points: int = Field(17, ge=5, le=101)


class ReservoirApp(BaseModel):
    task: Literal["narma2", "narma10", "sine_square", "memory"] = "narma2"
    n_osc: int = Field(64, ge=1, le=4096)
    n_samples: int = Field(1000, ge=50, le=100000)
    input_ns: float = Field(1.0, gt=0, description="hold time of one input sample")
    J_min: float = -0.9e11
    J_max: float = -1.4e11
    input_scale: float = Field(0.3, ge=0, description="relative current modulation")
    washout: int = Field(50, ge=0)
    train_frac: float = Field(0.7, gt=0.1, lt=0.95)
    ridge: float = Field(1e-6, ge=0)
    feature: Literal["mz", "mx", "mz+mx2"] = "mz"
    seed: int = 5


class MacroConfig(BaseModel):
    name: str = "pMTJ switching"
    device: Device = Device()
    stack: Stack = Stack()
    field: BField = BField()
    drive: Drive = Drive()
    study: MStudy = MStudy()
    pbit: PBitApp = PBitApp()
    reservoir: ReservoirApp = ReservoirApp()


def pmtj_preset() -> MacroConfig:
    """Perpendicular MTJ (CoFeB/MgO), STT write pulse: switching probability."""
    return MacroConfig(
        name="pMTJ STT switching",
        study=MStudy(kind="sweep", sweep_param="J", sweep_from=-0.1e11, sweep_to=-1.2e11, sweep_points=34,
                     duration_ns=12, n_devices=512, init="up", readout="mz", dt_ps=0.5),
        drive=Drive(J_Am2=-2e11, waveform="pulse", t_start_ns=1, width_ns=10),
    )


def smtj_preset() -> MacroConfig:
    """Low-barrier in-plane sMTJ: p-bit transfer curve and p-bit network."""
    return MacroConfig(
        name="sMTJ p-bit (in-plane, low barrier)",
        device=Device(name="low-barrier CoFeB", a_nm=30, b_nm=28, t_nm=1.5, Ms_kAm=1000, alpha=0.02,
                      Ku_MJm3=0.0, anis_axis="x"),
        stack=Stack(torque="STT", P=0.6, px=1, py=0, pz=0, rx=1, ry=0, rz=0, TMR_pct=150),
        drive=Drive(J_Am2=0.0, waveform="dc", t_start_ns=0, rise_ns=0),
        study=MStudy(kind="sweep", sweep_param="J", sweep_from=-1e11, sweep_to=1e11, sweep_points=21,
                     duration_ns=50, n_devices=256, init="thermal", readout="mx", dt_ps=1.0,
                     temperature_K=300),
        pbit=PBitApp(problem="AND", mode="physical", replicas=64, duration_ns=300, beta=1.0),
    )


def sto_preset() -> MacroConfig:
    """Spin-torque oscillator: in-plane free layer + perpendicular polariser."""
    return MacroConfig(
        name="STO (OPP precession) + reservoir",
        device=Device(name="in-plane CoFeB", a_nm=60, b_nm=60, t_nm=2.0, Ms_kAm=1000, alpha=0.01,
                      Ku_MJm3=0.0, anis_axis="x"),
        stack=Stack(torque="STT", P=0.5, px=0, py=0, pz=1, rx=1, ry=0, rz=0),
        field=BField(Bx_mT=20),
        drive=Drive(J_Am2=-1e11, waveform="dc", t_start_ns=0, rise_ns=0.1),
        study=MStudy(kind="spectrum", sweep_param="J", sweep_from=-0.6e11, sweep_to=-1.6e11, sweep_points=21,
                     duration_ns=40, n_devices=4, n_traces=4, temperature_K=0, init="random", dt_ps=1.0,
                     readout="mx"),
        reservoir=ReservoirApp(task="narma2", n_osc=64, J_min=-0.9e11, J_max=-1.4e11),
    )


MACRO_PRESETS = {"pmtj": pmtj_preset, "smtj_pbit": smtj_preset, "sto_reservoir": sto_preset}


# ----------------------------------------------------------------------------
# device physics
# ----------------------------------------------------------------------------
def ellipsoid_demag(a, b, c):
    """Demag factors of an ellipsoid with full axes a, b, c (Osborn integrals)."""
    a, b, c = a / 2, b / 2, c / 2
    u = np.linspace(0, 1, 200001)[1:-1]
    s = (u / (1 - u)) ** 2 * max(a, b, c) ** 2          # map [0,1) -> [0, inf)
    ds = 2 * u / (1 - u) ** 3 * max(a, b, c) ** 2
    R = np.sqrt((s + a * a) * (s + b * b) * (s + c * c))
    N = [a * b * c / 2 * np.trapezoid(ds / ((s + q * q) * R), u) for q in (a, b, c)]
    N = np.array(N)
    return N / N.sum()


def device_info(cfg: MacroConfig) -> dict:
    d, st = cfg.device, cfg.stack
    a, b, t = d.a_nm * 1e-9, d.b_nm * 1e-9, d.t_nm * 1e-9
    area = math.pi * a * b / 4 if d.shape == "ellipse" else a * b
    V = area * t
    N = ellipsoid_demag(a, b, t) if d.demag == "auto" else np.array([d.Nx, d.Ny, d.Nz])
    Ms, Ku = d.Ms_kAm * 1e3, d.Ku_MJm3 * 1e6
    k = AX[d.anis_axis]
    # effective anisotropy along the easy axis: Ku - mu0 Ms^2/2 (N_easy - N_hard_in_shape)
    others = [i for i in range(3) if i != k]
    Nhard = min(N[i] for i in others)
    Keff = Ku - 0.5 * MU0 * Ms ** 2 * (N[k] - Nhard)
    T = cfg.study.temperature_K
    out = {"area_nm2": area * 1e18, "volume_nm3": V * 1e27, "N": N.tolist(), "Keff_MJm3": Keff * 1e-6,
           "Bk_eff_mT": 2 * Keff / Ms * 1e3}
    if T > 0:
        out["Delta"] = Keff * V / (KB * T)
        out["tau_retention_s"] = 1e-9 * math.exp(min(out["Delta"], 700))
    R_P = st.RA_Ohm_um2 * 1e-12 / area
    out["R_P_Ohm"] = R_P
    out["R_AP_Ohm"] = R_P * (1 + st.TMR_pct / 100)
    eff = st.P if st.torque == "STT" else st.theta_SH
    out["B_DL_per_1e11_mT"] = HBAR * eff * 1e11 / (2 * QE * Ms * t) * 1e3
    if Keff > 0 and eff > 0:
        # T = 0 instability threshold of the easy-axis state (collinear STT): B_DL = alpha * B_k
        out["Jc0_Am2"] = 2 * QE * d.alpha * Ms * t * (2 * Keff / Ms) / (HBAR * eff)
    out["I_per_1e11_uA"] = 1e11 * area * 1e6
    return out


def _signal(t, kind: str, t0, w, period, n, f, rise):
    """Per-device current shape at time t (tensors broadcast over devices)."""
    def pulse(ts):
        if rise > 0:
            return ((t - ts) / rise).clamp(0, 1) * ((ts + w - t) / rise).clamp(0, 1)
        return ((t >= ts) & (t < ts + w)).to(t.dtype)
    if kind == "dc":
        return ((t - t0) / rise).clamp(0, 1) if rise > 0 else (t >= t0).to(t.dtype)
    if kind == "pulse":
        return pulse(t0)
    if kind == "pulse_train":
        k = torch.floor((t - t0) / period).clamp(0, n - 1)
        return pulse(t0 + k * period)
    return torch.sin(2 * math.pi * f * (t - t0)) * (t >= t0).to(t.dtype)


def _cross(a, b):
    return torch.stack((a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                        a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                        a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]), -1)


def _field(m, P, bdl, bth):
    """Effective field (..., 3): Zeeman + anisotropy + demag + spin torque (+ thermal)."""
    B = P["Bext"] + P["kuf"] * (m * P["u"]).sum(-1, keepdim=True) * P["u"] - P["dmf"] * m
    B = B + bdl * (_cross(m, P["p"]) + P["xi"] * P["p"])
    return B + bth


def _llg(m, B, P):
    mxB = _cross(m, B)
    return -P["pre"] * (mxB + P["alpha"] * _cross(m, mxB))


def _norm(m):
    return m * torch.rsqrt((m * m).sum(-1, keepdim=True))


def _step(m, t, P, noise, h: float, kind: str, stoch: bool):
    """One stochastic-Heun (T > 0) or RK4 (T = 0) step for all devices."""
    bdl = P["bdl0"] * _signal(t, kind, P["t0"], P["w"], P["period"], P["n"], P["f"], P["rise"])
    if "bdl_in" in P:                                  # network / reservoir input current
        bdl = bdl + P["bdl_in"]
    bdl = bdl.unsqueeze(-1)
    if stoch:
        bth = P["sig"] * noise
        k1 = _llg(m, _field(m, P, bdl, bth), P)
        mp = _norm(m + h * k1)
        k2 = _llg(mp, _field(mp, P, bdl, bth), P)
        return _norm(m + 0.5 * h * (k1 + k2))
    z = 0.0
    k1 = _llg(m, _field(m, P, bdl, z), P)
    k2 = _llg(m + 0.5 * h * k1, _field(m + 0.5 * h * k1, P, bdl, z), P)
    k3 = _llg(m + 0.5 * h * k2, _field(m + 0.5 * h * k2, P, bdl, z), P)
    k4 = _llg(m + h * k3, _field(m + h * k3, P, bdl, z), P)
    return _norm(m + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))


_COMPILED = {}


def _compiled(name, fn, dev):
    if dev.type != "cuda":
        return fn
    if name not in _COMPILED:
        _COMPILED[name] = torch.compile(fn, dynamic=False)
    return _COMPILED[name]


class MacroModel:
    """N independent macrospins. Per-device parameters allow sweeps in one batch."""

    def __init__(self, cfg: MacroConfig, device="cuda", n=None, overrides=None):
        self.cfg = cfg
        self.dev = torch.device(device if torch.cuda.is_available() else "cpu")
        d, st, dr, s = cfg.device, cfg.stack, cfg.drive, cfg.study
        self.n = n or s.n_devices
        self.dt = s.dt_ps * 1e-12
        self.h = d.gamma * self.dt
        self.info = device_info(cfg)
        kw = dict(dtype=torch.float32, device=self.dev)
        o = overrides or {}
        Ms = d.Ms_kAm * 1e3
        t = d.t_nm * 1e-9
        V = self.info["volume_nm3"] * 1e-27
        N = torch.tensor(self.info["N"], **kw)

        def per(name, default):
            v = o.get(name, default)
            return torch.as_tensor(v, **kw).expand(self.n).clone() if np.ndim(v) == 0 else torch.as_tensor(v, **kw)

        J = per("J", dr.J_Am2)
        T = per("T", s.temperature_K)
        eff = st.P if st.torque == "STT" else st.theta_SH
        p = np.array([st.px, st.py, st.pz], float)
        p = p / max(np.linalg.norm(p), 1e-12)
        r = np.array([st.rx, st.ry, st.rz], float)
        self.ref = torch.tensor(r / max(np.linalg.norm(r), 1e-12), **kw)
        u = torch.zeros(3, **kw)
        u[AX[d.anis_axis]] = 1
        Bext = torch.tensor([cfg.field.Bx_mT, cfg.field.By_mT, cfg.field.Bz_mT], **kw) * 1e-3
        Bext = Bext.expand(self.n, 3).clone()
        if "Bz" in o:
            Bext[:, 2] = torch.as_tensor(o["Bz"], **kw) * 1e-3
        if "Bx" in o:
            Bext[:, 0] = torch.as_tensor(o["Bx"], **kw) * 1e-3
        self.bdl_unit = HBAR * eff / (2 * QE * Ms * t)        # T per A/m^2
        self.P = {
            "Bext": Bext, "kuf": 2 * d.Ku_MJm3 * 1e6 / Ms, "u": u, "dmf": MU0 * Ms * N,
            "p": torch.tensor(p, **kw), "xi": st.xi_FL, "alpha": d.alpha, "pre": 1 / (1 + d.alpha ** 2),
            "bdl0": self.bdl_unit * J,
            "t0": dr.t_start_ns * 1e-9, "w": per("width", dr.width_ns) * 1e-9 if "width" in o else dr.width_ns * 1e-9,
            "period": dr.period_ns * 1e-9, "n": dr.n_pulses, "f": dr.f_GHz * 1e9, "rise": dr.rise_ns * 1e-9,
            "sig": torch.sqrt(2 * d.alpha * KB * T / (d.gamma * Ms * V * self.dt)).unsqueeze(-1),
        }
        self.kind = dr.waveform
        self.stoch = bool((T > 0).any())
        self.easy = u

    def initial(self, gen):
        s, n = self.cfg.study, self.n
        kw = dict(dtype=torch.float32, device=self.dev)
        u = self.easy.expand(n, 3)
        if s.init == "up":
            m = u + 0.03 * torch.randn((n, 3), generator=gen, **kw)
        elif s.init == "down":
            m = -u + 0.03 * torch.randn((n, 3), generator=gen, **kw)
        elif s.init == "random":
            m = torch.randn((n, 3), generator=gen, **kw)
        else:   # thermal: random sign along the easy axis plus spread
            sgn = torch.where(torch.rand((n, 1), generator=gen, **kw) < 0.5, -1.0, 1.0)
            m = sgn * u + 0.3 * torch.randn((n, 3), generator=gen, **kw)
        return _norm(m)

    def resistance(self, m):
        info = self.info
        c = (m * self.ref).sum(-1)
        return info["R_P_Ohm"] + (info["R_AP_Ohm"] - info["R_P_Ohm"]) * (1 - c) / 2

    @torch.no_grad()
    def run(self, steps, record_every=10, n_traces=8, progress=None, should_stop=None,
            m=None, t_offset=0.0, keep_full=False, K=64):
        """Integrate all devices (chunks of K steps replayed as CUDA graphs).

        Returns traces (every record_every steps) and per-device statistics.
        """
        s = self.cfg.study
        gen = torch.Generator(device=self.dev)
        gen.manual_seed(s.seed)
        torch.manual_seed(s.seed)
        n = self.n
        kw = dict(dtype=torch.float32, device=self.dev)
        step = _compiled("macro_step", _step, self.dev)
        easy, dt = self.easy, self.dt
        rd = self.ref if s.readout == "R" else torch.eye(3, **kw)[RD.get(s.readout, 2)]
        nt = min(n_traces, n)
        rec = max(1, record_every)
        # static state
        st_m = (self.initial(gen) if m is None else m).clone()
        sgn0 = torch.sign((st_m * easy).sum(-1))
        sgn0 = torch.where(sgn0 == 0, torch.ones_like(sgn0), sgn0)
        st_base = torch.zeros((), **kw)
        st_ts = torch.full((n,), float("inf"), **kw)
        st_sacc = torch.zeros(n, **kw)
        st_macc = torch.zeros((n, 3), **kw)
        b_tr = torch.zeros((K, nt, 3), **kw)
        b_ens = torch.zeros((K, 3), **kw)
        b_frac = torch.zeros(K, **kw)
        b_full = torch.zeros((K, n, 3), **kw) if keep_full else None
        zero_noise = torch.zeros((n, 3), **kw)

        def chunk():
            for j in range(K):
                t = (st_base + j) * dt + t_offset
                noise = torch.randn((n, 3), **kw) if self.stoch else zero_noise
                mn = step(st_m, t, self.P, noise, self.h, self.kind, self.stoch)
                st_m.copy_(mn)
                proj = (mn * easy).sum(-1)
                st_ts.copy_(torch.where(torch.isinf(st_ts) & (proj * sgn0 < -0.5), t + dt, st_ts))
                st_sacc.add_(torch.sign((mn * rd).sum(-1)))
                st_macc.add_(mn)
                b_tr[j].copy_(mn[:nt])
                b_ens[j].copy_(mn.mean(0))
                b_frac[j].copy_((proj * sgn0 < 0).float().mean())
                if b_full is not None:
                    b_full[j].copy_(mn)
            st_base.add_(K)

        replay, self.graph_used = graph_chunk(chunk, [st_m, st_base, st_ts, st_sacc, st_macc], self.dev)
        n_chunks = -(-steps // K)
        steps = n_chunks * K
        R = steps // rec + 1
        traces = torch.zeros((R, nt, 3), **kw)
        ens = torch.zeros((R, 3), **kw)
        frac = torch.zeros(R, **kw)
        full = torch.zeros((R, n, 3), **kw) if keep_full else None
        traces[0], ens[0] = st_m[:nt], st_m.mean(0)
        if full is not None:
            full[0] = st_m
        r = 1
        t0w = time.perf_counter()
        last = t0w
        stopped = False
        done = 0
        for c in range(n_chunks):
            replay()
            js = [j for j in range(K) if (c * K + j + 1) % rec == 0]
            if js:
                idx = torch.tensor(js, device=self.dev)
                traces[r:r + len(js)] = b_tr[idx]
                ens[r:r + len(js)] = b_ens[idx]
                frac[r:r + len(js)] = b_frac[idx]
                if full is not None:
                    full[r:r + len(js)] = b_full[idx]
                r += len(js)
            done = (c + 1) * K
            now = time.perf_counter()
            if progress and now - last > 0.3:
                last = now
                torch.cuda.synchronize() if self.dev.type == "cuda" else None
                progress(done, steps, done / (time.perf_counter() - t0w))
            if should_stop and (c % 50 == 49) and should_stop():
                stopped = True
                break
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0w
        return {
            "m": st_m, "t_switch": st_ts, "s_avg": st_sacc / done, "m_avg": st_macc / done,
            "traces": traces[:r], "ens": ens[:r], "frac": frac[:r], "full": full[:r] if full is not None else None,
            "t": (np.arange(r) * rec * dt + t_offset), "wall": wall, "steps": done,
            "rate": done / max(wall, 1e-9), "stopped": stopped,
        }


def graph_chunk(fn, state, dev):
    """Capture fn (which updates the static tensors in `state` in place) as a CUDA graph.

    Returns (replay callable, used_graph). The warm-up run is undone.
    """
    if dev.type != "cuda":
        return fn, False
    saved = [x.clone() for x in state]
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fn()                                    # compiles kernels
        torch.cuda.current_stream().wait_stream(side)
        for x, v in zip(state, saved):
            x.copy_(v)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        for x, v in zip(state, saved):
            x.copy_(v)
        return g.replay, True
    except Exception:  # noqa: BLE001
        for x, v in zip(state, saved):
            x.copy_(v)
        return fn, False


# ----------------------------------------------------------------------------
# studies
# ----------------------------------------------------------------------------
def sweep_values(s: MStudy):
    if s.sweep_points == 1:
        return np.array([s.sweep_from])
    if s.sweep_log and s.sweep_from * s.sweep_to > 0:
        return np.sign(s.sweep_from) * np.geomspace(abs(s.sweep_from), abs(s.sweep_to), s.sweep_points)
    return np.linspace(s.sweep_from, s.sweep_to, s.sweep_points)


def _psd(x, dt):
    """One-sided power spectrum (Hann) of x (..., T)."""
    x = x - x.mean(-1, keepdims=True)
    w = np.hanning(x.shape[-1])
    X = np.fft.rfft(x * w, axis=-1)
    return np.fft.rfftfreq(x.shape[-1], dt), (np.abs(X) ** 2).mean(axis=tuple(range(x.ndim - 1))) if x.ndim > 1 else np.abs(X) ** 2


def run_study(cfg: MacroConfig, progress=None, should_stop=None) -> dict:
    s = cfg.study
    steps = int(round(s.duration_ns * 1e-9 / (s.dt_ps * 1e-12)))
    info = device_info(cfg)
    if s.kind == "transient":
        mm = MacroModel(cfg)
        res = mm.run(steps, s.record_every, s.n_traces, progress, should_stop)
        R = mm.resistance(res["traces"]).cpu().numpy()
        ts = res["t_switch"].cpu().numpy()
        return {
            "kind": "transient", "info": info, "t_ns": (res["t"] * 1e9).tolist(),
            "traces": res["traces"].cpu().numpy().transpose(1, 2, 0).tolist(),     # (nt, 3, R)
            "R": R.T.tolist(), "ens": res["ens"].cpu().numpy().T.tolist(),
            "p_switch_t": res["frac"].cpu().numpy().tolist(),
            "p_switch": float(np.isfinite(ts).mean()),
            "t_switch_ns": (ts[np.isfinite(ts)] * 1e9).tolist(),
            "steps_per_s": res["rate"], "wall": res["wall"], "n_devices": mm.n,
            "drive": _drive_curve(cfg, res["t"]),
        }
    vals = sweep_values(s)
    P = len(vals)
    nd = s.n_devices if s.kind == "sweep" else max(1, s.n_traces)
    key = {"J": "J", "Bz": "Bz", "Bx": "Bx", "width": "width", "T": "T"}[s.sweep_param]
    ov = {key: np.repeat(vals, nd)}
    mm = MacroModel(cfg, n=P * nd, overrides=ov)
    keep = s.kind == "spectrum"
    rec = 1 if keep else s.record_every
    if keep and steps * P * nd * 3 * 4 > 2e9:
        raise ValueError("Spectrum study too large: reduce duration, points or traces")
    res = mm.run(steps, rec, min(s.n_traces, P * nd), progress, should_stop, keep_full=keep)
    out = {"kind": s.kind, "info": info, "param": s.sweep_param, "values": vals.tolist(),
           "steps_per_s": res["rate"], "wall": res["wall"], "n_devices": mm.n}
    ts = res["t_switch"].cpu().numpy().reshape(P, nd)
    if s.kind == "sweep":
        sw = np.isfinite(ts)
        p = sw.mean(1)
        out["p_switch"] = p.tolist()
        out["p_err"] = np.sqrt(np.maximum(p * (1 - p), 1 / nd ** 2) / nd).tolist()
        out["t_switch_ns"] = [float(np.median(r[np.isfinite(r)]) * 1e9) if np.isfinite(r).any() else None for r in ts]
        sa = res["s_avg"].cpu().numpy().reshape(P, nd)
        out["s_mean"] = sa.mean(1).tolist()
        out["s_err"] = (sa.std(1) / np.sqrt(nd)).tolist()
        out["m_final"] = res["m"].cpu().numpy().reshape(P, nd, 3).mean(1).T.tolist()
        out["fit"] = _fit_curves(vals, p, sa.mean(1))
    else:
        full = res["full"].cpu().numpy()                     # (R, P*nd, 3)
        comp = RD.get(s.readout, 0)
        x = full[:, :, comp].T.reshape(P, nd, -1)
        burn = x.shape[-1] // 4
        f, _ = _psd(x[0, :, burn:], mm.dt)
        spec = np.stack([_psd(x[i, :, burn:], mm.dt)[1] for i in range(P)])
        fmask = f <= 40e9
        peak = [float(f[1:][np.argmax(sp[1:])] / 1e9) if sp[1:].max() > 1e-12 * max(spec.max(), 1e-30) else None
                for sp in spec]
        amp = x[:, :, burn:].std(-1).mean(1)
        out.update({"f_GHz": (f[fmask] / 1e9).tolist(),
                    "psd": np.log10(spec[:, fmask] + spec.max() * 1e-12).tolist(),
                    "peak_GHz": peak, "amplitude": amp.tolist(),
                    "trace_t_ns": (res["t"][-min(2000, len(res['t'])):] * 1e9).tolist(),
                    "trace": x[P // 2, 0, -min(2000, x.shape[-1]):].tolist(),
                    "trace_param": float(vals[P // 2])})
    return out


def _drive_curve(cfg, t):
    dr = cfg.drive
    tt = torch.tensor(t, dtype=torch.float64)
    sh = _signal(tt, dr.waveform, dr.t_start_ns * 1e-9, dr.width_ns * 1e-9, dr.period_ns * 1e-9,
                 dr.n_pulses, dr.f_GHz * 1e9, dr.rise_ns * 1e-9)
    return (sh.numpy() * dr.J_Am2).tolist()


def _fit_curves(x, p_switch, s_mean):
    """50 % switching point and p-bit tanh width by least squares (numpy only)."""
    out = {}
    x = np.asarray(x, float)
    if np.any(p_switch > 0.5) and np.any(p_switch < 0.5):
        i = np.where(np.diff(np.sign(p_switch - 0.5)))[0][0]
        x0, x1, y0, y1 = x[i], x[i + 1], p_switch[i], p_switch[i + 1]
        out["x50"] = float(x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0))
    if np.ptp(s_mean) > 0.5:
        best = None
        for scale in np.geomspace(np.ptp(x) * 1e-3, np.ptp(x) * 10, 400):
            for off in np.linspace(x.min(), x.max(), 81):
                e = np.sum((np.tanh((x - off) / scale) - s_mean) ** 2)
                if best is None or e < best[0]:
                    best = (e, scale, off)
        out["tanh_scale"], out["tanh_offset"] = float(best[1]), float(best[2])
        out["tanh_rms"] = float(np.sqrt(best[0] / len(x)))
    return out


# ----------------------------------------------------------------------------
# p-bit network (probabilistic computing)
# ----------------------------------------------------------------------------
def pbit_problem(app: PBitApp):
    """Coupling matrix J, bias h (Ising convention E = -1/2 s.J.s - h.s), labels."""
    rng = np.random.default_rng(app.seed)
    if app.problem == "AND":
        J = np.array([[0, -1, 2], [-1, 0, 2], [2, 2, 0]], float)
        h = np.array([1, 1, -2], float)
        return J, h, ["A", "B", "C = A AND B"]
    n = app.n_nodes
    if app.problem == "maxcut":
        W = (rng.random((n, n)) < app.density).astype(float)
        W = np.triu(W, 1)
        W = W + W.T
        return -W, np.zeros(n), [f"n{i}" for i in range(n)]
    Jm = rng.normal(size=(n, n)) / math.sqrt(n)
    Jm = np.triu(Jm, 1)
    return Jm + Jm.T, np.zeros(n), [f"s{i}" for i in range(n)]


def ising_energy(S, J, h):
    return -0.5 * np.einsum("...i,ij,...j->...", S, J, S) - S @ h


def exact_boltzmann(J, h, beta):
    n = len(h)
    if n > 20:
        return None
    S = np.array(list(itertools.product([-1, 1], repeat=n)))[:, ::-1]   # index bit i = s_i
    E = ising_energy(S.astype(float), J, h)
    w = np.exp(-beta * (E - E.min()))
    return {"E": E, "p": w / w.sum(), "S": S}


def calibrate_pbit(cfg: MacroConfig, points=17, log=None):
    """Measure the sMTJ transfer curve <s>(J) and fit <s> = tanh((J - J_off) / J0)."""
    info = device_info(cfg)
    st = cfg.stack
    eff = st.P if st.torque == "STT" else st.theta_SH
    d = cfg.device
    V = info["volume_nm3"] * 1e-27
    T = max(cfg.study.temperature_K, 1.0)
    # current scale at which the spin torque competes with thermal agitation
    lim = 4 * 2 * QE * d.t_nm * 1e-9 * max(d.alpha, 1e-3) * KB * T / (HBAR * max(eff, 1e-3) * V)
    # ... and with the anisotropy / easy-plane (demag) stiffness of the free layer
    N = np.array(info["N"])
    Bs = abs(info["Bk_eff_mT"]) * 1e-3 + 0.5 * MU0 * d.Ms_kAm * 1e3 * (N.max() - N.min())
    lim = max(lim, 2 * 2 * QE * d.alpha * d.Ms_kAm * 1e3 * d.t_nm * 1e-9 * Bs / (HBAR * max(eff, 1e-3)))
    for it in range(6):
        c = cfg.model_copy(deep=True)
        c.study.kind, c.study.sweep_param = "sweep", "J"
        c.study.n_devices = 128
        c.study.duration_ns = min(max(cfg.study.duration_ns, 20.0), 40.0)
        c.drive.waveform, c.drive.t_start_ns, c.drive.rise_ns = "dc", 0.0, 0.0
        c.study.init = "thermal"
        c.study.sweep_from, c.study.sweep_to, c.study.sweep_points = -lim, lim, points
        cal = run_study(c)
        s = np.array(cal["s_mean"])
        if log:
            log(f"  calibration range ±{lim:.2g} A/m²: <s> from {s[0]:+.2f} to {s[-1]:+.2f}")
        if abs(s[0]) < 0.8 or abs(s[-1]) < 0.8:
            lim *= 3                      # not saturated: widen
            continue
        if np.sum(np.abs(s) < 0.8) < 4:
            lim /= 2.5                    # too steep for a good fit: zoom in
            continue
        break
    fit = cal["fit"]
    if "tanh_scale" not in fit:
        raise ValueError("The device does not behave as a p-bit (no tanh-like response). "
                         "Lower the energy barrier (smaller Ku / volume) or raise T.")
    return {"J": cal["values"], "s": cal["s_mean"], "s_err": cal["s_err"], "J0": fit["tanh_scale"],
            "J_off": fit["tanh_offset"], "rms": fit["tanh_rms"]}


@torch.no_grad()
def run_pbit(cfg: MacroConfig, progress=None, should_stop=None, log=None) -> dict:
    app = cfg.pbit
    J, h, labels = pbit_problem(app)
    n = len(h)
    R = app.replicas
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kw = dict(dtype=torch.float32, device=dev)
    Jt, ht = torch.tensor(J, **kw), torch.tensor(h, **kw)
    dt = cfg.study.dt_ps * 1e-12
    every = max(1, int(round(app.sample_ns * 1e-9 / dt)))
    K = min(every, 512)
    every = max(K, int(round(every / K)) * K)
    n_chunks = max(1, int(round(app.duration_ns * 1e-9 / dt)) // K)
    steps = n_chunks * K
    calib = None
    torch.manual_seed(app.seed)
    st_base = torch.zeros((), **kw)
    if app.mode == "physical":
        if log:
            log("Calibrating the sMTJ p-bit transfer curve <s>(J)...")
        calib = calibrate_pbit(cfg, app.calib_points, log)
        J0, Joff = calib["J0"], calib["J_off"]
        if log:
            log(f"p-bit: <s> = tanh((J - {Joff:.3g}) / {J0:.3g}) A/m², fit rms {calib['rms']:.3f}")
        c2 = cfg.model_copy(deep=True)
        c2.study.init = "thermal"
        c2.drive.waveform, c2.drive.t_start_ns, c2.drive.rise_ns = "dc", 0.0, 0.0
        mm = MacroModel(c2, n=R * n, overrides={"J": np.zeros(R * n)})
        mm.P["bdl_in"] = torch.zeros(R * n, **kw)
        gen = torch.Generator(device=dev)
        gen.manual_seed(app.seed)
        st_m = mm.initial(gen)
        step = _compiled("macro_step", _step, dev)
        rd = mm.ref
        bdl_unit = mm.bdl_unit

        def chunk():
            for j in range(K):
                s = torch.sign((st_m * rd).sum(-1)).view(R, n)
                I = app.beta * (s @ Jt + ht)                                   # local fields
                mm.P["bdl_in"].copy_((bdl_unit * (Joff + J0 * I)).view(-1))    # input currents
                mn = step(st_m, (st_base + j) * dt, mm.P, torch.randn((R * n, 3), **kw), mm.h, "dc", True)
                st_m.copy_(mn)
            st_base.add_(K)

        state = [st_m, st_base, mm.P["bdl_in"]]

        def states():
            return torch.sign((st_m * rd).sum(-1)).view(R, n)
    else:
        st_s = torch.where(torch.rand((R, n), device=dev) < 0.5, -1.0, 1.0)
        p_upd = dt / (app.tau_ns * 1e-9)

        def chunk():
            for _ in range(K):
                I = app.beta * (st_s @ Jt + ht)
                upd = torch.rand((R, n), device=dev) < p_upd
                new = torch.where(torch.rand((R, n), device=dev) * 2 - 1 < torch.tanh(I), 1.0, -1.0)
                st_s.copy_(torch.where(upd, new, st_s))
            st_base.add_(K)

        state = [st_s, st_base]

        def states():
            return st_s
    replay, used = graph_chunk(chunk, state, dev)
    if log:
        log(f"p-bit network: {n} p-bits × {R} replicas, {steps} steps, CUDA graph {'on' if used else 'off'}")
    counts = torch.zeros(2 ** n if n <= 20 else 1, **kw)
    pow2 = (2 ** torch.arange(n, device=dev)).float() if n <= 20 else None
    E_trace, best_trace, t_trace = [], [], []
    best_E, best_S = float("inf"), None
    t0w = time.perf_counter()
    last = t0w
    burn = n_chunks // 10
    for c in range(n_chunks):
        replay()
        if ((c + 1) * K) % every:
            continue
        s = states()
        E = -0.5 * ((s @ Jt) * s).sum(1) - s @ ht
        if pow2 is not None and c >= burn:
            idx = ((s > 0).float() @ pow2).long()
            counts += torch.bincount(idx, minlength=counts.numel()).float()
        Emin, k = E.min(0)
        if float(Emin) < best_E:
            best_E, best_S = float(Emin), s[k].cpu().numpy()
        E_trace.append(float(E.mean()))
        best_trace.append(best_E)
        t_trace.append((c + 1) * K * dt * 1e9)
        now = time.perf_counter()
        if progress and now - last > 0.3:
            last = now
            progress((c + 1) * K, steps, (c + 1) * K / (now - t0w))
        if should_stop and should_stop():
            break
    out = {"kind": "pbit", "problem": app.problem, "labels": labels, "n": n, "J": J.tolist(), "h": h.tolist(),
           "t_ns": t_trace, "E_mean": E_trace, "E_best": best_trace, "best_E": best_E,
           "best_state": best_S.tolist() if best_S is not None else None, "calibration": calib,
           "mode": app.mode, "beta": app.beta, "wall": time.perf_counter() - t0w}
    ex = exact_boltzmann(J, h, app.beta)
    if ex is not None:
        p_sim = counts.cpu().numpy()
        p_sim = p_sim / max(p_sim.sum(), 1)
        out["states"] = ["".join("1" if b > 0 else "0" for b in st) for st in ex["S"]]
        out["p_sim"] = p_sim.tolist()
        out["p_exact"] = ex["p"].tolist()
        out["E_states"] = ex["E"].tolist()
        out["E_ground"] = float(ex["E"].min())
        nz = (p_sim > 0) & (ex["p"] > 0)
        out["KL"] = float(np.sum(p_sim[nz] * np.log(p_sim[nz] / ex["p"][nz])))
    if app.problem == "maxcut" and best_S is not None:
        W = -J
        out["cut_best"] = float(0.25 * np.sum(W * (1 - np.outer(best_S, best_S))))
        if ex is not None:
            S = ex["S"].astype(float)
            cuts = 0.25 * np.einsum("ij,kij->k", W, 1 - S[:, :, None] * S[:, None, :])
            out["cut_optimum"] = float(cuts.max())
    return out


# ----------------------------------------------------------------------------
# reservoir computing with spin-torque oscillators
# ----------------------------------------------------------------------------
def reservoir_task(app: ReservoirApp, rng):
    N = app.n_samples
    if app.task == "narma2":
        u = rng.uniform(0, 0.5, N)
        y = np.zeros(N)
        for k in range(1, N - 1):
            y[k + 1] = 0.4 * y[k] + 0.4 * y[k] * y[k - 1] + 0.6 * u[k] ** 3 + 0.1
        return u, y[:, None], "regression"
    if app.task == "narma10":
        u = rng.uniform(0, 0.5, N)
        y = np.zeros(N)
        for k in range(9, N - 1):
            y[k + 1] = 0.3 * y[k] + 0.05 * y[k] * y[k - 9:k + 1].sum() + 1.5 * u[k - 9] * u[k] + 0.1
        return u, y[:, None], "regression"
    if app.task == "memory":
        u = rng.uniform(-1, 1, N)
        K = 30
        Y = np.zeros((N, K))
        for k in range(1, K + 1):
            Y[k:, k - 1] = u[:-k]
        return u, Y, "memory"
    # sine / square classification: 8 points per period
    sine = np.sin(np.arange(8) / 8 * 2 * np.pi)
    sq = np.where(np.arange(8) < 4, 1.0, -1.0)
    u, y = [], []
    for _ in range(N // 8):
        c = rng.integers(2)
        u.extend(sine if c == 0 else sq)
        y.extend([c] * 8)
    return np.array(u), np.array(y, float)[:, None], "classification"


@torch.no_grad()
def run_reservoir(cfg: MacroConfig, progress=None, should_stop=None, log=None) -> dict:
    app = cfg.reservoir
    rng = np.random.default_rng(app.seed)
    u, Y, kind = reservoir_task(app, rng)
    N = len(u)
    un = 2 * (u - u.min()) / max(np.ptp(u), 1e-12) - 1                  # input in [-1, 1]
    no = app.n_osc
    Jb = rng.uniform(min(app.J_min, app.J_max), max(app.J_min, app.J_max), no)
    win = rng.uniform(-1, 1, no)
    c = cfg.model_copy(deep=True)
    c.drive.waveform, c.drive.t_start_ns, c.drive.rise_ns = "dc", 0.0, 0.0
    c.study.init = "random"
    mm = MacroModel(c, n=no, overrides={"J": Jb})
    dev = mm.dev
    kw = dict(dtype=torch.float32, device=dev)
    dt = mm.dt
    per = max(2, int(round(app.input_ns * 1e-9 / dt)))
    half = per // 2
    gen = torch.Generator(device=dev)
    gen.manual_seed(app.seed)
    torch.manual_seed(app.seed)
    st_m = mm.initial(gen)
    step = _compiled("macro_step", _step, dev)
    Jb_t, win_t = torch.tensor(Jb, **kw), torch.tensor(win, **kw)
    bdl0 = mm.P["bdl0"]                     # static tensor updated before every sample
    st_base = torch.zeros((), **kw)
    acc_z = torch.zeros(no, **kw)
    acc_x2 = torch.zeros(no, **kw)
    tr0 = torch.zeros((per, 3), **kw)
    zero_noise = torch.zeros((no, 3), **kw)
    nf = no * (2 if app.feature == "mz+mx2" else 1)
    feats_g = torch.zeros((N, nf), **kw)

    def chunk():                            # one input sample = `per` time steps
        acc_z.zero_()
        acc_x2.zero_()
        for j in range(per):
            noise = torch.randn((no, 3), **kw) if mm.stoch else zero_noise
            mn = step(st_m, (st_base + j) * dt, mm.P, noise, mm.h, "dc", mm.stoch)
            st_m.copy_(mn)
            if j >= half:
                acc_z.add_(mn[:, 2])
                acc_x2.add_(mn[:, 0] ** 2)
            tr0[j].copy_(mn[0])
        st_base.add_(per)

    bdl0.copy_(mm.bdl_unit * Jb_t)
    replay, used = graph_chunk(chunk, [st_m, st_base, acc_z, acc_x2], dev)
    if log:
        log(f"Reservoir: {no} oscillators, {N} samples × {per} steps, CUDA graph {'on' if used else 'off'}")
    for _ in range(5):                      # settle without input
        replay()
    t0w = time.perf_counter()
    last = t0w
    trace_t, trace_mx, trace_mz, trace_u = [], [], [], []
    stopped = False
    for k in range(N):
        bdl0.copy_(mm.bdl_unit * Jb_t * (1 + app.input_scale * win_t * float(un[k])))
        replay()
        z = acc_z / (per - half)
        if app.feature == "mz":
            feats_g[k] = z
        elif app.feature == "mx":
            feats_g[k] = st_m[:, 0]
        else:
            feats_g[k] = torch.cat([z, acc_x2 / (per - half)])
        if k < 30:
            tr = tr0[::5].cpu().numpy()
            trace_t.extend(((k + 5) * per + np.arange(0, per, 5)) * dt * 1e9)
            trace_mx.extend(tr[:, 0].tolist())
            trace_mz.extend(tr[:, 2].tolist())
            trace_u.extend([float(un[k])] * len(tr))
        now = time.perf_counter()
        if progress and now - last > 0.3:
            last = now
            torch.cuda.synchronize() if dev.type == "cuda" else None
            progress(k + 1, N, (k + 1) * per / (time.perf_counter() - t0w))
        if should_stop and k % 20 == 19 and should_stop():
            stopped = True
            N = k + 1
            break
    feats = feats_g[:N].cpu().numpy()
    Y, u = Y[:N], u[:N]
    # ridge regression read-out
    w0 = app.washout
    ntr = int((N - w0) * app.train_frac) + w0
    Xf = np.hstack([feats, np.ones((N, 1))])
    mu, sd = Xf[w0:ntr].mean(0), Xf[w0:ntr].std(0) + 1e-9
    mu[-1], sd[-1] = 0, 1
    Xs = (Xf - mu) / sd
    A = Xs[w0:ntr]
    Wout = np.linalg.solve(A.T @ A + app.ridge * len(A) * np.eye(A.shape[1]), A.T @ Y[w0:ntr])
    pred = Xs @ Wout
    te = slice(ntr, N)
    out = {"kind": "reservoir", "task": app.task, "n_osc": no, "n_samples": N, "train_end": ntr,
           "washout": w0, "wall": time.perf_counter() - t0w, "stopped": stopped,
           "trace": {"t_ns": trace_t, "mx": trace_mx, "mz": trace_mz, "u": trace_u},
           "features": feats[w0:w0 + 200, :min(no, 64)].T.tolist()}
    if kind == "memory":
        r2 = []
        for kk in range(Y.shape[1]):
            yt, yp = Y[te, kk], pred[te, kk]
            r = np.corrcoef(yt, yp)[0, 1] if np.std(yp) > 0 else 0.0
            r2.append(float(r ** 2))
        out.update({"delays": list(range(1, Y.shape[1] + 1)), "r2": r2, "MC": float(np.sum(r2))})
        # baseline: linear memory of the input itself
        out["metric"] = f"memory capacity MC = {out['MC']:.2f}"
    else:
        yt, yp = Y[te, 0], pred[te, 0]
        if kind == "classification":
            acc = float(np.mean((yp > 0.5) == (yt > 0.5)))
            out.update({"accuracy": acc, "metric": f"accuracy = {100 * acc:.1f} %"})
        else:
            nmse = float(np.mean((yt - yp) ** 2) / np.var(yt))
            out.update({"NMSE": nmse, "metric": f"NMSE = {nmse:.4f}"})
            # linear baseline: ridge on the delayed raw inputs (no reservoir)
            D = 10
            U = np.stack([np.roll(u, d) for d in range(D)] + [np.ones(N)], 1)
            Wl = np.linalg.lstsq(U[w0:ntr], Y[w0:ntr, 0], rcond=None)[0]
            out["NMSE_linear"] = float(np.mean((yt - U[te] @ Wl) ** 2) / np.var(yt))
        n_show = min(300, N - ntr)
        out.update({"test_k": list(range(ntr, ntr + n_show)), "target": yt[:n_show].tolist(),
                    "pred": yp[:n_show].tolist()})
    return out
