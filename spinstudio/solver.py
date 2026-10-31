"""Fast GPU micromagnetic solver for 2D thin films.

Physics is the same as SpinTorch (LLG + exchange + thin-film demag + absorbing
boundaries, RK4 time stepping), rewritten for speed:
  * demag via real FFTs; the 2D kernel spectra are purely real, so they are
    stored as real tensors (half the memory and multiplications),
  * the whole RK4 step (incl. sources and probes) is one torch.compile graph,
    optionally replayed with CUDA graphs (removes Python/launch overhead),
  * no host-device synchronisation inside the time loop,
  * one level of gradient checkpointing (SpinTorch recomputes twice).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch._dynamo
from torch.utils.checkpoint import checkpoint

from .config import Config

MU0 = 4e-7 * math.pi
torch._dynamo.config.cache_size_limit = 64   # many configs per server session
COMP = {"x": 0, "y": 1, "z": 2}


# ----------------------------------------------------------------------------
# Demagnetisation kernel
# ----------------------------------------------------------------------------
def demag_kernel_2d(nx: int, ny: int, dx: float, dy: float, dz: float):
    """Thin-film (single layer) demag tensor, same formula as SpinTorch.

    Returns real numpy arrays Kxx, Kyy, Kzz, Kxy of shape (2nx, 2ny) with the
    zero lag stored at index (0, 0) (wrap-around order, ready for FFT).
    """
    I = np.arange(-nx + 1, nx, dtype=np.float64)[:, None]
    J = np.arange(-ny + 1, ny, dtype=np.float64)[None, :]
    shape = (2 * nx - 1, 2 * ny - 1)
    Kxx, Kyy, Kzz, Kxy = (np.zeros(shape) for _ in range(4))
    for i in (-0.5, 0.5):
        for j in (-0.5, 0.5):
            for k in (-0.5, 0.5):
                sgn = (-1.0) ** int(round(i + j + k + 1.5)) / (4 * math.pi)
                X, Y, Z = (I + i) * dx, (J + j) * dy, k * dz
                r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
                Kxx += sgn * np.arctan(Z * Y / (r * X))
                Kyy += sgn * np.arctan(X * Z / (r * Y))
                Kzz += sgn * np.arctan(Y * X / (r * Z))
                Kxy -= sgn * np.log(np.abs(Z + r))
    out = []
    for K in (Kxx, Kyy, Kzz, Kxy):
        full = np.zeros((2 * nx, 2 * ny))
        full[: 2 * nx - 1, : 2 * ny - 1] = K
        out.append(np.roll(full, (-(nx - 1), -(ny - 1)), axis=(0, 1)))
    return out


def demag_kernel_fft(nx, ny, dx, dy, dz, dtype, device):
    """Real-valued rFFT spectra of the demag kernel, stacked (4, 2nx, ny+1)."""
    specs = []
    for K in demag_kernel_2d(nx, ny, dx, dy, dz):
        S = np.fft.rfft2(K)
        # Kxx, Kyy, Kzz are even and Kxy is odd in both lags -> spectra are real
        assert np.abs(S.imag).max() <= 1e-9 * max(np.abs(S.real).max(), 1e-30)
        specs.append(S.real)
    return torch.tensor(np.stack(specs), dtype=dtype, device=device)


# ----------------------------------------------------------------------------
# Pure tensor physics (compiled)
#
# S is a dict with the static material tensors of a run (see Model.materials):
#   Msat (1,1,nx,ny) [A/m], exf = 2A/Ms, dmf = 2D/Ms, bcx/bcy = D/(2A)*dx (DMI
#   boundary condition), kuf = 2Ku/Ms, ax (1,3,1,1) anisotropy axis, nb* neighbour
#   masks, K demag spectra, Bext, src (source masks), tq_* torque maps, P probe
#   matrix, sigma_th thermal field std, scalars and flags.
# ----------------------------------------------------------------------------
def _cross(a, b):
    ax, ay, az = a[:, 0], a[:, 1], a[:, 2]
    bx, by, bz = b[:, 0], b[:, 1], b[:, 2]
    return torch.stack((ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx), 1)


def _neighbours(m, S):
    """m at x+1, x-1, y+1, y-1 with ghost cells at edges and next to vacuum.

    The ghost values implement the natural boundary condition of the energy
    E = A (grad m)^2 + D [mz div m - (m . grad) mz]  (the one the bulk DMI field below
    derives from):  dm/dn = -(D/2A) (z x n) x m.  Without DMI: free (Neumann) boundary.
    """
    if S["dmi"]:
        mx, my, mz = m[:, 0:1], m[:, 1:2], m[:, 2:3]
        z = torch.zeros_like(mx)
        gx = S["bcx"] * torch.cat((mz, z, -mx), 1)      # dx * (D/2A) (z x x) x m
        gy = S["bcy"] * torch.cat((z, mz, -my), 1)      # dy * (D/2A) (z x y) x m
        mR = torch.where(S["nbR"], torch.roll(m, -1, 2), m - gx)
        mL = torch.where(S["nbL"], torch.roll(m, 1, 2), m + gx)
        mU = torch.where(S["nbU"], torch.roll(m, -1, 3), m - gy)
        mD = torch.where(S["nbD"], torch.roll(m, 1, 3), m + gy)
    else:
        mR = torch.where(S["nbR"], torch.roll(m, -1, 2), m)
        mL = torch.where(S["nbL"], torch.roll(m, 1, 2), m)
        mU = torch.where(S["nbU"], torch.roll(m, -1, 3), m)
        mD = torch.where(S["nbD"], torch.roll(m, 1, 3), m)
    return mR, mL, mU, mD


def _demag(m, S):
    nx, ny = m.shape[-2], m.shape[-1]
    Mf = torch.fft.rfft2(m * S["Msat"], s=(2 * nx, 2 * ny))
    K = S["K"]
    Bf = torch.stack((K[0] * Mf[:, 0] + K[3] * Mf[:, 1],
                      K[3] * Mf[:, 0] + K[1] * Mf[:, 1],
                      K[2] * Mf[:, 2]), 1)
    return MU0 * torch.fft.irfft2(Bf, s=(2 * nx, 2 * ny))[..., :nx, :ny]


def _terms(m, S):
    """Internal field terms: exchange, demag, DMI, anisotropy (DMI/anisotropy may be None)."""
    mR, mL, mU, mD = _neighbours(m, S)
    Bex = S["exf"] * ((mR + mL - 2 * m) * S["idx2"] + (mU + mD - 2 * m) * S["idy2"])
    Bd = _demag(m, S) if S["demag"] else None
    Bdmi = Ban = None
    if S["dmi"]:
        hx, hy = S["hdx"], S["hdy"]
        Bdmi = S["dmf"] * torch.cat(((mR[:, 2:3] - mL[:, 2:3]) * hx,
                                     (mU[:, 2:3] - mD[:, 2:3]) * hy,
                                     -(mR[:, 0:1] - mL[:, 0:1]) * hx - (mU[:, 1:2] - mD[:, 1:2]) * hy), 1)
    if S["anis"]:
        Ban = S["kuf"] * (m * S["ax"]).sum(1, keepdim=True) * S["ax"]
    return Bex, Bd, Bdmi, Ban, (mR, mL, mU, mD)


def _b_eff(m, B, tqA, zx, zy, S):
    Bex, Bd, Bdmi, Ban, nb = _terms(m, S)
    Be = B + Bex
    if Bd is not None:
        Be = Be + Bd
    if Bdmi is not None:
        Be = Be + Bdmi
    if Ban is not None:
        Be = Be + Ban
    if S["tq"]:
        Be = Be + _cross(m, tqA)                  # damping-like torque as an effective field
    if S["zl"]:
        mR, mL, mU, mD = nb
        v = zx * (mR - mL) * S["hdx"] + zy * (mU - mD) * S["hdy"]     # (u . grad) m
        Be = Be - (_cross(m, v) + S["beta"] * v) / S["gam"]   # Zhang-Li as effective field
    return Be


def _torque(m, B, tqA, zx, zy, S, alpha, pre):
    mxB = _cross(m, _b_eff(m, B, tqA, zx, zy, S))
    return -pre * (mxB + alpha * _cross(m, mxB))


def _norm(m):
    return m * torch.rsqrt((m * m).sum(1, keepdim=True))


def _rk4(m, B, tqA, zx, zy, S, alpha, pre, h: float, normalize: bool):
    k1 = _torque(m, B, tqA, zx, zy, S, alpha, pre)
    k2 = _torque(m + (0.5 * h) * k1, B, tqA, zx, zy, S, alpha, pre)
    k3 = _torque(m + (0.5 * h) * k2, B, tqA, zx, zy, S, alpha, pre)
    k4 = _torque(m + h * k3, B, tqA, zx, zy, S, alpha, pre)
    m = m + (h / 6) * ((k1 + 2 * k2) + (2 * k3 + k4))
    return _norm(m) if normalize else m


def _heun(m, B, tqA, zx, zy, S, alpha, pre, h: float):
    """Stochastic Heun (Stratonovich) step; B already contains the thermal field."""
    k1 = _torque(m, B, tqA, zx, zy, S, alpha, pre)
    mp = _norm(m + h * k1)
    k2 = _torque(mp, B, tqA, zx, zy, S, alpha, pre)
    return _norm(m + (0.5 * h) * (k1 + k2))


def _drive(S, sig, tsig, zsig):
    """Everything that is constant during one step."""
    B = S["Bext"] + (sig[:, None, None, None] * S["src"]).sum(0, keepdim=True)
    tqA = zx = zy = None
    if S["tq"]:
        w = tsig[:, None, None, None]
        tqA = (w * S["tq_a"] * S["tq_p"]).sum(0, keepdim=True)
        B = B + (w * S["tq_f"] * S["tq_p"]).sum(0, keepdim=True)
    if S["zl"]:
        w = zsig[:, None, None, None]
        zx = (w * S["zl_x"]).sum(0, keepdim=True)
        zy = (w * S["zl_y"]).sum(0, keepdim=True)
    return B, tqA, zx, zy


def _advance(m, sig, tsig, zsig, S, alpha, pre, h: float, normalize: bool, stoch: bool):
    B, tqA, zx, zy = _drive(S, sig, tsig, zsig)
    if stoch:
        B = B + S["bth"]
        return _heun(m, B, tqA, zx, zy, S, alpha, pre, h)
    return _rk4(m, B, tqA, zx, zy, S, alpha, pre, h, normalize)


def _sim_step(m, m0, sig, tsig, zsig, S, alpha, pre, h: float, normalize: bool, stoch: bool,
              inten, dre, dim, cw, sw):
    """One recorded time step: drive -> integrate -> probes, intensity and running DFT."""
    m = _advance(m, sig, tsig, zsig, S, alpha, pre, h, normalize, stoch)
    dm = m - m0
    probe = (dm * S["Msat"]).reshape(3, -1) @ S["P"].t()          # (3, Np)
    return m, dm, probe, inten + dm * dm, dre + dm * cw, dim - dm * sw


def _train_step(m, m0, sig, tsig, zsig, S, alpha, pre, h: float, normalize: bool, stoch: bool, comp: int):
    m = _advance(m, sig, tsig, zsig, S, alpha, pre, h, normalize, stoch)
    v = ((m - m0)[:, comp] * S["Msat"][:, 0]).reshape(-1) @ S["P"].t()   # (Np,)
    return m, v * v


def _table(m, S):
    """Scalar observables: <m>, topological charge, energies (J), texture position."""
    Bex, Bd, Bdmi, Ban, nb = _terms(m, S)
    mag = S["magf"]
    n = mag.sum()
    V = S["Vcell"]
    Ms = S["Msat"]
    avg = (m * mag).sum((0, 2, 3)) / n
    mR, mL, mU, mD = nb
    q = (m * _cross((mR - mL) * S["hdx"], (mU - mD) * S["hdy"])).sum(1, keepdim=True)
    Q = (q * mag).sum() * S["dxdy"] / (4 * math.pi)

    def e(Bt, half=True):
        if Bt is None:
            return torch.zeros((), dtype=m.dtype, device=m.device)
        return -(0.5 if half else 1.0) * V * (Ms * (m * Bt).sum(1, keepdim=True)).sum()
    Eex, Ed, Edmi, Ean, Ez = e(Bex), e(Bd), e(Bdmi), e(Ban), e(S["Bext"], False)
    # texture = region whose mz is opposite to the average (skyrmion core)
    sgn = torch.where(avg[2] >= 0, 1.0, -1.0)
    w = (sgn * m[:, 2:3] < 0).to(m.dtype) * mag              # inside the mz = 0 contour
    W = w.sum().clamp_min(1e-12)
    cx = (w * S["X"]).sum() / W
    cy = (w * S["Y"]).sum() / W
    area = W * S["dxdy"]
    return torch.stack((avg[0], avg[1], avg[2], Q, Eex + Ed + Edmi + Ean + Ez, Eex, Ed, Edmi, Ean, Ez,
                        cx * 1e9, cy * 1e9, torch.sqrt(area / math.pi) * 1e9,
                        (1 + avg[2]) * 0.5 * S["Lx"] * 1e9))


TABLE_COLS = ["mx", "my", "mz", "Q", "E_total", "E_exch", "E_demag", "E_DMI", "E_anis", "E_Zeeman",
              "core_x", "core_y", "core_r", "x_DW"]
TABLE_UNITS = ["", "", "", "", "J", "J", "J", "J", "J", "J", "nm", "nm", "nm", "nm"]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _line_cells(x0, y0, x1, y1):
    from skimage.draw import line
    return line(x0, y0, x1, y1)


def _disk_cells(cx, cy, r, shape):
    from skimage.draw import disk
    return disk((cx, cy), r, shape=shape)


def waveforms(cfg: Config, n_steps: int) -> np.ndarray:
    """Source signals sampled at t_n = n*dt (value applied during step n)."""
    dt = cfg.study.dt_ps * 1e-12
    t = np.arange(n_steps) * dt
    out = []
    for s in cfg.sources:
        if not s.enabled:
            continue
        A = s.amp_mT * 1e-3
        w = 2 * np.pi * s.f_GHz * 1e9
        ph = np.deg2rad(s.phase_deg)
        if s.waveform == "sine":
            sig = A * np.sin(w * t + ph)
            if s.ramp_ns > 0:
                tr = s.ramp_ns * 1e-9
                sig *= np.where(t < tr, 0.5 * (1 - np.cos(np.pi * t / tr)), 1.0)
        elif s.waveform == "gaussian":
            t0, sg = s.t0_ns * 1e-9, s.width_ns * 1e-9
            sig = A * np.exp(-0.5 * ((t - t0) / sg) ** 2) * np.cos(w * (t - t0) + ph)
        else:  # sinc: flat spectrum from 0 to f_GHz
            t0 = s.t0_ns * 1e-9
            sig = A * np.sinc(2 * s.f_GHz * 1e9 * (t - t0))
        out.append(sig)
    if not out:
        return np.zeros((0, n_steps))
    sig = np.stack(out)
    if cfg.study.legacy_source:
        # SpinTorch adds the source to B_ext cumulatively inside each checkpoint stage
        n_stage = int(np.sqrt(n_steps))
        chunk = int(np.ceil(n_steps / n_stage))
        for a in range(0, n_steps, chunk):
            sig[:, a:a + chunk] = np.cumsum(sig[:, a:a + chunk], axis=1)
    return sig


HBAR = 1.054571817e-34
QE = 1.602176634e-19
MUB = 9.2740100783e-24
KB = 1.380649e-23


def _shape_signal(t, kind, t0, width, period, n, f, rise):
    """Dimensionless time shape of a current (1 = full amplitude)."""
    def pulse(ts):
        if rise > 0:
            return np.clip((t - ts) / rise, 0, 1) * np.clip((ts + width - t) / rise, 0, 1)
        return ((t >= ts) & (t < ts + width)).astype(float)
    if kind == "dc":
        return np.clip((t - t0) / rise, 0, 1) if rise > 0 else (t >= t0).astype(float)
    if kind == "pulse":
        return pulse(t0)
    if kind == "pulse_train":
        return np.clip(sum(pulse(t0 + k * period) for k in range(n)), 0, 1)
    return np.sin(2 * np.pi * f * (t - t0)) * (t >= t0)


def torque_waveforms(cfg: Config, n_steps: int):
    """(damping-like torques (Kt, T), Zhang-Li torques (Kz, T)) time shapes."""
    t = np.arange(n_steps) * cfg.study.dt_ps * 1e-12
    tq, zl = [], []
    for q in cfg.torques:
        if not q.enabled:
            continue
        s = _shape_signal(t, q.waveform, q.t_start_ns * 1e-9, q.width_ns * 1e-9, q.period_ns * 1e-9,
                          q.n_pulses, q.f_GHz * 1e9, q.rise_ns * 1e-9)
        (zl if q.kind == "ZhangLi" else tq).append(s)
    tq = np.stack(tq) if tq else np.zeros((1, n_steps))
    zl = np.stack(zl) if zl else np.zeros((1, n_steps))
    return tq, zl


def anisotropy_info(cfg: Config):
    """Effective anisotropy (thin film), domain-wall width and critical DMI."""
    mt = cfg.material
    Ms, A, Ku, D = mt.Ms_kAm * 1e3, mt.Aex_pJm * 1e-12, mt.Ku_MJm3 * 1e6, mt.D_mJm2 * 1e-3
    Keff = Ku - (0.5 * MU0 * Ms ** 2 if mt.anis_axis == "z" else 0.0)
    out = {"Keff": Keff}
    if Keff > 0 and A > 0:
        out["dw_width_nm"] = math.sqrt(A / Keff) * 1e9
        out["Dc_mJm2"] = 4 * math.sqrt(A * Keff) / math.pi * 1e3
        out["dw_energy_mJm2"] = (4 * math.sqrt(A * Keff) - math.pi * abs(D)) * 1e3
    return out


def initial_magnetization(cfg: Config) -> np.ndarray:
    """Initial m (3, nx, ny) from cfg.init."""
    g, ini, mt = cfg.grid, cfg.init, cfg.material
    nx, ny = g.nx, g.ny
    X = (np.arange(nx) * g.dx_nm)[:, None] * np.ones((1, ny))
    Y = (np.arange(ny) * g.dy_nm)[None, :] * np.ones((nx, 1))
    m = np.zeros((3, nx, ny))
    if ini.kind == "field":
        B0 = np.array([cfg.field.Bx_mT, cfg.field.By_mT, cfg.field.Bz_mT])
        if np.linalg.norm(B0) > 0:
            u = B0 / np.linalg.norm(B0)
        elif mt.Ku_MJm3 > 0:
            u = np.eye(3)[COMP[mt.anis_axis]]
        else:
            u = np.array([0.0, 1.0, 0.0])
        m[:] = u[:, None, None]
    elif ini.kind == "uniform":
        u = np.array([ini.mx, ini.my, ini.mz], dtype=float)
        m[:] = (u / max(np.linalg.norm(u), 1e-12))[:, None, None]
    elif ini.kind == "random":
        m = np.random.default_rng(ini.seed).normal(size=(3, nx, ny))
    else:
        info = anisotropy_info(cfg)
        w = info.get("dw_width_nm", 5.0)
        # chirality +1 = the rotation sense favoured by the DMI sign
        # (E_DMI = D [mz div m - (m.grad) mz] is lowered by c = -sign(D) for these profiles)
        c = (1.0 if ini.chirality >= 0 else -1.0) * (-1.0 if mt.D_mJm2 >= 0 else 1.0)
        if ini.kind == "domain_wall":
            th = 2 * np.arctan(np.exp((X - ini.wall_x_nm) / w))      # up (x < x0) -> down
            if ini.wall_type == "Neel":
                m = np.stack((c * np.sin(th), 0 * th, np.cos(th)))
            else:
                m = np.stack((0 * th, c * np.sin(th), np.cos(th)))
        elif ini.kind == "skyrmion":
            r = np.hypot(X - ini.cx_nm, Y - ini.cy_nm)
            ph = np.arctan2(Y - ini.cy_nm, X - ini.cx_nm)
            th = 2 * np.arctan(np.exp(-(r - ini.radius_nm) / w))     # pi in the core, 0 outside
            # along r the polar angle decreases (core down -> up), so the favoured
            # radial component has the opposite sign to the domain-wall case
            m = np.stack((-c * np.sin(th) * np.cos(ph), -c * np.sin(th) * np.sin(ph), np.cos(th)))
            if ini.core == "up":
                m = -m
        else:  # vortex
            r = np.hypot(X - ini.cx_nm, Y - ini.cy_nm)
            rc = max(ini.radius_nm, 1e-3)
            mz = np.exp(-(r / rc) ** 2) * (1 if ini.core == "up" else -1)
            ip = np.sqrt(np.clip(1 - mz ** 2, 0, 1))
            m = np.stack((-c * (Y - ini.cy_nm) / np.maximum(r, 1e-9) * ip,
                          c * (X - ini.cx_nm) / np.maximum(r, 1e-9) * ip, mz))
    return m / np.maximum(np.linalg.norm(m, axis=0, keepdims=True), 1e-12)


def kalinikos_slavin(cfg: Config, k, phi_k):
    """Lowest thickness mode dispersion (Kalinikos-Slavin), M along B0.

    k: wavenumber array (rad/m), phi_k: in-plane angle of k (rad, from x).
    Returns f in Hz (nan where no real solution).
    """
    g = cfg.material.gamma
    Ms = cfg.material.Ms_kAm * 1e3
    A = cfg.material.Aex_pJm * 1e-12
    d = cfg.grid.dz_nm * 1e-9
    B = np.array([cfg.field.Bx_mT, cfg.field.By_mT, cfg.field.Bz_mT]) * 1e-3
    Bn = np.linalg.norm(B)
    if Bn == 0:
        return np.full_like(k, np.nan)
    u = B / Bn
    theta = np.arccos(np.clip(u[2], -1, 1))            # angle of M from film normal
    Bint = Bn - MU0 * Ms * np.cos(theta) ** 2           # internal field (thin film)
    wH = g * Bint
    wM = g * MU0 * Ms
    lam = 2 * A / (MU0 * Ms ** 2) if Ms > 0 else 0.0
    phi_M = np.arctan2(u[1], u[0])
    phi = phi_k - phi_M
    kd = np.maximum(k * d, 1e-12)
    P = 1 - (1 - np.exp(-kd)) / kd
    wHk = wH + wM * lam * k ** 2
    Fnn = P + np.sin(theta) ** 2 * (1 - P * (1 + np.cos(phi) ** 2)
                                    + wM * P * (1 - P) * np.sin(phi) ** 2 / wHk)
    w2 = wHk * (wHk + wM * Fnn)
    return np.where(w2 > 0, np.sqrt(np.abs(w2)) / (2 * np.pi), np.nan)


def physics_report(cfg: Config) -> dict:
    """Quick checks shown in the GUI before running."""
    g, mt, st = cfg.grid, cfg.material, cfg.study
    gm = mt.gamma
    dx, dy, dz = g.dx_nm * 1e-9, g.dy_nm * 1e-9, g.dz_nm * 1e-9
    dt = st.dt_ps * 1e-12
    Ms = mt.Ms_kAm * 1e3
    A = mt.Aex_pJm * 1e-12
    Ku, D = mt.Ku_MJm3 * 1e6, mt.D_mJm2 * 1e-3
    rep = {"warnings": [], "info": []}
    if Ms > 0:
        rep["exchange_length_nm"] = math.sqrt(2 * A / (MU0 * Ms ** 2)) * 1e9
        if max(g.dx_nm, g.dy_nm) > 1.5 * rep["exchange_length_nm"] and (Ku or D):
            rep["warnings"].append(f"Cell size > exchange length ({rep['exchange_length_nm']:.1f} nm): "
                                   f"textures (domain walls, skyrmions) are under-resolved")
    ai = anisotropy_info(cfg)
    rep.update({k: v for k, v in ai.items() if k != "Keff"})
    rep["Keff_MJm3"] = ai["Keff"] * 1e-6
    if "dw_width_nm" in ai and min(g.dx_nm, g.dy_nm) > 0.5 * ai["dw_width_nm"] * math.pi:
        rep["warnings"].append(f"Domain wall ({ai['dw_width_nm']:.1f} nm) spans < 2 cells: refine the mesh")
    if D and "Dc_mJm2" in ai and abs(mt.D_mJm2) > ai["Dc_mJm2"]:
        rep["info"].append(f"|D| > Dc = {ai['Dc_mJm2']:.2f} mJ/m²: the uniform state is unstable "
                           f"(spin spirals / skyrmion lattice expected)")
    # explicit stability estimate: largest precession frequency
    Bmax = (np.linalg.norm([cfg.field.Bx_mT, cfg.field.By_mT, cfg.field.Bz_mT]) * 1e-3 + MU0 * Ms
            + ((2 * A / Ms) * 4 * (1 / dx ** 2 + 1 / dy ** 2) if Ms > 0 else 0)
            + (2 * abs(Ku) / Ms if Ms > 0 else 0) + (2 * abs(D) / Ms * 2 / min(dx, dy) if Ms > 0 else 0))
    wmax = gm * Bmax
    rep["omega_max_dt"] = wmax * dt
    lim = 1.0 if st.temperature_K > 0 else 2.5          # Heun is less stable than RK4
    rep["dt_max_ps"] = lim / wmax * 1e12
    if wmax * dt > lim:
        rep["warnings"].append(f"Time step too large: omega_max*dt = {wmax*dt:.2f} > {lim}, "
                               f"use dt < {rep['dt_max_ps']:.3f} ps")
    # currents
    rep["torques"] = []
    for q in cfg.torques:
        if not q.enabled or Ms <= 0:
            continue
        if q.kind == "ZhangLi":
            u = MUB * q.P * q.J_Am2 / (QE * Ms * (1 + q.beta ** 2))
            item = {"name": q.name, "u_ms": u}
            if mt.alpha > 0:
                item["v_dw_ms"] = q.beta / mt.alpha * u
        else:
            eff = q.theta_SH if q.kind == "SOT" else q.P
            item = {"name": q.name, "B_DL_mT": HBAR * eff * q.J_Am2 / (2 * QE * Ms * dz) * 1e3}
        rep["torques"].append(item)
    if st.temperature_K > 0 and Ms > 0:
        V = dx * dy * dz
        rep["thermal_B_mT"] = math.sqrt(2 * mt.alpha * KB * st.temperature_K / (gm * Ms * V * dt)) * 1e3
        if ai["Keff"] > 0:
            rep["info"].append("Thermal noise uses a stochastic Heun integrator; results depend on the seed.")
    # spin waves: sources vs dispersion
    rep["sources"] = []
    rep["fmr_GHz"] = float(np.nan_to_num(kalinikos_slavin(cfg, np.array([1.0]), 0.0))[0] / 1e9)
    fs = sorted({s.f_GHz for s in cfg.sources if s.enabled and s.waveform != "sinc"})
    if fs and (Ku or D):
        rep["info"].append("The analytic dispersion neglects anisotropy and DMI.")
    k = np.linspace(1e3, math.pi / dx, 4000)
    f_k = kalinikos_slavin(cfg, k, 0.0)
    for f in fs:
        item = {"f_GHz": f}
        idx = np.where(np.diff(np.sign(np.nan_to_num(f_k) - f * 1e9)))[0]
        if len(idx):
            kk = k[idx[0]]
            item["k_rad_um"] = kk * 1e-6
            item["lambda_nm"] = 2 * np.pi / kk * 1e9
            item["cells_per_lambda"] = 2 * np.pi / kk / dx
            if item["cells_per_lambda"] < 6:
                rep["warnings"].append(f"{f} GHz: wavelength {item['lambda_nm']:.0f} nm is only "
                                       f"{item['cells_per_lambda']:.1f} cells, refine dx")
        elif not (Ku or D):
            rep["warnings"].append(f"{f} GHz: no propagating spin wave along x "
                                   f"(outside the band of the lowest mode)")
        if f > 0:
            spp = 1 / (f * 1e9 * dt)
            item["samples_per_period"] = spp
            if spp < 8:
                rep["warnings"].append(f"{f} GHz: only {spp:.1f} time steps per period")
        rep["sources"].append(item)
    if (g.pbc_x or g.pbc_y) and st.demag:
        rep["info"].append("Periodic boundaries apply to exchange/DMI only; the demag field is not periodic.")
    w = cfg.boundary.width
    if w > 0 and 2 * w > 0.5 * min(g.nx, g.ny):
        rep["warnings"].append(f"Absorbing layer ({w} cells per side) covers more than half of the "
                               f"grid; waves in the interior will be damped. Reduce the width or enlarge the grid.")
    n = g.nx * g.ny
    frames = st.steps // st.save_every
    rep["frames_mb"] = frames * 3 * n * 4 / 2 ** 20
    if rep["frames_mb"] > st.max_frames_mb:
        rep["info"].append(f"Stored frames would need {rep['frames_mb']:.0f} MB; "
                           f"save_every will be increased automatically.")
    rep["sim_time_ns"] = st.steps * dt * 1e9
    return rep


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
@dataclass
class SimResult:
    dt: float
    steps: int
    save_every: int
    frame_steps: np.ndarray            # step index (1-based) of each stored frame
    frames: torch.Tensor               # (F, 3, nx, ny) float32 on CPU: dm = m - m0, or m
    probe: np.ndarray                  # (T, 3, Np) probe signals, sum over disk of dm*Ms (A/m)
    intensity: np.ndarray              # (3, nx, ny) sum_t dm^2
    dft: Optional[np.ndarray]          # (3, nx, ny) complex amplitude at f_dft
    f_dft: float
    m0: np.ndarray                     # (3, nx, ny)
    Msat: np.ndarray                   # (nx, ny) A/m
    Bext: np.ndarray                   # (3, nx, ny) T
    wall_time: float
    steps_per_s: float
    record: str = "dm"
    table_t: np.ndarray = None         # (R,) time of table rows (s)
    table: np.ndarray = None           # (R, len(TABLE_COLS))
    m_final: np.ndarray = None         # (3, nx, ny)
    extra: dict = field(default_factory=dict)


class Model(torch.nn.Module):
    def __init__(self, cfg: Config, device: str = "cuda", train: bool = False,
                 build_kernel: bool = True):
        super().__init__()
        self.cfg = cfg
        self.dev = torch.device(device)
        self.dtype = torch.float64 if cfg.study.precision == "float64" else torch.float32
        g, st, mt = cfg.grid, cfg.study, cfg.material
        nx, ny = g.nx, g.ny
        self.nx, self.ny = nx, ny
        dx, dy, dz = g.dx_nm * 1e-9, g.dy_nm * 1e-9, g.dz_nm * 1e-9
        self.dx, self.dy, self.dz = dx, dy, dz
        self.idx2, self.idy2 = 1.0 / dx ** 2, 1.0 / dy ** 2
        self.dt = st.dt_ps * 1e-12
        self.h = mt.gamma * self.dt
        kw = dict(dtype=self.dtype, device=self.dev)

        if build_kernel:
            self.register_buffer("K", demag_kernel_fft(nx, ny, dx, dy, dz, self.dtype, self.dev))
        else:
            self.K = None

        # material maps (regions override in order)
        Xum = (np.arange(nx) * dx * 1e6)[:, None] * np.ones((1, ny))
        Ms = np.full((nx, ny), mt.Ms_kAm * 1e3)
        Ku = mt.Ku_MJm3 * 1e6 + mt.Ku_grad_MJm3_um * 1e6 * Xum
        D = np.full((nx, ny), mt.D_mJm2 * 1e-3)
        alpha = np.full((nx, ny), mt.alpha)
        for r in cfg.regions:
            if not r.enabled:
                continue
            sel = np.zeros((nx, ny), bool)
            if r.shape == "rect":
                sel[max(r.x0, 0):r.x1, max(r.y0, 0):r.y1] = True
            else:
                xx, yy = _disk_cells(r.cx, r.cy, r.r, (nx, ny))
                sel[xx, yy] = True
            if r.Ms_kAm is not None:
                Ms[sel] = r.Ms_kAm * 1e3
            if r.Ku_MJm3 is not None:
                Ku[sel] = r.Ku_MJm3 * 1e6
            if r.D_mJm2 is not None:
                D[sel] = r.D_mJm2 * 1e-3
            if r.alpha is not None:
                alpha[sel] = r.alpha
        mag = Ms > 0
        self.register_buffer("Ms_base", torch.tensor(Ms, **kw))
        self.register_buffer("Ku_map", torch.tensor(Ku, **kw).view(1, 1, nx, ny))
        self.register_buffer("D_map", torch.tensor(D, **kw).view(1, 1, nx, ny))
        self.register_buffer("magf", torch.tensor(mag, **kw).view(1, 1, nx, ny))
        self.has_dmi = bool(np.any(D != 0)) and mt.Aex_pJm > 0
        self.has_anis = bool(np.any(Ku != 0))
        # neighbour masks: is the neighbour inside the grid and magnetic?
        nb = {k: np.zeros((nx, ny), bool) for k in "RLUD"}
        nb["R"][:-1, :] = mag[1:, :]
        nb["L"][1:, :] = mag[:-1, :]
        nb["U"][:, :-1] = mag[:, 1:]
        nb["D"][:, 1:] = mag[:, :-1]
        if g.pbc_x:
            nb["R"][-1, :] = mag[0, :]
            nb["L"][0, :] = mag[-1, :]
        if g.pbc_y:
            nb["U"][:, -1] = mag[:, 0]
            nb["D"][:, 0] = mag[:, -1]
        for k, v in nb.items():
            self.register_buffer("nb" + k, torch.tensor(v, device=self.dev).view(1, 1, nx, ny))
        self.register_buffer("Xm", torch.tensor(np.arange(nx) * dx, **kw).view(1, 1, nx, 1).expand(1, 1, nx, ny).clone())
        self.register_buffer("Ym", torch.tensor(np.arange(ny) * dy, **kw).view(1, 1, 1, ny).expand(1, 1, nx, ny).clone())

        # external field and its unit direction
        B0 = np.array([cfg.field.Bx_mT, cfg.field.By_mT, cfg.field.Bz_mT]) * 1e-3
        u = B0 / np.linalg.norm(B0) if np.linalg.norm(B0) > 0 else np.array([0.0, 1.0, 0.0])
        self.B0dir = u
        self.register_buffer("B0", torch.tensor(B0, **kw).view(1, 3, 1, 1).expand(1, 3, nx, ny).clone())
        self.register_buffer("ax", torch.tensor(np.eye(3)[COMP[mt.anis_axis]], **kw).view(1, 3, 1, 1))

        # design region and trainable parameter
        tr = cfg.training
        mask = np.zeros((nx, ny))
        mask[max(tr.region_x0, 0):tr.region_x1, max(tr.region_y0, 0):tr.region_y1] = 1
        self.register_buffer("design_mask", torch.tensor(mask, **kw))
        rho0 = 1.0 if tr.design == "Ms" else 0.0
        self.rho = torch.nn.Parameter(torch.full((nx, ny), rho0, **kw), requires_grad=train)

        # damping with absorbing frame
        x = np.arange(nx)[:, None]
        y = np.arange(ny)[None, :]
        dist = np.minimum(np.minimum(x, nx - 1 - x), np.minimum(y, ny - 1 - y))
        w, amax = cfg.boundary.width, cfg.boundary.alpha_max
        alpha = np.where(dist < w, (1 - dist / max(w, 1)) ** 2 * (amax - alpha) + alpha, alpha)
        self.register_buffer("alpha", torch.tensor(alpha, **kw).view(1, 1, nx, ny))
        self.register_buffer("pre", 1 / (1 + self.alpha ** 2))

        # RF field sources
        masks = []
        for s in cfg.sources:
            if not s.enabled:
                continue
            mk = np.zeros((3, nx, ny))
            xx, yy = _line_cells(s.x0, s.y0, s.x1, s.y1)
            mk[COMP[s.component], xx, yy] = 1
            masks.append(mk)
        if not masks:
            masks = [np.zeros((3, nx, ny))]
        self.register_buffer("src_masks", torch.tensor(np.stack(masks), **kw))
        self.n_src = sum(s.enabled for s in cfg.sources)

        # currents: J * region mask and the physical prefactors
        tq_J, tq_c, tq_cf, tq_p, zl_J, zl_c, zl_d = [], [], [], [], [], [], []
        self.beta = 0.0
        for q in cfg.torques:
            if not q.enabled:
                continue
            Jm = np.zeros((nx, ny))
            Jm[max(q.x0, 0):q.x1, max(q.y0, 0):q.y1] = q.J_Am2
            if q.kind == "ZhangLi":
                zl_J.append(Jm)
                zl_c.append(MUB * q.P / (QE * (1 + q.beta ** 2)))         # u = c * J / Ms
                zl_d.append(q.direction)
                self.beta = q.beta
            else:
                eff = q.theta_SH if q.kind == "SOT" else q.P
                c = HBAR * eff / (2 * QE * dz)                            # B_DL = c * J / Ms
                p = np.array([q.px, q.py, q.pz], float)
                tq_J.append(Jm)
                tq_c.append(c)
                tq_cf.append(c * q.xi_FL)
                tq_p.append(p / max(np.linalg.norm(p), 1e-12))
        self.has_tq, self.has_zl = len(tq_J) > 0, len(zl_J) > 0
        z1 = np.zeros((1, nx, ny))
        self.register_buffer("tq_Ja", torch.tensor(np.stack([J * c for J, c in zip(tq_J, tq_c)]) if tq_J else z1, **kw).unsqueeze(1))
        self.register_buffer("tq_Jf", torch.tensor(np.stack([J * c for J, c in zip(tq_J, tq_cf)]) if tq_J else z1, **kw).unsqueeze(1))
        self.register_buffer("tq_p", torch.tensor(np.stack(tq_p) if tq_p else np.zeros((1, 3)), **kw).view(-1, 3, 1, 1))
        zx = [J * c * (d == "x") for J, c, d in zip(zl_J, zl_c, zl_d)]
        zy = [J * c * (d == "y") for J, c, d in zip(zl_J, zl_c, zl_d)]
        self.register_buffer("zl_Jx", torch.tensor(np.stack(zx) if zx else z1, **kw).unsqueeze(1))
        self.register_buffer("zl_Jy", torch.tensor(np.stack(zy) if zy else z1, **kw).unsqueeze(1))

        # probes: one row per probe, weights over the flattened grid
        P = np.zeros((max(len(cfg.probes), 1), nx * ny))
        self.probe_cells = []
        for i, p in enumerate(cfg.probes):
            xx, yy = _disk_cells(p.x, p.y, p.r, (nx, ny))
            P[i, xx * ny + yy] = 1
            self.probe_cells.append((xx.tolist(), yy.tolist()))
        self.n_probes = len(cfg.probes)
        self.register_buffer("P", torch.tensor(P, **kw))

        m_init = torch.tensor(initial_magnetization(cfg), **kw).unsqueeze(0)
        self.register_buffer("m_init", m_init)
        self.register_buffer("m0", m_init.clone())
        self.stoch = st.temperature_K > 0
        self._compiled = {}

    # --------------------------------------------------------------- materials
    def materials(self):
        """Static tensors of a run for the current design rho (dict S, see _terms)."""
        cfg, tr, mt = self.cfg, self.cfg.training, self.cfg.material
        if tr.design == "Ms":
            Msat = self.Ms_base * (1 + self.design_mask * (self.rho - 1))
            Bext = self.B0
        else:
            Msat = self.Ms_base
            u = torch.tensor(self.B0dir, dtype=self.dtype, device=self.dev).view(1, 3, 1, 1)
            Bext = self.B0 + u * (tr.B1_mT * 1e-3 * self.rho * self.design_mask)
        Msat = Msat.view(1, 1, self.nx, self.ny)
        pos = Msat > 0
        inv = torch.where(pos, 1 / torch.where(pos, Msat, torch.ones_like(Msat)), torch.zeros_like(Msat))
        A = mt.Aex_pJm * 1e-12
        S = {
            "Msat": Msat, "Bext": Bext, "K": self.K, "src": self.src_masks, "P": self.P,
            "exf": 2 * A * inv, "idx2": self.idx2, "idy2": self.idy2,
            "hdx": 0.5 / self.dx, "hdy": 0.5 / self.dy, "gam": mt.gamma, "beta": self.beta,
            "nbR": self.nbR, "nbL": self.nbL, "nbU": self.nbU, "nbD": self.nbD,
            "dmi": self.has_dmi, "anis": self.has_anis, "tq": self.has_tq, "zl": self.has_zl,
            "demag": cfg.study.demag,
            "magf": self.magf, "Vcell": self.dx * self.dy * self.dz, "dxdy": self.dx * self.dy,
            "X": self.Xm, "Y": self.Ym, "Lx": self.nx * self.dx,
        }
        if self.has_dmi:
            S["dmf"] = 2 * self.D_map * inv
            bc = self.D_map / (2 * A)
            S["bcx"], S["bcy"] = bc * self.dx, bc * self.dy
        if self.has_anis:
            S["kuf"] = 2 * self.Ku_map * inv
            S["ax"] = self.ax
        if self.has_tq:
            S["tq_a"], S["tq_f"], S["tq_p"] = self.tq_Ja * inv, self.tq_Jf * inv, self.tq_p
        if self.has_zl:
            S["zl_x"], S["zl_y"] = self.zl_Jx * inv, self.zl_Jy * inv
        if self.stoch:
            T = self.cfg.study.temperature_K
            S["sigma_th"] = torch.sqrt(2 * self.alpha * KB * T / (mt.gamma * S["Vcell"] * self.dt) * inv) * self.magf
        return S

    def drive_signals(self, T):
        sig = torch.tensor(waveforms(self.cfg, T), dtype=self.dtype, device=self.dev)
        if sig.shape[0] == 0:
            sig = torch.zeros((1, T), dtype=self.dtype, device=self.dev)
        tq, zl = torque_waveforms(self.cfg, T)
        return (sig, torch.tensor(tq, dtype=self.dtype, device=self.dev),
                torch.tensor(zl, dtype=self.dtype, device=self.dev))

    # --------------------------------------------------------------- compile
    def _fn(self, name, fn, mode):
        key = (name, mode)
        if key not in self._compiled:
            if mode == "off" or self.dev.type != "cuda":
                self._compiled[key] = fn
            else:
                self._compiled[key] = torch.compile(fn, dynamic=False, fullgraph=False)
        return self._compiled[key]

    @torch.no_grad()
    def relax(self, S, steps=None, warm=False):
        st = self.cfg.study
        steps = st.relax_steps if steps is None else steps
        m = self.m0 if warm else self.m_init.clone()
        a = st.relax_alpha
        Sr = dict(S, tq=False, zl=False)
        fn = self._fn("rk4", _rk4, "default" if st.compile != "off" else "off")

        def step(m):
            return fn(m, Sr["Bext"], None, None, None, Sr, a, 1 / (1 + a * a), self.h, True)

        for _ in range(steps):
            m = step(m)
        n, resid = steps, None
        if st.relax_tol > 0:
            # continue until the magnetisation stops changing (only magnetic cells count)
            mag = self.magf
            while n < max(st.relax_max, steps):
                for _ in range(49):
                    m = step(m)
                m_new = step(m)
                n += 50
                resid = float(((m_new - m).abs() * mag).max())
                m = m_new
                if resid < st.relax_tol:
                    break
        self.relax_info = {"steps": n, "residual": resid,
                           "converged": resid is None or resid < st.relax_tol}
        self.m0 = m.clone()
        return self.m0

    # --------------------------------------------------------------- simulate
    @torch.no_grad()
    def simulate(self, steps=None, progress: Optional[Callable] = None,
                 should_stop: Optional[Callable] = None, live_every_s: float = 0.3,
                 relax: bool = True, warm: bool = False) -> SimResult:
        cfg, st = self.cfg, self.cfg.study
        T = st.steps if steps is None else steps
        nx, ny = self.nx, self.ny
        S = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in self.materials().items()}
        if relax:
            if progress:
                progress({"phase": "relax", "step": 0, "total": st.relax_steps})
            self.relax(S, warm=warm)
        if st.seed:
            torch.manual_seed(st.seed)
        m0 = self.m0
        sig, tsig, zsig = self.drive_signals(T)
        rec_m = st.record == "m"
        stoch = self.stoch

        # frame storage (respect memory budget)
        save_every = st.save_every
        while (T // save_every) * 3 * nx * ny * 4 / 2 ** 20 > st.max_frames_mb:
            save_every += 1
        n_frames = T // save_every
        frames = torch.empty((n_frames, 3, nx, ny), dtype=torch.float32,
                             pin_memory=self.dev.type == "cuda")
        frame_steps = np.arange(1, n_frames + 1) * save_every
        probe_buf = torch.zeros((T, 3, max(self.n_probes, 1)), dtype=self.dtype, device=self.dev)

        # running DFT at the first source frequency (frequency-domain result),
        # taken over an integer number of periods in the second half of the run
        f_dft = next((s.f_GHz * 1e9 for s in cfg.sources if s.enabled and s.f_GHz > 0), 0.0)
        dft_start = T
        if f_dft > 0:
            per = 1 / (f_dft * self.dt)
            n_per = max(1, int((T // 2) / per))
            dft_start = max(0, T - int(round(n_per * per)))
        kw = dict(dtype=self.dtype, device=self.dev)
        t_steps = torch.arange(1, T + 1, **kw) * self.dt
        win = (torch.arange(T, device=self.dev) >= dft_start).to(self.dtype)
        cosw = torch.cos(2 * math.pi * f_dft * t_steps) * win
        sinw = torch.sin(2 * math.pi * f_dft * t_steps) * win
        sigT, tsigT, zsigT = sig.t().contiguous(), tsig.t().contiguous(), zsig.t().contiguous()

        # static buffers, updated in place (required for CUDA-graph replay).
        # A chunk has ~16 steps; frames are stored inside the chunk when save_every is
        # small, otherwise at chunk ends (save_every is rounded to a multiple of K).
        if save_every <= 16:
            K = save_every * max(1, round(16 / save_every))
            in_chunk = True
        else:
            K = 16
            save_every = max(K, int(round(save_every / K)) * K)
            in_chunk = False
            n_frames = T // save_every
            frames = torch.empty((n_frames, 3, nx, ny), dtype=torch.float32,
                                 pin_memory=self.dev.type == "cuda")
            frame_steps = np.arange(1, n_frames + 1) * save_every
        st_m = m0.clone()
        st_inten, st_dre, st_dim = (torch.zeros_like(m0) for _ in range(3))
        st_sig = torch.zeros((K, sigT.shape[1]), **kw)
        st_tsig = torch.zeros((K, tsigT.shape[1]), **kw)
        st_zsig = torch.zeros((K, zsigT.shape[1]), **kw)
        st_cw, st_sw = torch.zeros(K, **kw), torch.zeros(K, **kw)
        st_probe = torch.zeros((K, 3, max(self.n_probes, 1)), **kw)
        st_frames = torch.zeros((max(K // save_every, 1), 3, nx, ny), dtype=torch.float32, device=self.dev)
        st_live = torch.zeros((3, nx, ny), dtype=torch.float32, device=self.dev)
        fn = self._fn("sim", _sim_step, "off" if st.compile == "off" else "default")
        tab_fn = self._fn("table", _table, "off" if st.compile == "off" else "default")
        alpha, pre = self.alpha, self.pre

        def body(k):
            for j in range(k):
                Sj = S
                if stoch:
                    Sj = dict(S)
                    Sj["bth"] = S["sigma_th"] * torch.randn_like(st_m)   # eager RNG: graph-safe
                m, dm, pr, it, dr, di = fn(st_m, m0, st_sig[j], st_tsig[j], st_zsig[j], Sj, alpha, pre,
                                           self.h, st.normalize, stoch, st_inten, st_dre, st_dim,
                                           st_cw[j], st_sw[j])
                st_m.copy_(m)
                st_inten.copy_(it)
                st_dre.copy_(dr)
                st_dim.copy_(di)
                st_probe[j].copy_(pr)
                if in_chunk and (j + 1) % save_every == 0:
                    st_frames[(j + 1) // save_every - 1].copy_((m if rec_m else dm)[0])
            st_live.copy_((m if rec_m else dm)[0])

        graph = None
        self.graph_error = None
        if st.compile == "cudagraphs" and self.dev.type == "cuda" and T >= K:
            state = [b.clone() for b in (st_m, st_inten, st_dre, st_dim)]
            try:
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    body(K)                               # warm-up (compiles kernels)
                torch.cuda.current_stream().wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    body(K)                               # capture, does not execute
            except Exception as e:  # noqa: BLE001
                graph = None
                self.graph_error = f"{type(e).__name__}: {e}"
            for b, v in zip((st_m, st_inten, st_dre, st_dim), state):
                b.copy_(v)

        n_rows = T // K + 2
        table = torch.zeros((n_rows, len(TABLE_COLS)), **kw)
        table_t = np.zeros(n_rows)
        table[0] = tab_fn(st_m, S)
        rows = 1
        t0 = time.perf_counter()
        last = t0
        stopped = False
        fi = 0
        n = 0
        while n < T:
            k = min(K, T - n)
            st_sig[:k].copy_(sigT[n:n + k])
            st_tsig[:k].copy_(tsigT[n:n + k])
            st_zsig[:k].copy_(zsigT[n:n + k])
            st_cw[:k].copy_(cosw[n:n + k])
            st_sw[:k].copy_(sinw[n:n + k])
            if graph is not None and k == K:
                graph.replay()
            else:
                body(k)
            probe_buf[n:n + k].copy_(st_probe[:k])
            if in_chunk:
                nf = min(k // save_every, n_frames - fi)
                if nf > 0:
                    frames[fi:fi + nf].copy_(st_frames[:nf], non_blocking=True)
                    fi += nf
            elif (n + k) % save_every == 0 and fi < n_frames:
                frames[fi].copy_(st_live, non_blocking=True)
                fi += 1
            n += k
            table[rows] = tab_fn(st_m, S)
            table_t[rows] = n * self.dt
            rows += 1
            now = time.perf_counter()
            if progress and (now - last > live_every_s or n == T):
                last = now
                live = st_live.cpu()                     # synchronises
                progress({"phase": "run", "step": n, "total": T,
                          "rate": n / (time.perf_counter() - t0), "frame": fi - 1, "live": live})
            if should_stop and should_stop():
                stopped = n < T
                T = n
                break
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        frames = frames[:fi]
        frame_steps = frame_steps[:fi]
        dft = None
        if f_dft > 0 and T > dft_start:
            # X(f) = 2/N sum dm(t) exp(-i w t)   ->   dm(t) ~ Re(X exp(i w t))
            scale = 2.0 / (T - dft_start)
            dft = scale * (st_dre[0].cpu().numpy() + 1j * st_dim[0].cpu().numpy())
        return SimResult(
            dt=self.dt, steps=T, save_every=save_every, frame_steps=frame_steps, frames=frames,
            probe=probe_buf[:T, :, : self.n_probes].float().cpu().numpy(),
            intensity=st_inten[0].float().cpu().numpy(), dft=dft, f_dft=f_dft,
            m0=m0[0].float().cpu().numpy(), Msat=S["Msat"][0, 0].float().cpu().numpy(),
            Bext=S["Bext"][0].float().cpu().numpy(), wall_time=wall, steps_per_s=T / max(wall, 1e-9),
            record=st.record, table_t=table_t[:rows], table=table[:rows].double().cpu().numpy(),
            m_final=st_m[0].float().cpu().numpy(),
            extra={"stopped": stopped, "cuda_graph": graph is not None,
                   "relax": getattr(self, "relax_info", None), "stochastic": stoch,
                   "graph_error": self.graph_error, "chunk": K},
        )

    # --------------------------------------------------------------- training forward
    def train_forward(self, steps=None, record_intensity=True, progress=None):
        """Return time-integrated probe intensities u (Np,) with autograd enabled.

        Thermal noise is not used during training (deterministic gradients).
        """
        cfg, st = self.cfg, self.cfg.study
        T = st.steps if steps is None else steps
        S = self.materials()
        self.relax({k: (v.detach() if torch.is_tensor(v) else v) for k, v in S.items()},
                   warm=getattr(self, "_warm", False))
        self._warm = True
        m0 = self.m0
        sig, tsig, zsig = self.drive_signals(T)
        comp = COMP[st.probe_component]
        step = self._fn("train", _train_step, "default" if st.compile != "off" else "off")
        seg = max(1, int(math.ceil(math.sqrt(T))))
        inten = torch.zeros_like(m0) if record_intensity else None
        state = {"record": True}
        alpha, pre = self.alpha, self.pre

        def run_segment(m, Msat, Bext, s, ts, zs):
            Sx = dict(S, Msat=Msat, Bext=Bext)
            outs = []
            for j in range(s.shape[1]):
                m, v = step(m, m0, s[:, j], ts[:, j], zs[:, j], Sx, alpha, pre, self.h,
                            st.normalize, False, comp)
                outs.append(v)
                if inten is not None and state["record"]:
                    dm = (m - m0).detach()
                    inten.add_(dm * dm)
            return m, torch.stack(outs).sum(0)

        m = m0.clone()
        u = 0
        for a in range(0, T, seg):
            m, us = checkpoint(run_segment, m, S["Msat"], S["Bext"], sig[:, a:a + seg],
                               tsig[:, a:a + seg], zsig[:, a:a + seg], use_reentrant=False)
            u = u + us
            if progress:
                progress(min(a + seg, T), T)
        state["record"] = False      # recomputation during backward must not add again
        self._last_intensity = inten
        return u[: self.n_probes], S["Msat"], S["Bext"]

    def clamp_(self):
        tr = self.cfg.training
        if tr.clamp:
            with torch.no_grad():
                self.rho.clamp_(tr.rho_min, tr.rho_max)


def focus_loss(u, target, kind="focus_log"):
    """SpinTorch loss: (sum of all outputs / target output - 1), log10."""
    r = u.sum() / u[target] - 1
    return r.log10() if kind == "focus_log" else r
