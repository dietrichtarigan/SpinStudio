"""Micromagnetic simulation configuration (validated with pydantic).

All values are stored in the units shown in the GUI (nm, mT, kA/m, GHz, ps, ...).
The solver converts them to SI.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class Grid(BaseModel):
    nx: int = Field(100, ge=1, le=4096, description="cells along x")
    ny: int = Field(100, ge=1, le=4096, description="cells along y")
    dx_nm: float = Field(50.0, gt=0)
    dy_nm: float = Field(50.0, gt=0)
    dz_nm: float = Field(20.0, gt=0, description="film thickness")
    pbc_x: bool = Field(False, description="periodic exchange/DMI along x (demag stays open)")
    pbc_y: bool = Field(False, description="periodic exchange/DMI along y (demag stays open)")


class Material(BaseModel):
    name: str = "YIG"
    Ms_kAm: float = Field(140.0, ge=0, description="saturation magnetization")
    Aex_pJm: float = Field(3.65, ge=0, description="exchange stiffness")
    alpha: float = Field(1e-4, ge=0, description="Gilbert damping")
    gamma: float = Field(1.7595e11, gt=0, description="gyromagnetic ratio (rad/(s T))")
    Ku_MJm3: float = Field(0.0, description="uniaxial anisotropy constant (>0 easy axis)")
    anis_axis: Literal["x", "y", "z"] = "z"
    Ku_grad_MJm3_um: float = Field(0.0, description="linear Ku gradient along x (per um)")
    D_mJm2: float = Field(0.0, description="interfacial DMI constant")


class ExternalField(BaseModel):
    Bx_mT: float = 0.0
    By_mT: float = 60.0
    Bz_mT: float = 0.0


class Boundary(BaseModel):
    width: int = Field(10, ge=0, description="absorbing layer width (cells)")
    alpha_max: float = Field(0.5, ge=0, description="damping at the outer edge")


class Region(BaseModel):
    """Geometry primitive that overrides material parameters inside its area.

    A value of None keeps the material value; Ms = 0 makes a hole.
    """
    name: str = "region"
    shape: Literal["rect", "disk"] = "rect"
    x0: int = 0
    y0: int = 0
    x1: int = 10   # rect: exclusive end
    y1: int = 10
    cx: int = 50   # disk
    cy: int = 50
    r: float = 10
    Ms_kAm: Optional[float] = 0.0
    Ku_MJm3: Optional[float] = None
    D_mJm2: Optional[float] = None
    alpha: Optional[float] = None
    enabled: bool = True


class Source(BaseModel):
    name: str = "source"
    x0: int = 10
    y0: int = 0
    x1: int = 10
    y1: int = 99
    component: Literal["x", "y", "z"] = "z"
    waveform: Literal["sine", "gaussian", "sinc"] = "sine"
    amp_mT: float = 1.0
    f_GHz: float = Field(4.0, ge=0, description="sine / carrier frequency, sinc: max frequency")
    phase_deg: float = 0.0
    ramp_ns: float = Field(0.0, ge=0, description="smooth turn-on time (sine)")
    t0_ns: float = Field(1.0, description="pulse center (gaussian / sinc)")
    width_ns: float = Field(0.2, gt=0, description="pulse width sigma (gaussian)")
    enabled: bool = True


class Torque(BaseModel):
    """Electrical current driving the magnetisation.

    SOT: spin-orbit torque from a heavy-metal underlayer (spin Hall angle).
    STT_CPP: Slonczewski torque from a fixed layer (MTJ / spin valve, current perpendicular).
    ZhangLi: spin-transfer torque of an in-plane current (domain-wall / skyrmion motion).
    """
    name: str = "current"
    kind: Literal["SOT", "STT_CPP", "ZhangLi"] = "SOT"
    enabled: bool = True
    J_Am2: float = Field(1e11, description="current density amplitude")
    waveform: Literal["dc", "pulse", "pulse_train", "sine"] = "pulse"
    t_start_ns: float = 0.0
    width_ns: float = Field(1.0, gt=0)
    period_ns: float = Field(4.0, gt=0)
    n_pulses: int = Field(3, ge=1)
    f_GHz: float = 1.0
    rise_ns: float = Field(0.05, ge=0, description="rise / fall time of pulses")
    x0: int = 0
    y0: int = 0
    x1: int = 100000
    y1: int = 100000
    theta_SH: float = Field(0.1, description="spin Hall angle (SOT)")
    P: float = Field(0.5, description="spin polarisation (STT)")
    xi_FL: float = Field(0.0, description="field-like / damping-like ratio")
    beta: float = Field(0.04, description="non-adiabaticity (Zhang-Li)")
    px: float = 0.0
    py: float = 1.0
    pz: float = 0.0
    direction: Literal["x", "y"] = "x"


class InitialState(BaseModel):
    kind: Literal["field", "uniform", "random", "skyrmion", "domain_wall", "vortex"] = "field"
    mx: float = 0.0
    my: float = 0.0
    mz: float = 1.0
    seed: int = 1
    # skyrmion / vortex / domain wall
    cx_nm: float = 100.0
    cy_nm: float = 50.0
    radius_nm: float = 10.0
    core: Literal["down", "up"] = "down"
    wall_x_nm: float = 100.0
    wall_type: Literal["Neel", "Bloch"] = "Neel"
    chirality: int = Field(1, description="+1 / -1")


class Probe(BaseModel):
    x: int
    y: int
    r: float = 2.0


class Study(BaseModel):
    dt_ps: float = Field(20.0, gt=0)
    steps: int = Field(600, ge=1, le=20_000_000)
    relax_steps: int = Field(100, ge=0)
    relax_alpha: float = Field(0.5, ge=0)
    relax_tol: float = Field(1e-8, ge=0, description="relax until max |dm| per step < tol (0: fixed steps)")
    relax_max: int = Field(20000, ge=0, description="upper limit of relaxation steps")
    demag: bool = Field(True, description="include the magnetostatic (demag) field")
    temperature_K: float = Field(0.0, ge=0, description="thermal fluctuations (stochastic LLG, Heun)")
    seed: int = 1
    save_every: int = Field(1, ge=1, description="store a field frame every N steps")
    record: Literal["dm", "m"] = Field("dm", description="store m - m0 (spin waves) or m (textures)")
    normalize: bool = Field(True, description="renormalize |m|=1 after every step")
    legacy_source: bool = Field(False, description="reproduce SpinTorch's accumulating source bug")
    probe_component: Literal["x", "y", "z"] = "x"
    precision: Literal["float32", "float64"] = "float32"
    compile: Literal["off", "default", "cudagraphs"] = "cudagraphs"
    max_frames_mb: float = Field(1500, gt=0, description="memory budget for stored frames")


class Training(BaseModel):
    design: Literal["Ms", "field"] = "Ms"
    B1_mT: float = Field(50.0, description="field design: B = B0 + B1*rho")
    region_x0: int = 0
    region_y0: int = 0
    region_x1: int = 100
    region_y1: int = 100
    epochs: int = Field(10, ge=1, le=10000)
    lr: float = Field(0.01, gt=0)
    target: int = Field(9, ge=0, description="index of the probe that should receive the wave")
    loss: Literal["focus_log", "focus_ratio"] = "focus_log"
    clamp: bool = Field(False, description="keep rho inside [rho_min, rho_max]")
    rho_min: float = 0.0
    rho_max: float = 2.0
    simulate_after: bool = Field(True, description="run a full recorded simulation of the trained design")


class Config(BaseModel):
    name: str = "untitled"
    grid: Grid = Grid()
    material: Material = Material()
    field: ExternalField = ExternalField()
    boundary: Boundary = Boundary()
    regions: List[Region] = []
    sources: List[Source] = [Source()]
    torques: List[Torque] = []
    init: InitialState = InitialState()
    probes: List[Probe] = []
    study: Study = Study()
    training: Training = Training()

    @model_validator(mode="after")
    def _check(self):
        nx, ny = self.grid.nx, self.grid.ny
        for s in self.sources:
            for x, y in ((s.x0, s.y0), (s.x1, s.y1)):
                if not (0 <= x < nx and 0 <= y < ny):
                    raise ValueError(f"source '{s.name}' point ({x},{y}) is outside the grid")
        for i, p in enumerate(self.probes):
            if not (0 <= p.x < nx and 0 <= p.y < ny):
                raise ValueError(f"probe {i} at ({p.x},{p.y}) is outside the grid")
        return self


# ----------------------------------------------------------------------------
# presets
# ----------------------------------------------------------------------------
def focus_preset() -> Config:
    """The focusing example from SpinTorch (focus.py)."""
    nx = ny = 100
    Np = 19
    probes = [Probe(x=nx - 15, y=int(ny * (p + 1) / (Np + 1)), r=2) for p in range(Np)]
    return Config(
        name="focus (SpinTorch example)",
        probes=probes,
        sources=[Source(name="line source", x0=10, y0=0, x1=10, y1=ny - 1,
                        component="z", waveform="sine", amp_mT=1.0, f_GHz=4.0)],
        training=Training(target=int(Np / 2), region_x1=nx, region_y1=ny),
    )


def waveguide_preset() -> Config:
    """A narrow waveguide with a broadband pulse: good for checking dispersion."""
    nx, ny = 256, 64
    return Config(
        name="waveguide pulse",
        grid=Grid(nx=nx, ny=ny, dx_nm=50, dy_nm=50, dz_nm=20),
        sources=[Source(name="antenna", x0=20, y0=0, x1=20, y1=ny - 1, component="z",
                        waveform="sinc", amp_mT=0.5, f_GHz=6.0, t0_ns=0.5)],
        probes=[Probe(x=x, y=ny // 2, r=2) for x in (60, 120, 180)],
        study=Study(steps=2000, dt_ps=20, save_every=5),
        training=Training(target=1, region_x1=nx, region_y1=ny),
    )


def dw_neuron_preset() -> Config:
    """Leaky integrate-and-fire neuron: a Neel domain wall in a PMA track pushed by SOT pulses.

    Each input spike (SOT current pulse) moves the wall to the right (integrate); a Ku
    gradient pushes it back between spikes (leak); the read-out MTJ (probe) at the
    right detects when the wall passes the threshold (fire).
    """
    nx, ny = 256, 32
    return Config(
        name="DW LIF neuron (SOT)",
        grid=Grid(nx=nx, ny=ny, dx_nm=2, dy_nm=2, dz_nm=1),
        material=Material(name="CoFeB/Pt", Ms_kAm=1100, Aex_pJm=15, alpha=0.05, Ku_MJm3=1.0,
                          anis_axis="z", D_mJm2=1.0, Ku_grad_MJm3_um=0.1),
        field=ExternalField(Bx_mT=0, By_mT=0, Bz_mT=0),
        boundary=Boundary(width=0),
        sources=[],
        torques=[Torque(name="SOT input spikes", kind="SOT", J_Am2=1e11, waveform="pulse_train",
                        t_start_ns=0.2, width_ns=0.5, period_ns=2.0, n_pulses=6, theta_SH=0.1,
                        px=0, py=-1, pz=0)],
        init=InitialState(kind="domain_wall", wall_x_nm=120, wall_type="Neel", chirality=1),
        probes=[Probe(x=110, y=16, r=6)],
        study=Study(dt_ps=0.1, steps=130000, save_every=500, record="m", relax_steps=3000,
                    relax_tol=0, probe_component="z"),
        training=Training(target=0, region_x1=nx, region_y1=ny),
    )


def skyrmion_preset() -> Config:
    """Skyrmion racetrack: a Neel skyrmion driven by spin-orbit torque."""
    nx, ny = 200, 80
    return Config(
        name="skyrmion racetrack (SOT)",
        grid=Grid(nx=nx, ny=ny, dx_nm=2, dy_nm=2, dz_nm=0.6),
        material=Material(name="Co/Pt (SP DMI)", Ms_kAm=580, Aex_pJm=15, alpha=0.3, Ku_MJm3=0.8,
                          anis_axis="z", D_mJm2=3.0),
        field=ExternalField(Bx_mT=0, By_mT=0, Bz_mT=0),
        boundary=Boundary(width=0),
        sources=[],
        torques=[Torque(name="SOT drive", kind="SOT", J_Am2=2e11, waveform="pulse", t_start_ns=0.1,
                        width_ns=1.5, theta_SH=0.1, px=0, py=-1, pz=0)],
        init=InitialState(kind="skyrmion", cx_nm=80, cy_nm=80, radius_nm=10, core="down"),
        probes=[],
        study=Study(dt_ps=0.1, steps=30000, save_every=300, record="m", relax_steps=5000, relax_tol=0),
        training=Training(target=0, region_x1=nx, region_y1=ny),
    )


PRESETS = {
    "focus": focus_preset,
    "waveguide": waveguide_preset,
    "dw_neuron": dw_neuron_preset,
    "skyrmion": skyrmion_preset,
    "empty": lambda: Config(),
}
