import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spinstudio.config import focus_preset
from spinstudio.solver import Model
N = int(os.environ.get("NX", 100))
for mode in sys.argv[1:]:
    cfg = focus_preset(); cfg.study.compile = mode; cfg.study.steps = 2000
    cfg.grid.nx = cfg.grid.ny = N
    cfg.training.region_x1 = cfg.training.region_y1 = N
    cfg.sources[0].y1 = N - 1
    cfg.probes = [p.model_copy(update={"x": min(p.x, N-1), "y": min(p.y, N-1)}) for p in cfg.probes]
    m = Model(cfg, "cuda")
    m.simulate(steps=30, relax=True)
    torch.cuda.synchronize(); t = time.perf_counter()
    r = m.simulate(relax=False)
    torch.cuda.synchronize()
    print(f"{mode:10s} N={N}: {r.steps_per_s:7.0f} steps/s  ({time.perf_counter()-t:.2f}s)", flush=True)
