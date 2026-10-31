import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spinstudio.config import dw_neuron_preset
from spinstudio.solver import Model, TABLE_COLS, physics_report
cfg = dw_neuron_preset()
grad = float(sys.argv[1]) if len(sys.argv) > 1 else cfg.material.Ku_grad_MJm3_um
cfg.material.Ku_grad_MJm3_um = grad
cfg.torques[0].py = float(os.environ.get("PY", cfg.torques[0].py))
cfg.torques[0].enabled = os.environ.get("TQ", "1") == "1"
cfg.study.steps = int(os.environ.get("STEPS", cfg.study.steps))
rep = physics_report(cfg)
print("warnings:", rep["warnings"], "torques:", rep["torques"], "dw width", rep.get("dw_width_nm"))
r = Model(cfg, "cuda").simulate()
c = TABLE_COLS.index("x_DW")
t, x = r.table_t * 1e9, r.table[:, c]
for tt in np.arange(0, t[-1] + 0.01, float(os.environ.get("DT", 0.5))):
    i = np.argmin(abs(t - tt))
    print(f"t={t[i]:5.2f} ns  x_DW={x[i]:7.1f} nm  Q={r.table[i, 3]:+.2f} mz={r.table[i,2]:+.3f}")
print("steps/s", round(r.steps_per_s), "relax", r.extra["relax"])
