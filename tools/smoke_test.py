"""End-to-end API test of the SpinStudio server (runs in-process, no browser)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient  # noqa: E402

from spinstudio.server import app  # noqa: E402


def wait(c, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = c.get("/api/status").json()
        if s["status"] != "running":
            return s
        time.sleep(0.5)
    raise TimeoutError


with TestClient(app) as c:
    print("info:", c.get("/api/info").json())
    cfg = c.get("/api/preset/focus").json()
    r = c.post("/api/check", json=cfg).json()
    print("warnings:", r["warnings"], "fmr", round(r["fmr_GHz"], 3), "sources", r["sources"])
    bad = dict(cfg)
    bad["probes"] = cfg["probes"] + [{"x": 500, "y": 3, "r": 2}]
    print("invalid config ->", c.post("/api/check", json=bad).status_code)

    # simulation
    assert c.post("/api/run", json={"config": cfg, "mode": "simulate"}).status_code == 200
    s = wait(c)
    print("simulate:", s["status"], [m["text"] for m in s["log"][-3:]])
    meta = c.get("/api/result/meta").json()
    print("frames", meta["frames"], "steps/s", round(meta["steps_per_s"]), meta["extra"])
    fr = c.get("/api/result/frames?start=0&count=4&comp=z").json()
    print("frame batch", fr["idx"], fr["frames"][0]["shape"])
    for name in ("intensity", "dft_amp", "dft_phase", "Msat", "B"):
        assert c.get(f"/api/result/map/{name}?comp=z").status_code == 200, name
    pr = c.get("/api/result/probes?comp=x").json()
    print("probe intensity argmax:", max(range(len(pr["intensity"])), key=lambda i: pr["intensity"][i]))
    k = c.get(f"/api/result/kspace/{meta['frames'] - 1}?comp=z").json()
    print("kspace iso points:", sum(v is not None for v in k["iso"]["kx"]))
    assert c.get("/api/result/linecut/10").status_code == 200
    assert c.get("/api/result/range?comp=abs").status_code == 200
    print("export bytes:", len(c.get("/api/result/export.npz").content))

    # training, 3 epochs
    cfg["training"]["epochs"] = 3
    cfg["training"]["simulate_after"] = True
    assert c.post("/api/run", json={"config": cfg, "mode": "train"}).status_code == 200
    s = wait(c)
    print("train:", s["status"])
    for m in s["log"]:
        if "Epoch" in m["text"] or m["level"] in ("error", "warn"):
            print("  ", m["text"][:300])
    print("history loss:", c.get("/api/train/history").json()["loss"])
    print("runs:", c.get("/api/runs").json()[:2])
