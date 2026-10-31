"""GUI test of the spintronic / neuromorphic features (headless Chromium)."""
import os
import subprocess
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "ui_shots2")
os.makedirs(OUT, exist_ok=True)
PORT = 8798
srv = subprocess.Popen([sys.executable, "-m", "spinstudio.server", "--port", str(PORT)], cwd=ROOT)
for _ in range(120):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/info", timeout=2)
        break
    except Exception:
        time.sleep(0.5)
errors = []


def done(pg, timeout=900000):
    pg.wait_for_function("['Finished','Stopped'].includes(document.querySelector('#progLabel').textContent) || "
                         "document.querySelector('#progLabel').textContent.startsWith('Failed')", timeout=timeout)
    pg.wait_for_timeout(2500)


try:
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1680, "height": 960})
        pg.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type in ("error", "warning") else None)
        pg.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        pg.goto(f"http://127.0.0.1:{PORT}/")
        pg.wait_for_timeout(3000)
        # --- micromagnetics: DW neuron
        pg.select_option("#preset", "dw_neuron")
        pg.click("#btnNew")
        pg.wait_for_timeout(2000)
        pg.click("#tree >> text=Currents & Torques")
        pg.wait_for_timeout(1000)
        pg.screenshot(path=f"{OUT}/01_dw_torques.png")
        pg.click("#btnSim")
        pg.wait_for_timeout(1500)
        pg.click("#btnSim") if False else None
        done(pg)
        pg.click(".tab >> text=Field")
        pg.wait_for_timeout(2000)
        pg.screenshot(path=f"{OUT}/02_dw_field.png")
        pg.click(".tab >> text=Table")
        pg.wait_for_timeout(1500)
        pg.select_option("#toolbar select >> nth=0", "Texture position")
        pg.wait_for_timeout(1500)
        pg.screenshot(path=f"{OUT}/03_dw_table.png")
        pg.click(".tab >> text=Probes")
        pg.wait_for_timeout(2000)
        pg.screenshot(path=f"{OUT}/04_dw_probe.png")
        # --- skyrmion
        pg.select_option("#preset", "skyrmion")
        pg.click("#btnNew")
        pg.wait_for_timeout(2000)
        pg.click("#tree >> text=Material")
        pg.wait_for_timeout(800)
        pg.screenshot(path=f"{OUT}/05_sk_material.png")
        pg.click("#btnSim")
        pg.wait_for_timeout(1500)
        done(pg)
        pg.click(".tab >> text=Field")
        pg.wait_for_timeout(2000)
        pg.screenshot(path=f"{OUT}/06_sk_field.png")
        pg.click(".tab >> text=Table")
        pg.wait_for_timeout(1500)
        pg.select_option("#toolbar select >> nth=0", "Texture position")
        pg.wait_for_timeout(1500)
        pg.screenshot(path=f"{OUT}/07_sk_table.png")
        # --- macrospin module
        pg.select_option("#module", "macro")
        pg.wait_for_timeout(2500)
        pg.screenshot(path=f"{OUT}/08_macro_device.png")
        pg.click("#btnMStudy")
        pg.wait_for_timeout(1500)
        done(pg)
        pg.screenshot(path=f"{OUT}/09_macro_sweep.png")
        pg.select_option("#preset", "smtj_pbit")
        pg.click("#btnNew")
        pg.wait_for_timeout(2000)
        pg.click("#tree >> text=p-bit network")
        pg.wait_for_timeout(800)
        pg.screenshot(path=f"{OUT}/10_pbit_settings.png")
        pg.click("#btnMPbit")
        pg.wait_for_timeout(1500)
        done(pg)
        pg.screenshot(path=f"{OUT}/11_pbit_result.png")
        pg.select_option("#preset", "sto_reservoir")
        pg.click("#btnNew")
        pg.wait_for_timeout(2000)
        pg.click("#btnMStudy")
        pg.wait_for_timeout(1500)
        done(pg)
        pg.screenshot(path=f"{OUT}/12_sto_spectrum.png")
        pg.click("#btnMRes")
        pg.wait_for_timeout(1500)
        done(pg)
        pg.screenshot(path=f"{OUT}/13_reservoir.png")
        b.close()
finally:
    srv.terminate()
print("\n".join(errors) if errors else "no console errors")
