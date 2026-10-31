"""Drive the GUI in headless Chromium and take screenshots of every view."""
import os
import subprocess
import sys
import time

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "ui_shots")
os.makedirs(OUT, exist_ok=True)
PORT = 8799
srv = subprocess.Popen([sys.executable, "-m", "spinstudio.server", "--port", str(PORT)], cwd=ROOT)
import urllib.request
for _ in range(120):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/info", timeout=2)
        break
    except Exception:
        time.sleep(0.5)
errors = []
try:
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1680, "height": 960})
        pg.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type in ("error", "warning") else None)
        pg.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        pg.goto(f"http://127.0.0.1:{PORT}/")
        pg.wait_for_timeout(2500)
        pg.screenshot(path=f"{OUT}/01_geometry.png")

        pg.click("text=Sources")
        pg.wait_for_timeout(800)
        pg.screenshot(path=f"{OUT}/02_sources.png")

        pg.click("#btnSim")
        pg.wait_for_timeout(2500)
        pg.screenshot(path=f"{OUT}/02b_running.png")
        pg.wait_for_function("document.querySelector('#progLabel').textContent === 'Finished'", timeout=300000)
        pg.wait_for_timeout(1500)
        pg.click(".tab >> text=Field")
        pg.wait_for_timeout(1500)
        pg.screenshot(path=f"{OUT}/03_field.png")
        pg.check("text=Line cut >> input") if pg.query_selector("text=Line cut") else None
        pg.wait_for_timeout(1200)
        pg.screenshot(path=f"{OUT}/04_field_linecut.png")
        for name, fn in (("Intensity", "05_intensity"), ("Frequency domain", "06_freq"), ("k-space", "07_kspace"),
                         ("Probes", "08_probes"), ("Dispersion", "09_dispersion")):
            pg.click(f".tab >> text={name}")
            pg.wait_for_timeout(2000)
            pg.screenshot(path=f"{OUT}/{fn}.png")
        pg.click(".tab >> text=Probes")
        pg.wait_for_timeout(1200)
        pg.select_option("#toolbar select >> nth=0", "spectrum")
        pg.wait_for_timeout(1500)
        pg.screenshot(path=f"{OUT}/10_spectrum.png")

        # training: 3 epochs
        pg.click("#tree >> text=Inverse Design")
        pg.wait_for_timeout(500)
        ep = pg.locator(".field:has(label:text('Epochs')) input")
        ep.fill("3")
        ep.dispatch_event("change")
        pg.wait_for_timeout(600)
        pg.screenshot(path=f"{OUT}/11_training_settings.png")
        pg.click("#btnTrain")
        pg.wait_for_timeout(9000)
        pg.screenshot(path=f"{OUT}/12_training_running.png")
        pg.wait_for_function("document.querySelector('#progLabel').textContent === 'Finished'", timeout=600000)
        pg.wait_for_timeout(2000)
        pg.click(".tab >> text=Training")
        pg.wait_for_timeout(1500)
        pg.screenshot(path=f"{OUT}/13_training_done.png")
        # live preview + stop on a long run
        pg.select_option("#preset", "waveguide")
        pg.click("#btnNew")
        pg.wait_for_timeout(1500)
        pg.click("#tree >> text=Time Dependent")
        st = pg.locator(".field:has(label:text('Number of steps')) input")
        st.fill("200000")
        st.dispatch_event("change")
        pg.wait_for_timeout(800)
        pg.click("#btnSim")
        pg.wait_for_function("document.querySelector('#progLabel').textContent.startsWith('Time stepping')", timeout=300000)
        pg.wait_for_timeout(2500)
        pg.screenshot(path=f"{OUT}/14_live.png")
        pg.click("#btnStop")
        pg.wait_for_function("document.querySelector('#progLabel').textContent === 'Stopped'", timeout=60000)
        pg.wait_for_timeout(2000)
        pg.screenshot(path=f"{OUT}/15_stopped.png")
        b.close()
finally:
    srv.terminate()
print("\n".join(errors) if errors else "no console errors")
