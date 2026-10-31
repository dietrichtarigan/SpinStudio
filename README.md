# SpinStudio

A GPU simulator for spintronic and neuromorphic devices, with a browser GUI in the style of COMSOL.

There are two modules:

- **Micromagnetics 2D** covers spin waves, domain walls and skyrmions. Physics: DMI, anisotropy, SOT/STT/Zhang–Li torques and thermal noise. It also supports inverse design as in SpinTorch.
- **Macrospin devices** covers MTJ switching statistics, stochastic MTJs used as p-bits (Ising and Boltzmann networks), and spin-torque oscillators used for reservoir computing.

Full documentation, the physics, and the validation against analytic results are in [spinstudio/README.md](spinstudio/README.md).

## Quick start

**Requirements:** a Python environment with an NVIDIA GPU and the packages in `spinstudio/requirements.txt`.

**Run the server directly:**

```bash
pip install -r spinstudio/requirements.txt
python -m spinstudio.server --port 8765      # then open http://localhost:8765
```

**Run it on a remote GPU node:** `SpinStudio.bat` / `SpinStudio.ps1` upload the code, start the server on the node and open an SSH tunnel to it. Edit the node names in `SpinStudio.ps1` to match your SSH config.

## Tests

`tools/` contains the validation suite. Run all of it with `sh tools/run_all.sh`.

Most tests need nothing extra. `tools/verify.py` compares results against the original SpinTorch, so it needs a clone of https://github.com/a-papp/SpinTorch in `./SpinTorch`.

## Acknowledgement

The spin-wave solver started as a GPU-optimised re-implementation of SpinTorch (Á. Papp, W. Porod, G. Csaba, *Nanoscale neural network using non-linear spin-wave interference*, Nat. Commun. 12, 6422 (2021)).
