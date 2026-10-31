# SpinStudio

A GPU simulator for spintronic and neuromorphic devices, with a browser GUI
in the style of COMSOL (model builder, settings, physics check, result tabs).
It started from the physics of [SpinTorch](https://github.com/a-papp/SpinTorch)
(Papp, Porod & Csaba, Nat. Commun. 12, 6422, 2021) and extends it.

## Starting it

1. Double-click `SpinStudio.bat`. For the other node, run `SpinStudio.bat quasi09` instead.
2. The script uploads the code to `~/micromagnetics/` on the GPU node and starts the server there.
3. It forwards the server over SSH and opens http://localhost:8765 in the browser.
4. Keep the window open while you work. Closing it, or pressing Ctrl+C, stops the server.

Every run is saved on the node in `~/micromagnetics/runs/<date>_<mode>_<name>/`, with the configuration and the results.

## Modules

Pick the module in the **Module** box at the top.

### Micromagnetics 2D

This is a finite-difference LLG solver on a GPU for thin films.

**Physics:**
- Field terms: exchange, thin-film demag (FFT), uniaxial anisotropy (with an optional Kᵤ gradient), interfacial DMI with the correct boundary condition, and the Zeeman field.
- Driving terms: RF field sources, spin-orbit torque (SOT), Slonczewski STT (current perpendicular, CPP), and Zhang–Li STT (current in-plane, CIP).
- Thermal field: stochastic LLG solved with the Heun integrator.
- Absorbing boundaries.
- Geometry regions that override Mₛ, Kᵤ, D or α locally.
- Initial states: field, uniform, random, skyrmion, domain wall or vortex.
- Exchange and DMI can be made periodic along x or y.

**Studies:**
- Time-dependent simulation.
- Inverse design: gradient training of an Mₛ map or a bias-field map, as in SpinTorch.

**Result tabs:**
- Field: plays back step by step, with a line cut.
- Intensity.
- Frequency domain: amplitude and phase at the drive frequency.
- k-space, with the analytic iso-frequency curve overlaid.
- Probes: signals and spectra.
- Table: ⟨m⟩, topological charge Q, energy terms, skyrmion core position and radius, domain-wall position.
- Dispersion.
- Training.

**Presets:**
- `focus`: the SpinTorch example.
- `waveguide`: spin-wave dispersion.
- `dw_neuron`: a leaky integrate-and-fire neuron made from a domain wall pushed by SOT.
- `skyrmion`: an SOT-driven skyrmion racetrack.

### Macrospin devices

Thousands of stochastic LLGS macrospins are integrated in parallel on the GPU, for MTJs, stochastic MTJs and spin-torque oscillators (STOs).

**Device studies:**
- transient: time traces, switched fraction and the switching-time histogram.
- sweep: switching probability and write-error rate, and the p-bit transfer curve ⟨s⟩(J) with a tanh fit.
- spectrum: oscillator spectra as a function of current or field.

**p-bit network (probabilistic computing):**
- Each p-bit is a physical stochastic MTJ. The tanh transfer curve is calibrated automatically.
- The p-bits are coupled into an Ising or Boltzmann network. The problems available are the invertible AND gate, MaxCut and a spin glass.
- The sampled state distribution is compared with the exact Boltzmann distribution (KL divergence).

**Reservoir computing with STOs:**
- An array of oscillators with random bias currents and input weights is driven by the input, and a ridge-regression read-out is trained on their states.
- Benchmark tasks: NARMA2, NARMA10, sine/square classification and memory capacity.
- Results are compared with a linear baseline that has no reservoir.

**Presets:**
- `pmtj`: STT switching statistics.
- `smtj_pbit`: low-barrier stochastic MTJ used as a p-bit.
- `sto_reservoir`: STO with out-of-plane precession, used as a reservoir.

## Validation

All tests are in `tools/`. Run them all with `sh tools/run_all.sh` on the node; the log goes to `tests_all.log`.

| Test | Result |
|---|---|
| Against the original SpinTorch (legacy options) | probe outputs, gradient and field agree to ~1e-5 (float32) |
| Spin-wave dispersion vs. Kalinikos–Slavin | 1–3.5 % |
| DMI edge canting vs. sin θ₀ = D / 2√(AK) | 0.01 % |
| Domain wall at rest without current, energy decreasing during relaxation | consistency of the DMI boundary condition |
| Zhang–Li domain-wall velocity (β = α, and the precessional regime) | 0.16 % |
| Thermal noise vs. Boltzmann distribution (micromagnetic and macrospin) | < 1 % |
| STT critical current J_c0 (T = 0) vs. Sun's formula | 0.5 % |
| STO frequency vs. f = γB_DL / (2πα) | < 0.12 % |
| p-bit AND gate vs. exact Boltzmann distribution | KL ≈ 0.01–0.07 |
| p-bit MaxCut (14 nodes) | finds the optimum |
| STO reservoir on NARMA2 | NMSE 0.11 (linear baseline 0.20) |

## Differences from SpinTorch

- The RF source no longer accumulates. SpinTorch's behaviour is still available through the *legacy source* option.
- The ground state is relaxed until it converges.
- |m| is renormalised to 1 after each step.
- Time stepping uses CUDA graphs: about 10× faster for the forward simulation and 3–4× faster for a training epoch.

## Files

- `config.py`: parameters, validated with pydantic, and the micromagnetic presets.
- `solver.py`: the micromagnetic engine. It contains the physics and its compiled kernels, the CUDA-graph time stepping and the training forward pass.
- `macrospin.py`: the macrospin engine, with the device studies, the p-bit network and reservoir computing.
- `server.py`: the FastAPI backend (REST and WebSocket).
- `static/`: the GUI. `app.js` holds the shared code and the micromagnetic module; `macro.js` holds the macrospin module. Plots use Plotly.
