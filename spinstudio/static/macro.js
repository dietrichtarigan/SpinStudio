'use strict';
/* SpinStudio: macrospin device module (MTJ, sMTJ / p-bits, STO reservoir). Uses helpers from app.js. */

const SECTIONS_M = {
  device: { title: 'Free layer', path: 'device', fields: [
    F('name', 'Name', '', 'text'), F('shape', 'Shape', '', 'select', { options: ['ellipse', 'rect'] }),
    F('a_nm', 'Length a (along x)', 'nm'), F('b_nm', 'Width b (along y)', 'nm'), F('t_nm', 'Thickness t', 'nm'),
    F('Ms_kAm', 'Saturation magnetization Mₛ', 'kA/m'), F('alpha', 'Gilbert damping α'),
    F('gamma', 'Gyromagnetic ratio γ', 'rad/sT'),
    F('Ku_MJm3', 'Anisotropy Kᵤ (interface/crystal)', 'MJ/m³'),
    F('anis_axis', 'Easy axis', '', 'select', { options: ['z', 'x', 'y'] }),
    F('demag', 'Demag factors', '', 'select', { options: ['auto', 'manual'] }),
    F('Nx', 'Nx', '', 'num', { show: o => o.demag === 'manual' }), F('Ny', 'Ny', '', 'num', { show: o => o.demag === 'manual' }),
    F('Nz', 'Nz', '', 'num', { show: o => o.demag === 'manual' })],
    help: 'Single-domain (macrospin) free layer. Demag factors "auto" use the ellipsoid with axes a × b × t.' },
  stack: { title: 'Stack, torque and read-out', path: 'stack', fields: [
    F('torque', 'Spin torque', '', 'select', { options: ['STT', 'SOT'] }),
    F('P', 'Spin polarisation P', '', 'num', { show: o => o.torque === 'STT' }),
    F('theta_SH', 'Spin Hall angle θ_SH', '', 'num', { show: o => o.torque === 'SOT' }),
    F('xi_FL', 'Field-like / damping-like ratio'),
    F('px', 'Polariser pₓ'), F('py', 'Polariser p_y'), F('pz', 'Polariser p_z'),
    F('rx', 'Read-out layer rₓ'), F('ry', 'Read-out layer r_y'), F('rz', 'Read-out layer r_z'),
    F('TMR_pct', 'TMR', '%'), F('RA_Ohm_um2', 'Resistance-area RA', 'Ω·µm²')],
    help: 'STT: torque from the reference layer p (MTJ, current perpendicular). SOT: p is the spin polarisation σ of the heavy-metal current. ' +
      'Positive current pushes m towards p. Resistance R = R_P + (R_AP − R_P)(1 − m·r)/2.' },
  mfield: { title: 'Bias field', path: 'field', fields: [F('Bx_mT', 'Bx', 'mT'), F('By_mT', 'By', 'mT'), F('Bz_mT', 'Bz', 'mT')] },
  drive: { title: 'Current drive', path: 'drive', fields: [
    F('J_Am2', 'Current density J', 'A/m²'),
    F('waveform', 'Waveform', '', 'select', { options: ['dc', 'pulse', 'pulse_train', 'sine'] }),
    F('t_start_ns', 'Start time', 'ns'),
    F('width_ns', 'Pulse width', 'ns', 'num', { show: o => o.waveform.startsWith('pulse') }),
    F('period_ns', 'Pulse period', 'ns', 'num', { show: o => o.waveform === 'pulse_train' }),
    F('n_pulses', 'Number of pulses', '', 'int', { show: o => o.waveform === 'pulse_train' }),
    F('f_GHz', 'Frequency', 'GHz', 'num', { show: o => o.waveform === 'sine' }),
    F('rise_ns', 'Rise / fall time', 'ns')],
    derived: () => mreport ? `${fmt(mcfg.drive.J_Am2 / 1e11 * mreport.info.I_per_1e11_uA)} µA through the junction, ` +
      `B_DL = ${fmt(mcfg.drive.J_Am2 / 1e11 * mreport.info.B_DL_per_1e11_mT)} mT` : '' },
  mstudy: { title: 'Study', path: 'study', fields: [
    F('kind', 'Study type', '', 'select', { options: ['transient', 'sweep', 'spectrum'] }),
    F('dt_ps', 'Time step Δt', 'ps'), F('duration_ns', 'Duration', 'ns'), F('temperature_K', 'Temperature', 'K'),
    F('n_devices', 'Devices (per sweep point)', '', 'int', { show: o => o.kind !== 'spectrum' }),
    F('n_traces', 'Traces to record', '', 'int'),
    F('init', 'Initial state', '', 'select', { options: ['up', 'down', 'thermal', 'random'] }),
    F('seed', 'Random seed', '', 'int'),
    F('sweep_param', 'Sweep parameter', '', 'select', { options: ['J', 'Bz', 'Bx', 'width', 'T'], show: o => o.kind !== 'transient' }),
    F('sweep_from', 'Sweep from', '', 'num', { show: o => o.kind !== 'transient' }),
    F('sweep_to', 'Sweep to', '', 'num', { show: o => o.kind !== 'transient' }),
    F('sweep_points', 'Sweep points', '', 'int', { show: o => o.kind !== 'transient' }),
    F('sweep_log', 'Logarithmic sweep', '', 'bool', { show: o => o.kind !== 'transient' }),
    F('record_every', 'Record every', 'steps', 'int'),
    F('readout', 'Read-out', '', 'select', { options: ['mz', 'mx', 'my', 'R'] })],
    help: 'transient: time traces of many devices (switching statistics). sweep: one batch per parameter value ' +
      '(switching probability / write-error rate, p-bit transfer curve <s>). spectrum: oscillator power spectrum vs parameter. ' +
      'Units of the sweep: J in A/m², B in mT, width in ns, T in K.' },
  pbit: { title: 'p-bit network (probabilistic computing)', path: 'pbit', fields: [
    F('problem', 'Problem', '', 'select', { options: ['AND', 'maxcut', 'spin_glass'] }),
    F('n_nodes', 'Number of p-bits', '', 'int', { show: o => o.problem !== 'AND' }),
    F('density', 'Graph density', '', 'num', { show: o => o.problem === 'maxcut' }),
    F('seed', 'Problem seed', '', 'int'), F('beta', 'Inverse temperature β'),
    F('mode', 'p-bit model', '', 'select', { options: ['physical', 'behavioral'] }),
    F('replicas', 'Parallel replicas', '', 'int'), F('duration_ns', 'Duration', 'ns'), F('sample_ns', 'Sample every', 'ns'),
    F('tau_ns', 'Behavioral update time τ', 'ns', 'num', { show: o => o.mode === 'behavioral' }),
    F('calib_points', 'Calibration points', '', 'int', { show: o => o.mode === 'physical' })],
    help: 'Each p-bit is a stochastic MTJ (the free layer above, at the study temperature). Its output s = sign(m·r) feeds the ' +
      'input currents of the others, I_i = β(Σ J_ij s_j + h_i), through the measured tanh transfer curve (calibrated automatically). ' +
      'The network samples the Boltzmann distribution of the Ising problem; "behavioral" uses ideal p-bits for comparison.' },
  reservoir: { title: 'Reservoir computing (spin-torque oscillators)', path: 'reservoir', fields: [
    F('task', 'Benchmark task', '', 'select', { options: ['narma2', 'narma10', 'sine_square', 'memory'] }),
    F('n_osc', 'Oscillators', '', 'int'), F('n_samples', 'Input samples', '', 'int'), F('input_ns', 'Hold time per sample', 'ns'),
    F('J_min', 'Bias current from', 'A/m²'), F('J_max', 'Bias current to', 'A/m²'),
    F('input_scale', 'Input modulation (relative)'), F('feature', 'Read-out feature', '', 'select', { options: ['mz', 'mx', 'mz+mx2'] }),
    F('washout', 'Washout samples', '', 'int'), F('train_frac', 'Training fraction'), F('ridge', 'Ridge regularisation'),
    F('seed', 'Seed', '', 'int')],
    help: 'An array of STOs (the device above) with random bias currents and input weights is driven by the input sequence; ' +
      'a linear ridge read-out is trained on the oscillator states. NMSE is compared with a linear baseline without reservoir.' },
};
const TREE_M = [
  { head: 'Device' },
  { id: 'device', ic: '◆', label: 'Free layer' },
  { id: 'stack', ic: '☰', label: 'Stack & torque' },
  { id: 'mfield', ic: '➝', label: 'Bias field' },
  { id: 'drive', ic: '⏚', label: 'Current drive' },
  { head: 'Study' },
  { id: 'mstudy', ic: '⏱', label: 'Device study' },
  { head: 'Neuromorphic apps' },
  { id: 'pbit', ic: '⚄', label: 'p-bit network' },
  { id: 'reservoir', ic: '≋', label: 'Reservoir computing' },
  { head: 'Results' },
  { id: 'res:mtraces', ic: '∿', label: 'Traces', tab: 'mtraces' },
  { id: 'res:msweep', ic: '⤴', label: 'Sweep / statistics', tab: 'msweep' },
  { id: 'res:mspectrum', ic: '▥', label: 'Spectrum', tab: 'mspectrum' },
  { id: 'res:mpbit', ic: '⚄', label: 'p-bit results', tab: 'mpbit' },
  { id: 'res:mres', ic: '≋', label: 'Reservoir results', tab: 'mres' },
];
const TABS_M = [['mdevice', 'Device'], ['mtraces', 'Traces'], ['msweep', 'Sweep'], ['mspectrum', 'Spectrum'],
  ['mpbit', 'p-bit network'], ['mres', 'Reservoir']];
const MACRO_MATERIALS = {
  'CoFeB/MgO (PMA)': { Ms_kAm: 1100, alpha: 0.01, Ku_MJm3: 1.0, anis_axis: 'z' },
  'CoFeB in-plane': { Ms_kAm: 1000, alpha: 0.01, Ku_MJm3: 0.0, anis_axis: 'x' },
  'Permalloy': { Ms_kAm: 800, alpha: 0.01, Ku_MJm3: 0.0, anis_axis: 'x' },
  'Co/Pt': { Ms_kAm: 580, alpha: 0.1, Ku_MJm3: 0.8, anis_axis: 'z' },
};

function macroRenderSettings(body, title) {
  const S = SECTIONS_M[node];
  if (!S) {
    const t = TREE_M.find(x => x.id === node);
    title.textContent = t ? t.label : 'Results';
    body.append(h('div', { class: 'help' }, {
      'res:mtraces': 'Magnetisation and resistance of the first devices vs time, ensemble average and switched fraction.',
      'res:msweep': 'Switching probability (write-error rate) and time-averaged output <s> vs the swept parameter, with fits.',
      'res:mspectrum': 'Power spectral density of the read-out component vs the swept parameter (oscillator tunability).',
      'res:mpbit': 'State histogram of the p-bit network against the exact Boltzmann distribution, energy vs time.',
      'res:mres': 'Reservoir prediction vs target on the test set, memory-capacity curve, oscillator responses.',
    }[node] || ''));
    return;
  }
  title.textContent = S.title;
  const obj = mcfg[S.path];
  const rows = S.fields.map(f => fieldRow(obj, f));
  if (node === 'device') {
    rows.unshift(h('div', { class: 'field' }, h('label', {}, 'Library'),
      h('select', { onchange: e => { const m = MACRO_MATERIALS[e.target.value]; if (m) { Object.assign(mcfg.device, m, { name: e.target.value }); changed(true); } } },
        h('option', { value: '' }, '— choose —'), ...Object.keys(MACRO_MATERIALS).map(k => h('option', { value: k }, k))), h('span')));
  }
  body.append(section(S.title, [S.help ? h('div', { class: 'help' }, S.help) : null, ...rows,
    S.derived ? h('div', { class: 'derived' }, S.derived()) : null]));
  const btn = { mstudy: ['▶ Run device study', 'macro_study'], pbit: ['⚄ Run p-bit network', 'macro_pbit'], reservoir: ['≋ Run reservoir', 'macro_reservoir'] }[node];
  if (btn) body.append(h('button', { class: 'sbtn', style: 'padding:5px 12px', disabled: running, onclick: () => startMacro(btn[1]) }, btn[0]));
}

async function mcheck() {
  try {
    mreport = await post('/api/macro/check', mcfg);
    renderMReport();
    const d = $('.derived'); if (d && SECTIONS_M[node] && SECTIONS_M[node].derived) d.textContent = SECTIONS_M[node].derived();
    if (tab === 'mdevice') renderTab();
  } catch (e) {
    mreport = null; $('#report').innerHTML = '';
    $('#report').append(h('div', { class: 'e' }, '✕ ' + e.message));
  }
}
function renderMReport() {
  const el = $('#report'); el.innerHTML = '';
  if (!mreport) return;
  const i = mreport.info;
  for (const w of mreport.warnings) el.append(h('div', { class: 'w' }, '⚠ ' + w));
  for (const w of mreport.notes) el.append(h('div', {}, 'ℹ ' + w));
  if (!mreport.warnings.length) el.append(h('div', { class: 'ok' }, '✓ Parameters look consistent'));
  const rows = [
    ['Area / volume', `${fmt(i.area_nm2)} nm² / ${fmt(i.volume_nm3)} nm³`],
    ['Demag factors Nx, Ny, Nz', i.N.map(v => v.toFixed(3)).join(', ')],
    ['K_eff (easy axis)', fmt(i.Keff_MJm3) + ' MJ/m³'],
    ['Anisotropy field B_k,eff', fmt(i.Bk_eff_mT) + ' mT'],
    ['Thermal barrier Δ', i.Delta !== undefined ? fmt(i.Delta) + ' kT' : '–'],
    ['R_P / R_AP', `${fmt(i.R_P_Ohm)} Ω / ${fmt(i.R_AP_Ohm)} Ω`],
    ['B_DL per 10¹¹ A/m²', fmt(i.B_DL_per_1e11_mT) + ' mT'],
    ['Current per 10¹¹ A/m²', fmt(i.I_per_1e11_uA) + ' µA'],
    ['Critical current J_c0 (T = 0)', i.Jc0_Am2 ? fmt(i.Jc0_Am2) + ' A/m²' : '–'],
    ['ω·Δt', fmt(mreport.omega_dt)],
    ['Devices × steps', `${mreport.devices.toLocaleString()} × ${Math.round(mreport.steps).toLocaleString()}`],
  ];
  el.append(h('table', {}, ...rows.map(r => h('tr', {}, h('td', {}, r[0]), h('td', {}, r[1])))));
}

async function startMacro(mode) {
  try {
    await mcheck();
    if (!mreport) { addMsg({ level: 'error', text: 'Fix the parameter errors first.' }); return; }
    runStart = performance.now();
    $('#progLabel').textContent = 'Starting…';
    await post('/api/macro/run', { config: mcfg, mode });
    setRunning(true, mode);
  } catch (e) { addMsg({ level: 'error', text: e.message }); }
}
async function loadMacroResult(select = true) {
  try { mres = await api('/api/macro/result'); } catch (e) { mres = null; return; }
  $('#btnExport').classList.remove('disabled');
  if (select && mod === 'macro') {
    const t = mres.kind === 'pbit' ? 'mpbit' : mres.kind === 'reservoir' ? 'mres' :
      mres.kind === 'transient' ? 'mtraces' : mres.kind === 'sweep' ? 'msweep' : 'mspectrum';
    selectTab(t);
  } else renderTabs();
}
function macroTabEnabled(id) {
  if (id === 'mdevice') return true;
  if (!mres) return false;
  return { mtraces: mres.kind === 'transient', msweep: mres.kind === 'sweep', mspectrum: mres.kind === 'spectrum',
    mpbit: mres.kind === 'pbit', mres: mres.kind === 'reservoir' }[id];
}

// ------------------------------------------------------------------ macro plots
const LAY = (title, x, y, extra = {}) => Object.assign({
  margin: { l: 70, r: 24, t: 34, b: 48 }, title: { text: title, font: { size: 13 }, x: 0.02 },
  xaxis: { title: { text: x }, showline: true, mirror: true, ticks: 'outside', zeroline: false },
  yaxis: { title: { text: y }, showline: true, mirror: true, ticks: 'outside', exponentformat: 'power' },
  paper_bgcolor: '#fff', plot_bgcolor: '#fff', legend: { font: { size: 11 } }, font: { family: 'Segoe UI, Arial', size: 12 },
}, extra);
const PNAME = { J: 'J (A/m²)', Bz: 'Bz (mT)', Bx: 'Bx (mT)', width: 'pulse width (ns)', T: 'T (K)' };
const MCOL = ['#1b6ac9', '#c2410c', '#12a150'];

function drawMacroTab(tb) {
  showSub(false);
  if (tab === 'mdevice') return drawMDevice(tb);
  if (!mres || !macroTabEnabled(tab)) {
    const what = { mtraces: 'a transient study', msweep: 'a sweep study', mspectrum: 'a spectrum study', mpbit: 'the p-bit network', mres: 'the reservoir' }[tab];
    return emptyPlot(`No result for this view yet.<br>Run ${what} (Model Builder → ${tab === 'mpbit' ? 'p-bit network' : tab === 'mres' ? 'Reservoir computing' : 'Device study'}).`);
  }
  ({ mtraces: drawMTraces, msweep: drawMSweep, mspectrum: drawMSpectrum, mpbit: drawMPbit, mres: drawMRes })[tab](tb);
}

function drawMDevice(tb) {
  if (!mcfg) return;
  const d = mcfg.device, st = mcfg.stack;
  tb.append(h('span', { class: 'muted' }, 'Top view of the free layer with easy axis (blue), polariser p (orange), read-out layer r (green) and bias field (grey). Out-of-plane vectors are drawn as ⊙.'));
  const a = d.a_nm / 2, b = d.b_nm / 2, L = Math.max(a, b);
  const shapes = [d.shape === 'ellipse'
    ? { type: 'circle', x0: -a, x1: a, y0: -b, y1: b, line: { color: '#26313d', width: 2 }, fillcolor: 'rgba(27,106,201,0.08)' }
    : { type: 'rect', x0: -a, x1: a, y0: -b, y1: b, line: { color: '#26313d', width: 2 }, fillcolor: 'rgba(27,106,201,0.08)' }];
  const ann = [];
  const arrow = (v, col, label, off) => {
    const n = Math.hypot(v[0], v[1], v[2]) || 1;
    const vx = v[0] / n, vy = v[1] / n, vz = v[2] / n;
    if (Math.hypot(vx, vy) < 0.2) {
      ann.push({ x: off * L * 0.35, y: -L * 0.15 * off, text: `${vz >= 0 ? '⊙' : '⊗'} ${label}`, showarrow: false, font: { color: col, size: 15 } });
    } else {
      ann.push({ x: vx * L * 0.9, y: vy * L * 0.9, ax: 0, ay: 0, axref: 'x', ayref: 'y', xref: 'x', yref: 'y', showarrow: true,
        arrowhead: 2, arrowwidth: 2.5, arrowcolor: col, text: label, font: { color: col, size: 13 } });
    }
  };
  const ea = { x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1] }[d.anis_axis];
  arrow(ea, '#1b6ac9', 'easy', -1);
  arrow([st.px, st.py, st.pz], '#c2410c', st.torque === 'STT' ? 'p (pol.)' : 'σ (SOT)', 0);
  arrow([st.rx, st.ry, st.rz], '#12a150', 'r (TMR)', 1);
  const f = mcfg.field;
  if (Math.hypot(f.Bx_mT, f.By_mT, f.Bz_mT) > 0) arrow([f.Bx_mT, f.By_mT, f.Bz_mT], '#666', 'B', 2);
  Plotly.react('plotMain', [{ x: [-L * 1.4, L * 1.4], y: [-L * 1.4, L * 1.4], mode: 'markers', marker: { opacity: 0 }, hoverinfo: 'skip' }],
    LAY(`${d.name} — ${d.a_nm} × ${d.b_nm} × ${d.t_nm} nm³`, 'x (nm)', 'y (nm)', {
      shapes, annotations: ann, showlegend: false, yaxis: { scaleanchor: 'x', title: { text: 'y (nm)' }, showline: true, mirror: true },
    }), PCONF);
}

function drawMTraces(tb) {
  const r = mres;
  const comps = ['mx', 'my', 'mz'];
  view.mcomp = view.mcomp || 'mz';
  tb.append(sel('Show', view.mcomp, [['mx', 'mₓ'], ['my', 'm_y'], ['mz', 'm_z'], ['R', 'Resistance R']], v => { view.mcomp = v; renderTab(); }));
  tb.append(h('span', {}, `P(switched) at the end = ${(100 * r.p_switch).toFixed(1)} % of ${r.n_devices} devices · ${Math.round(r.steps_per_s).toLocaleString()} steps/s`));
  const traces = [];
  r.traces.forEach((tr, i) => {
    const y = view.mcomp === 'R' ? r.R[i] : tr[comps.indexOf(view.mcomp)];
    traces.push({ x: r.t_ns, y, mode: 'lines', name: `device ${i}`, line: { width: 1.2 }, opacity: 0.8 });
  });
  if (view.mcomp !== 'R') traces.push({ x: r.t_ns, y: r.ens[comps.indexOf(view.mcomp)], mode: 'lines', name: 'ensemble ⟨m⟩', line: { width: 3, color: '#111' } });
  traces.push({ x: r.t_ns, y: r.drive, mode: 'lines', name: 'J(t)', yaxis: 'y2', line: { color: '#c2410c', dash: 'dot', width: 1.5 } });
  Plotly.react('plotMain', traces, LAY('Magnetisation / resistance vs time', 't (ns)', view.mcomp === 'R' ? 'R (Ω)' : view.mcomp, {
    yaxis2: { title: { text: 'J (A/m²)' }, overlaying: 'y', side: 'right', exponentformat: 'power', showgrid: false },
  }), PCONF);
  showSub(true);
  const hist = r.t_switch_ns.length ? [{ type: 'histogram', x: r.t_switch_ns, nbinsx: 40, marker: { color: '#7a3fc2' }, name: 'switching time' }] : [];
  Plotly.react('plotSub', [{ x: r.t_ns, y: r.p_switch_t, mode: 'lines', name: 'switched fraction', line: { color: '#1b6ac9', width: 2 } },
    ...hist.map(t => Object.assign(t, { xaxis: 'x2', yaxis: 'y2' }))], LAY('Switched fraction and switching-time histogram', 't (ns)', 'fraction', {
    grid: { rows: 1, columns: 2, pattern: 'independent' }, xaxis: { domain: [0, 0.46], title: { text: 't (ns)' } },
    xaxis2: { domain: [0.56, 1], title: { text: 'switching time (ns)' }, anchor: 'y2' }, yaxis2: { anchor: 'x2', title: { text: 'count' } }, showlegend: false,
  }), PCONF);
}

function drawMSweep(tb) {
  const r = mres;
  const x = r.values;
  view.mscale = view.mscale || 'lin';
  tb.append(sel('Probability axis', view.mscale, [['lin', 'linear'], ['wer', 'write-error rate (log)']], v => { view.mscale = v; renderTab(); }));
  const fit = r.fit || {};
  let info = `${r.n_devices.toLocaleString()} devices · ${Math.round(r.steps_per_s).toLocaleString()} steps/s`;
  if (fit.x50 !== undefined) info += ` · 50 % switching at ${fmt(fit.x50)} ${PNAME[r.param] || ''}`;
  if (fit.tanh_scale !== undefined) info += ` · ⟨s⟩ ≈ tanh((x − ${fmt(fit.tanh_offset)}) / ${fmt(fit.tanh_scale)})`;
  tb.append(h('span', {}, info));
  const wer = view.mscale === 'wer';
  const y = wer ? r.p_switch.map(p => Math.max(1 - p, 0.5 / r.n_devices * 1)) : r.p_switch;
  Plotly.react('plotMain', [
    { x, y, mode: 'lines+markers', name: wer ? 'WER = 1 − P(switch)' : 'P(switch)', error_y: wer ? undefined : { array: r.p_err, thickness: 1 },
      line: { color: '#1b6ac9', width: 2 } },
    { x, y: r.t_switch_ns, mode: 'lines+markers', name: 'median switching time', yaxis: 'y2', line: { color: '#7a3fc2', dash: 'dot' } },
  ], LAY('Switching statistics', PNAME[r.param] || r.param, wer ? 'write-error rate' : 'switching probability', {
    yaxis: { type: wer ? 'log' : 'linear', title: { text: wer ? 'write-error rate' : 'P(switch)' }, showline: true, mirror: true, exponentformat: 'power' },
    yaxis2: { title: { text: 'median t_sw (ns)' }, overlaying: 'y', side: 'right', showgrid: false },
    legend: { x: 0.01, y: 0.99, bgcolor: 'rgba(255,255,255,0.8)' },
  }), PCONF);
  showSub(true);
  const tr = [{ x, y: r.s_mean, error_y: { array: r.s_err, thickness: 1 }, mode: 'markers', name: '⟨s⟩ = ⟨sign(m·r)⟩', marker: { color: '#12a150', size: 7 } }];
  if (fit.tanh_scale !== undefined) {
    const xs = []; const n = 200; const a = Math.min(...x), b = Math.max(...x);
    for (let i = 0; i <= n; i++) xs.push(a + (b - a) * i / n);
    tr.push({ x: xs, y: xs.map(v => Math.tanh((v - fit.tanh_offset) / fit.tanh_scale)), mode: 'lines', name: 'tanh fit', line: { color: '#111', dash: 'dash' } });
  }
  tr.push({ x, y: r.m_final[2], mode: 'lines', name: '⟨m_z⟩ final', line: { color: '#c2410c', width: 1 }, yaxis: 'y' });
  Plotly.react('plotSub', tr, LAY('Time-averaged output ⟨s⟩ (p-bit transfer curve)', PNAME[r.param] || r.param, '⟨s⟩', { yaxis: { range: [-1.05, 1.05], title: { text: '⟨s⟩' } } }), PCONF);
}

function drawMSpectrum(tb) {
  const r = mres;
  tb.append(h('span', {}, `Read-out ${r.config.study.readout} · ${Math.round(r.steps_per_s).toLocaleString()} steps/s`));
  const multi = r.values.length > 1;
  if (multi) {
    Plotly.react('plotMain', [
      { type: 'heatmap', x: r.values, y: r.f_GHz, z: r.psd[0].map((_, j) => r.psd.map(row => row[j])), colorscale: CMAPS.magma,
        colorbar: { title: { text: 'log₁₀ PSD', side: 'right' }, thickness: 14 } },
      { x: r.values, y: r.peak_GHz, mode: 'markers', name: 'peak', marker: { color: '#5fe0ff', size: 6 } },
    ], LAY('Oscillator spectrum vs ' + r.param, PNAME[r.param] || r.param, 'f (GHz)', { showlegend: false }), PCONF);
  } else {
    Plotly.react('plotMain', [{ x: r.f_GHz, y: r.psd[0], mode: 'lines', line: { color: '#1b6ac9' } }],
      LAY(`Spectrum (peak ${fmt(r.peak_GHz[0])} GHz)`, 'f (GHz)', 'log₁₀ PSD'), PCONF);
  }
  showSub(true);
  Plotly.react('plotSub', [{ x: r.trace_t_ns, y: r.trace, mode: 'lines', line: { color: '#12a150', width: 1 }, name: 'trace' },
    { x: r.values, y: r.peak_GHz, mode: 'lines+markers', xaxis: 'x2', yaxis: 'y2', name: 'peak f', line: { color: '#c2410c' } }],
    LAY(`Time trace at ${r.param} = ${fmt(r.trace_param)} and peak frequency`, 't (ns)', r.config.study.readout, {
      grid: { rows: 1, columns: 2, pattern: 'independent' }, xaxis: { domain: [0, 0.55], title: { text: 't (ns)' } },
      xaxis2: { domain: [0.63, 1], title: { text: PNAME[r.param] || r.param }, anchor: 'y2' },
      yaxis2: { anchor: 'x2', title: { text: 'f_peak (GHz)' } }, showlegend: false,
    }), PCONF);
}

function drawMPbit(tb) {
  const r = mres;
  let info = `${r.problem} · ${r.mode} p-bits · β = ${r.beta} · best energy ${fmt(r.best_E)}`;
  if (r.E_ground !== undefined) info += ` (ground state ${fmt(r.E_ground)})`;
  if (r.KL !== undefined) info += ` · KL(sim‖Boltzmann) = ${r.KL.toFixed(4)}`;
  if (r.cut_best !== undefined) info += ` · best cut ${r.cut_best}` + (r.cut_optimum !== undefined ? ` / optimum ${r.cut_optimum}` : '');
  tb.append(h('span', {}, info));
  if (r.states && r.states.length <= 256) {
    const lab = r.problem === 'AND' ? r.states.map(s => `${s[0]}${s[1]}→${s[2]}`) : r.states;
    Plotly.react('plotMain', [
      { type: 'bar', x: lab, y: r.p_exact, name: 'exact Boltzmann', marker: { color: 'rgba(59,75,94,0.35)' } },
      { type: 'bar', x: lab, y: r.p_sim, name: 'p-bit network', marker: { color: '#7a3fc2' } },
    ], LAY(r.problem === 'AND' ? 'Invertible AND gate: state probabilities (A B → C)' : 'State probabilities', 'state', 'probability',
      { barmode: 'group', legend: { x: 0.99, xanchor: 'right', y: 0.99 } }), PCONF);
  } else {
    Plotly.react('plotMain', [{ x: r.t_ns, y: r.E_mean, mode: 'lines', name: 'mean energy' }, { x: r.t_ns, y: r.E_best, mode: 'lines', name: 'best energy', line: { width: 2.5 } }],
      LAY('Ising energy vs time', 't (ns)', 'E'), PCONF);
  }
  showSub(true);
  const tr = [{ x: r.t_ns, y: r.E_mean, mode: 'lines', name: 'mean energy (replicas)', line: { color: '#1b6ac9' } },
    { x: r.t_ns, y: r.E_best, mode: 'lines', name: 'best energy', line: { color: '#c2410c', width: 2.5 } }];
  const lay = LAY('Energy vs time' + (r.calibration ? ' · sMTJ transfer curve (calibration)' : ''), 't (ns)', 'E');
  if (r.calibration) {
    const c = r.calibration;
    tr.push({ x: c.J, y: c.s, error_y: { array: c.s_err, thickness: 1 }, mode: 'markers', xaxis: 'x2', yaxis: 'y2', name: 'measured ⟨s⟩', marker: { color: '#12a150' } });
    tr.push({ x: c.J, y: c.J.map(v => Math.tanh((v - c.J_off) / c.J0)), mode: 'lines', xaxis: 'x2', yaxis: 'y2', name: 'tanh fit', line: { color: '#111', dash: 'dash' } });
    Object.assign(lay, { grid: { rows: 1, columns: 2, pattern: 'independent' }, xaxis: { domain: [0, 0.5], title: { text: 't (ns)' } },
      xaxis2: { domain: [0.6, 1], title: { text: 'J (A/m²)' }, anchor: 'y2', exponentformat: 'power' }, yaxis2: { anchor: 'x2', title: { text: '⟨s⟩' } } });
  }
  Plotly.react('plotSub', tr, lay, PCONF);
}

function drawMRes(tb) {
  const r = mres;
  tb.append(h('span', {}, `${r.task} · ${r.n_osc} oscillators · ${r.n_samples} samples · ${r.metric}` +
    (r.NMSE_linear !== undefined ? ` (linear baseline NMSE ${r.NMSE_linear.toFixed(4)})` : '')));
  if (r.task === 'memory') {
    Plotly.react('plotMain', [{ type: 'bar', x: r.delays, y: r.r2, marker: { color: '#1b6ac9' } }],
      LAY(`Short-term memory: r² per delay, MC = ${r.MC.toFixed(2)}`, 'delay k (samples)', 'r²', { yaxis: { range: [0, 1.05], title: { text: 'r²' } } }), PCONF);
  } else {
    Plotly.react('plotMain', [{ x: r.test_k, y: r.target, mode: 'lines', name: 'target', line: { color: '#111', width: 2 } },
      { x: r.test_k, y: r.pred, mode: 'lines', name: 'reservoir prediction', line: { color: '#c2410c', width: 1.6 } }],
      LAY('Test set: prediction vs target', 'sample k', 'y', { legend: { x: 0.99, xanchor: 'right', y: 0.99 } }), PCONF);
  }
  showSub(true);
  const t = r.trace;
  Plotly.react('plotSub', [
    { x: t.t_ns, y: t.mx, mode: 'lines', name: 'mₓ (oscillator 0)', line: { width: 0.8, color: '#1b6ac9' } },
    { x: t.t_ns, y: t.mz, mode: 'lines', name: 'm_z (oscillator 0)', line: { width: 2, color: '#c2410c' } },
    { x: t.t_ns, y: t.u, mode: 'lines', name: 'input u', line: { width: 1.5, color: '#12a150', shape: 'hv' } },
    { type: 'heatmap', z: r.features, xaxis: 'x2', yaxis: 'y2', colorscale: CMAPS.wave, showscale: false },
  ], LAY('Oscillator response to the input · feature matrix (oscillators × samples)', 't (ns)', 'm', {
    grid: { rows: 1, columns: 2, pattern: 'independent' }, xaxis: { domain: [0, 0.55], title: { text: 't (ns)' } },
    xaxis2: { domain: [0.62, 1], title: { text: 'sample' }, anchor: 'y2' }, yaxis2: { anchor: 'x2', title: { text: 'oscillator' } },
    legend: { orientation: 'h', y: -0.25 },
  }), PCONF);
}
