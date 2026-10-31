'use strict';
/* SpinStudio front end: model builder, settings, live graphics. */

// ------------------------------------------------------------------ utils
const $ = (s, el = document) => el.querySelector(s);
function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (k === 'style') el.style.cssText = v;
    else if (k in el && k !== 'list') el[k] = v;
    else el.setAttribute(k, v);
  }
  for (const c of kids.flat()) if (c !== null && c !== undefined && c !== false) el.append(c instanceof Node ? c : document.createTextNode(c));
  return el;
}
function fmt(v, d = 3) {
  if (v === null || v === undefined || !isFinite(v)) return '–';
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  return (+v.toPrecision(d)).toString();
}
async function api(path, opt = {}) {
  const r = await fetch(path, opt);
  if (!r.ok) {
    let t = await r.text();
    try {
      const j = JSON.parse(t);
      t = Array.isArray(j.detail) ? j.detail.map(d => `${(d.loc || []).slice(1).join('.')}: ${d.msg}`).join('\n') : (j.detail || t);
    } catch (e) { /* plain text */ }
    throw new Error(t);
  }
  return r.json();
}
const post = (p, body) => api(p, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
function b64f32(b64) {
  const bin = atob(b64); const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(u8.buffer);
}
function decode(e) {
  const f = b64f32(e.b64); const [r, c] = e.shape; const z = new Array(r);
  for (let i = 0; i < r; i++) z[i] = Array.from(f.subarray(i * c, (i + 1) * c));
  return { z, min: e.min, max: e.max, rows: r, cols: c };
}
let debTimer = null;
const debounce = (fn, ms) => { clearTimeout(debTimer); debTimer = setTimeout(fn, ms); };

// ------------------------------------------------------------------ state
let cfg = null, geom = null, report = null, meta = null;
let mod = 'micro', mcfg = null, mreport = null, mres = null;   // module: micromagnetics | macrospin
let node = 'grid', tab = 'geometry';
let running = false, runMode = null;
const train = { loss: [], outputs: [], design: null, intensity: null, unit: '', target: 0 };
const cache = new Map();
const ranges = {};
const view = {
  comp: 'z', cmap: 'wave', scale: 'global', frame: 0, playing: false, fps: 15, linecut: false,
  live: null, liveStride: 1, liveT: 0, geomMap: 'Ms', icomp: 'z', idec: 3, dkind: 'dft_amp', dcomp: 'z',
  pcomp: 'x', pview: 'signals', psel: 'all', plog: true, kcomp: 'z', overlay: true,
};

const CMAPS = {
  wave: [[0, '#053061'], [0.15, '#2166ac'], [0.32, '#92c5de'], [0.5, '#f7f7f7'], [0.68, '#f4a582'], [0.85, '#b2182b'], [1, '#67001f']],
  magma: [[0, '#000004'], [0.2, '#3b0f70'], [0.4, '#8c2981'], [0.6, '#de4968'], [0.8, '#fe9f6d'], [1, '#fcfdbf']],
  viridis: 'Viridis', jet: 'Jet', greys: 'Greys', cividis: 'Cividis',
  phase: [[0, '#e6194b'], [0.17, '#f5d000'], [0.33, '#3cb44b'], [0.5, '#00bcd4'], [0.67, '#2050e0'], [0.83, '#c030c0'], [1, '#e6194b']],
  summer: [[0, '#008066'], [1, '#ffff66']],
};
const PCONF = { responsive: true, displaylogo: false, toImageButtonOptions: { format: 'png', scale: 3 },
  modeBarButtonsToRemove: ['lasso2d', 'select2d'] };
const MATERIALS = {
  YIG: { Ms_kAm: 140, Aex_pJm: 3.65, alpha: 1e-4, Ku_MJm3: 0, D_mJm2: 0 },
  Permalloy: { Ms_kAm: 800, Aex_pJm: 13, alpha: 0.008, Ku_MJm3: 0, D_mJm2: 0 },
  'CoFeB (in-plane)': { Ms_kAm: 1250, Aex_pJm: 15, alpha: 0.004, Ku_MJm3: 0, D_mJm2: 0 },
  'Co2MnSi (Heusler)': { Ms_kAm: 1000, Aex_pJm: 20, alpha: 0.003, Ku_MJm3: 0, D_mJm2: 0 },
  'CoFeB/MgO (PMA)': { Ms_kAm: 1100, Aex_pJm: 15, alpha: 0.01, Ku_MJm3: 1.0, anis_axis: 'z', D_mJm2: 0 },
  'CoFeB/Pt (PMA + DMI)': { Ms_kAm: 1100, Aex_pJm: 15, alpha: 0.05, Ku_MJm3: 1.0, anis_axis: 'z', D_mJm2: 1.0 },
  'Pt/Co/AlOx (skyrmions)': { Ms_kAm: 580, Aex_pJm: 15, alpha: 0.3, Ku_MJm3: 0.8, anis_axis: 'z', D_mJm2: 3.0 },
};

// ------------------------------------------------------------------ schema
const F = (key, label, unit = '', type = 'num', extra = {}) => ({ key, label, unit, type, ...extra });
const SECTIONS = {
  grid: { title: 'Grid & Mesh', path: 'grid', fields: [
    F('nx', 'Cells in x', '', 'int'), F('ny', 'Cells in y', '', 'int'),
    F('dx_nm', 'Cell size Δx', 'nm'), F('dy_nm', 'Cell size Δy', 'nm'), F('dz_nm', 'Film thickness', 'nm'),
    F('pbc_x', 'Periodic in x (exchange/DMI)', '', 'bool'), F('pbc_y', 'Periodic in y (exchange/DMI)', '', 'bool')],
    derived: () => {
      const g = cfg.grid;
      return `Domain ${fmt(g.nx * g.dx_nm / 1000)} × ${fmt(g.ny * g.dy_nm / 1000)} µm, ` +
        `${(g.nx * g.ny).toLocaleString()} cells` + (report && report.exchange_length_nm ? `, exchange length ${fmt(report.exchange_length_nm)} nm` : '');
    } },
  material: { title: 'Magnetic material', path: 'material', fields: [
    F('name', 'Name', '', 'text'), F('Ms_kAm', 'Saturation magnetization Mₛ', 'kA/m'),
    F('Aex_pJm', 'Exchange stiffness A', 'pJ/m'), F('alpha', 'Gilbert damping α'),
    F('gamma', 'Gyromagnetic ratio γ', 'rad/sT'),
    F('Ku_MJm3', 'Uniaxial anisotropy Kᵤ', 'MJ/m³'), F('anis_axis', 'Anisotropy axis', '', 'select', { options: ['z', 'x', 'y'] }),
    F('Ku_grad_MJm3_um', 'Kᵤ gradient along x', 'MJ/m³/µm', 'num', { title: 'Linear change of Ku along x, e.g. a restoring force (leak) for domain-wall neurons' }),
    F('D_mJm2', 'Interfacial DMI D', 'mJ/m²')],
    derived: () => !report ? '' : (`K_eff = ${fmt(report.Keff_MJm3)} MJ/m³` +
      (report.dw_width_nm ? `, DW width Δ = ${fmt(report.dw_width_nm)} nm, D_c = ${fmt(report.Dc_mJm2)} mJ/m²` : '') +
      (report.exchange_length_nm ? `, exchange length ${fmt(report.exchange_length_nm)} nm` : '')) },
  field: { title: 'External bias field', path: 'field', fields: [
    F('Bx_mT', 'Bx', 'mT'), F('By_mT', 'By', 'mT'), F('Bz_mT', 'Bz', 'mT')],
    help: 'Uniform bias field. The ground state is relaxed along this field before the dynamics starts.' },
  boundary: { title: 'Absorbing boundary', path: 'boundary', fields: [
    F('width', 'Layer width', 'cells', 'int'), F('alpha_max', 'Damping at outer edge')],
    help: 'Damping grows quadratically towards the edges so spin waves are absorbed instead of reflected.' },
  study: { title: 'Time-dependent study', path: 'study', fields: [
    F('dt_ps', 'Time step Δt', 'ps'), F('steps', 'Number of steps', '', 'int'),
    F('temperature_K', 'Temperature', 'K', 'num', { title: 'T > 0 adds the thermal field (stochastic LLG, Heun integrator)' }),
    F('seed', 'Random seed', '', 'int'),
    F('demag', 'Demagnetising field', '', 'bool'),
    F('record', 'Store frames as', '', 'select', { options: ['dm', 'm'], title: 'dm = m − m₀ (spin waves), m = full magnetisation (domain walls, skyrmions)' }),
    F('relax_steps', 'Min. relaxation steps', '', 'int'), F('relax_alpha', 'Relaxation damping'),
    F('relax_tol', 'Relaxation tolerance', '|Δm|', 'num', { title: 'Relax until the largest change of m per step is below this (0 = fixed number of steps, like SpinTorch)' }),
    F('relax_max', 'Max. relaxation steps', '', 'int'),
    F('save_every', 'Store field every', 'steps', 'int'),
    F('probe_component', 'Probe component', '', 'select', { options: ['x', 'y', 'z'] }),
    F('normalize', 'Renormalize |m| = 1', '', 'bool'),
    F('precision', 'Precision', '', 'select', { options: ['float32', 'float64'] }),
    F('compile', 'GPU acceleration', '', 'select', { options: ['cudagraphs', 'default', 'off'] }),
    F('legacy_source', 'SpinTorch legacy source', '', 'bool', { title: 'Reproduce the accumulating source of the original SpinTorch code (for comparison only)' }),
    F('max_frames_mb', 'Frame memory budget', 'MB')],
    derived: () => {
      const s = cfg.study;
      return `Simulated time ${fmt(s.steps * s.dt_ps / 1000)} ns, ${Math.floor(s.steps / s.save_every)} stored frames`;
    } },
  training: { title: 'Inverse design (training)', path: 'training', fields: [
    F('design', 'Design variable', '', 'select', { options: ['Ms', 'field'] }),
    F('B1_mT', 'Field design scale B₁', 'mT'),
    F('region_x0', 'Design region x from', 'cell', 'int'), F('region_x1', 'x to (excl.)', 'cell', 'int'),
    F('region_y0', 'Design region y from', 'cell', 'int'), F('region_y1', 'y to (excl.)', 'cell', 'int'),
    F('target', 'Target probe #', '', 'int'), F('epochs', 'Epochs', '', 'int'), F('lr', 'Learning rate (Adam)'),
    F('loss', 'Loss', '', 'select', { options: ['focus_log', 'focus_ratio'] }),
    F('clamp', 'Clamp ρ', '', 'bool'), F('rho_min', 'ρ min'), F('rho_max', 'ρ max'),
    F('simulate_after', 'Simulate trained design', '', 'bool')],
    help: 'Ms design: Mₛ = Mₛ·ρ inside the region (ρ starts at 1). Field design: B = B₀ + B₁·ρ along B₀ (ρ starts at 0). ' +
      'Loss = log10(Σ outputs / output[target] − 1), as in SpinTorch.' },
};

SECTIONS.init = { title: 'Initial state', path: 'init', fields: [
  F('kind', 'Initial magnetisation', '', 'select', { options: ['field', 'uniform', 'random', 'skyrmion', 'domain_wall', 'vortex'] }),
  F('mx', 'mₓ', '', 'num', { show: o => o.kind === 'uniform' }), F('my', 'm_y', '', 'num', { show: o => o.kind === 'uniform' }),
  F('mz', 'm_z', '', 'num', { show: o => o.kind === 'uniform' }),
  F('seed', 'Seed', '', 'int', { show: o => o.kind === 'random' }),
  F('cx_nm', 'Center x', 'nm', 'num', { show: o => o.kind === 'skyrmion' || o.kind === 'vortex' }),
  F('cy_nm', 'Center y', 'nm', 'num', { show: o => o.kind === 'skyrmion' || o.kind === 'vortex' }),
  F('radius_nm', 'Radius', 'nm', 'num', { show: o => o.kind === 'skyrmion' || o.kind === 'vortex' }),
  F('core', 'Core', '', 'select', { options: ['down', 'up'], show: o => o.kind === 'skyrmion' || o.kind === 'vortex' }),
  F('wall_x_nm', 'Wall position x', 'nm', 'num', { show: o => o.kind === 'domain_wall' }),
  F('wall_type', 'Wall type', '', 'select', { options: ['Neel', 'Bloch'], show: o => o.kind === 'domain_wall' }),
  F('chirality', 'Chirality (+1 = DMI-favoured)', '', 'int', { show: o => ['skyrmion', 'domain_wall', 'vortex'].includes(o.kind) })],
  help: '"field" aligns m with the bias field (or the easy axis). The state is relaxed (high damping) before the dynamics starts; ' +
    'domain walls go from up (left) to down (right).' };

const TORQUE_FIELDS = [
  F('kind', 'Type', '', 'select', { options: ['SOT', 'STT_CPP', 'ZhangLi'] }),
  F('J_Am2', 'Current density J', 'A/m²'),
  F('waveform', 'Waveform', '', 'select', { options: ['dc', 'pulse', 'pulse_train', 'sine'] }),
  F('t_start_ns', 'Start', 'ns'), F('width_ns', 'Pulse width', 'ns', 'num', { show: o => o.waveform.startsWith('pulse') }),
  F('period_ns', 'Period', 'ns', 'num', { show: o => o.waveform === 'pulse_train' }),
  F('n_pulses', 'Pulses', '', 'int', { show: o => o.waveform === 'pulse_train' }),
  F('f_GHz', 'Frequency', 'GHz', 'num', { show: o => o.waveform === 'sine' }), F('rise_ns', 'Rise / fall', 'ns'),
  F('theta_SH', 'Spin Hall angle θ_SH', '', 'num', { show: o => o.kind === 'SOT' }),
  F('P', 'Spin polarisation P', '', 'num', { show: o => o.kind !== 'SOT' }),
  F('xi_FL', 'Field-like / damping-like', '', 'num', { show: o => o.kind !== 'ZhangLi' }),
  F('beta', 'Non-adiabaticity β', '', 'num', { show: o => o.kind === 'ZhangLi' }),
  F('direction', 'Current direction', '', 'select', { options: ['x', 'y'], show: o => o.kind === 'ZhangLi' }),
  F('px', 'Polarisation pₓ', '', 'num', { show: o => o.kind !== 'ZhangLi' }), F('py', 'Polarisation p_y', '', 'num', { show: o => o.kind !== 'ZhangLi' }),
  F('pz', 'Polarisation p_z', '', 'num', { show: o => o.kind !== 'ZhangLi' }),
  F('x0', 'Region x from', 'cell', 'int'), F('x1', 'x to (excl.)', 'cell', 'int'), F('y0', 'Region y from', 'cell', 'int'), F('y1', 'y to (excl.)', 'cell', 'int')];

const REGION_FIELDS = [F('shape', 'Shape', '', 'select', { options: ['rect', 'disk'] }),
  F('x0', 'x from', 'cell', 'int', { show: o => o.shape === 'rect' }), F('x1', 'x to (excl.)', 'cell', 'int', { show: o => o.shape === 'rect' }),
  F('y0', 'y from', 'cell', 'int', { show: o => o.shape === 'rect' }), F('y1', 'y to (excl.)', 'cell', 'int', { show: o => o.shape === 'rect' }),
  F('cx', 'Center x', 'cell', 'int', { show: o => o.shape === 'disk' }), F('cy', 'Center y', 'cell', 'int', { show: o => o.shape === 'disk' }),
  F('r', 'Radius', 'cells', 'num', { show: o => o.shape === 'disk' }), F('Ms_kAm', 'Mₛ inside (0 = hole)', 'kA/m', 'numopt'),
  F('Ku_MJm3', 'Kᵤ inside (empty = material)', 'MJ/m³', 'numopt'), F('D_mJm2', 'D inside (empty = material)', 'mJ/m²', 'numopt'),
  F('alpha', 'α inside (empty = material)', '', 'numopt')];
const SOURCE_FIELDS = [F('x0', 'Start x', 'cell', 'int'), F('y0', 'Start y', 'cell', 'int'),
  F('x1', 'End x', 'cell', 'int'), F('y1', 'End y', 'cell', 'int'),
  F('component', 'Field component', '', 'select', { options: ['x', 'y', 'z'] }),
  F('waveform', 'Waveform', '', 'select', { options: ['sine', 'gaussian', 'sinc'] }),
  F('amp_mT', 'Amplitude', 'mT'),
  F('f_GHz', 'Frequency', 'GHz', 'num', { label2: o => o.waveform === 'sinc' ? 'Max. frequency' : null }),
  F('phase_deg', 'Phase', 'deg', 'num', { show: o => o.waveform !== 'sinc' }),
  F('ramp_ns', 'Smooth turn-on', 'ns', 'num', { show: o => o.waveform === 'sine' }),
  F('t0_ns', 'Pulse center', 'ns', 'num', { show: o => o.waveform !== 'sine' }),
  F('width_ns', 'Pulse width σ', 'ns', 'num', { show: o => o.waveform === 'gaussian' })];

const TREE = [
  { head: 'Global Definitions' },
  { id: 'grid', ic: '▦', label: 'Grid & Mesh' },
  { id: 'material', ic: '◆', label: 'Material' },
  { head: 'Physics (LLG)' },
  { id: 'field', ic: '➝', label: 'External Field' },
  { id: 'boundary', ic: '▢', label: 'Absorbing Boundary' },
  { id: 'init', ic: '↑', label: 'Initial State' },
  { id: 'regions', ic: '◩', label: 'Geometry Regions', badge: () => cfg.regions.length },
  { id: 'sources', ic: '≋', label: 'RF Field Sources', badge: () => cfg.sources.length },
  { id: 'torques', ic: '⏚', label: 'Currents & Torques', badge: () => (cfg.torques || []).length },
  { id: 'probes', ic: '◎', label: 'Probes', badge: () => cfg.probes.length },
  { head: 'Study' },
  { id: 'study', ic: '⏱', label: 'Time Dependent' },
  { id: 'training', ic: '✦', label: 'Inverse Design' },
  { head: 'Results' },
  { id: 'res:field', ic: '▤', label: 'Field vs. time', tab: 'field' },
  { id: 'res:intensity', ic: '▥', label: 'Intensity', tab: 'intensity' },
  { id: 'res:freq', ic: '∿', label: 'Frequency domain', tab: 'freq' },
  { id: 'res:kspace', ic: '✳', label: 'k-space', tab: 'kspace' },
  { id: 'res:probes', ic: '◎', label: 'Probe signals', tab: 'probes' },
  { id: 'res:table', ic: '⤴', label: 'Table (m, Q, energy)', tab: 'table' },
  { id: 'res:dispersion', ic: '⌒', label: 'Dispersion', tab: 'dispersion' },
  { id: 'res:training', ic: '✦', label: 'Training', tab: 'training' },
];
const TABS = [['geometry', 'Geometry'], ['field', 'Field'], ['intensity', 'Intensity'], ['freq', 'Frequency domain'],
  ['kspace', 'k-space'], ['probes', 'Probes'], ['table', 'Table'], ['dispersion', 'Dispersion'], ['training', 'Training']];

// ------------------------------------------------------------------ tree & settings
function curTree() { return mod === 'macro' ? TREE_M : TREE; }
function renderTree() {
  const ul = $('#treeList'); ul.innerHTML = '';
  for (const t of curTree()) {
    if (t.head) { ul.append(h('li', { class: 'head' }, t.head)); continue; }
    const li = h('li', { class: node === t.id ? 'sel' : '', style: '--lvl:1', onclick: () => selectNode(t.id) },
      h('span', { class: 'ic' }, t.ic), t.label, t.badge ? h('span', { class: 'badge' }, String(t.badge())) : null);
    ul.append(li);
  }
}
function selectNode(id) {
  node = id; renderTree(); renderSettings();
  const t = curTree().find(x => x.id === id);
  if (t && t.tab) selectTab(t.tab);
  else if (mod === 'macro') { if (['device', 'stack', 'mfield'].includes(id) && !running) selectTab('mdevice'); }
  else if (['regions', 'sources', 'probes', 'boundary', 'grid', 'material', 'field'].includes(id) && tab !== 'geometry' && !running) selectTab('geometry');
  else if (id === 'training' && train.loss.length) selectTab('training');
}
function changed(redrawSettings = false) {
  renderTree();
  if (redrawSettings) renderSettings();
  debounce(mod === 'macro' ? mcheck : check, 350);
}
function inputFor(obj, f) {
  const set = v => { obj[f.key] = v; changed(['shape', 'waveform', 'design', 'kind', 'torque', 'demag', 'problem', 'mode'].includes(f.key)); };
  if (f.type === 'bool') return h('input', { type: 'checkbox', checked: !!obj[f.key], onchange: e => set(e.target.checked) });
  if (f.type === 'select') {
    const s = h('select', { onchange: e => set(e.target.value) }, ...f.options.map(o => h('option', { value: o, selected: obj[f.key] === o }, o)));
    return s;
  }
  if (f.type === 'text') return h('input', { type: 'text', value: obj[f.key] ?? '', onchange: e => set(e.target.value) });
  if (f.type === 'numopt') return h('input', { type: 'number', step: 'any', value: obj[f.key] ?? '', placeholder: 'material',
    onchange: e => { const t = e.target.value.trim(); const v = t === '' ? null : parseFloat(t); if (v !== null && Number.isNaN(v)) { e.target.value = obj[f.key] ?? ''; return; } set(v); } });
  return h('input', {
    type: 'number', step: 'any', value: obj[f.key],
    onchange: e => {
      const v = f.type === 'int' ? parseInt(e.target.value, 10) : parseFloat(e.target.value);
      if (Number.isNaN(v)) { e.target.value = obj[f.key]; return; }
      e.target.value = v; set(v);
    },
  });
}
function fieldRow(obj, f) {
  if (f.show && !f.show(obj)) return null;
  const label = (f.label2 && f.label2(obj)) || f.label;
  return h('div', { class: 'field', title: f.title || '' }, h('label', {}, label), inputFor(obj, f), h('span', { class: 'unit' }, f.unit));
}
function section(title, body, tools = null) {
  return h('div', { class: 'section' }, h('div', { class: 'sec-h' }, title, tools ? h('span', { class: 'tools' }, tools) : null), h('div', { class: 'sec-b' }, body));
}
function renderSettings() {
  const body = $('#settingsBody'); body.innerHTML = '';
  const title = $('#settingsTitle');
  if (mod === 'macro') { if (mcfg) macroRenderSettings(body, title); return; }
  if (!cfg) return;
  const S = SECTIONS[node];
  if (S) {
    title.textContent = S.title;
    const obj = cfg[S.path];
    const rows = S.fields.map(f => fieldRow(obj, f));
    if (node === 'material') {
      rows.unshift(h('div', { class: 'field' }, h('label', {}, 'Library'),
        h('select', { onchange: e => { const m = MATERIALS[e.target.value]; if (m) { Object.assign(cfg.material, m, { name: e.target.value }); changed(true); } } },
          h('option', { value: '' }, '— choose —'), ...Object.keys(MATERIALS).map(k => h('option', { value: k }, k))), h('span')));
    }
    body.append(section(S.title, [S.help ? h('div', { class: 'help' }, S.help) : null, ...rows,
      S.derived ? h('div', { class: 'derived' }, S.derived()) : null]));
    if (node === 'training') body.append(h('div', { class: 'help' }, `Probes are numbered from 0. Currently ${cfg.probes.length} probes defined.`));
    return;
  }
  if (node === 'regions') {
    title.textContent = 'Geometry regions';
    body.append(h('div', { class: 'help' }, 'The film has the material parameters everywhere. Each region overrides Mₛ, Kᵤ, D or α inside its area (empty = keep the material value), applied in order. Mₛ = 0 makes a hole. Coordinates are cell indices.'));
    cfg.regions.forEach((r, i) => body.append(listItem(cfg.regions, i, r, REGION_FIELDS)));
    body.append(h('button', { class: 'sbtn', onclick: () => {
      cfg.regions.push({ name: `region ${cfg.regions.length + 1}`, shape: 'rect', x0: 0, y0: 0, x1: Math.round(cfg.grid.nx / 4), y1: Math.round(cfg.grid.ny / 4), cx: Math.round(cfg.grid.nx / 2), cy: Math.round(cfg.grid.ny / 2), r: 10, Ms_kAm: 0, enabled: true });
      changed(true);
    } }, '+ Add region'));
    return;
  }
  if (node === 'sources') {
    title.textContent = 'Sources';
    body.append(h('div', { class: 'help' }, 'Line antennas adding an RF field to B along the cells of the line. Coordinates are cell indices.'));
    cfg.sources.forEach((s, i) => body.append(listItem(cfg.sources, i, s, SOURCE_FIELDS)));
    body.append(h('button', { class: 'sbtn', onclick: () => {
      cfg.sources.push({ name: `source ${cfg.sources.length + 1}`, x0: 10, y0: 0, x1: 10, y1: cfg.grid.ny - 1, component: 'z', waveform: 'sine', amp_mT: 1, f_GHz: 4, phase_deg: 0, ramp_ns: 0, t0_ns: 1, width_ns: 0.2, enabled: true });
      changed(true);
    } }, '+ Add source'));
    return;
  }
  if (node === 'probes') { title.textContent = 'Probes'; renderProbes(body); return; }
  if (node === 'torques') {
    title.textContent = 'Currents & torques';
    cfg.torques = cfg.torques || [];
    body.append(h('div', { class: 'help' }, 'SOT: damping-like spin-orbit torque from a heavy-metal current, B_DL = ħθ_SH J / (2e Mₛ t). ' +
      'STT_CPP: Slonczewski torque from a fixed layer p (MTJ). ZhangLi: in-plane current moving domain walls / skyrmions (u = μ_B P J / (e Mₛ)). ' +
      'Region coordinates are cell indices (defaults cover the whole film).'));
    cfg.torques.forEach((q, i) => body.append(listItem(cfg.torques, i, q, TORQUE_FIELDS)));
    body.append(h('button', { class: 'sbtn', onclick: () => {
      cfg.torques.push({ name: `current ${cfg.torques.length + 1}`, kind: 'SOT', enabled: true, J_Am2: 1e11, waveform: 'pulse', t_start_ns: 0.1,
        width_ns: 1, period_ns: 4, n_pulses: 3, f_GHz: 1, rise_ns: 0.05, x0: 0, y0: 0, x1: cfg.grid.nx, y1: cfg.grid.ny,
        theta_SH: 0.1, P: 0.5, xi_FL: 0, beta: 0.04, px: 0, py: 1, pz: 0, direction: 'x' });
      changed(true);
    } }, '+ Add current'));
    if (report && report.torques && report.torques.length) {
      body.append(h('div', { class: 'derived' }, report.torques.map(t => t.B_DL_mT !== undefined ? `${t.name}: B_DL = ${fmt(t.B_DL_mT)} mT`
        : `${t.name}: u = ${fmt(t.u_ms)} m/s` + (t.v_dw_ms ? `, DW velocity βu/α = ${fmt(t.v_dw_ms)} m/s` : '')).join(' · ')));
    }
    return;
  }
  const t = TREE.find(x => x.id === node);
  title.textContent = t ? t.label : 'Settings';
  body.append(h('div', { class: 'help' }, resultHelp(node)));
}
function resultHelp(id) {
  return {
    'res:field': 'Δm = m(t) − m₀ for each stored frame. Use the slider or ▶ to step through time. Space = play/pause, ←/→ = step.',
    'res:intensity': 'Time-integrated Σₜ Δm² (log scale), equivalent to SpinTorch\'s "integrated" plot.',
    'res:freq': 'Running Fourier transform at the source frequency over an integer number of periods at the end of the run: steady-state amplitude and phase (like a frequency-domain study).',
    'res:kspace': '|FFT| of the current frame. The dashed curve is the analytic iso-frequency contour (Kalinikos–Slavin) at the source frequency: the bright spots should lie on it.',
    'res:probes': 'Signals at the probes (sum over the disk of Δm·Mₛ) and their spectra.',
    'res:dispersion': 'Analytic dispersion of the lowest thickness mode for the current parameters, with the source frequencies.',
    'res:training': 'Loss and output distribution per epoch, and the current design.',
  }[id] || '';
}
function listItem(list, i, obj, fields) {
  const tools = [
    h('input', { type: 'checkbox', checked: obj.enabled, title: 'enabled', onchange: e => { obj.enabled = e.target.checked; changed(); } }),
    h('button', { class: 'sbtn', title: 'Duplicate', onclick: () => { list.splice(i + 1, 0, JSON.parse(JSON.stringify(obj))); changed(true); } }, '⧉'),
    h('button', { class: 'sbtn del', title: 'Delete', onclick: () => { list.splice(i, 1); changed(true); } }, '✕'),
  ];
  const name = h('input', { type: 'text', value: obj.name, style: 'border:1px solid transparent;background:transparent;font-weight:600;width:150px', onchange: e => { obj.name = e.target.value; changed(); } });
  return section(name, fields.map(f => fieldRow(obj, f)), tools);
}
function renderProbes(body) {
  const g = cfg.grid;
  const gen = { n: 19, x: g.nx - 15, y0: 0, y1: g.ny, r: 2 };
  const inp = (k) => h('label', {}, { n: 'Count', x: 'x', y0: 'y from', y1: 'y to', r: 'radius' }[k],
    h('input', { type: 'number', value: gen[k], step: 'any', onchange: e => { gen[k] = parseFloat(e.target.value); } }));
  body.append(section('Generate a probe column', [
    h('div', { class: 'help' }, 'Places N probes evenly between y from/to at column x (same layout as SpinTorch).'),
    h('div', { class: 'gen' }, inp('n'), inp('x'), inp('y0'), inp('y1'), inp('r')),
    h('button', { class: 'sbtn', onclick: () => {
      const out = [];
      for (let p = 0; p < gen.n; p++) out.push({ x: Math.round(gen.x), y: Math.floor(gen.y0 + (gen.y1 - gen.y0) * (p + 1) / (gen.n + 1)), r: gen.r });
      cfg.probes = out; changed(true);
    } }, 'Replace probes'), ' ',
    h('button', { class: 'sbtn', onclick: () => { cfg.probes = []; changed(true); } }, 'Clear all')]));
  const tb = h('tbody');
  cfg.probes.forEach((p, i) => {
    const num = (k) => h('input', { type: 'number', value: p[k], step: k === 'r' ? 'any' : '1', onchange: e => { p[k] = k === 'r' ? parseFloat(e.target.value) : parseInt(e.target.value, 10); changed(); } });
    tb.append(h('tr', { class: i === cfg.training.target ? 'target' : '' }, h('td', {}, String(i)), h('td', {}, num('x')), h('td', {}, num('y')), h('td', {}, num('r')),
      h('td', {}, h('button', { class: 'sbtn del', onclick: () => { cfg.probes.splice(i, 1); changed(true); } }, '✕'))));
  });
  body.append(section(`Probes (${cfg.probes.length})`, [
    h('div', { class: 'help' }, 'Disk probes; the training target is highlighted.'),
    h('table', { class: 'probes' }, h('thead', {}, h('tr', {}, h('th', {}, '#'), h('th', {}, 'x'), h('th', {}, 'y'), h('th', {}, 'r'), h('th'))), tb),
    h('button', { class: 'sbtn', onclick: () => { cfg.probes.push({ x: Math.round(g.nx / 2), y: Math.round(g.ny / 2), r: 2 }); changed(true); } }, '+ Add probe')]));
}

// ------------------------------------------------------------------ check / report
async function check() {
  if (mod === 'macro') return mcheck();
  if (!cfg) return;
  try {
    const r = await post('/api/check', cfg);
    report = r; geom = r.geometry;
    renderReport();
    if (SECTIONS[node] && SECTIONS[node].derived) { const d = $('.derived'); if (d) d.textContent = SECTIONS[node].derived(); }
    if (tab === 'geometry' || tab === 'dispersion') renderTab();
  } catch (e) {
    report = null;
    $('#report').innerHTML = '';
    $('#report').append(h('div', { class: 'e' }, '✕ ' + e.message));
  }
}
function renderReport() {
  const el = $('#report'); el.innerHTML = '';
  if (!report) return;
  for (const w of report.warnings) el.append(h('div', { class: 'w' }, '⚠ ' + w));
  for (const w of report.info) el.append(h('div', {}, 'ℹ ' + w));
  if (!report.warnings.length) el.append(h('div', { class: 'ok' }, '✓ Parameters look consistent'));
  const rows = [
    ['FMR frequency (k→0)', fmt(report.fmr_GHz) + ' GHz'],
    ['Exchange length', report.exchange_length_nm ? fmt(report.exchange_length_nm) + ' nm' : '–'],
    ['ω_max·Δt (RK4 limit ≈ 2.8)', fmt(report.omega_max_dt)],
    ['Max. stable Δt (est.)', fmt(report.dt_max_ps) + ' ps'],
  ];
  for (const s of report.sources) {
    rows.push([`λ at ${s.f_GHz} GHz (k ∥ x)`, s.lambda_nm ? fmt(s.lambda_nm) + ' nm' : '–']);
    if (s.cells_per_lambda) rows.push(['  cells per wavelength', fmt(s.cells_per_lambda)]);
    if (s.samples_per_period) rows.push(['  steps per period', fmt(s.samples_per_period)]);
  }
  rows.push(['Stored frames memory', fmt(report.frames_mb) + ' MB']);
  el.append(h('table', {}, ...rows.map(r => h('tr', {}, h('td', {}, r[0]), h('td', {}, r[1])))));
}

// ------------------------------------------------------------------ plotting helpers
function overlayShapes(g, opt = {}) {
  if (!g || !view.overlay) return [];
  const sh = []; const dx = g.dx, dy = g.dy;
  const lineCol = opt.dark ? 'rgba(255,255,255,0.8)' : 'rgba(0,0,0,0.65)';
  if (g.boundary > 0) {
    const w = g.boundary;
    sh.push({ type: 'rect', x0: (w - 0.5) * dx, x1: (g.nx - w - 0.5) * dx, y0: (w - 0.5) * dy, y1: (g.ny - w - 0.5) * dy,
      line: { color: lineCol, width: 1, dash: 'dot' } });
  }
  if (opt.design) {
    const [x0, y0, x1, y1] = g.design;
    sh.push({ type: 'rect', x0: (x0 - 0.5) * dx, x1: (x1 - 0.5) * dx, y0: (y0 - 0.5) * dy, y1: (y1 - 0.5) * dy,
      line: { color: '#7a3fc2', width: 1.5, dash: 'dash' } });
  }
  for (const s of g.sources) {
    if (!s.enabled) continue;
    sh.push({ type: 'line', x0: s.x0 * dx, x1: s.x1 * dx, y0: s.y0 * dy, y1: s.y1 * dy, line: { color: '#12a150', width: 3 } });
  }
  g.probes.forEach((p, i) => {
    const tgt = opt.target !== undefined && i === opt.target;
    sh.push({ type: 'circle', x0: (p.x - p.r) * dx, x1: (p.x + p.r) * dx, y0: (p.y - p.r) * dy, y1: (p.y + p.r) * dy,
      line: { color: tgt ? '#ff8c00' : lineCol, width: tgt ? 2.5 : 1 } });
  });
  return sh;
}
function axisLayout(title, extra = {}) {
  return Object.assign({
    margin: { l: 60, r: 20, t: 34, b: 46 }, title: { text: title, font: { size: 13 }, x: 0.02 },
    xaxis: { title: { text: 'x (µm)' }, constrain: 'domain', zeroline: false, showgrid: false, mirror: true, ticks: 'outside', showline: true },
    yaxis: { title: { text: 'y (µm)' }, scaleanchor: 'x', constrain: 'domain', zeroline: false, showgrid: false, mirror: true, ticks: 'outside', showline: true },
    paper_bgcolor: '#fff', plot_bgcolor: '#fff', uirevision: 'map', font: { family: 'Segoe UI, Arial', size: 12 },
  }, extra);
}
function heat(div, d, g, opt = {}) {
  const stride = opt.stride || 1;
  const trace = {
    type: 'heatmap', z: d.z, x0: 0, dx: g.dx * stride, y0: 0, dy: g.dy * stride, zsmooth: opt.smooth ? 'best' : false,
    colorscale: CMAPS[opt.cmap] || opt.cmap || 'Viridis', zmin: opt.zmin, zmax: opt.zmax, zauto: opt.zmin === undefined,
    colorbar: { title: { text: opt.cb || '', side: 'right' }, thickness: 14, outlinewidth: 0, exponentformat: 'power' },
    hovertemplate: `x = %{x:.3f} µm<br>y = %{y:.3f} µm<br>${opt.cb || 'value'} = %{z:.4g}<extra></extra>`,
  };
  const lay = axisLayout(opt.title || '', { shapes: overlayShapes(g, opt.overlay || {}) });
  lay.xaxis.range = [-0.5 * g.dx, (g.nx - 0.5) * g.dx];
  lay.yaxis.range = [-0.5 * g.dy, (g.ny - 0.5) * g.dy];
  if (opt.annotations) lay.annotations = opt.annotations;
  Plotly.react(div, [trace, ...(opt.extra || [])], lay, PCONF);
}
function showSub(on) {
  const sub = $('#plotSub');
  const was = !sub.classList.contains('hidden');
  sub.classList.toggle('hidden', !on);
  if (!on) sub.style.flex = '';
  if (was !== on) requestAnimationFrame(() => {
    for (const id of ['plotMain', 'plotSub']) { const d = document.getElementById(id); if (d && d.data && d.offsetParent !== null) Plotly.Plots.resize(d); }
  });
}
function emptyPlot(msg) {
  Plotly.purge('plotMain'); Plotly.purge('plotSub'); showSub(false);
  $('#plotMain').innerHTML = `<div class="empty"><div>${msg}</div></div>`;
}
function clearEmpty() { const e = $('#plotMain .empty'); if (e) e.remove(); }

// ------------------------------------------------------------------ toolbar controls
function sel(label, value, options, onchange) {
  return h('label', {}, label, h('select', { onchange: e => onchange(e.target.value) },
    ...options.map(o => Array.isArray(o) ? h('option', { value: o[0], selected: o[0] === value }, o[1]) : h('option', { value: o, selected: o === value }, o))));
}
const COMP_OPTS = [['x', 'Δmₓ'], ['y', 'Δm_y'], ['z', 'Δm_z'], ['abs', '|Δm|']];
const CMAP_OPTS = [['wave', 'Wave (blue–red)'], ['magma', 'Magma'], ['viridis', 'Viridis'], ['jet', 'Rainbow'], ['greys', 'Grey'], ['cividis', 'Cividis']];

// ------------------------------------------------------------------ tabs
function renderTabs() {
  const el = $('#tabs'); el.innerHTML = '';
  if (mod === 'macro') {
    for (const [id, label] of TABS_M) {
      el.append(h('div', { class: 'tab' + (id === tab ? ' sel' : '') + (macroTabEnabled(id) ? '' : ' dis'), onclick: () => selectTab(id) }, label));
    }
    return;
  }
  for (const [id, label] of TABS) {
    const needRes = ['field', 'intensity', 'freq', 'kspace', 'probes', 'table'].includes(id);
    const dis = (needRes && !meta && !(id === 'field' && view.live)) || (id === 'training' && !train.loss.length);
    el.append(h('div', { class: 'tab' + (id === tab ? ' sel' : '') + (dis ? ' dis' : ''), onclick: () => selectTab(id) }, label));
  }
}
function selectTab(id) {
  if (view.playing && id !== 'field') view.playing = false;
  tab = id; renderTabs(); renderTab();
}
async function renderTab() {
  const tb = $('#toolbar'); tb.innerHTML = '';
  clearEmpty();
  try {
    if (mod === 'macro') return drawMacroTab(tb);
    if (tab === 'geometry') return drawGeometry(tb);
    if (tab === 'dispersion') return drawDispersion(tb);
    if (tab === 'training') return drawTraining(tb);
    if (tab === 'field') return drawFieldTab(tb);
    if (!meta) return emptyPlot('No results yet.<br>Press <b>▶ Compute</b> to run a simulation.');
    if (tab === 'intensity') return drawIntensity(tb);
    if (tab === 'freq') return drawFreq(tb);
    if (tab === 'kspace') return drawKspace(tb);
    if (tab === 'probes') return drawProbes(tb);
    if (tab === 'table') return drawTable(tb);
  } catch (e) { addMsg({ level: 'error', text: 'Plot error: ' + e.message }); }
}

// geometry ---------------------------------------------------------------
function drawGeometry(tb) {
  tb.append(sel('Show', view.geomMap, [['Ms', 'Saturation magnetization'], ['B', 'Bias field |B|'], ['alpha', 'Damping (log₁₀ α)']], v => { view.geomMap = v; renderTab(); }));
  tb.append(h('label', {}, h('input', { type: 'checkbox', checked: view.overlay, onchange: e => { view.overlay = e.target.checked; renderTab(); } }), 'Overlays'));
  tb.append(h('span', { class: 'muted' }, '  green = sources · circles = probes · dotted = absorbing layer · purple dashed = design region'));
  showSub(false);
  if (!geom) return emptyPlot('Press ✓ Check to preview the geometry.');
  const m = view.geomMap;
  const d = decode(geom[m]);
  const cb = { Ms: 'Mₛ (kA/m)', B: '|B| (mT)', alpha: 'log₁₀ α' }[m];
  heat('plotMain', d, geom, { cmap: m === 'alpha' ? 'magma' : 'summer', cb, title: `Geometry — ${cfg.name}`,
    overlay: { design: true, target: cfg.training.target } });
}

// dispersion -------------------------------------------------------------
function drawDispersion(tb) {
  showSub(false);
  if (!report) return emptyPlot('Press ✓ Check first.');
  const d = report.dispersion;
  const traces = d.curves.map((c, i) => ({ x: d.k, y: c.f, name: c.name, mode: 'lines', line: { width: 2.5, color: ['#1b6ac9', '#c2410c'][i] } }));
  const kmax = d.k[d.k.length - 1];
  for (const f of d.sources) traces.push({ x: [0, kmax], y: [f, f], name: `source ${f} GHz`, mode: 'lines', line: { dash: 'dash', color: '#12a150', width: 1.5 } });
  for (const s of report.sources) if (s.k_rad_um) traces.push({ x: [s.k_rad_um], y: [s.f_GHz], mode: 'markers', name: `λ = ${fmt(s.lambda_nm)} nm`, marker: { size: 10, color: '#12a150', symbol: 'circle-open', line: { width: 2 } } });
  Plotly.react('plotMain', traces, {
    margin: { l: 60, r: 20, t: 34, b: 50 }, title: { text: 'Spin-wave dispersion (Kalinikos–Slavin, lowest mode)', font: { size: 13 }, x: 0.02 },
    xaxis: { title: { text: 'k (rad/µm)' }, range: [0, kmax], showline: true, mirror: true, ticks: 'outside' },
    yaxis: { title: { text: 'f (GHz)' }, showline: true, mirror: true, ticks: 'outside' },
    legend: { x: 0.99, xanchor: 'right', y: 0.02, bgcolor: 'rgba(255,255,255,0.8)' }, paper_bgcolor: '#fff', plot_bgcolor: '#fff',
  }, PCONF);
}

// field (time) ------------------------------------------------------------
function frameKey(i, c) { return `${c}:${i}`; }
async function getFrame(i, comp) {
  const k = frameKey(i, comp);
  if (cache.has(k)) return cache.get(k);
  const r = await api(`/api/result/frames?start=${i}&count=16&comp=${comp}`);
  r.idx.forEach((j, n) => cache.set(frameKey(j, comp), decode(r.frames[n])));
  while (cache.size > 600) cache.delete(cache.keys().next().value);
  return cache.get(k);
}
async function getRange(comp) {
  if (ranges[comp] === undefined) ranges[comp] = (await api(`/api/result/range?comp=${comp}`)).lim;
  return ranges[comp];
}
function drawFieldTab(tb) {
  const c = view;
  tb.append(sel('Component', c.comp, ['x', 'y', 'z', 'abs'].map(k => [k, COMP_LABEL[k]]), v => { c.comp = v; drawField(); }));
  tb.append(sel('Colormap', c.cmap, CMAP_OPTS, v => { c.cmap = v; drawField(); }));
  tb.append(sel('Scale', c.scale, [['global', 'Fixed (all frames)'], ['frame', 'Per frame']], v => { c.scale = v; drawField(); }));
  const n = meta ? meta.frames : 0;
  tb.append(h('button', { class: 'tbtn', title: 'First', onclick: () => gotoFrame(0) }, '⏮'));
  tb.append(h('button', { class: 'tbtn', title: 'Previous (←)', onclick: () => gotoFrame(c.frame - 1) }, '◀'));
  tb.append(h('button', { class: 'tbtn' + (c.playing ? ' on' : ''), id: 'playBtn', title: 'Play / pause (space)', onclick: togglePlay }, c.playing ? '❚❚' : '▶'));
  tb.append(h('button', { class: 'tbtn', title: 'Next (→)', onclick: () => gotoFrame(c.frame + 1) }, '▶|'));
  tb.append(h('button', { class: 'tbtn', title: 'Last', onclick: () => gotoFrame(n - 1) }, '⏭'));
  tb.append(h('input', { type: 'range', id: 'frameSlider', min: 0, max: Math.max(0, n - 1), value: c.frame, disabled: !meta || running,
    oninput: e => gotoFrame(parseInt(e.target.value, 10)) }));
  tb.append(h('label', {}, 'fps', h('input', { type: 'number', value: c.fps, min: 1, max: 60, style: 'width:50px', onchange: e => { c.fps = Math.max(1, parseInt(e.target.value, 10) || 15); } })));
  tb.append(h('label', {}, h('input', { type: 'checkbox', checked: c.linecut, onchange: e => { c.linecut = e.target.checked; drawField(); } }), 'Line cut'));
  tb.append(h('span', { class: 'time', id: 'timeLabel' }, ''));
  tb.append(h('span', { id: 'liveBadge' }));
  drawField();
}
function gotoFrame(i) {
  if (!meta) return;
  view.frame = Math.max(0, Math.min(meta.frames - 1, i));
  const s = $('#frameSlider'); if (s) s.value = view.frame;
  if (tab === 'field') drawField();
  if (tab === 'kspace') drawKspacePlot();
}
function togglePlay() {
  if (!meta || running) return;
  view.playing = !view.playing;
  const b = $('#playBtn'); if (b) { b.textContent = view.playing ? '❚❚' : '▶'; b.classList.toggle('on', view.playing); }
  if (view.playing) playLoop();
}
async function playLoop() {
  while (view.playing && tab === 'field' && meta) {
    const t0 = performance.now();
    view.frame = (view.frame + 1) % meta.frames;
    const s = $('#frameSlider'); if (s) s.value = view.frame;
    await drawField();
    const wait = 1000 / view.fps - (performance.now() - t0);
    await new Promise(r => setTimeout(r, Math.max(0, wait)));
  }
}
function absOf(ds) {
  const [a, b, c] = ds; const z = a.z.map((row, i) => row.map((v, j) => Math.hypot(v, b.z[i][j], c.z[i][j])));
  return { z };
}
const COMP_LABEL_DM = { x: 'Δmₓ', y: 'Δm_y', z: 'Δm_z', abs: '|Δm|' };
const COMP_LABEL_M = { x: 'mₓ', y: 'm_y', z: 'm_z', abs: '|m|' };
let COMP_LABEL = COMP_LABEL_DM;
let drawing = false;
async function drawField() {
  if (tab !== 'field') return;
  const c = view;
  const g = (meta && !running) ? meta.geometry : (geom || (meta && meta.geometry));
  const badge = $('#liveBadge');
  if (running && c.live) {
    // live preview from the running job
    const ds = c.live.map(e => decode(e));
    const d = c.comp === 'abs' ? absOf(ds) : ds['xyz'.indexOf(c.comp)];
    let lim = 0; for (const row of d.z) for (const v of row) lim = Math.max(lim, Math.abs(v));
    lim = lim || 1;
    heat('plotMain', d, g, { stride: c.liveStride, cmap: c.comp === 'abs' && c.cmap === 'wave' ? 'magma' : c.cmap,
      zmin: c.comp === 'abs' ? 0 : -lim, zmax: lim, cb: COMP_LABEL[c.comp], title: `Live — t = ${fmt(c.liveT, 4)} ns`, overlay: {} });
    if (badge) { badge.className = 'live'; badge.textContent = 'LIVE'; }
    const tl = $('#timeLabel'); if (tl) tl.textContent = `t = ${c.liveT.toFixed(3)} ns`;
    showSub(false);
    return;
  }
  if (badge) { badge.className = ''; badge.textContent = ''; }
  if (running) return emptyPlot('Running… relaxing the ground state and preparing GPU kernels.<br>The field appears here live as soon as time stepping starts.');
  if (!meta) return emptyPlot('No results yet.<br>Press <b>▶ Compute</b> to run a simulation. The field is shown live while it runs.');
  if (drawing) return; drawing = true;
  try {
    const d = await getFrame(c.frame, c.comp);
    let zmin, zmax;
    if (meta.record === 'm' && c.comp !== 'abs') { zmin = -1; zmax = 1; } else if (c.scale === 'global') { const lim = await getRange(c.comp); zmax = lim; zmin = c.comp === 'abs' ? 0 : -lim; }
    else { let lim = 0; for (const row of d.z) for (const v of row) lim = Math.max(lim, Math.abs(v)); lim = lim || 1; zmax = lim; zmin = c.comp === 'abs' ? 0 : -lim; }
    const t = meta.frame_t_ns[c.frame];
    heat('plotMain', d, meta.geometry, { cmap: c.comp === 'abs' && c.cmap === 'wave' ? 'magma' : c.cmap, zmin, zmax, cb: COMP_LABEL[c.comp],
      title: `${COMP_LABEL[c.comp]} — frame ${c.frame + 1}/${meta.frames}, t = ${fmt(t, 5)} ns`, overlay: {} });
    const tl = $('#timeLabel'); if (tl) tl.textContent = `t = ${t.toFixed(3)} ns  (step ${meta.frame_t_ns.length ? Math.round(t / meta.dt_ns) : 0})`;
    if (c.linecut) {
      showSub(true);
      const g2 = meta.geometry; const row = Math.floor(d.z.length / 2);
      const xs = d.z[row].map((_, i) => i * g2.dx);
      Plotly.react('plotSub', [{ x: xs, y: d.z[row], mode: 'lines', line: { color: '#1b6ac9', width: 2 }, name: 'cut' }], {
        margin: { l: 60, r: 20, t: 26, b: 40 }, title: { text: `Line cut at y = ${fmt(row * g2.dy)} µm`, font: { size: 12 }, x: 0.02 },
        xaxis: { title: { text: 'x (µm)' }, showline: true, mirror: true }, yaxis: { title: { text: COMP_LABEL[c.comp] }, range: [zmin, zmax], showline: true, mirror: true },
        paper_bgcolor: '#fff', plot_bgcolor: '#fff', uirevision: 'cut',
      }, PCONF);
    } else showSub(false);
    // prefetch ahead while playing
    if (view.playing) { const nxt = c.frame + 16; if (nxt < meta.frames && !cache.has(frameKey(nxt, c.comp))) getFrame(nxt, c.comp); }
  } finally { drawing = false; }
}

// intensity ----------------------------------------------------------------
async function drawIntensity(tb) {
  tb.append(sel('Component', view.icomp, [['x', 'Σ Δmₓ²'], ['y', 'Σ Δm_y²'], ['z', 'Σ Δm_z²'], ['abs', 'Σ |Δm|²']], v => { view.icomp = v; renderTab(); }));
  tb.append(sel('Dynamic range', String(view.idec), [['1', '1 decade'], ['2', '2 decades'], ['3', '3 decades'], ['4', '4 decades'], ['6', '6 decades']], v => { view.idec = +v; renderTab(); }));
  tb.append(sel('Colormap', view.cmap === 'wave' ? 'viridis' : view.cmap, CMAP_OPTS.slice(1), v => { view.cmap = v; renderTab(); }));
  showSub(false);
  const r = await api(`/api/result/map/intensity?comp=${view.icomp}`);
  const d = decode(r.map);
  const mx = d.max || 1; const lo = Math.log10(mx) - view.idec;
  d.z = d.z.map(row => row.map(v => Math.max(Math.log10(Math.max(v, 1e-300)), lo)));
  heat('plotMain', d, meta.geometry, { cmap: view.cmap === 'wave' ? 'viridis' : view.cmap, zmin: lo, zmax: Math.log10(mx), cb: 'log₁₀ Σ Δm²',
    title: 'Time-integrated intensity', overlay: { dark: true, target: meta.config.training.target } });
}

// frequency domain ---------------------------------------------------------
async function drawFreq(tb) {
  if (!meta.has_dft) { showSub(false); return emptyPlot('No frequency-domain data (needs a source with f > 0 and enough steps).'); }
  tb.append(sel('Quantity', view.dkind, [['dft_amp', 'Amplitude |Δm(f)|'], ['dft_phase', 'Phase arg Δm(f)'], ['dft_real', 'Re Δm(f) (snapshot)']], v => { view.dkind = v; renderTab(); }));
  tb.append(sel('Component', view.dcomp, COMP_OPTS.slice(0, 3), v => { view.dcomp = v; renderTab(); }));
  showSub(false);
  const r = await api(`/api/result/map/${view.dkind}?comp=${view.dcomp}`);
  const d = decode(r.map);
  const opts = { title: `${{ dft_amp: 'Amplitude', dft_phase: 'Phase', dft_real: 'Real part' }[view.dkind]} of Δm_${view.dcomp} at f = ${fmt(r.f_GHz)} GHz`, overlay: {} };
  if (view.dkind === 'dft_amp') Object.assign(opts, { cmap: 'magma', zmin: 0, zmax: d.max, cb: '|Δm|', overlay: { dark: true } });
  else if (view.dkind === 'dft_phase') Object.assign(opts, { cmap: 'phase', zmin: -Math.PI, zmax: Math.PI, cb: 'phase (rad)' });
  else { const lim = Math.max(Math.abs(d.min), Math.abs(d.max)) || 1; Object.assign(opts, { cmap: 'wave', zmin: -lim, zmax: lim, cb: 'Re Δm' }); }
  heat('plotMain', d, meta.geometry, opts);
}

// k-space ------------------------------------------------------------------
function drawKspace(tb) {
  tb.append(sel('Component', view.kcomp, COMP_OPTS.slice(0, 3), v => { view.kcomp = v; drawKspacePlot(); }));
  tb.append(h('input', { type: 'range', id: 'frameSlider', min: 0, max: meta.frames - 1, value: view.frame, oninput: e => gotoFrame(parseInt(e.target.value, 10)) }));
  tb.append(h('span', { class: 'time', id: 'timeLabel' }));
  showSub(false);
  drawKspacePlot();
}
let kbusy = false;
async function drawKspacePlot() {
  if (tab !== 'kspace' || kbusy) return; kbusy = true;
  try {
    const r = await api(`/api/result/kspace/${view.frame}?comp=${view.kcomp}`);
    const d = decode(r.map);
    const traces = [{ type: 'heatmap', z: d.z, x0: r.kx0, dx: r.dkx, y0: r.ky0, dy: r.dky, colorscale: CMAPS.magma,
      colorbar: { title: { text: 'log₁₀|FFT|', side: 'right' }, thickness: 14 }, hovertemplate: 'kx=%{x:.2f}<br>ky=%{y:.2f} rad/µm<extra></extra>' }];
    if (r.iso) traces.push({ x: r.iso.kx, y: r.iso.ky, mode: 'lines', line: { color: '#5fe0ff', dash: 'dash', width: 2 }, name: `analytic @ ${fmt(r.iso.f_GHz)} GHz`, connectgaps: false });
    const lay = axisLayout(`k-space of Δm_${view.kcomp}, t = ${fmt(meta.frame_t_ns[view.frame], 4)} ns`, { showlegend: true,
      legend: { x: 0.01, y: 0.99, bgcolor: 'rgba(255,255,255,0.8)' } });
    lay.xaxis.title.text = 'kₓ (rad/µm)'; lay.yaxis.title.text = 'k_y (rad/µm)'; lay.uirevision = 'k';
    Plotly.react('plotMain', traces, lay, PCONF);
    const tl = $('#timeLabel'); if (tl) tl.textContent = `t = ${meta.frame_t_ns[view.frame].toFixed(3)} ns`;
  } finally { kbusy = false; }
}

// probes -------------------------------------------------------------------
async function drawProbes(tb) {
  tb.append(sel('View', view.pview, [['signals', 'Time signals'], ['spectrum', 'Spectra'], ['bars', 'Output intensity']], v => { view.pview = v; renderTab(); }));
  tb.append(sel('Component', view.pcomp, [['x', 'x'], ['y', 'y'], ['z', 'z'], ['abs', '|·|']], v => { view.pcomp = v; renderTab(); }));
  tb.append(h('label', {}, 'Probes', h('input', { type: 'text', value: view.psel, style: 'width:110px', title: 'all, or a list like 0,9,18 or 5-12',
    onchange: e => { view.psel = e.target.value.trim() || 'all'; renderTab(); } })));
  if (view.pview === 'spectrum') tb.append(h('label', {}, h('input', { type: 'checkbox', checked: view.plog, onchange: e => { view.plog = e.target.checked; renderTab(); } }), 'log scale'));
  showSub(false);
  const r = await api(`/api/result/probes?comp=${view.pcomp}`);
  const n = r.signals.length;
  if (!n || !meta.config.probes.length) return emptyPlot('No probes defined.');
  const pick = parseSel(view.psel, n);
  const target = meta.config.training.target;
  const base = { margin: { l: 70, r: 20, t: 34, b: 46 }, paper_bgcolor: '#fff', plot_bgcolor: '#fff', legend: { font: { size: 11 } } };
  if (view.pview === 'bars') {
    const cols = r.intensity.map((_, i) => i === target ? '#ff8c00' : '#3b4b5e');
    Plotly.react('plotMain', [{ type: 'bar', x: r.intensity.map((_, i) => i), y: r.intensity, marker: { color: cols } }], Object.assign(base, {
      title: { text: `Time-integrated probe intensity (Δm_${meta.config.study.probe_component}), target = #${target}`, font: { size: 13 }, x: 0.02 },
      xaxis: { title: { text: 'probe #' }, dtick: 1 }, yaxis: { title: { text: 'Σₜ (Σ Δm·Mₛ)²' }, exponentformat: 'power' } }), PCONF);
    return;
  }
  const xs = view.pview === 'signals' ? b64f32(r.t) : b64f32(r.f);
  const traces = pick.map(i => ({ x: xs, y: b64f32(view.pview === 'signals' ? r.signals[i] : r.spectra[i]), mode: 'lines', name: `#${i}`,
    line: { width: i === target ? 2.5 : 1.2, color: i === target ? '#ff8c00' : undefined } }));
  if (view.pview === 'signals') {
    Plotly.react('plotMain', traces, Object.assign(base, { title: { text: 'Probe signals', font: { size: 13 }, x: 0.02 },
      xaxis: { title: { text: 't (ns)' }, showline: true, mirror: true }, yaxis: { title: { text: r.unit }, exponentformat: 'power', showline: true, mirror: true } }), PCONF);
  } else {
    const fs = (meta.config.sources || []).filter(s => s.enabled && s.waveform !== 'sinc').map(s => s.f_GHz);
    Plotly.react('plotMain', traces, Object.assign(base, { title: { text: 'Probe spectra (Hann window)', font: { size: 13 }, x: 0.02 },
      shapes: fs.map(f => ({ type: 'line', x0: f, x1: f, yref: 'paper', y0: 0, y1: 1, line: { color: '#12a150', dash: 'dash' } })),
      xaxis: { title: { text: 'f (GHz)' }, showline: true, mirror: true }, yaxis: { title: { text: 'amplitude' }, type: view.plog ? 'log' : 'linear', exponentformat: 'power', showline: true, mirror: true } }), PCONF);
  }
}
function parseSel(s, n) {
  if (!s || s === 'all') return [...Array(n).keys()];
  const out = new Set();
  for (const part of s.split(',')) {
    const m = part.trim().match(/^(\d+)(?:-(\d+))?$/);
    if (!m) continue;
    const a = +m[1], b = m[2] ? +m[2] : a;
    for (let i = a; i <= b && i < n; i++) out.add(i);
  }
  return [...out];
}

// table ---------------------------------------------------------------------
let tableData = null;
async function drawTable(tb) {
  if (!meta.has_table) { showSub(false); return emptyPlot('No table data.'); }
  if (!tableData || tableData._meta !== meta) { tableData = await api('/api/result/table'); tableData._meta = meta; }
  const d = tableData;
  view.tcols = view.tcols || ['mx', 'my', 'mz'];
  const groups = [['m components', ['mx', 'my', 'mz']], ['Topological charge', ['Q']], ['Energies', ['E_total', 'E_exch', 'E_demag', 'E_DMI', 'E_anis', 'E_Zeeman']],
    ['Texture position', ['core_x', 'core_y', 'core_r', 'x_DW']]];
  const preset = groups.find(g => g[1].join() === view.tcols.join());
  tb.append(sel('Quantities', preset ? preset[0] : 'custom', groups.map(g => [g[0], g[0]]).concat(preset ? [] : [['custom', 'custom']]),
    v => { const g = groups.find(x => x[0] === v); if (g) { view.tcols = g[1]; renderTab(); } }));
  tb.append(h('span', { class: 'muted' }, 'core_* = region with m_z opposite to the average (skyrmion core); x_DW = (1 + ⟨m_z⟩)/2 · Lₓ (wall between up on the left and down on the right).'));
  showSub(false);
  const traces = view.tcols.map(c => {
    const i = d.cols.indexOf(c);
    return { x: d.t_ns, y: d.data[i], mode: 'lines', name: c + (d.units[i] ? ` (${d.units[i]})` : ''), line: { width: 2 } };
  });
  const unit = d.units[d.cols.indexOf(view.tcols[0])];
  Plotly.react('plotMain', traces, {
    margin: { l: 70, r: 20, t: 34, b: 46 }, title: { text: 'Table (sampled every chunk of ' + meta.extra.chunk + ' steps)', font: { size: 13 }, x: 0.02 },
    xaxis: { title: { text: 't (ns)' }, showline: true, mirror: true }, yaxis: { title: { text: unit || 'value' }, exponentformat: 'power', showline: true, mirror: true },
    paper_bgcolor: '#fff', plot_bgcolor: '#fff', legend: { x: 0.99, xanchor: 'right', y: 0.99, bgcolor: 'rgba(255,255,255,0.8)' },
  }, PCONF);
}

// training -----------------------------------------------------------------
function drawTraining(tb) {
  if (!train.loss.length) { showSub(false); return emptyPlot('No training yet.<br>Set up <b>Inverse Design</b> and press <b>✦ Train</b>.'); }
  const last = train.outputs[train.outputs.length - 1];
  const tg = train.target;
  const share = last ? 100 * last[tg] / last.reduce((a, b) => a + b, 0) : 0;
  tb.append(h('span', {}, `Epoch ${train.loss.length}: loss = ${fmt(train.loss[train.loss.length - 1], 5)}, target #${tg} receives ${share.toFixed(1)}% of the output`));
  const ep = train.loss.map((_, i) => i + 1);
  Plotly.react('plotMain', [
    { x: ep, y: train.loss, mode: 'lines+markers', name: 'loss', line: { color: '#7a3fc2', width: 2 }, marker: { size: 6 } },
    { type: 'bar', x: last.map((_, i) => i), y: last, xaxis: 'x2', yaxis: 'y2', name: 'outputs', marker: { color: last.map((_, i) => i === tg ? '#ff8c00' : '#3b4b5e') } },
  ], {
    margin: { l: 60, r: 20, t: 34, b: 46 }, showlegend: false, paper_bgcolor: '#fff', plot_bgcolor: '#fff',
    grid: { rows: 1, columns: 2, pattern: 'independent' },
    xaxis: { title: { text: 'epoch' }, domain: [0, 0.44], dtick: train.loss.length > 20 ? undefined : 1, showline: true, mirror: true },
    yaxis: { title: { text: 'loss' }, showline: true, mirror: true },
    xaxis2: { title: { text: 'probe #' }, domain: [0.56, 1], dtick: 1, anchor: 'y2' },
    yaxis2: { title: { text: 'output (last epoch)' }, anchor: 'x2', exponentformat: 'power' },
    annotations: [{ text: 'Loss', x: 0.22, xref: 'paper', y: 1.06, yref: 'paper', showarrow: false, font: { size: 13 } },
      { text: 'Output distribution', x: 0.78, xref: 'paper', y: 1.06, yref: 'paper', showarrow: false, font: { size: 13 } }],
  }, PCONF);
  if (train.design) {
    showSub(true);
    $('#plotSub').style.flex = '0 0 50%';
    const g = (meta && meta.geometry) || geom;
    const d = decode(train.design);
    heat('plotSub', d, g, { cmap: 'summer', cb: train.unit, title: `Design (${train.unit})`, overlay: { design: true, target: tg } });
  } else showSub(false);
}

// ------------------------------------------------------------------ messages / progress
function addMsg(m) {
  const box = $('#messages');
  const t = m.t || new Date().toLocaleTimeString('en-GB');
  box.append(h('div', { class: m.level || '' }, h('span', { class: 't' }, t), m.text));
  while (box.children.length > 500) box.removeChild(box.firstChild);
  box.scrollTop = box.scrollHeight;
}
let runStart = 0;
function setProgress(p) {
  const frac = p.total ? Math.min(1, p.step / p.total) : 0;
  $('#progBar').style.width = (100 * frac).toFixed(1) + '%';
  $('#progLabel').textContent = p.label || '';
  let txt = '';
  if (p.rate) {
    txt = `${Math.round(p.rate).toLocaleString()} steps/s`;
    const rem = (p.total - p.step) / p.rate;
    if (rem > 0.5) txt += ` · ${rem < 60 ? rem.toFixed(0) + ' s' : (rem / 60).toFixed(1) + ' min'} left`;
  }
  if (p.phase === 'train' && p.epochs) {
    const el = (performance.now() - runStart) / 1000;
    txt = `elapsed ${el.toFixed(0)} s`;
  }
  $('#progRate').textContent = txt;
}
function setRunning(on, mode) {
  running = on; runMode = mode;
  for (const id of ['btnSim', 'btnTrain', 'btnMStudy', 'btnMPbit', 'btnMRes']) $('#' + id).disabled = on;
  $('#btnStop').disabled = !on;
  const chip = $('#device'); chip.classList.toggle('busy', on);
  if (!on) { view.live = null; $('#progBar').style.width = '0'; $('#progRate').textContent = ''; }
  renderTabs();
}

// ------------------------------------------------------------------ websocket
function connectWS() {
  const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');
  ws.onmessage = ev => handle(JSON.parse(ev.data));
  ws.onclose = () => { setTimeout(connectWS, 1500); };
}
let liveScheduled = false;
async function handle(m) {
  if (m.type === 'log') addMsg(m);
  else if (m.type === 'progress') {
    setProgress(m);
    if (m.live) {
      view.live = m.live; view.liveStride = m.live_stride || 1; view.liveT = m.t_ns || 0;
      if (tab === 'field' && !liveScheduled) {
        liveScheduled = true;
        requestAnimationFrame(() => { liveScheduled = false; drawField(); });
      }
    }
  } else if (m.type === 'epoch') {
    train.loss.push(m.loss); train.outputs.push(m.outputs); train.design = m.design; train.unit = m.design_unit; train.target = m.target;
    renderTabs();
    if (tab === 'training') renderTab();
  } else if (m.type === 'status') {
    if (m.status === 'running') setRunning(true, m.mode);
    else {
      setRunning(false, null);
      $('#progLabel').textContent = { done: 'Finished', stopped: 'Stopped', error: 'Failed — see messages' }[m.status] || m.status;
      if (m.mode && m.mode.startsWith('macro_')) { if (m.has_macro) await loadMacroResult(true); }
      else if (m.has_result) await loadResult();
      renderTab();
    }
  } else if (m.type === 'hello') {
    if (m.status === 'running') setRunning(true, null);
  }
}
async function loadResult() {
  try {
    meta = await api('/api/result/meta');
    COMP_LABEL = meta.record === 'm' ? COMP_LABEL_M : COMP_LABEL_DM;
    tableData = null;
    cache.clear(); for (const k in ranges) delete ranges[k];
    view.frame = Math.max(0, meta.frames - 1);
    $('#btnExport').classList.remove('disabled');
    renderTabs();
  } catch (e) { meta = null; }
}

// ------------------------------------------------------------------ actions
async function startRun(mode) {
  if (!cfg) return;
  try {
    await check();
    if (!report) { addMsg({ level: 'error', text: 'Fix the parameter errors first (see Physics check).' }); return; }
    if (mode === 'train') { train.loss = []; train.outputs = []; train.design = null; }
    view.playing = false; view.live = null;
    runStart = performance.now();
    $('#progLabel').textContent = 'Starting…';
    await post('/api/run', { config: cfg, mode });
    setRunning(true, mode);
    selectTab(mode === 'train' ? 'training' : 'field');
  } catch (e) { addMsg({ level: 'error', text: e.message }); }
}
async function loadPreset(name) {
  if (mod === 'macro') {
    mcfg = await api(`/api/macro/preset/${name}`);
    renderTree(); renderSettings(); await mcheck(); renderTab();
    addMsg({ text: `Loaded device preset "${mcfg.name}"` });
    return;
  }
  cfg = await api(`/api/preset/${name}`);
  renderTree(); renderSettings(); await check();
  addMsg({ text: `Loaded preset "${cfg.name}"` });
}
function saveConfig() {
  const c = mod === 'macro' ? mcfg : cfg;
  const blob = new Blob([JSON.stringify(c, null, 2)], { type: 'application/json' });
  const a = h('a', { href: URL.createObjectURL(blob), download: (c.name || 'config').replace(/[^\w-]+/g, '_') + '.json' });
  document.body.append(a); a.click(); a.remove();
}
async function showRuns() {
  const ul = $('#runsList'); ul.innerHTML = '';
  const runs = await api('/api/runs');
  if (!runs.length) ul.append(h('li', {}, 'No runs yet'));
  for (const r of runs) ul.append(h('li', { onclick: async () => {
    const obj = await api(`/api/runs/${encodeURIComponent(r.id)}/config`); $('#runsDialog').close();
    if (obj.device && obj.stack) { mcfg = obj; await setModule('macro'); } else { cfg = obj; await setModule('micro'); }
    addMsg({ text: `Loaded configuration of run ${r.id}` });
  } }, r.id));
  $('#runsDialog').showModal();
}

// ------------------------------------------------------------------ modules
let INFO = null;
async function setModule(m) {
  mod = m;
  $('#module').value = m;
  document.querySelectorAll('.micro-only').forEach(e => e.classList.toggle('hidden', m !== 'micro'));
  document.querySelectorAll('.macro-only').forEach(e => e.classList.toggle('hidden', m !== 'macro'));
  const pl = m === 'macro' ? INFO.macro_presets : INFO.presets;
  $('#preset').innerHTML = '';
  for (const p of pl) $('#preset').append(h('option', { value: p }, p));
  if (m === 'macro' && !mcfg) mcfg = await api('/api/macro/preset/' + INFO.macro_presets[0]);
  node = m === 'macro' ? 'device' : 'grid';
  tab = m === 'macro' ? 'mdevice' : 'geometry';
  $('#btnExport').href = m === 'macro' ? '/api/macro/export.json' : '/api/result/export.npz';
  $('#btnExport').classList.toggle('disabled', m === 'macro' ? !mres : !meta);
  renderTree(); renderSettings(); renderTabs();
  if (m === 'macro') { await mcheck(); if (mres) await loadMacroResult(true); else renderTab(); }
  else { await check(); if (meta) selectTab('field'); else renderTab(); }
}

// ------------------------------------------------------------------ init
async function init() {
  $('#module').onchange = e => setModule(e.target.value);
  $('#btnMStudy').onclick = () => startMacro('macro_study');
  $('#btnMPbit').onclick = () => startMacro('macro_pbit');
  $('#btnMRes').onclick = () => startMacro('macro_reservoir');
  $('#btnNew').onclick = () => loadPreset($('#preset').value);
  $('#btnSave').onclick = saveConfig;
  $('#btnOpen').onclick = () => $('#fileOpen').click();
  $('#fileOpen').onchange = async e => {
    const f = e.target.files[0]; if (!f) return;
    try {
      const obj = JSON.parse(await f.text());
      if (obj.device && obj.stack) { mcfg = obj; await setModule('macro'); } else { cfg = obj; await setModule('micro'); }
      addMsg({ text: `Opened ${f.name}` });
    }
    catch (err) { addMsg({ level: 'error', text: 'Cannot open file: ' + err.message }); }
    e.target.value = '';
  };
  $('#btnRuns').onclick = showRuns;
  $('#btnCheck').onclick = check;
  $('#btnSim').onclick = () => startRun('simulate');
  $('#btnTrain').onclick = () => startRun('train');
  $('#btnStop').onclick = () => post('/api/stop', {});
  document.addEventListener('keydown', e => {
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
    if (e.key === 'F8') { e.preventDefault(); if (mod === 'macro') startMacro('macro_study'); else startRun('simulate'); }
    if (tab !== 'field' && tab !== 'kspace') return;
    if (e.key === ' ') { e.preventDefault(); togglePlay(); }
    if (e.key === 'ArrowRight') { e.preventDefault(); gotoFrame(view.frame + 1); }
    if (e.key === 'ArrowLeft') { e.preventDefault(); gotoFrame(view.frame - 1); }
  });

  const info = await api('/api/info');
  INFO = info;
  $('#device').textContent = info.cuda ? `${info.gpu} · ${info.mem_free_gb.toFixed(1)}/${info.mem_total_gb.toFixed(1)} GB free · ${info.host}` : `CPU only · ${info.host}`;
  $('#preset').innerHTML = '';
  for (const p of info.presets) $('#preset').append(h('option', { value: p }, p));

  renderTabs();
  const st = await api('/api/status');
  for (const m of st.log) addMsg(m);
  if (st.has_result) { await loadResult(); cfg = meta.config; }
  else cfg = await api('/api/preset/focus');
  if (st.status === 'running') setRunning(true, st.mode);
  if (st.has_macro) { try { mres = await api('/api/macro/result'); } catch (e) { mres = null; } }
  if (st.has_training) {
    const th = await api('/api/train/history');
    train.loss = th.loss; train.outputs = th.outputs; train.target = th.target; train.design = th.design;
    train.unit = cfg.training.design === 'Ms' ? 'kA/m' : 'mT';
  }
  const startMod = st.mode && st.mode.startsWith('macro_') ? 'macro' : 'micro';
  if (startMod === 'macro' && mres) mcfg = mres.config;
  await setModule(startMod);
  connectWS();
  addMsg({ level: 'ok', text: 'SpinStudio ready.' });
}
init().catch(e => addMsg({ level: 'error', text: 'Init failed: ' + e.message }));
