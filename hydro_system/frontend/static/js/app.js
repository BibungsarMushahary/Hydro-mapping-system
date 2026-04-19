/* ═══════════════════════════════════════════════════════════
   HydroML — Frontend Application
   ═══════════════════════════════════════════════════════════ */

'use strict';

// ── State ──────────────────────────────────────────────────
const state = {
  jobId      : null,
  polling    : null,
  layers     : {},   // name → Leaflet ImageOverlay
  opacity    : 0.75,
  bounds     : null,
  currentStats: null,
};

// ── Map init ───────────────────────────────────────────────
const map = L.map('map', {
  center: [20, 0],
  zoom  : 2,
  zoomControl: true,
  attributionControl: true,
});

L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
  subdomains : 'abcd',
  maxZoom    : 19,
}).addTo(map);

// Coordinates display
map.on('mousemove', (e) => {
  document.getElementById('coordsBar').textContent =
    `${e.latlng.lat.toFixed(4)}°, ${e.latlng.lng.toFixed(4)}°`;
});

// ── Panel navigation ────────────────────────────────────────
document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const panel = btn.dataset.panel;
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(`panel-${panel}`).classList.add('active');
  });
});

// ── Upload zone ────────────────────────────────────────────
const uploadZone  = document.getElementById('uploadZone');
const fileInput   = document.getElementById('fileInput');
const uploadStatus = document.getElementById('uploadStatus');

uploadZone.addEventListener('click',      () => fileInput.click());
uploadZone.addEventListener('dragover',   e => { e.preventDefault(); uploadZone.style.borderColor='var(--accent)'; });
uploadZone.addEventListener('dragleave',  () => uploadZone.style.borderColor='');
uploadZone.addEventListener('drop',       e => { e.preventDefault(); handleFile(e.dataTransfer.files[0]); });
fileInput.addEventListener('change',      () => handleFile(fileInput.files[0]));

async function handleFile(file) {
  if (!file) return;
  uploadStatus.textContent = `Uploading ${file.name} …`;
  setChip('active', 'Uploading');

  const fd = new FormData();
  fd.append('file', file);

  try {
    const res  = await fetch('/api/upload', { method: 'POST', body: fd });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    state.jobId = data.job_id;
    uploadStatus.textContent = `✓ ${data.filename} uploaded`;
    document.getElementById('processBtn').disabled = false;
    setChip('done', 'Uploaded');
  } catch(err) {
    uploadStatus.textContent = `✗ ${err.message}`;
    setChip('error', 'Error');
  }
}

// ── Sliders ────────────────────────────────────────────────
const demSizeSlider     = document.getElementById('demSize');
const demSizeLabel      = document.getElementById('demSizeLabel');
const rainfallSlider    = document.getElementById('rainfallSlider');
const rainfallLabel     = document.getElementById('rainfallLabel');
const rainfallUpdate    = document.getElementById('rainfallUpdate');
const rainfallUpdateLbl = document.getElementById('rainfallUpdateLabel');
const opacitySlider     = document.getElementById('opacitySlider');
const opacityLabel      = document.getElementById('opacityLabel');

demSizeSlider.addEventListener('input', () => {
  demSizeLabel.textContent = `${demSizeSlider.value} × ${demSizeSlider.value}`;
});
rainfallSlider.addEventListener('input', () => {
  rainfallLabel.textContent = `${rainfallSlider.value} mm`;
});
rainfallUpdate.addEventListener('input', () => {
  rainfallUpdateLbl.textContent = `${rainfallUpdate.value} mm`;
});
opacitySlider.addEventListener('input', () => {
  state.opacity = parseFloat(opacitySlider.value);
  opacityLabel.textContent = `${Math.round(state.opacity * 100)}%`;
  Object.values(state.layers).forEach(l => { if (l) l.setOpacity(state.opacity); });
});

// ── Synthetic DEM ──────────────────────────────────────────
document.getElementById('syntheticBtn').addEventListener('click', async () => {
  const size     = parseInt(demSizeSlider.value);
  const rainfall = parseFloat(rainfallSlider.value);

  setChip('active', 'Generating');
  try {
    const res  = await fetch('/api/generate_synthetic', {
      method : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body   : JSON.stringify({ size, rainfall_mm: rainfall }),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    state.jobId = data.job_id;
    uploadStatus.textContent = `✓ Synthetic DEM (${size}×${size}) created`;
    document.getElementById('processBtn').disabled = false;
    setChip('done', 'Ready');
    startPolling();
  } catch(err) {
    uploadStatus.textContent = `✗ ${err.message}`;
    setChip('error', 'Error');
  }
});

// ── Process ────────────────────────────────────────────────
document.getElementById('processBtn').addEventListener('click', async () => {
  if (!state.jobId) return;
  const rainfall = parseFloat(rainfallSlider.value);

  try {
    const res  = await fetch('/api/process', {
      method : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body   : JSON.stringify({ job_id: state.jobId, rainfall_mm: rainfall }),
    });
    const data = await res.json();
    if (data.error && data.error !== 'Already processing') throw new Error(data.error);
    startPolling();
  } catch(err) {
    showError(err.message);
  }
});

// ── Polling ────────────────────────────────────────────────
function startPolling() {
  if (state.polling) clearInterval(state.polling);
  showOverlay(true, 'Starting …', 0);
  state.polling = setInterval(pollStatus, 1200);
}

async function pollStatus() {
  if (!state.jobId) return;
  try {
    const res  = await fetch(`/api/status/${state.jobId}`);
    const data = await res.json();

    updateProgress(data.progress || 0, data.message || '');
    setChip('active', `${data.progress || 0}%`);

    if (data.status === 'done') {
      clearInterval(state.polling);
      showOverlay(false);
      setChip('done', 'Complete');
      onProcessingDone(data);
    } else if (data.status === 'error') {
      clearInterval(state.polling);
      showOverlay(false);
      setChip('error', 'Error');
      showError(data.message);
    }
  } catch(e) { /* network hiccup */ }
}

function onProcessingDone(data) {
  enableTabs();
  loadAllLayers();
  if (data.bounds) fitBounds(data.bounds);
  if (data.stats)  renderStats(data.stats);
  loadRainfallChart();
}

// ── Layers ────────────────────────────────────────────────
const LAYER_ORDER = [
  'hillshade', 'dem', 'stream_prob', 'cnn_prob',
  'xgb_prob', 'classical_labels', 'stream_mask', 'flood_risk',
];

function loadAllLayers() {
  LAYER_ORDER.forEach(name => loadLayer(name));
}

function loadLayer(name) {
  if (!state.bounds) return;

  // Remove existing
  if (state.layers[name]) {
    map.removeLayer(state.layers[name]);
    state.layers[name] = null;
  }

  const checkbox = document.querySelector(`.layer-toggle[data-layer="${name}"]`);
  if (checkbox && !checkbox.checked) return;

  const url = `/api/layer/${state.jobId}/${name}?t=${Date.now()}`;
  const overlay = L.imageOverlay(url, state.bounds, {
    opacity  : state.opacity,
    className: `layer-${name}`,
  }).addTo(map);
  state.layers[name] = overlay;
}

// Layer toggle checkboxes
document.querySelectorAll('.layer-toggle').forEach(cb => {
  cb.addEventListener('change', () => {
    const name = cb.dataset.layer;
    if (cb.checked) {
      loadLayer(name);
    } else {
      if (state.layers[name]) {
        map.removeLayer(state.layers[name]);
        state.layers[name] = null;
      }
    }
  });
});

function fitBounds(bounds) {
  // bounds: [south, west, north, east]  OR  [minx, miny, maxx, maxy]
  let leafletBounds;
  if (bounds[0] < -90 || bounds[0] > 90) {
    // Pixel coordinates — use arbitrary lat/lng box for demo
    leafletBounds = L.latLngBounds([[-30, -50], [30, 50]]);
  } else {
    leafletBounds = L.latLngBounds([[bounds[0], bounds[1]], [bounds[2], bounds[3]]]);
  }
  state.bounds = leafletBounds;
  map.fitBounds(leafletBounds, { padding: [20, 20] });
  loadAllLayers();
}

// ── Stats rendering ────────────────────────────────────────
function renderStats(stats) {
  const grid = document.getElementById('statsGrid');
  const cards = [
    { label: 'Grid Size', value: `${stats.grid_cols}×${stats.grid_rows}` },
    { label: 'Elev. Range (m)', value: `${stats.dem_min.toFixed(0)}–${stats.dem_max.toFixed(0)}` },
    { label: 'Mean Elevation (m)', value: stats.dem_mean.toFixed(1) },
    { label: 'Stream Coverage', value: `${stats.stream_fraction}%` },
    { label: 'Mean Stream Prob.', value: stats.mean_stream_prob },
    { label: 'High Risk Area', value: `${stats.high_risk_pct}%` },
    { label: 'Medium Risk Area', value: `${stats.med_risk_pct}%` },
    { label: 'Low Risk Area', value: `${stats.low_risk_pct}%` },
  ];
  grid.innerHTML = cards.map(c => `
    <div class="stat-card">
      <div class="stat-value">${c.value}</div>
      <div class="stat-label">${c.label}</div>
    </div>
  `).join('');
  state.currentStats = stats;
}

// ── Rainfall chart ─────────────────────────────────────────
async function loadRainfallChart() {
  try {
    const res  = await fetch('/api/rainfall');
    const data = await res.json();
    drawRainfallChart(data.data);
  } catch(e) {}
}

function drawRainfallChart(data) {
  const canvas = document.getElementById('rainfallChart');
  const ctx    = canvas.getContext('2d');
  const W = canvas.offsetWidth || 230;
  const H = 120;
  canvas.width  = W;
  canvas.height = H;

  const max = Math.max(...data.map(d => d.mm));
  const pad = { l: 30, r: 10, t: 10, b: 25 };
  const bw  = (W - pad.l - pad.r) / data.length - 3;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = 'rgba(255,255,255,0.03)';
  ctx.fillRect(0, 0, W, H);

  data.forEach((d, i) => {
    const x = pad.l + i * ((W - pad.l - pad.r) / data.length);
    const bh = ((d.mm / max) * (H - pad.t - pad.b));
    const y  = H - pad.b - bh;
    const alpha = d.mm / max;

    // Bar gradient
    const grad = ctx.createLinearGradient(0, y, 0, H - pad.b);
    grad.addColorStop(0, `rgba(78,205,196,${alpha})`);
    grad.addColorStop(1, `rgba(78,205,196,0.1)`);
    ctx.fillStyle = grad;
    ctx.fillRect(x, y, bw, bh);

    // Month label
    ctx.fillStyle = 'rgba(107,122,153,0.8)';
    ctx.font      = '8px "Space Mono"';
    ctx.fillText(d.month.substring(0, 3), x, H - 6);
  });

  // Y-axis
  ctx.fillStyle = 'rgba(107,122,153,0.6)';
  ctx.font      = '8px "Space Mono"';
  ctx.fillText(`${max}`, 2, pad.t + 6);
  ctx.fillText('0', 2, H - pad.b + 6);
}

// ── Downloads ─────────────────────────────────────────────
document.getElementById('dlStreamTif').addEventListener('click', () => {
  if (state.jobId) window.open(`/api/download/${state.jobId}/stream_probability.tif`);
});
document.getElementById('dlFloodTif').addEventListener('click', () => {
  if (state.jobId) window.open(`/api/download/${state.jobId}/flood_risk.tif`);
});
document.getElementById('dlStats').addEventListener('click', () => {
  if (state.jobId) window.open(`/api/download/${state.jobId}/stats.json`);
});

// ── UI helpers ─────────────────────────────────────────────
function setChip(cls, text) {
  const dot  = document.querySelector('.dot');
  const span = document.getElementById('chipText');
  dot.className  = `dot ${cls}`;
  span.textContent = text;
}

function showOverlay(show, msg = '', pct = 0) {
  const el = document.getElementById('progressOverlay');
  if (show) {
    el.removeAttribute('hidden');
    updateProgress(pct, msg);
  } else {
    el.setAttribute('hidden', '');
  }
}

function updateProgress(pct, msg) {
  document.getElementById('progressFill').style.width = `${pct}%`;
  document.getElementById('progressPct').textContent   = `${pct}%`;
  if (msg) document.getElementById('progressMsg').textContent = msg;
}

function showError(msg) {
  uploadStatus.textContent = `✗ ${msg}`;
  console.error('[HydroML]', msg);
}

function enableTabs() {
  ['resultsBtn', 'floodBtn', 'statsBtn'].forEach(id => {
    document.getElementById(id).disabled = false;
  });
}

// ── Init default bounds for synthetic demo ─────────────────
state.bounds = L.latLngBounds([[-20, -40], [20, 40]]);
