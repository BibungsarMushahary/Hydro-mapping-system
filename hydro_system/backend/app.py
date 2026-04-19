"""
Flask API — Drainage Network Extraction & Flood Risk Mapping
============================================================
Endpoints:
  POST /api/upload          Upload DEM file
  POST /api/process         Process uploaded DEM
  GET  /api/status/<job_id> Poll processing status
  GET  /api/layer/<job_id>/<layer>  Retrieve output image
  GET  /api/download/<job_id>/<file> Download output file
  GET  /api/rainfall        Fetch rainfall data (simulated)
  GET  /                    Serve web app
"""

import os
import sys
import json
import time
import uuid
import threading
import traceback
import numpy as np
from io import BytesIO

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import (
    Flask, request, jsonify, send_file, render_template_string,
    send_from_directory
)
from flask_cors import CORS
from werkzeug.utils import secure_filename

from backend.models.ensemble_model import EnsembleModel
from backend.utils.geo_utils       import (
    read_dem, write_geotiff, write_png,
    generate_hillshade, apply_colormap, array_to_png_bytes
)

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------

app = Flask(
    __name__,
    static_folder='../frontend/static',
    template_folder='../frontend/templates',
)
CORS(app)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'uploads')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'outputs')
MODEL_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data', 'models')

for d in [UPLOAD_DIR, OUTPUT_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

ALLOWED_EXTENSIONS = {'.tif', '.tiff', '.geotiff'}

# In-memory job store
jobs: dict[str, dict] = {}
model_cache: dict[str, EnsembleModel] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS


def get_model() -> EnsembleModel:
    """Return a cached (or new) EnsembleModel."""
    if 'default' not in model_cache:
        model_cache['default'] = EnsembleModel()
    return model_cache['default']


def update_job(job_id, **kwargs):
    if job_id in jobs:
        jobs[job_id].update(kwargs)


# ---------------------------------------------------------------------------
# Processing thread
# ---------------------------------------------------------------------------

def process_dem_task(job_id: str, dem_path: str, rainfall_mm: float):
    try:
        update_job(job_id, status='processing', progress=5,
                   message='Reading DEM …')

        dem, meta = read_dem(dem_path)
        update_job(job_id, progress=15, message='Extracting features …')

        model = get_model()
        if not model.is_trained:
            update_job(job_id, progress=20, message='Training ML models …')
            model.train(dem, flow_threshold=600)
            model.save(MODEL_DIR)

        update_job(job_id, progress=50, message='Running ML inference …')
        results = model.predict(dem)

        update_job(job_id, progress=70, message='Computing flood risk …')
        flood = model.compute_flood_risk(
            dem, results['ensemble_prob'], rainfall_mm
        )

        update_job(job_id, progress=80, message='Generating visualisations …')

        out_dir = os.path.join(OUTPUT_DIR, job_id)
        os.makedirs(out_dir, exist_ok=True)

        # --- Hillshade ---
        hs = generate_hillshade(dem)
        hs_rgba = np.stack([hs, hs, hs, np.full_like(hs, 255)], axis=-1)
        _save_png(os.path.join(out_dir, 'hillshade.png'), hs_rgba)

        # --- DEM colourmap ---
        dem_n = (dem - dem.min()) / max(dem.max() - dem.min(), 1)
        dem_rgba = apply_colormap(dem_n, 'viridis')
        _save_png(os.path.join(out_dir, 'dem.png'), dem_rgba)

        # --- Stream probability ---
        stream_rgba = apply_colormap(results['ensemble_prob'], 'stream')
        _save_png(os.path.join(out_dir, 'stream_prob.png'), stream_rgba)

        # --- CNN probability ---
        cnn_rgba = apply_colormap(results['cnn_prob'], 'stream')
        _save_png(os.path.join(out_dir, 'cnn_prob.png'), cnn_rgba)

        # --- XGB probability ---
        xgb_rgba = apply_colormap(results['xgb_prob'], 'stream')
        _save_png(os.path.join(out_dir, 'xgb_prob.png'), xgb_rgba)

        # --- Flood risk ---
        risk_rgba = apply_colormap(flood['risk_index'], 'risk')
        _save_png(os.path.join(out_dir, 'flood_risk.png'), risk_rgba)

        # --- Stream mask (binary) ---
        mask = (results['stream_mask'] * 255).astype(np.uint8)
        mask_rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        mask_rgba[mask > 0] = [30, 120, 255, 230]
        _save_png(os.path.join(out_dir, 'stream_mask.png'), mask_rgba)

        # --- Classical labels ---
        lab = (results['labels'] * 255).astype(np.uint8)
        lab_rgba = np.zeros((*lab.shape, 4), dtype=np.uint8)
        lab_rgba[lab > 0] = [0, 200, 80, 210]
        _save_png(os.path.join(out_dir, 'classical_labels.png'), lab_rgba)

        update_job(job_id, progress=90, message='Saving outputs …')

        # --- GeoTIFF outputs ---
        write_geotiff(
            os.path.join(out_dir, 'stream_probability.tif'),
            results['ensemble_prob'], meta
        )
        write_geotiff(
            os.path.join(out_dir, 'flood_risk.tif'),
            flood['risk_index'], meta
        )

        # --- Statistics ---
        stats = _compute_stats(dem, results, flood)
        with open(os.path.join(out_dir, 'stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)

        update_job(
            job_id,
            status='done', progress=100,
            message='Processing complete.',
            stats=stats,
            bounds=_meta_bounds(meta, dem),
        )

    except Exception as exc:
        traceback.print_exc()
        update_job(job_id, status='error', progress=0,
                   message=f'Error: {exc}')


def _save_png(path: str, rgba: np.ndarray):
    png_bytes = array_to_png_bytes(rgba.astype(np.uint8))
    with open(path, 'wb') as f:
        f.write(png_bytes)


def _compute_stats(dem, results, flood):
    stream_px = int(results['stream_mask'].sum())
    total_px  = int(results['stream_mask'].size)
    risk      = flood['risk_index']
    return {
        'dem_min'         : float(dem.min()),
        'dem_max'         : float(dem.max()),
        'dem_mean'        : float(dem.mean()),
        'stream_pixels'   : stream_px,
        'stream_fraction' : round(stream_px / total_px * 100, 2),
        'mean_stream_prob': round(float(results['ensemble_prob'].mean()), 4),
        'high_risk_pct'   : round(float((risk > 0.66).mean() * 100), 2),
        'med_risk_pct'    : round(float(((risk > 0.33) & (risk <= 0.66)).mean() * 100), 2),
        'low_risk_pct'    : round(float((risk <= 0.33).mean() * 100), 2),
        'grid_rows'       : int(dem.shape[0]),
        'grid_cols'       : int(dem.shape[1]),
    }


def _meta_bounds(meta, dem):
    if meta.get('bounds'):
        b = meta['bounds']
        if hasattr(b, 'left'):
            return [b.bottom, b.left, b.top, b.right]
        return list(b)
    rows, cols = dem.shape
    return [0, 0, rows * 30, cols * 30]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return send_from_directory('../frontend/templates', 'index.html')


@app.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    f = request.files['file']
    if not f.filename or not allowed_file(f.filename):
        return jsonify({'error': 'Invalid file type. Upload a GeoTIFF (.tif)'}), 400

    job_id   = str(uuid.uuid4())[:8]
    filename = secure_filename(f.filename)
    save_path = os.path.join(UPLOAD_DIR, f'{job_id}_{filename}')
    f.save(save_path)

    jobs[job_id] = {
        'status'  : 'uploaded',
        'progress': 0,
        'message' : 'File uploaded. Ready to process.',
        'path'    : save_path,
        'filename': filename,
        'created' : time.time(),
    }
    return jsonify({'job_id': job_id, 'filename': filename})


@app.route('/api/process', methods=['POST'])
def process():
    data       = request.get_json() or {}
    job_id     = data.get('job_id')
    rainfall   = float(data.get('rainfall_mm', 50.0))

    if not job_id or job_id not in jobs:
        return jsonify({'error': 'Unknown job_id'}), 404

    job = jobs[job_id]
    if job['status'] == 'processing':
        return jsonify({'error': 'Already processing'}), 409

    thread = threading.Thread(
        target=process_dem_task,
        args=(job_id, job['path'], rainfall),
        daemon=True,
    )
    thread.start()
    return jsonify({'job_id': job_id, 'status': 'started'})


@app.route('/api/status/<job_id>')
def status(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Unknown job_id'}), 404
    return jsonify(jobs[job_id])


@app.route('/api/layer/<job_id>/<layer_name>')
def get_layer(job_id, layer_name):
    safe_names = {
        'hillshade', 'dem', 'stream_prob', 'cnn_prob', 'xgb_prob',
        'flood_risk', 'stream_mask', 'classical_labels',
    }
    if layer_name not in safe_names:
        return jsonify({'error': 'Unknown layer'}), 404

    path = os.path.join(OUTPUT_DIR, job_id, f'{layer_name}.png')
    if not os.path.exists(path):
        return jsonify({'error': 'Layer not ready'}), 404

    return send_file(path, mimetype='image/png')


@app.route('/api/download/<job_id>/<filename>')
def download(job_id, filename):
    safe = {'stream_probability.tif', 'flood_risk.tif', 'stats.json'}
    if filename not in safe:
        return jsonify({'error': 'Not available'}), 404
    path = os.path.join(OUTPUT_DIR, job_id, filename)
    if not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404
    return send_file(path, as_attachment=True)


@app.route('/api/stats/<job_id>')
def get_stats(job_id):
    path = os.path.join(OUTPUT_DIR, job_id, 'stats.json')
    if not os.path.exists(path):
        return jsonify({'error': 'Stats not available'}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@app.route('/api/generate_synthetic', methods=['POST'])
def generate_synthetic():
    """Generate and process a synthetic DEM for demo purposes."""
    from backend.utils.geo_utils import _synthetic_dem, _synthetic_meta

    data     = request.get_json() or {}
    size     = min(int(data.get('size', 256)), 512)
    rainfall = float(data.get('rainfall_mm', 50.0))

    job_id = str(uuid.uuid4())[:8]
    dem    = _synthetic_dem(size, size)
    meta   = _synthetic_meta(dem.shape)

    # Save as NPY (bypass TIFF)
    npy_path = os.path.join(UPLOAD_DIR, f'{job_id}_synthetic.npy')
    np.save(npy_path, dem)

    jobs[job_id] = {
        'status'  : 'uploaded',
        'progress': 0,
        'message' : 'Synthetic DEM generated.',
        'path'    : npy_path,
        'filename': 'synthetic.npy',
        'created' : time.time(),
    }

    # Override read_dem for NPY
    def _process_npy(jid, path, rain):
        try:
            update_job(jid, status='processing', progress=10,
                       message='Loading synthetic DEM …')
            dem_arr = np.load(path).astype(np.float32)
            meta_d  = _synthetic_meta(dem_arr.shape)
            update_job(jid, progress=20, message='Training ML models …')

            m = get_model()
            if not m.is_trained:
                m.train(dem_arr)
                m.save(MODEL_DIR)

            update_job(jid, progress=50, message='Running inference …')
            res   = m.predict(dem_arr)
            flood = m.compute_flood_risk(dem_arr, res['ensemble_prob'], rain)

            update_job(jid, progress=75, message='Rendering layers …')
            out_dir = os.path.join(OUTPUT_DIR, jid)
            os.makedirs(out_dir, exist_ok=True)

            hs = generate_hillshade(dem_arr)
            _save_png(os.path.join(out_dir, 'hillshade.png'),
                      np.stack([hs,hs,hs,np.full_like(hs,255)], -1))
            dem_n = (dem_arr - dem_arr.min()) / max(dem_arr.max()-dem_arr.min(),1)
            _save_png(os.path.join(out_dir, 'dem.png'),
                      apply_colormap(dem_n, 'viridis'))
            _save_png(os.path.join(out_dir, 'stream_prob.png'),
                      apply_colormap(res['ensemble_prob'], 'stream'))
            _save_png(os.path.join(out_dir, 'cnn_prob.png'),
                      apply_colormap(res['cnn_prob'], 'stream'))
            _save_png(os.path.join(out_dir, 'xgb_prob.png'),
                      apply_colormap(res['xgb_prob'], 'stream'))
            _save_png(os.path.join(out_dir, 'flood_risk.png'),
                      apply_colormap(flood['risk_index'], 'risk'))
            mask = (res['stream_mask']*255).astype(np.uint8)
            m_rgba = np.zeros((*mask.shape,4), np.uint8)
            m_rgba[mask>0] = [30,120,255,230]
            _save_png(os.path.join(out_dir, 'stream_mask.png'), m_rgba)
            lab = (res['labels']*255).astype(np.uint8)
            l_rgba = np.zeros((*lab.shape,4), np.uint8)
            l_rgba[lab>0] = [0,200,80,210]
            _save_png(os.path.join(out_dir, 'classical_labels.png'), l_rgba)

            stats = _compute_stats(dem_arr, res, flood)
            with open(os.path.join(out_dir, 'stats.json'), 'w') as fh:
                json.dump(stats, fh, indent=2)

            update_job(jid, status='done', progress=100,
                       message='Complete.', stats=stats,
                       bounds=_meta_bounds(meta_d, dem_arr))
        except Exception as e:
            traceback.print_exc()
            update_job(jid, status='error', message=str(e))

    t = threading.Thread(target=_process_npy,
                         args=(job_id, npy_path, rainfall), daemon=True)
    t.start()
    return jsonify({'job_id': job_id})


@app.route('/api/rainfall', methods=['GET'])
def rainfall_data():
    """Simulated historical rainfall data."""
    import random
    random.seed(42)
    months = ['Jan','Feb','Mar','Apr','May','Jun',
              'Jul','Aug','Sep','Oct','Nov','Dec']
    data = [{'month': m, 'mm': random.randint(20, 200)} for m in months]
    return jsonify({'data': data, 'annual_mean_mm': 85.4})


if __name__ == '__main__':
    print("🌊 Hydro ML System starting on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
