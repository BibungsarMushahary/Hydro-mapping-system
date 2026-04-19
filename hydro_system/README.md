# 🌊 HydroML — AI-Powered Drainage Network Extraction & Flood Risk Mapping

A full-stack machine learning system for hydrological analysis combining CNN, XGBoost ensemble
inference with an interactive web application.

---

## Architecture Overview

```
hydro_system/
├── backend/
│   ├── app.py                    # Flask API server
│   ├── models/
│   │   ├── cnn_model.py          # Convolutional Neural Network
│   │   ├── xgb_model.py          # Gradient Boosted Trees (XGBoost)
│   │   └── ensemble_model.py     # Ensemble + full pipeline
│   └── utils/
│       ├── dem_processor.py      # Feature extraction (13 channels)
│       └── geo_utils.py          # GeoTIFF I/O, hillshade, colormaps
├── frontend/
│   ├── templates/index.html      # Single-page web app
│   └── static/
│       ├── css/style.css         # Industrial Noir stylesheet
│       └── js/app.js             # Leaflet map + API client
├── notebooks/
│   └── analysis.ipynb            # Exploratory analysis notebook
├── data/
│   ├── uploads/                  # Uploaded DEMs
│   ├── outputs/                  # Per-job output layers
│   └── models/                   # Saved model weights
├── train.py                      # Standalone training script
├── run.py                        # Server entry point
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** GDAL/rasterio can be tricky. On Ubuntu:
> ```bash
> sudo apt-get install libgdal-dev
> pip install GDAL==$(gdal-config --version)
> ```
> The system includes pure-Python fallbacks for all geospatial operations.

### 2. (Optional) Pre-train models on a DEM

```bash
# Use a real GeoTIFF
python train.py --dem path/to/your/dem.tif --output data/models

# Or use the built-in synthetic DEM generator
python train.py --synthetic --size 256 --output data/models
```

### 3. Start the web server

```bash
python run.py --port 5000 --debug
```

Open **http://localhost:5000** in your browser.

---

## Web Application Usage

1. **Upload DEM** tab — Upload a GeoTIFF, or click **Generate Synthetic DEM** for a demo
2. Set **Rainfall Intensity** (mm) and click **▶ Run ML Pipeline**
3. Progress bar shows: feature extraction → CNN training → XGBoost training → inference → flood risk
4. **Analysis** tab — Toggle individual layers on the map:
   - Hillshade / DEM elevation
   - Ensemble / CNN / XGBoost stream probability
   - ML stream network (binary)
   - Classical D8 labels (reference)
5. **Flood Risk** tab — View risk index and classification with rainfall control
6. **Statistics** tab — Download GeoTIFF outputs and JSON stats

---

## ML Pipeline

### Feature Extraction (13 channels)
| # | Feature | Description |
|---|---------|-------------|
| 0 | Elevation | Normalised DEM value |
| 1 | Slope | Gradient magnitude (degrees) |
| 2 | Aspect | Flow direction (degrees) |
| 3 | Plan Curvature | Contour curvature |
| 4 | Profile Curvature | Slope-direction curvature |
| 5 | Texture Mean | Local window mean |
| 6 | Texture Variance | Local window variance |
| 7 | Flow Accumulation | log(D8 upstream area) |
| 8 | TWI | Topographic Wetness Index |
| 9 | SPI | Stream Power Index |
| 10 | Ridge Distance | Distance to nearest ridge |
| 11 | Y coordinate | Normalised row position |
| 12 | X coordinate | Normalised col position |

### Models
- **CNN** (`SimpleCNN`): 3-layer spatial feature extractor with patch-wise training
- **XGBoost** (`XGBoostModel`): 80-round gradient boosted stumps, pixel-wise features
- **Ensemble**: Weighted average (CNN × 0.45 + XGB × 0.55)

### Label Generation
Automatic using D8 flow accumulation — pixels with ≥ `flow_threshold` upstream cells
are classified as stream channels.

### Flood Risk
```
risk = 0.35 × (1 - elev_norm)
     + 0.30 × proximity_to_streams
     + 0.20 × flow_accumulation_norm
     + 0.15 × stream_probability
     × rainfall_factor(mm)
```
Classified into: **Low** (< 0.33), **Medium** (0.33–0.66), **High** (> 0.66)

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload GeoTIFF DEM |
| POST | `/api/process` | Start ML pipeline |
| GET | `/api/status/<job_id>` | Poll processing progress |
| GET | `/api/layer/<job_id>/<layer>` | Get PNG map layer |
| GET | `/api/download/<job_id>/<file>` | Download output file |
| POST | `/api/generate_synthetic` | Create & process synthetic DEM |
| GET | `/api/rainfall` | Historical rainfall data |
| GET | `/api/stats/<job_id>` | Analysis statistics JSON |

**Available layers:** `hillshade`, `dem`, `stream_prob`, `cnn_prob`, `xgb_prob`,
`stream_mask`, `classical_labels`, `flood_risk`

---

## Extending the System

### Swap in real XGBoost
```python
# In backend/models/xgb_model.py, replace XGBoostModel.train() with:
import xgboost as xgb
self.clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05)
self.clf.fit(X_train, y_train)
```

### Swap in TensorFlow CNN
```python
# In backend/models/cnn_model.py, replace SimpleCNN with a tf.keras UNet:
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
    tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
    tf.keras.layers.Conv2D(1,  1, activation='sigmoid'),
])
```

### Real rainfall data
```python
# In backend/app.py rainfall_data(), replace with:
import requests
resp = requests.get(f'https://api.open-meteo.com/v1/forecast?...')
```

---

## License
MIT — free for academic and commercial use.
