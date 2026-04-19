#!/usr/bin/env python3
"""
Standalone Training Script
===========================
Usage:
    python train.py --dem path/to/dem.tif --output data/models
    python train.py --synthetic --size 256 --output data/models
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.models.ensemble_model import EnsembleModel
from backend.utils.geo_utils import read_dem, _synthetic_dem, write_geotiff, _synthetic_meta


def train(dem_path=None, synthetic=False, size=256,
          output_dir='data/models', flow_threshold=800):

    if synthetic or dem_path is None:
        print(f"[Train] Generating synthetic DEM ({size}×{size}) …")
        from backend.utils.geo_utils import _synthetic_dem
        dem = _synthetic_dem(size, size)
        meta = _synthetic_meta(dem.shape)
    else:
        print(f"[Train] Reading DEM from {dem_path} …")
        dem, meta = read_dem(dem_path)

    print(f"[Train] DEM shape: {dem.shape}  range: {dem.min():.1f}–{dem.max():.1f} m")

    model = EnsembleModel(cell_size=30.0)
    features, labels = model.train(dem, flow_threshold=flow_threshold)

    print(f"[Train] Saving models to {output_dir} …")
    model.save(output_dir)

    # Quick evaluation
    results = model.predict(dem)
    ens     = results['ensemble_prob']
    pred    = (ens >= 0.5).astype(np.uint8)

    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())

    prec  = tp / (tp + fp + 1e-9)
    rec   = tp / (tp + fn + 1e-9)
    f1    = 2 * prec * rec / (prec + rec + 1e-9)
    acc   = (tp + tn) / labels.size

    print("\n──────────────── Evaluation ────────────────")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1 Score  : {f1*100:.2f}%")
    print("─────────────────────────────────────────────\n")

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train HydroML models')
    parser.add_argument('--dem',       type=str, default=None,
                        help='Path to GeoTIFF DEM')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic DEM for training')
    parser.add_argument('--size',      type=int, default=256,
                        help='Synthetic DEM size (default 256)')
    parser.add_argument('--output',    type=str, default='data/models',
                        help='Output directory for model weights')
    parser.add_argument('--threshold', type=int, default=800,
                        help='Flow accumulation threshold for stream labels')
    args = parser.parse_args()

    train(
        dem_path       = args.dem,
        synthetic      = args.synthetic or (args.dem is None),
        size           = args.size,
        output_dir     = args.output,
        flow_threshold = args.threshold,
    )
