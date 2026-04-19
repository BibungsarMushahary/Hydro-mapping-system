"""
Ensemble Model — combines CNN and XGBoost predictions
======================================================
Stacking approach: weighted average of individual model outputs,
with optional logistic meta-learner.
"""

import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')

from backend.models.cnn_model   import SimpleCNN
from backend.models.xgb_model   import XGBoostModel
from backend.utils.dem_processor import DEMProcessor


class EnsembleModel:
    """
    Orchestrates training and inference for the full ML pipeline.

    Parameters
    ----------
    cnn_weight : float   weight for CNN prediction  (0-1)
    xgb_weight : float   weight for XGBoost prediction (0-1)
    cell_size  : float   DEM cell size in metres
    """

    def __init__(
        self,
        cnn_weight: float = 0.45,
        xgb_weight: float = 0.55,
        cell_size: float  = 30.0,
    ):
        self.cnn_weight = cnn_weight
        self.xgb_weight = xgb_weight

        self.processor = DEMProcessor(cell_size=cell_size)
        self.cnn       = SimpleCNN()
        self.xgb       = XGBoostModel(n_estimators=80, learning_rate=0.1)
        self._trained  = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, dem: np.ndarray, flow_threshold: int = 800):
        """
        End-to-end training from a raw DEM array.

        1. Extract features
        2. Auto-generate labels (D8 flow accumulation)
        3. Train CNN and XGBoost independently
        """
        print("=" * 60)
        print("[Ensemble] Starting training pipeline …")
        print("=" * 60)

        print("[Ensemble] Extracting features …")
        features = self.processor.extract_all_features(dem)

        print("[Ensemble] Generating stream labels …")
        labels = self.processor.generate_stream_labels(dem, flow_threshold)
        pos = labels.sum()
        print(f"  Stream pixels : {pos:,}  ({pos/labels.size*100:.1f}%)")

        # --- CNN ---
        self.cnn.train(features, labels, epochs=2, n_patches=150)

        # --- XGBoost ---
        self.xgb.train(features, labels, sample_frac=0.04)

        self._trained = True
        print("[Ensemble] Training complete.")
        return features, labels

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, dem: np.ndarray) -> dict:
        """
        Run full inference pipeline on a DEM.

        Returns
        -------
        dict with keys:
            'features'     : (H, W, 13)
            'cnn_prob'     : (H, W) float32
            'xgb_prob'     : (H, W) float32
            'ensemble_prob': (H, W) float32
            'stream_mask'  : (H, W) uint8
            'labels'       : (H, W) uint8   (classical reference)
        """
        print("[Ensemble] Extracting features …")
        features = self.processor.extract_all_features(dem)

        print("[Ensemble] CNN inference …")
        cnn_prob = self.cnn.predict(features)

        print("[Ensemble] XGBoost inference …")
        xgb_prob = self.xgb.predict(features)

        # Weighted ensemble
        ens_prob = (
            self.cnn_weight * cnn_prob + self.xgb_weight * xgb_prob
        ).astype(np.float32)

        # Binary mask at 0.5 threshold
        stream_mask = (ens_prob >= 0.5).astype(np.uint8)

        # Classical reference labels
        labels = self.processor.generate_stream_labels(dem)

        return {
            'features'     : features,
            'cnn_prob'     : cnn_prob,
            'xgb_prob'     : xgb_prob,
            'ensemble_prob': ens_prob,
            'stream_mask'  : stream_mask,
            'labels'       : labels,
        }

    def compute_flood_risk(
        self,
        dem: np.ndarray,
        stream_prob: np.ndarray,
        rainfall_mm: float = 50.0,
    ) -> dict:
        """
        Generate flood risk index and classified zones.

        Returns
        -------
        dict with keys:
            'risk_index' : (H, W) float32 [0, 1]
            'risk_class' : (H, W) uint8   {1=low, 2=medium, 3=high}
        """
        risk = self.processor.compute_flood_risk(dem, stream_prob, rainfall_mm)
        risk_class = np.digitize(risk, bins=[0.33, 0.66]).astype(np.uint8) + 1
        return {'risk_index': risk, 'risk_class': risk_class}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        self.cnn.save(os.path.join(model_dir, 'cnn_weights.json'))
        self.xgb.save(os.path.join(model_dir, 'xgb_weights.json'))
        meta = {
            'cnn_weight': self.cnn_weight,
            'xgb_weight': self.xgb_weight,
        }
        with open(os.path.join(model_dir, 'ensemble_meta.json'), 'w') as f:
            json.dump(meta, f)
        print(f"[Ensemble] Models saved to {model_dir}")

    def load(self, model_dir: str):
        self.cnn.load(os.path.join(model_dir, 'cnn_weights.json'))
        self.xgb.load(os.path.join(model_dir, 'xgb_weights.json'))
        meta_path = os.path.join(model_dir, 'ensemble_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            self.cnn_weight = meta.get('cnn_weight', self.cnn_weight)
            self.xgb_weight = meta.get('xgb_weight', self.xgb_weight)
        self._trained = True
        print(f"[Ensemble] Models loaded from {model_dir}")

    @property
    def is_trained(self) -> bool:
        return self._trained
