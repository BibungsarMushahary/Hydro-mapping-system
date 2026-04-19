"""
XGBoost Model for Pixel-wise Stream Classification
===================================================
Operates on per-pixel feature vectors derived from the DEM.
"""

import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Pure-NumPy Gradient Boosted Trees (simplified CART implementation)
# Used when XGBoost package is unavailable.  Replace with xgboost.XGBClassifier
# in production for full performance.
# ---------------------------------------------------------------------------

class DecisionStump:
    """Single-feature threshold split."""

    def __init__(self):
        self.feature_idx = 0
        self.threshold   = 0.0
        self.left_val    = 0.0
        self.right_val   = 0.0

    def fit(self, X, residuals):
        n_feat = X.shape[1]
        best_loss = np.inf
        rng = np.random.default_rng(int(abs(residuals.sum() * 1e6)) % (2**31))
        feat_subset = rng.choice(n_feat, size=min(6, n_feat), replace=False)

        for fi in feat_subset:
            thresholds = np.percentile(X[:, fi], [25, 50, 75])
            for thr in thresholds:
                left  = residuals[X[:, fi] <  thr]
                right = residuals[X[:, fi] >= thr]
                loss  = (left.var()  * len(left) +
                         right.var() * len(right)) if len(left)>0 and len(right)>0 else np.inf
                if loss < best_loss:
                    best_loss = loss
                    self.feature_idx = fi
                    self.threshold   = thr
                    self.left_val    = left.mean()  if len(left)  else 0.0
                    self.right_val   = right.mean() if len(right) else 0.0

    def predict(self, X):
        mask = X[:, self.feature_idx] < self.threshold
        out  = np.where(mask, self.left_val, self.right_val)
        return out.astype(np.float32)

    def to_dict(self):
        return {
            'fi': int(self.feature_idx),
            'th': float(self.threshold),
            'lv': float(self.left_val),
            'rv': float(self.right_val),
        }

    @classmethod
    def from_dict(cls, d):
        s = cls()
        s.feature_idx = d['fi']
        s.threshold   = d['th']
        s.left_val    = d['lv']
        s.right_val   = d['rv']
        return s


class XGBoostModel:
    """
    Gradient Boosted Tree ensemble for stream/non-stream classification.

    Parameters
    ----------
    n_estimators : int   number of boosting rounds
    learning_rate: float shrinkage parameter
    subsample    : float fraction of samples used per tree
    """

    def __init__(
        self,
        n_estimators: int  = 100,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
    ):
        self.n_estimators  = n_estimators
        self.lr            = learning_rate
        self.subsample     = subsample
        self.trees: list[DecisionStump] = []
        self.init_pred     = 0.0
        self._trained      = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_frac: float = 0.05,
    ):
        """
        Parameters
        ----------
        features   : (H, W, C) or (N, C)
        labels     : (H, W)    or (N,)  binary uint8
        sample_frac: fraction of pixels to use for training
        """
        X, y = self._flatten(features, labels)

        # Down-sample for speed
        rng  = np.random.default_rng(42)
        n    = max(1000, int(len(X) * sample_frac))
        idx  = rng.choice(len(X), size=min(n, len(X)), replace=False)
        X, y = X[idx], y[idx].astype(np.float32)

        print(f"[XGB] Training on {len(X):,} samples, {self.n_estimators} trees …")
        self.init_pred = np.log(y.mean() / (1 - y.mean() + 1e-9) + 1e-9)
        F = np.full(len(X), self.init_pred, dtype=np.float32)

        for t in range(self.n_estimators):
            p = 1.0 / (1.0 + np.exp(-F))              # sigmoid
            g = p - y                                   # gradient
            h = p * (1 - p) + 1e-6                     # hessian

            # Subsample
            sub_idx = rng.choice(len(X), size=int(len(X)*self.subsample), replace=False)
            Xs = X[sub_idx]
            gs = g[sub_idx]
            hs_sub = h[sub_idx]
            residuals = -(gs / hs_sub)

            stump = DecisionStump()
            stump.fit(Xs, residuals)
            F += self.lr * stump.predict(X)
            self.trees.append(stump)

            if (t + 1) % 20 == 0:
                loss = self._log_loss(y, F)
                print(f"  Round {t+1:3d}/{self.n_estimators}  log-loss={loss:.4f}")

        self._trained = True
        print("[XGB] Training complete.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        features : (H, W, C) or (N, C)

        Returns
        -------
        prob : same spatial shape as features[..., 0], float32 [0,1]
        """
        spatial = features.ndim == 3
        if spatial:
            H, W, C = features.shape
            X = features.reshape(-1, C)
        else:
            X = features

        F = np.full(len(X), self.init_pred, dtype=np.float32)
        for stump in self.trees:
            F += self.lr * stump.predict(X)
        prob = (1.0 / (1.0 + np.exp(-F))).astype(np.float32)

        if spatial:
            prob = prob.reshape(H, W)
        return prob

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        data = {
            'init_pred':     float(self.init_pred),
            'lr':            float(self.lr),
            'n_estimators':  self.n_estimators,
            'trees':         [t.to_dict() for t in self.trees],
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"[XGB] Saved to {path}")

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.init_pred = data['init_pred']
        self.lr        = data['lr']
        self.trees     = [DecisionStump.from_dict(d) for d in data['trees']]
        self._trained  = True
        print(f"[XGB] Loaded from {path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten(features, labels):
        if features.ndim == 3:
            H, W, C = features.shape
            X = features.reshape(-1, C)
            y = labels.reshape(-1)
        else:
            X, y = features, labels
        return X.astype(np.float32), y.astype(np.float32)

    @staticmethod
    def _log_loss(y, F):
        p = np.clip(1.0 / (1.0 + np.exp(-F)), 1e-7, 1 - 1e-7)
        return -(y * np.log(p) + (1-y) * np.log(1-p)).mean()
