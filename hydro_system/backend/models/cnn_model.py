"""
CNN Model for Drainage Network Extraction
==========================================
A lightweight UNet-style CNN operating on DEM feature patches.
"""

import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Pure-NumPy UNet implementation (no TensorFlow/PyTorch dependency at import)
# ---------------------------------------------------------------------------

class ConvLayer:
    """2-D convolution + ReLU, implemented in NumPy (inference only)."""

    def __init__(self, in_ch, out_ch, kernel=3):
        self.in_ch  = in_ch
        self.out_ch = out_ch
        self.k      = kernel
        rng = np.random.default_rng(42 + in_ch * 100 + out_ch)
        fan_in = in_ch * kernel * kernel
        self.W = rng.standard_normal((out_ch, in_ch, kernel, kernel)).astype(
            np.float32
        ) * np.sqrt(2.0 / fan_in)
        self.b = np.zeros(out_ch, dtype=np.float32)

    def forward(self, x):
        """x: (H, W, C)  →  (H, W, out_ch)."""
        H, W, _ = x.shape
        pad = self.k // 2
        x_p = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
        out = np.zeros((H, W, self.out_ch), dtype=np.float32)
        for oc in range(self.out_ch):
            for ic in range(self.in_ch):
                w = self.W[oc, ic]                    # (k, k)
                for ki in range(self.k):
                    for kj in range(self.k):
                        out[:, :, oc] += (
                            x_p[ki:ki+H, kj:kj+W, ic] * w[ki, kj]
                        )
            out[:, :, oc] += self.b[oc]
        return np.maximum(out, 0)                     # ReLU


class SimpleCNN:
    """
    Lightweight 3-layer CNN for DEM patch classification.

    Architecture:
        Conv(13→32) → Conv(32→64) → Conv(64→1) → Sigmoid

    Input  : (H, W, 13) feature map
    Output : (H, W, 1)  stream probability
    """

    ARCH = [(13, 32, 3), (32, 64, 3), (64, 1, 1)]

    def __init__(self):
        self.layers = [ConvLayer(ic, oc, k) for ic, oc, k in self.ARCH]
        self._trained = False

    # ------------------------------------------------------------------
    # Training (gradient-free approximation via random initialisation)
    # ------------------------------------------------------------------

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        epochs: int = 3,
        patch_size: int = 32,
        n_patches: int = 200,
    ):
        """
        Train the CNN using patch-wise binary cross-entropy minimisation
        via a simple evolutionary / perturbation strategy suitable for
        pure-NumPy environments.

        In production, replace with TF/PyTorch backprop.
        """
        print(f"[CNN] Training on {n_patches} patches × {epochs} epochs …")
        best_loss = np.inf
        best_weights = self._get_weights()

        rng = np.random.default_rng(0)
        H, W, _ = features.shape

        for ep in range(epochs):
            loss = self._eval_loss(features, labels, rng, patch_size, n_patches)
            print(f"  Epoch {ep+1}/{epochs}  loss={loss:.4f}")

            if loss < best_loss:
                best_loss  = loss
                best_weights = self._get_weights()

            # Perturb weights slightly
            self._perturb_weights(rng, scale=0.01)

        # Restore best weights
        self._set_weights(best_weights)
        self._trained = True
        print(f"[CNN] Training complete.  Best loss={best_loss:.4f}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        features : (H, W, 13) float32

        Returns
        -------
        prob : (H, W) float32  stream probability [0,1]
        """
        x = features
        for layer in self.layers[:-1]:
            x = layer.forward(x)
        # Final layer (no ReLU — apply sigmoid)
        last = self.layers[-1]
        x_p = np.pad(x, ((0,0),(0,0),(0,0)), mode='reflect')
        H, W, _ = x.shape
        oc_w = last.W[0, :, 0, 0]                    # (64,) for 1×1 conv
        logit = (x * oc_w).sum(axis=-1) + last.b[0]  # (H, W)
        prob = 1.0 / (1.0 + np.exp(-logit))
        return prob.astype(np.float32)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        weights = []
        for layer in self.layers:
            weights.append({'W': layer.W.tolist(), 'b': layer.b.tolist()})
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump({'arch': self.ARCH, 'weights': weights}, f)
        print(f"[CNN] Saved to {path}")

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        for i, (w_dict, layer) in enumerate(zip(data['weights'], self.layers)):
            layer.W = np.array(w_dict['W'], dtype=np.float32)
            layer.b = np.array(w_dict['b'], dtype=np.float32)
        self._trained = True
        print(f"[CNN] Loaded from {path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _eval_loss(self, feat, labels, rng, ps, n):
        H, W, _ = feat.shape
        total = 0.0
        for _ in range(n):
            r = rng.integers(0, H - ps)
            c = rng.integers(0, W - ps)
            patch_f = feat[r:r+ps, c:c+ps, :]
            patch_l = labels[r:r+ps, c:c+ps].astype(np.float32)
            prob = self.predict(patch_f)
            prob  = np.clip(prob, 1e-7, 1 - 1e-7)
            bce  = -(patch_l * np.log(prob) + (1-patch_l) * np.log(1-prob))
            total += bce.mean()
        return total / n

    def _get_weights(self):
        return [(l.W.copy(), l.b.copy()) for l in self.layers]

    def _set_weights(self, weights):
        for layer, (W, b) in zip(self.layers, weights):
            layer.W = W.copy()
            layer.b = b.copy()

    def _perturb_weights(self, rng, scale=0.01):
        for layer in self.layers:
            layer.W += rng.standard_normal(layer.W.shape).astype(np.float32) * scale
            layer.b += rng.standard_normal(layer.b.shape).astype(np.float32) * scale
