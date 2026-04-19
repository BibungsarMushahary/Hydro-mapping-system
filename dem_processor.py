"""
DEM Processing and Hydrological Feature Extraction
===================================================
Extracts terrain and hydrological features from Digital Elevation Models (DEMs).
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import uniform_filter, generic_filter
import warnings
warnings.filterwarnings('ignore')


class DEMProcessor:
    """
    Processes Digital Elevation Models to extract hydrological and terrain features.
    
    Features extracted:
    - Elevation (normalized)
    - Slope
    - Aspect
    - Plan Curvature
    - Profile Curvature
    - Texture Mean / Variance
    - Flow Accumulation (D8)
    - Topographic Wetness Index (TWI)
    - Stream Power Index (SPI)
    - Distance to Ridge
    - Normalized coordinates (X, Y)
    """

    def __init__(self, cell_size: float = 30.0):
        self.cell_size = cell_size  # meters per pixel

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_all_features(self, dem: np.ndarray) -> np.ndarray:
        """
        Extract all features and return a (H, W, C) array.

        Parameters
        ----------
        dem : np.ndarray  shape (H, W), float32/float64

        Returns
        -------
        features : np.ndarray  shape (H, W, 13)
        """
        dem = dem.astype(np.float32)
        dem = self._fill_nodata(dem)

        elev_norm   = self._normalize(dem)
        slope       = self._compute_slope(dem)
        aspect      = self._compute_aspect(dem)
        plan_curv   = self._compute_plan_curvature(dem)
        prof_curv   = self._compute_profile_curvature(dem)
        tex_mean    = self._texture_mean(dem)
        tex_var     = self._texture_variance(dem)
        flow_acc    = self._d8_flow_accumulation(dem)
        flow_acc_n  = self._normalize(np.log1p(flow_acc))
        twi         = self._compute_twi(slope, flow_acc)
        spi         = self._compute_spi(slope, flow_acc)
        ridge_dist  = self._distance_to_ridge(dem)
        y_coord, x_coord = self._spatial_coords(dem)

        stack = np.stack([
            elev_norm,
            self._normalize(slope),
            self._normalize(aspect),
            self._normalize(plan_curv),
            self._normalize(prof_curv),
            self._normalize(tex_mean),
            self._normalize(tex_var),
            flow_acc_n,
            self._normalize(twi),
            self._normalize(spi),
            self._normalize(ridge_dist),
            y_coord,
            x_coord,
        ], axis=-1)

        # Replace any remaining NaN / Inf
        stack = np.nan_to_num(stack, nan=0.0, posinf=1.0, neginf=0.0)
        return stack.astype(np.float32)

    def generate_stream_labels(
        self,
        dem: np.ndarray,
        flow_threshold: int = 1000,
    ) -> np.ndarray:
        """
        Auto-generate binary stream/non-stream labels via D8 flow accumulation.

        Parameters
        ----------
        dem            : np.ndarray (H, W)
        flow_threshold : pixels contributing to a cell before it is classified
                         as a stream channel

        Returns
        -------
        labels : np.ndarray (H, W)  dtype uint8  {0=land, 1=stream}
        """
        dem = self._fill_nodata(dem.astype(np.float32))
        flow_acc = self._d8_flow_accumulation(dem)
        labels = (flow_acc >= flow_threshold).astype(np.uint8)
        return labels

    def compute_flood_risk(
        self,
        dem: np.ndarray,
        stream_prob: np.ndarray,
        rainfall_mm: float = 50.0,
    ) -> np.ndarray:
        """
        Compute a continuous flood risk index [0, 1].

        Risk is driven by:
        - Proximity to / density of drainage network (stream probability)
        - Low relative elevation
        - High flow accumulation
        - Rainfall intensity scalar

        Parameters
        ----------
        dem          : (H, W) elevation array
        stream_prob  : (H, W) float32 [0, 1] from ML model
        rainfall_mm  : scalar rainfall depth in mm

        Returns
        -------
        risk_index : (H, W) float32 [0, 1]
        """
        dem = self._fill_nodata(dem.astype(np.float32))

        # Invert normalized elevation — low areas are high risk
        low_elev = 1.0 - self._normalize(dem)

        # Flow accumulation
        flow_acc = self._d8_flow_accumulation(dem)
        flow_n   = self._normalize(np.log1p(flow_acc))

        # Proximity to drainage (Gaussian blur of stream probability)
        proximity = ndimage.gaussian_filter(stream_prob.astype(np.float32), sigma=3)
        proximity = self._normalize(proximity)

        # Rainfall factor (logistic scaling)
        rain_factor = 1.0 / (1.0 + np.exp(-0.05 * (rainfall_mm - 50)))

        # Weighted combination
        risk = (
            0.35 * low_elev
            + 0.30 * proximity
            + 0.20 * flow_n
            + 0.15 * stream_prob
        ) * rain_factor

        risk = np.clip(self._normalize(risk), 0.0, 1.0)
        return risk.astype(np.float32)

    # ------------------------------------------------------------------
    # Terrain derivatives
    # ------------------------------------------------------------------

    def _compute_slope(self, dem: np.ndarray) -> np.ndarray:
        """Slope in degrees using finite differences."""
        dz_dy, dz_dx = np.gradient(dem, self.cell_size)
        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        return np.degrees(slope_rad).astype(np.float32)

    def _compute_aspect(self, dem: np.ndarray) -> np.ndarray:
        """Aspect in degrees [0, 360]."""
        dz_dy, dz_dx = np.gradient(dem, self.cell_size)
        aspect = np.degrees(np.arctan2(-dz_dy, dz_dx)) % 360
        return aspect.astype(np.float32)

    def _compute_plan_curvature(self, dem: np.ndarray) -> np.ndarray:
        """Plan (contour) curvature — negative=convergent, positive=divergent."""
        return self._second_derivative_curvature(dem, kind='plan')

    def _compute_profile_curvature(self, dem: np.ndarray) -> np.ndarray:
        """Profile curvature along the steepest descent direction."""
        return self._second_derivative_curvature(dem, kind='profile')

    def _second_derivative_curvature(
        self, dem: np.ndarray, kind: str = 'plan'
    ) -> np.ndarray:
        cs = self.cell_size
        dz_dy, dz_dx = np.gradient(dem, cs)
        d2z_dx2 = np.gradient(dz_dx, cs)[1]
        d2z_dy2 = np.gradient(dz_dy, cs)[0]
        d2z_dxdy = np.gradient(dz_dx, cs)[0]

        p = dz_dx**2
        q = dz_dy**2
        denom = (p + q + 1e-10)

        if kind == 'plan':
            curv = -(
                d2z_dx2 * q - 2 * d2z_dxdy * dz_dx * dz_dy + d2z_dy2 * p
            ) / (denom * np.sqrt(denom))
        else:  # profile
            curv = -(
                d2z_dx2 * p + 2 * d2z_dxdy * dz_dx * dz_dy + d2z_dy2 * q
            ) / (denom * np.sqrt(denom))

        return np.clip(curv, -5, 5).astype(np.float32)

    # ------------------------------------------------------------------
    # Texture features
    # ------------------------------------------------------------------

    def _texture_mean(self, dem: np.ndarray, size: int = 5) -> np.ndarray:
        return uniform_filter(dem, size=size).astype(np.float32)

    def _texture_variance(self, dem: np.ndarray, size: int = 5) -> np.ndarray:
        mean  = uniform_filter(dem, size=size)
        mean2 = uniform_filter(dem**2, size=size)
        var   = np.maximum(mean2 - mean**2, 0)
        return var.astype(np.float32)

    # ------------------------------------------------------------------
    # Hydrological indices
    # ------------------------------------------------------------------

    def _d8_flow_accumulation(self, dem: np.ndarray) -> np.ndarray:
        """
        D8 single-flow-direction flow accumulation.
        Returns number of upstream cells draining into each cell.
        """
        rows, cols = dem.shape
        # 8 neighbors: E, NE, N, NW, W, SW, S, SE
        dy = [0, -1, -1, -1,  0,  1, 1,  1]
        dx = [1,  1,  0, -1, -1, -1, 0,  1]
        # diagonal distance correction
        dist = [1, np.sqrt(2), 1, np.sqrt(2), 1, np.sqrt(2), 1, np.sqrt(2)]

        # Compute steepest descent direction for each cell
        flow_dir = np.full((rows, cols), -1, dtype=np.int8)
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                max_drop = -np.inf
                best_dir = -1
                for d in range(8):
                    nr, nc = r + dy[d], c + dx[d]
                    drop = (dem[r, c] - dem[nr, nc]) / dist[d]
                    if drop > max_drop:
                        max_drop = drop
                        best_dir = d
                flow_dir[r, c] = best_dir

        # Accumulate flow (iterative topological sort approximation)
        acc = np.ones((rows, cols), dtype=np.float32)
        # Sort cells by decreasing elevation
        flat_idx = np.argsort(dem.ravel())[::-1]
        for idx in flat_idx:
            r, c = divmod(int(idx), cols)
            d = flow_dir[r, c]
            if d >= 0:
                nr, nc = r + dy[d], c + dx[d]
                if 0 <= nr < rows and 0 <= nc < cols:
                    acc[nr, nc] += acc[r, c]

        return acc

    def _compute_twi(
        self, slope_deg: np.ndarray, flow_acc: np.ndarray
    ) -> np.ndarray:
        """Topographic Wetness Index: ln(a / tan(β)), a = specific catchment area."""
        slope_rad = np.radians(np.maximum(slope_deg, 0.001))
        sca = flow_acc * self.cell_size          # specific catchment area
        twi = np.log(sca / np.tan(slope_rad) + 1e-10)
        return twi.astype(np.float32)

    def _compute_spi(
        self, slope_deg: np.ndarray, flow_acc: np.ndarray
    ) -> np.ndarray:
        """Stream Power Index: a × tan(β)."""
        slope_rad = np.radians(np.maximum(slope_deg, 0.001))
        sca = flow_acc * self.cell_size
        spi = sca * np.tan(slope_rad)
        return np.log1p(spi).astype(np.float32)

    def _distance_to_ridge(self, dem: np.ndarray) -> np.ndarray:
        """
        Approximate ridge distance: distance from each cell to the nearest
        local maximum (ridge/peak).
        """
        # Local maxima = cells higher than all 8 neighbours
        max_filter = ndimage.maximum_filter(dem, size=5)
        ridges = (dem == max_filter).astype(np.uint8)
        # Distance transform
        dist = ndimage.distance_transform_edt(1 - ridges)
        return dist.astype(np.float32)

    # ------------------------------------------------------------------
    # Spatial coordinates
    # ------------------------------------------------------------------

    @staticmethod
    def _spatial_coords(dem: np.ndarray):
        rows, cols = dem.shape
        y = np.linspace(0, 1, rows, dtype=np.float32)
        x = np.linspace(0, 1, cols, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        return yy, xx

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - mn) / (mx - mn)).astype(np.float32)

    @staticmethod
    def _fill_nodata(dem: np.ndarray, nodata: float = -9999.0) -> np.ndarray:
        """Replace nodata values with local mean."""
        mask = (dem == nodata) | np.isnan(dem) | np.isinf(dem)
        if mask.any():
            dem = dem.copy()
            dem[mask] = np.nanmean(dem[~mask]) if (~mask).any() else 0.0
        return dem
