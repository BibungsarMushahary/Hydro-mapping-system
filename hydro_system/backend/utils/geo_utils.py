"""
Geospatial I/O and Visualisation Utilities
==========================================
Handles reading/writing GeoTIFFs and generating map-ready images.
"""

import numpy as np
import os
import json
import struct
import zlib
import warnings
warnings.filterwarnings('ignore')

try:
    import rasterio
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ---------------------------------------------------------------------------
# DEM I/O
# ---------------------------------------------------------------------------

def read_dem(path: str) -> tuple[np.ndarray, dict]:
    """
    Read a GeoTIFF DEM file.

    Returns
    -------
    dem      : (H, W) float32
    meta     : dict with keys 'transform', 'crs', 'width', 'height',
                               'bounds'  (minx, miny, maxx, maxy)
    """
    if HAS_RASTERIO:
        return _read_dem_rasterio(path)
    else:
        return _read_dem_fallback(path)


def write_geotiff(
    path: str,
    array: np.ndarray,
    meta: dict,
    dtype: str = 'float32',
    nodata: float = -9999.0,
):
    """
    Write a (H, W) array as a single-band GeoTIFF.
    Falls back to PNG if rasterio is unavailable.
    """
    if HAS_RASTERIO:
        _write_geotiff_rasterio(path, array, meta, dtype, nodata)
    else:
        _write_png_fallback(path.replace('.tif', '.png'), array)


def write_png(path: str, array: np.ndarray):
    """Write a normalised (H, W) float array as a PNG."""
    _write_png_fallback(path, array)


# ---------------------------------------------------------------------------
# Hillshade
# ---------------------------------------------------------------------------

def generate_hillshade(
    dem: np.ndarray,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    cell_size: float = 30.0,
    z_factor: float  = 1.0,
) -> np.ndarray:
    """
    Generate a hillshade image from a DEM.

    Returns
    -------
    hs : (H, W) uint8  [0, 255]
    """
    az_rad  = np.radians(360.0 - azimuth + 90)
    alt_rad = np.radians(altitude)

    dz_dy, dz_dx = np.gradient(dem * z_factor, cell_size)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    aspect = np.arctan2(-dz_dy, dz_dx)

    hs = (
        np.sin(alt_rad) * np.sin(slope)
        + np.cos(alt_rad) * np.cos(slope) * np.cos(az_rad - aspect)
    )
    hs = np.clip(hs, 0, 1)
    hs_uint8 = (hs * 255).astype(np.uint8)
    return hs_uint8


# ---------------------------------------------------------------------------
# Colourmap helpers
# ---------------------------------------------------------------------------

def apply_colormap(array: np.ndarray, cmap: str = 'viridis') -> np.ndarray:
    """
    Apply a matplotlib-compatible colourmap to a [0,1] normalised array.

    Returns
    -------
    rgba : (H, W, 4) uint8
    """
    arr = np.clip(array, 0, 1)
    if cmap == 'viridis':
        lut = _viridis_lut()
    elif cmap == 'risk':
        lut = _risk_lut()
    elif cmap == 'stream':
        lut = _stream_lut()
    else:
        lut = _viridis_lut()

    idx  = (arr * (len(lut) - 1)).astype(int)
    rgba = np.array(lut, dtype=np.uint8)[idx]      # (H, W, 4)
    return rgba


def array_to_png_bytes(rgba: np.ndarray) -> bytes:
    """Encode (H, W, 4) uint8 RGBA to PNG bytes without PIL."""
    if HAS_PIL:
        from io import BytesIO
        img = Image.fromarray(rgba, mode='RGBA')
        buf = BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()
    else:
        return _encode_png(rgba)


# ---------------------------------------------------------------------------
# Rasterio-backed implementations
# ---------------------------------------------------------------------------

def _read_dem_rasterio(path):
    import rasterio
    with rasterio.open(path) as src:
        dem = src.read(1).astype(np.float32)
        meta = {
            'transform': src.transform,
            'crs'      : src.crs,
            'width'    : src.width,
            'height'   : src.height,
            'bounds'   : src.bounds,
        }
    return dem, meta


def _write_geotiff_rasterio(path, array, meta, dtype, nodata):
    import rasterio
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(
        path, 'w',
        driver='GTiff',
        height=array.shape[0], width=array.shape[1],
        count=1,
        dtype=dtype,
        crs=meta.get('crs'),
        transform=meta.get('transform'),
        nodata=nodata,
    ) as dst:
        dst.write(array.astype(dtype), 1)


# ---------------------------------------------------------------------------
# Fallback implementations (no rasterio)
# ---------------------------------------------------------------------------

def _read_dem_fallback(path: str):
    """
    Try to read a GeoTIFF by parsing TIFF tags manually.
    Supports basic uncompressed / LZW-compressed single-band float/int GeoTIFFs.
    Falls back to generating a synthetic DEM if parsing fails.
    """
    try:
        dem = _parse_tiff(path)
        meta = _synthetic_meta(dem.shape)
        return dem, meta
    except Exception as e:
        print(f"[IO] Could not parse TIFF ({e}), generating synthetic DEM …")
        dem  = _synthetic_dem(256, 256)
        meta = _synthetic_meta(dem.shape)
        return dem, meta


def _parse_tiff(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        raw = f.read()

    # Byte order
    endian = '<' if raw[:2] == b'II' else '>'
    magic = struct.unpack_from(endian + 'H', raw, 2)[0]
    if magic not in (42, 43):
        raise ValueError("Not a TIFF")

    ifd_offset = struct.unpack_from(endian + 'I', raw, 4)[0]

    tags = {}
    n_entries = struct.unpack_from(endian + 'H', raw, ifd_offset)[0]
    for i in range(n_entries):
        base = ifd_offset + 2 + i * 12
        tag, typ, count = struct.unpack_from(endian + 'HHI', raw, base)
        val_off = base + 8
        tags[tag] = (typ, count, val_off)

    def tag_value(t):
        if t not in tags:
            return None
        typ, count, off = tags[t]
        fmt = {1:'B',2:'s',3:'H',4:'I',5:'II',12:'d',11:'f'}.get(typ,'I')
        if fmt == 's':
            return raw[off:off+count].decode('ascii','replace')
        size = struct.calcsize(endian + fmt)
        if count == 1:
            return struct.unpack_from(endian + fmt, raw, off)[0]
        return [struct.unpack_from(endian + fmt, raw, off + k*size)[0]
                for k in range(count)]

    width  = tag_value(256) or 256
    height = tag_value(257) or 256
    spp    = tag_value(277) or 1
    bps    = tag_value(258) or 32
    sample_fmt = tag_value(339) or 1   # 1=uint, 2=int, 3=float
    compression = tag_value(259) or 1

    strips = tag_value(273)
    if isinstance(strips, int):
        strips = [strips]
    strip_bytes = tag_value(279)
    if isinstance(strip_bytes, int):
        strip_bytes = [strip_bytes]

    pixel_bytes = bps // 8
    fmt_char = {(32, 3): 'f', (64, 3): 'd',
                (16, 1): 'H', (16, 2): 'h',
                (32, 1): 'I', (32, 2): 'i'}.get((bps, sample_fmt), 'f')

    data = bytearray()
    for offset, nbytes in zip(strips, strip_bytes):
        chunk = raw[offset:offset+nbytes]
        if compression == 5:   # LZW - approximate
            try:
                chunk = _lzw_decompress(chunk)
            except Exception:
                pass
        data.extend(chunk)

    n_pixels = width * height * spp
    arr = np.frombuffer(bytes(data[:n_pixels * pixel_bytes]),
                        dtype=np.dtype(endian + fmt_char))
    if len(arr) < n_pixels:
        arr = np.pad(arr, (0, n_pixels - len(arr)))
    return arr[:height*width].reshape(height, width).astype(np.float32)


def _lzw_decompress(data: bytes) -> bytes:
    """Minimal LZW decompression (TIFF variant)."""
    out = bytearray()
    table = {i: bytes([i]) for i in range(256)}
    table[256] = b''   # clear
    table[257] = b''   # EOI
    next_code = 258
    code_size = 9
    buf = 0
    bits = 0
    prev = None
    idx = 0

    def read_code():
        nonlocal buf, bits, idx
        while bits < code_size and idx < len(data):
            buf = (buf << 8) | data[idx]
            bits += 8
            idx += 1
        if bits < code_size:
            return None
        bits -= code_size
        return (buf >> bits) & ((1 << code_size) - 1)

    while True:
        code = read_code()
        if code is None or code == 257:
            break
        if code == 256:
            table = {i: bytes([i]) for i in range(256)}
            table[256] = b''
            table[257] = b''
            next_code = 258
            code_size = 9
            prev = None
            continue
        if code in table:
            entry = table[code]
        elif code == next_code and prev is not None:
            entry = table[prev] + table[prev][:1]
        else:
            break
        out.extend(entry)
        if prev is not None:
            new_entry = table[prev] + entry[:1]
            table[next_code] = new_entry
            next_code += 1
            if next_code >= (1 << code_size) and code_size < 12:
                code_size += 1
        prev = code
    return bytes(out)


def _synthetic_dem(rows: int = 256, cols: int = 256) -> np.ndarray:
    """Generate a realistic-looking synthetic DEM using fractional Brownian motion."""
    rng = np.random.default_rng(42)
    dem = np.zeros((rows, cols), dtype=np.float32)
    for octave in range(6):
        scale  = 2 ** octave
        amp    = 100.0 / (scale + 1)
        sr, sc = max(rows // scale, 2), max(cols // scale, 2)
        noise  = rng.standard_normal((sr, sc)).astype(np.float32)
        # Upsample (nearest)
        yr = np.linspace(0, sr - 1, rows)
        xr = np.linspace(0, sc - 1, cols)
        ri = yr.astype(int).clip(0, sr - 1)
        ci = xr.astype(int).clip(0, sc - 1)
        dem += amp * noise[np.ix_(ri, ci)]

    # Add a valley
    y = np.linspace(-1, 1, rows)
    x = np.linspace(-1, 1, cols)
    xx, yy = np.meshgrid(x, y)
    dem -= 80 * np.exp(-(yy**2 + (xx * 0.3)**2))
    dem = dem - dem.min()
    return dem


def _synthetic_meta(shape):
    rows, cols = shape
    return {
        'transform': None,
        'crs'      : None,
        'width'    : cols,
        'height'   : rows,
        'bounds'   : (0.0, 0.0, cols * 30.0, rows * 30.0),
    }


def _write_png_fallback(path: str, array: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if HAS_PIL:
        mn, mx = array.min(), array.max()
        if mx - mn > 1e-10:
            norm = ((array - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            norm = np.zeros_like(array, dtype=np.uint8)
        Image.fromarray(norm).save(path)
    else:
        mn, mx = array.min(), array.max()
        if mx - mn > 1e-10:
            norm = ((array - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            norm = np.zeros_like(array, dtype=np.uint8)
        rgba = np.stack([norm, norm, norm,
                         np.full_like(norm, 255)], axis=-1)
        png_bytes = _encode_png(rgba)
        with open(path, 'wb') as f:
            f.write(png_bytes)


# ---------------------------------------------------------------------------
# Pure-Python PNG encoder
# ---------------------------------------------------------------------------

def _encode_png(rgba: np.ndarray) -> bytes:
    """Minimal PNG encoder for RGBA uint8 arrays."""
    H, W, _ = rgba.shape

    def chunk(name: bytes, data: bytes) -> bytes:
        c = name + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xFFFFFFFF)

    sig = b'\x89PNG\r\n\x1a\n'

    ihdr_data = struct.pack('>IIBBBBB', W, H, 8, 6, 0, 0, 0)
    ihdr = chunk(b'IHDR', ihdr_data)

    raw_rows = bytearray()
    for r in range(H):
        raw_rows.append(0)             # filter type None
        raw_rows.extend(rgba[r].tobytes())

    idat = chunk(b'IDAT', zlib.compress(bytes(raw_rows), 9))
    iend = chunk(b'IEND', b'')

    return sig + ihdr + idat + iend


# ---------------------------------------------------------------------------
# Colour lookup tables
# ---------------------------------------------------------------------------

def _viridis_lut():
    colours = [
        (68,1,84,255),(72,27,109,255),(67,62,133,255),(56,89,140,255),
        (45,112,142,255),(37,133,142,255),(30,155,138,255),(42,176,127,255),
        (82,196,105,255),(134,213,73,255),(194,223,35,255),(253,231,37,255),
    ]
    n = 256
    lut = []
    for i in range(n):
        t   = i / (n - 1) * (len(colours) - 1)
        lo  = int(t)
        hi  = min(lo + 1, len(colours) - 1)
        f   = t - lo
        r = int(colours[lo][0] + f * (colours[hi][0] - colours[lo][0]))
        g = int(colours[lo][1] + f * (colours[hi][1] - colours[lo][1]))
        b = int(colours[lo][2] + f * (colours[hi][2] - colours[lo][2]))
        lut.append((r, g, b, 200))
    return lut


def _risk_lut():
    """Green → Yellow → Red risk gradient."""
    n = 256
    lut = []
    for i in range(n):
        t = i / (n - 1)
        if t < 0.5:
            r = int(t * 2 * 255)
            g = 200
            b = 50
        else:
            r = 220
            g = int((1 - (t - 0.5) * 2) * 200)
            b = 30
        lut.append((r, g, b, 180))
    return lut


def _stream_lut():
    """Transparent → Blue stream probability."""
    n = 256
    lut = []
    for i in range(n):
        t = i / (n - 1)
        r = int(30  + t * 10)
        g = int(100 + t * 50)
        b = int(180 + t * 75)
        a = int(t * 220)
        lut.append((r, g, b, a))
    return lut
