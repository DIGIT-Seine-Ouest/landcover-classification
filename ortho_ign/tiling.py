"""Découpage en tuiles GeoTIFF et chargement de tuiles."""

import os

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window

TILE_SIZE = 512


def tile_raster(
    src: rasterio.DatasetReader,
    output_dir: str,
    tile_size: int = TILE_SIZE,
    tile_offset: int = 0,
) -> list[str]:
    """Découpe un dataset rasterio en tuiles GeoTIFF carrées."""
    os.makedirs(output_dir, exist_ok=True)
    meta = src.meta.copy()
    meta.update(driver="GTiff", width=tile_size, height=tile_size)
    saved = []

    for local_idx, (row, col) in enumerate(
        (r, c)
        for r in range(0, src.height, tile_size)
        for c in range(0, src.width, tile_size)
    ):
        window = Window(col, row, tile_size, tile_size)
        data   = src.read(window=window)
        if data.shape[1] < tile_size or data.shape[2] < tile_size:
            continue
        meta.update(transform=src.window_transform(window), count=data.shape[0])
        out_path = os.path.join(output_dir, f"tile_{tile_offset + local_idx:05d}.tif")
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(data)
        saved.append(out_path)

    return saved


def load_tile_as_array(tile_path: str, size: int = TILE_SIZE) -> tuple[np.ndarray, dict]:
    """
    Charge une tuile GeoTIFF.

    Returns
    -------
    arr255 : (3, H, W) float32 dans [0, 255]
    meta   : dict rasterio (crs, transform, …)
    """
    with rasterio.open(tile_path) as src:
        data = src.read([1, 2, 3]).astype(np.float32)
        meta = src.meta.copy()
    if data.shape[1] != size or data.shape[2] != size:
        img  = Image.fromarray(np.moveaxis(data.astype(np.uint8), 0, -1))
        data = np.transpose(np.array(img.resize((size, size)), dtype=np.float32), (2, 0, 1))
    return data, meta


def load_tile_as_rgb(tile_path: str, size: int = TILE_SIZE) -> Image.Image:
    """Charge une tuile GeoTIFF → image PIL RGB."""
    arr, _ = load_tile_as_array(tile_path, size)
    return Image.fromarray(np.moveaxis(arr.astype(np.uint8), 0, -1))
