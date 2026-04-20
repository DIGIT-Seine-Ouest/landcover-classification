"""
Acquisition et chargement des données raster.

- Téléchargement WMS (IGN)
- Découpage en tuiles GeoTIFF géoréférencées (EPSG:2154)
- Chargement d'une tuile en tableau numpy ou image PIL
- Cache local par commune
"""

import glob
import os
from collections.abc import Callable

import numpy as np
import rasterio
import requests
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.windows import Window

from cv_routes.config import (
    City,
    GPSO_GEOJSON_URL,
    INPUT_SIZE,
    WMS_BLOCK_M,
    WMS_CRS,
    WMS_HEIGHT,
    WMS_LAYER,
    WMS_URL,
    WMS_WIDTH,
)
from cv_routes.territory import commune_bbox, grid_blocks, load_commune_polygons

ProgressCallback = Callable[[float, str], None]


def fetch_wms(
    url: str,
    layer: str,
    bbox: tuple[int, int, int, int],
    crs: str,
    width: int,
    height: int,
) -> rasterio.DatasetReader:
    """Télécharge une image WMS et retourne un dataset rasterio géoréférencé en mémoire."""
    xmin, ymin, xmax, ymax = bbox
    params = {
        "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap",
        "LAYERS": layer, "CRS": crs,
        "BBOX": f"{xmin},{ymin},{xmax},{ymax}",
        "WIDTH": width, "HEIGHT": height,
        "FORMAT": "image/tiff", "STYLES": "",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()

    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)
    with MemoryFile(r.content) as mf:
        with mf.open() as tmp:
            data, count, dtype = tmp.read(), tmp.count, tmp.dtypes[0]

    meta = {"driver": "GTiff", "count": count, "dtype": dtype,
            "width": width, "height": height, "crs": crs, "transform": transform}
    buf = MemoryFile()
    with buf.open(**meta) as dst:
        dst.write(data)
    return buf.open()


def tile_raster(
    src: rasterio.DatasetReader,
    output_dir: str,
    tile_size: int = INPUT_SIZE,
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


def load_tile_as_array(tile_path: str) -> tuple[np.ndarray, dict]:
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
    h, w = INPUT_SIZE, INPUT_SIZE
    if data.shape[1] != h or data.shape[2] != w:
        img  = Image.fromarray(np.moveaxis(data.astype(np.uint8), 0, -1))
        data = np.transpose(np.array(img.resize((w, h)), dtype=np.float32), (2, 0, 1))
    return data, meta


def load_tile_as_rgb(tile_path: str) -> Image.Image:
    """Charge une tuile GeoTIFF → image PIL RGB 512×512."""
    arr, _ = load_tile_as_array(tile_path)
    return Image.fromarray(np.moveaxis(arr.astype(np.uint8), 0, -1))


def fetch_city_tiles(
    city: City,
    tiles_dir: str,
    geojson_url: str = GPSO_GEOJSON_URL,
    wms_url: str = WMS_URL,
    wms_layer: str = WMS_LAYER,
    wms_crs: str = WMS_CRS,
    block_m: int = WMS_BLOCK_M,
    on_progress: ProgressCallback | None = None,
) -> list[str]:
    """
    Retourne les tuiles en cache ou les télécharge via WMS IGN.

    Parameters
    ----------
    city        : objet City (attribut .name = nom commune GPSO)
    tiles_dir   : répertoire de cache local
    geojson_url : URL GeoJSON GPSO (EPSG:2154)
    block_m     : taille d'un bloc WMS en mètres (défaut 800 m = 4096 px à 0.20 m/px)
    on_progress : callback optionnel (ratio 0→1, message)
    """
    def _prog(r: float, m: str) -> None:
        if on_progress:
            on_progress(r, m)

    cached = sorted(glob.glob(os.path.join(tiles_dir, "**", "*.tif"), recursive=True))
    if cached:
        _prog(1.0, f"Cache : {len(cached)} tuile(s)")
        return cached

    os.makedirs(tiles_dir, exist_ok=True)
    polygons = load_commune_polygons(geojson_url)
    bbox     = commune_bbox(polygons, city.name)
    blocks   = grid_blocks(bbox, block_m)

    all_tiles: list[str] = []
    tile_offset = 0
    n = len(blocks)

    for i, (bx_min, by_min, bx_max, by_max) in enumerate(blocks):
        _prog(i / n, f"Bloc {i + 1}/{n} — {city.name}…")
        px_w = min(max(1, round((bx_max - bx_min) / 0.20)), 4096)
        px_h = min(max(1, round((by_max - by_min) / 0.20)), 4096)

        src = fetch_wms(
            wms_url, wms_layer,
            (int(bx_min), int(by_min), int(bx_max), int(by_max)),
            wms_crs, px_w, px_h,
        )
        block_tiles = tile_raster(
            src,
            os.path.join(tiles_dir, f"block_{i:04d}"),
            tile_offset=tile_offset,
        )
        src.close()
        all_tiles.extend(block_tiles)
        tile_offset += len(block_tiles)

    _prog(1.0, f"{len(all_tiles)} tuiles prêtes — {city.name}")
    return all_tiles
