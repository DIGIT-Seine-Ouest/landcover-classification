"""Récupération des tuiles WMS pour une commune (avec cache local)."""

import glob
import math
import os
from collections.abc import Callable

from ortho_ign.territory import load_commune_polygons, commune_bbox, grid_blocks
from ortho_ign.wms import fetch_wms, IGN_WMS_URL, IGN_WMS_LAYER, IGN_WMS_CRS, IGN_BLOCK_M
from ortho_ign.tiling import tile_raster, TILE_SIZE

ProgressCallback = Callable[[float, str], None]


def fetch_city_tiles(
    commune_name: str,
    tiles_dir: str,
    geojson_url: str,
    wms_url: str = IGN_WMS_URL,
    wms_layer: str = IGN_WMS_LAYER,
    wms_crs: str = IGN_WMS_CRS,
    block_m: int = IGN_BLOCK_M,
    on_progress: ProgressCallback | None = None,
) -> list[str]:
    """
    Retourne les tuiles en cache ou les télécharge via WMS.

    Parameters
    ----------
    commune_name : nom de la commune (doit correspondre au GeoJSON)
    tiles_dir    : répertoire de cache local
    geojson_url  : URL du GeoJSON territoire (EPSG:2154)
    block_m      : taille d'un bloc WMS en mètres

    Returns
    -------
    Liste de chemins vers les tuiles .tif
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
    bbox     = commune_bbox(polygons, commune_name)
    blocks   = grid_blocks(bbox, block_m)

    all_tiles: list[str] = []
    tile_offset = 0
    n = len(blocks)

    for i, (bx_min, by_min, bx_max, by_max) in enumerate(blocks):
        _prog(i / n, f"Bloc {i + 1}/{n} — {commune_name}…")
        bxi_min, byi_min = int(bx_min), int(by_min)
        bxi_max, byi_max = int(bx_max), int(by_max)
        # Arrondi au multiple supérieur de TILE_SIZE pour que chaque bloc soit
        # exactement divisible en tuiles 512×512 sans pixel de bord orphelin.
        px_w = min(math.ceil(round((bxi_max - bxi_min) / 0.20) / TILE_SIZE) * TILE_SIZE, 4096)
        px_h = min(math.ceil(round((byi_max - byi_min) / 0.20) / TILE_SIZE) * TILE_SIZE, 4096)

        src = fetch_wms(
            wms_url, wms_layer,
            (bxi_min, byi_min, bxi_max, byi_max),
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

    _prog(1.0, f"{len(all_tiles)} tuiles prêtes — {commune_name}")
    return all_tiles
