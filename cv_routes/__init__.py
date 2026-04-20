"""
cv_routes — Segmentation sémantique de routes sur ortho-photos IGN (FLAIR-INC).

API publique
------------
>>> from cv_routes import FlairModel, run_on_image, run_on_tile, run_on_city, fetch_tiles

Exemple notebook
----------------
>>> model = FlairModel("models/flair_12cl_resnet34_unet.onnx")
>>>
>>> # Sur une image PIL
>>> result = run_on_image(model, pil_image)
>>> result["road_pct"]        # % surface route
>>> result["flair_map"]       # PIL.Image — segmentation colorisée
>>> result["road_overlay"]    # PIL.Image — routes (orange) sur ortho
>>>
>>> # Sur une tuile GeoTIFF (avec géoréférencement)
>>> result = run_on_tile(model, "tile_00001.tif")
>>>
>>> # Commune entière — fetch WMS + inférence + export
>>> tiles = fetch_tiles("MEUDON", "data/tiles/MEUDON")
>>> city_result = run_on_city(model, "MEUDON", tiles, export_dir="data/exports/MEUDON")
>>> city_result["road_pct"]   # moyenne sur toutes les tuiles
>>> city_result["gpkg"]       # chemin GeoPackage vecteur
"""

from cv_routes.inference import FlairModel
from cv_routes.pipeline import (
    fetch_tiles,
    run_and_export_tile,
    run_on_city,
    run_on_image,
    run_on_tile,
)
from cv_routes.visualization import (
    apply_overlay,
    flair_colormap,
    flair_stats,
    road_mask_from_flair,
)
from cv_routes.export import mask_to_geotiff, masks_to_gpkg
from cv_routes.download import download_model
from cv_routes.config import CITIES, FLAIR_CLASS_NAMES

__version__ = "0.1.0"
__all__ = [
    "FlairModel",
    "download_model",
    "run_on_image",
    "run_on_tile",
    "run_and_export_tile",
    "run_on_city",
    "fetch_tiles",
    "apply_overlay",
    "flair_colormap",
    "flair_stats",
    "road_mask_from_flair",
    "mask_to_geotiff",
    "masks_to_gpkg",
    "CITIES",
    "FLAIR_CLASS_NAMES",
]
