"""
Pipeline haut niveau — API publique pour notebooks.

Usage typique
-------------
>>> from cv_routes import FlairModel, run_on_tile, run_on_city, fetch_tiles
>>> model = FlairModel("models/flair_12cl_resnet34_unet.onnx")
>>>
>>> # Image locale
>>> result = run_on_image(model, pil_image)
>>>
>>> # Tuile GeoTIFF
>>> result = run_on_tile(model, "data/tiles/MEUDON/tile_00001.tif")
>>> result["road_pct"]       # float
>>> result["flair_map"]      # PIL.Image — 12 classes colorisées
>>> result["road_mask"]      # np.ndarray (512×512) uint8
>>>
>>> # Commune entière (avec export)
>>> tiles   = fetch_tiles("MEUDON", "data/tiles/MEUDON")
>>> results = run_on_city(model, "MEUDON", tiles, export_dir="data/exports/MEUDON")
"""

import os
from collections.abc import Callable

import numpy as np
from PIL import Image

from cv_routes.config import (
    CITIES,
    COLOR_FLAIR_ROAD,
    INPUT_SIZE,
)
from cv_routes.data import fetch_city_tiles, load_tile_as_array, load_tile_as_rgb
from cv_routes.export import (
    collect_mask_polys,
    mask_to_geotiff,
    masks_to_gpkg,
    write_polys_to_gpkg,
    zip_files,
)
from cv_routes.inference import FlairModel
from cv_routes.visualization import (
    apply_overlay,
    flair_colormap,
    flair_stats,
    road_mask_from_flair,
)

ProgressCallback = Callable[[float, str], None]


# ---------------------------------------------------------------------------
# Image libre (PIL)
# ---------------------------------------------------------------------------

def run_on_image(
    model: FlairModel,
    image: Image.Image,
) -> dict:
    """
    Segmentation FLAIR sur une image PIL.

    Returns
    -------
    {
        "ortho"      : PIL.Image — image redimensionnée 512×512
        "flair_map"  : PIL.Image — 12 classes colorisées
        "road_overlay": PIL.Image — routes FLAIR (orange) sur ortho
        "road_mask"  : np.ndarray (512×512) uint8
        "class_map"  : np.ndarray (512×512) uint8
        "road_pct"   : float — % de surface route
        "stats"      : dict   — couverture par classe
    }
    """
    img    = image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
    arr255 = np.transpose(np.array(img, dtype=np.float32), (2, 0, 1))
    return _process(model, img, arr255, tile_path=None)


# ---------------------------------------------------------------------------
# Tuile GeoTIFF
# ---------------------------------------------------------------------------

def run_on_tile(
    model: FlairModel,
    tile_path: str,
) -> dict:
    """
    Segmentation FLAIR sur une tuile GeoTIFF.

    Returns
    -------
    Même structure que run_on_image, plus :
        "tile_path" : str — chemin de la tuile source
    """
    img, arr255 = _load_tile(tile_path)
    result = _process(model, img, arr255, tile_path=tile_path)
    result["tile_path"] = tile_path
    return result


def run_and_export_tile(
    model: FlairModel,
    tile_path: str,
    export_dir: str,
) -> dict:
    """
    Segmentation + export GeoTIFF / GeoPackage pour une tuile.

    Returns
    -------
    Même structure que run_on_tile, plus :
        "flair_tif" : str — chemin GeoTIFF masque route
        "gpkg"      : str — chemin GeoPackage vecteur route_flair
    """
    result    = run_on_tile(model, tile_path)
    tile_name = os.path.splitext(os.path.basename(tile_path))[0]
    os.makedirs(export_dir, exist_ok=True)

    result["flair_tif"] = mask_to_geotiff(
        result["road_mask"], tile_path,
        os.path.join(export_dir, f"{tile_name}_flair.tif"),
    )
    result["gpkg"] = masks_to_gpkg(
        {"route_flair": result["road_mask"]}, tile_path,
        os.path.join(export_dir, f"{tile_name}_masks.gpkg"),
    )
    return result


# ---------------------------------------------------------------------------
# Commune entière
# ---------------------------------------------------------------------------

def fetch_tiles(
    commune: str,
    tiles_dir: str,
    on_progress: ProgressCallback | None = None,
) -> list[str]:
    """
    Télécharge (ou retourne depuis le cache) les tuiles d'une commune GPSO.

    Parameters
    ----------
    commune   : nom de la commune (ex. "MEUDON", "BOULOGNE-BILLANCOURT")
    tiles_dir : répertoire de cache local

    Returns
    -------
    Liste de chemins vers les tuiles .tif
    """
    key  = commune.upper()
    city = CITIES.get(key)
    if city is None:
        raise KeyError(f"Commune '{commune}' inconnue. Disponibles : {list(CITIES)}")
    return fetch_city_tiles(city, tiles_dir, on_progress=on_progress)


def run_on_city(
    model: FlairModel,
    commune: str,
    tiles: list[str],
    export_dir: str | None = None,
    on_progress: ProgressCallback | None = None,
) -> dict:
    """
    Segmentation FLAIR sur toutes les tuiles d'une commune.

    Parameters
    ----------
    model      : FlairModel instancié
    commune    : nom de la commune
    tiles      : liste de chemins vers les tuiles (résultat de fetch_tiles)
    export_dir : si fourni, exporte GeoTIFF + GeoPackage global dans ce dossier

    Returns
    -------
    {
        "commune"   : str
        "n_tiles"   : int
        "road_pct"  : float — moyenne % route sur toutes les tuiles
        "tile_results": list[dict]  — résultat par tuile (run_on_tile)
        "gpkg"      : str | None    — chemin GPKG global (si export_dir)
        "tifs_zip"  : str | None    — chemin ZIP GeoTIFFs (si export_dir)
    }
    """
    city      = CITIES.get(commune.upper(), type("C", (), {"name": commune})())
    n         = len(tiles)
    results   = []
    road_pcts = []
    all_polys = {"route_flair": []}
    tif_paths = []

    tif_dir = os.path.join(export_dir, "tifs") if export_dir else None
    if tif_dir:
        os.makedirs(tif_dir, exist_ok=True)

    for i, tile_path in enumerate(tiles):
        if on_progress:
            on_progress(i / n, f"Tuile {i + 1}/{n} — {city.name}")

        r = run_on_tile(model, tile_path)
        results.append(r)
        road_pcts.append(r["road_pct"])

        if export_dir:
            tile_name = os.path.splitext(os.path.basename(tile_path))[0]
            tif_path  = mask_to_geotiff(
                r["road_mask"], tile_path,
                os.path.join(tif_dir, f"{tile_name}_flair.tif"),
            )
            tif_paths.append(tif_path)
            all_polys["route_flair"] = collect_mask_polys(
                r["road_mask"], tile_path, all_polys["route_flair"]
            )

    gpkg_path = None
    zip_path  = None

    if export_dir:
        if on_progress:
            on_progress(0.9, "Écriture GPKG…")
        gpkg_path = write_polys_to_gpkg(
            all_polys, os.path.join(export_dir, f"{commune.lower()}_flair.gpkg")
        )
        if on_progress:
            on_progress(0.95, "Compression ZIP…")
        zip_path = zip_files(
            tif_paths,
            os.path.join(export_dir, f"{commune.lower()}_flair_tifs.zip"),
            tif_dir,
        )

    if on_progress:
        on_progress(1.0, "Terminé")

    return {
        "commune":      city.name,
        "n_tiles":      n,
        "road_pct":     round(sum(road_pcts) / n, 2) if road_pcts else 0.0,
        "tile_results": results,
        "gpkg":         gpkg_path,
        "tifs_zip":     zip_path,
    }


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------

def _load_tile(tile_path: str) -> tuple[Image.Image, np.ndarray]:
    img    = load_tile_as_rgb(tile_path)
    arr255 = np.transpose(np.array(img, dtype=np.float32), (2, 0, 1))
    return img, arr255


def _process(
    model: FlairModel,
    img: Image.Image,
    arr255: np.ndarray,
    tile_path: str | None,
) -> dict:
    class_map  = model.predict(arr255)
    road_mask  = road_mask_from_flair(class_map)
    stats      = flair_stats(class_map)

    return {
        "ortho":        img,
        "flair_map":    Image.fromarray(flair_colormap(class_map)),
        "road_overlay": apply_overlay(img, road_mask, COLOR_FLAIR_ROAD),
        "road_mask":    road_mask,
        "class_map":    class_map,
        "road_pct":     stats["road_pct"],
        "stats":        stats,
    }
