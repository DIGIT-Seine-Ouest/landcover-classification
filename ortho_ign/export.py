"""Export des masques : GeoTIFF géoréférencé, GeoPackage vectoriel, ZIP."""

import os
import zipfile

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape


def mask_to_geotiff(
    mask: np.ndarray,
    tile_path: str,
    output_path: str,
) -> str:
    """Exporte un masque binaire (H×W uint8, 0/1) en GeoTIFF géoréférencé."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(tile_path) as src:
        meta = src.meta.copy()
    meta.update(count=1, dtype="uint8", compress="lzw", nodata=255)
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(mask[np.newaxis].astype(np.uint8))
    return output_path


def masks_to_gpkg(
    masks: dict[str, np.ndarray],
    tile_path: str,
    output_path: str,
) -> str:
    """
    Vectorise plusieurs masques dans un GeoPackage (une couche par masque).

    Parameters
    ----------
    masks       : { layer_name: mask_array (H×W uint8, 0/1) }
    tile_path   : tuile source — fournit la géotransformation EPSG:2154
    output_path : chemin .gpkg
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    with rasterio.open(tile_path) as src:
        transform = src.transform

    for layer_name, mask in masks.items():
        polys = [
            shape(geom)
            for geom, val in shapes(mask.astype(np.uint8), transform=transform)
            if int(val) == 1
        ]
        gdf = gpd.GeoDataFrame(geometry=polys, crs="EPSG:2154")
        gdf.to_file(output_path, driver="GPKG", layer=layer_name)

    return output_path


def collect_mask_polys(
    mask: np.ndarray,
    tile_path: str,
    existing: list | None = None,
) -> list:
    """Vectorise un masque et ajoute les polygones à une liste existante."""
    with rasterio.open(tile_path) as src:
        transform = src.transform
    polys = [
        shape(geom)
        for geom, val in shapes(mask.astype(np.uint8), transform=transform)
        if int(val) == 1
    ]
    return (existing or []) + polys


def write_polys_to_gpkg(layers: dict[str, list], output_path: str) -> str:
    """Écrit un dict de couches de polygones dans un GeoPackage."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)
    for layer_name, polys in layers.items():
        gdf = gpd.GeoDataFrame(geometry=polys, crs="EPSG:2154")
        gdf.to_file(output_path, driver="GPKG", layer=layer_name)
    return output_path


def zip_files(file_paths: list[str], output_path: str, base_dir: str) -> str:
    """Zippe une liste de fichiers en conservant leur chemin relatif à base_dir."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in file_paths:
            if os.path.exists(fp):
                zf.write(fp, os.path.relpath(fp, base_dir))
    return output_path
