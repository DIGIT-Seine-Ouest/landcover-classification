"""
Gestion du territoire — polygones communes, bbox, grille de blocs WMS.
"""

import unicodedata
from functools import lru_cache

import requests
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry


def _normalize(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", name.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


@lru_cache(maxsize=4)
def load_commune_polygons(url: str) -> dict[str, BaseGeometry]:
    """Télécharge le GeoJSON et retourne { nom_normalisé: shapely_geometry }."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    geojson  = response.json()

    polygons: dict[str, BaseGeometry] = {}
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        name  = (
            props.get("nom_comm")
            or props.get("nom")
            or props.get("commune")
            or props.get("libelle")
            or ""
        )
        if name:
            polygons[_normalize(name)] = shape(feature["geometry"])
    return polygons


def commune_bbox(
    polygons: dict[str, BaseGeometry],
    commune_name: str,
) -> tuple[float, float, float, float]:
    """(xmin, ymin, xmax, ymax) en EPSG:2154 pour une commune."""
    key = _normalize(commune_name)
    if key not in polygons:
        available = ", ".join(polygons.keys())
        raise KeyError(f"Commune '{commune_name}' introuvable. Disponibles : {available}")
    b = polygons[key].bounds
    return b[0], b[1], b[2], b[3]


def territory_bbox(
    polygons: dict[str, BaseGeometry],
) -> tuple[float, float, float, float]:
    """Bbox global couvrant toutes les communes."""
    all_bounds = [p.bounds for p in polygons.values()]
    return (
        min(b[0] for b in all_bounds),
        min(b[1] for b in all_bounds),
        max(b[2] for b in all_bounds),
        max(b[3] for b in all_bounds),
    )


def grid_blocks(
    bbox: tuple[float, float, float, float],
    block_m: int,
) -> list[tuple[float, float, float, float]]:
    """Découpe un bbox en blocs carrés de `block_m` mètres."""
    xmin, ymin, xmax, ymax = bbox
    blocks = []
    x = xmin
    while x < xmax:
        y = ymin
        while y < ymax:
            blocks.append((x, y, min(x + block_m, xmax), min(y + block_m, ymax)))
            y += block_m
        x += block_m
    return blocks
