"""Téléchargement d'images WMS en mémoire (rasterio)."""

import rasterio
import requests
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

IGN_WMS_URL   = "https://data.geopf.fr/wms-r/wms"
IGN_WMS_LAYER = "ORTHOIMAGERY.ORTHOPHOTOS"
IGN_WMS_CRS   = "EPSG:2154"
IGN_BLOCK_M   = 800  # 0.20 m/px → 800 m ≈ 4000 px, arrondi à 4096 px (8×512) dans fetch.py


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

    meta = {
        "driver": "GTiff", "count": count, "dtype": dtype,
        "width": width, "height": height, "crs": crs, "transform": transform,
    }
    buf = MemoryFile()
    with buf.open(**meta) as dst:
        dst.write(data)
    return buf.open()
