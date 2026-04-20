"""
Constantes globales — classes FLAIR, WMS IGN, territoire GPSO.
"""

from dataclasses import dataclass

# --- Images ---
INPUT_SIZE = 512

# --- Données ---
TILES_DIR = "data/tiles"

# --- WMS IGN ---
WMS_URL    = "https://data.geopf.fr/wms-r/wms"
WMS_LAYER  = "ORTHOIMAGERY.ORTHOPHOTOS"
WMS_CRS    = "EPSG:2154"
WMS_WIDTH  = 4096
WMS_HEIGHT = 4096
WMS_BLOCK_M: int = 800   # 0.20 m/px → 800 m = 4096 px

# --- GeoJSON communes GPSO ---
GPSO_GEOJSON_URL = (
    "https://data.seineouest.fr/api/explore/v2.1/catalog/datasets/plu-de-gpso"
    "/exports/geojson?lang=fr&timezone=Europe%2FBerlin&epsg=2154"
)

# --- Classes FLAIR ---
FLAIR_CLASS_NAMES: dict[int, str] = {
    0:  "building",
    1:  "pervious_surface",
    2:  "impervious_surface",
    3:  "bare_soil",
    4:  "water",
    5:  "coniferous",
    6:  "deciduous",
    7:  "brushwood",
    8:  "vineyard",
    9:  "herbaceous",
    10: "agricultural_land",
    11: "plowed_land",
}

FLAIR_CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0:  (180, 110,  60),
    1:  (190, 190, 130),
    2:  (110, 110, 110),
    3:  (139,  90,  43),
    4:  ( 30, 144, 255),
    5:  (  0,  80,   0),
    6:  ( 34, 139,  34),
    7:  (107, 142,  35),
    8:  (128,   0, 128),
    9:  (124, 205, 124),
    10: (255, 215,   0),
    11: (160,  82,  45),
}

# Imperméable (2) + perméable (1) = route
FLAIR_ROAD_CLASSES: set[int] = {1, 2}

# --- Visualisation ---
COLOR_FLAIR_ROAD: tuple[int, int, int] = (255, 140, 0)
OVERLAY_ALPHA: int = 180


# --- Communes GPSO ---
@dataclass(frozen=True)
class City:
    name: str


CITIES: dict[str, City] = {
    "BOULOGNE-BILLANCOURT": City("BOULOGNE-BILLANCOURT"),
    "CHAVILLE":             City("CHAVILLE"),
    "ISSY-LES-MOULINEAUX":  City("ISSY-LES-MOULINEAUX"),
    "MARNES-LA-COQUETTE":   City("MARNES-LA-COQUETTE"),
    "MEUDON":               City("MEUDON"),
    "SEVRES":               City("SEVRES"),
    "VANVES":               City("VANVES"),
    "VILLE-D'AVRAY":        City("VILLE-D'AVRAY"),
}
