"""Configuration du use-case détection de routes — territoire GPSO."""

from dataclasses import dataclass

# --- Classes FLAIR cibles ---
ROAD_CLASSES: set[int] = {1, 2}   # pervious_surface + impervious_surface
ROAD_COLOR:   tuple[int, int, int] = (255, 140, 0)

# --- WMS IGN ---
WMS_URL     = "https://data.geopf.fr/wms-r/wms"
WMS_LAYER   = "ORTHOIMAGERY.ORTHOPHOTOS"
WMS_CRS     = "EPSG:2154"
WMS_BLOCK_M = 800  # 0.20 m/px → 800 m = 4096 px

# --- Territoire GPSO ---
GPSO_GEOJSON_URL = (
    "https://data.seineouest.fr/api/explore/v2.1/catalog/datasets/plu-de-gpso"
    "/exports/geojson?lang=fr&timezone=Europe%2FBerlin&epsg=2154"
)


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
