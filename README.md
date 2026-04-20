# landcover-classification

Segmentation sémantique de l'occupation du sol sur ortho-photos IGN.  
Architecture modulaire : les libs d'acquisition et d'inférence sont indépendantes des cas d'usage métier.

---

## Architecture

```
landcover-classification/
├── ortho_ign/              # lib 1 — acquisition ortho-photos IGN
├── flair_inference/        # lib 2 — inférence FLAIR-INC ONNX
└── use_cases/
    └── road_detection/     # cas métier : détection de routes
        ├── config.py
        ├── notebook.ipynb
        └── app.py
```

### `ortho_ign` — Acquisition ortho-photos

Télécharge des ortho-photos depuis n'importe quel serveur WMS et les découpe en tuiles GeoTIFF géoréférencées (EPSG:2154).

| Module | Contenu |
|--------|---------|
| `wms.py` | `fetch_wms()` — requête WMS → dataset rasterio en mémoire |
| `tiling.py` | `tile_raster()`, `load_tile_as_rgb()`, `load_tile_as_array()` |
| `territory.py` | `load_commune_polygons()`, `commune_bbox()`, `grid_blocks()` |
| `fetch.py` | `fetch_city_tiles()` — pipeline complet avec cache local |
| `export.py` | `mask_to_geotiff()`, `masks_to_gpkg()`, `write_polys_to_gpkg()` |

### `flair_inference` — Inférence FLAIR-INC

Encapsule le modèle [FLAIR-INC](https://ignf.github.io/FLAIR/) (ResNet34-UNet, 12 classes, format ONNX) et expose des fonctions de visualisation **génériques** — sans connaissance du cas d'usage.

| Module | Contenu |
|--------|---------|
| `model.py` | `FlairModel` — chargement ONNX + `predict(arr255) → class_map` |
| `classes.py` | `FLAIR_CLASS_NAMES`, `FLAIR_CLASS_COLORS`, `INPUT_SIZE` |
| `visualization.py` | `colorize()`, `mask_from_classes()`, `apply_overlay()`, `class_stats()` |
| `download.py` | `download_model()` — HuggingFace Hub ou URL directe |

**12 classes FLAIR**

| ID | Classe | Couleur |
|----|--------|---------|
| 0 | building | ![#b46e3c](https://placehold.co/12x12/b46e3c/b46e3c.png) |
| 1 | pervious\_surface | ![#bebebe](https://placehold.co/12x12/bebebe/bebebe.png) |
| 2 | impervious\_surface | ![#6e6e6e](https://placehold.co/12x12/6e6e6e/6e6e6e.png) |
| 3 | bare\_soil | |
| 4 | water | |
| 5 | coniferous | |
| 6 | deciduous | |
| 7 | brushwood | |
| 8 | vineyard | |
| 9 | herbaceous | |
| 10 | agricultural\_land | |
| 11 | plowed\_land | |

---

## Cas métier — Détection de routes (`use_cases/road_detection`)

### Principe

Les routes sont définies comme l'union des classes FLAIR `pervious_surface` (1) et `impervious_surface` (2).  
Cette définition vit **dans le use-case**, pas dans les libs — pour détecter la végétation il suffit de changer le set de classes.

```python
ROAD_CLASSES = {1, 2}   # ← toute la "logique métier route" est ici
```

### Pipeline

```
fetch_city_tiles()          ortho_ign
      ↓
load_tile_as_rgb()          ortho_ign
      ↓
FlairModel.predict()        flair_inference   →  class_map (512×512, 12 classes)
      ↓
mask_from_classes(..., ROAD_CLASSES)          →  road_mask (binaire)
      ↓
mask_to_geotiff() / write_polys_to_gpkg()    →  GeoTIFF + GeoPackage
```

### Territoire supporté — GPSO (Grand Paris Seine Ouest)

`BOULOGNE-BILLANCOURT` · `CHAVILLE` · `ISSY-LES-MOULINEAUX` · `MARNES-LA-COQUETTE`  
`MEUDON` · `SEVRES` · `VANVES` · `VILLE-D'AVRAY`

Source ortho : WMS IGN `ORTHOIMAGERY.ORTHOPHOTOS` · résolution 0.20 m/px · EPSG:2154

### Notebook

`use_cases/road_detection/notebook.ipynb` — pipeline complet en 7 sections, conçu pour ArcGIS Online Notebooks.

```python
from ortho_ign import fetch_city_tiles, load_tile_as_rgb
from flair_inference import FlairModel, mask_from_classes, colorize, class_stats

ROAD_CLASSES = {1, 2}

tiles     = fetch_city_tiles("MEUDON", TILES_DIR, GPSO_GEOJSON_URL)
model     = FlairModel(MODEL_PATH)

for tile_path in tiles:
    img       = load_tile_as_rgb(tile_path)
    class_map = model.predict(...)
    road_mask = mask_from_classes(class_map, ROAD_CLASSES)
    stats     = class_stats(class_map, target_ids=ROAD_CLASSES)
    # → stats["target_pct"] : % surface route
```

### Démo Gradio

`use_cases/road_detection/app.py` — déployable sur [HuggingFace Spaces](https://huggingface.co/spaces).  
Upload une ortho-photo → segmentation FLAIR + overlay routes + statistiques.

---

## Installation

```bash
# Dépendances conda (recommandé pour rasterio / fiona)
conda env create -f environment.yml
conda activate landcover

# Ou pip
pip install -e ".[full]"
```

### Téléchargement du modèle

```python
from flair_inference import download_model
model_path = download_model()   # depuis HuggingFace Hub
```

Le modèle `flair_12cl_resnet34_unet.onnx` est hébergé sur [`mandresyandri/road-landcover-detection`](https://huggingface.co/mandresyandri/road-landcover-detection).

---

## Étendre à un nouveau cas d'usage

Dupliquer `use_cases/road_detection/` et changer le set de classes cibles :

```python
# use_cases/vegetation/config.py
VEGETATION_CLASSES = {5, 6, 7, 9}   # coniferous + deciduous + brushwood + herbaceous
```

Les libs `ortho_ign` et `flair_inference` n'ont pas besoin d'être modifiées.
