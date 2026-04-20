"""
Visualisation : overlays, colormaps, statistiques textuelles.
"""

import numpy as np
from PIL import Image

from cv_routes.config import (
    FLAIR_CLASS_COLORS,
    FLAIR_CLASS_NAMES,
    FLAIR_ROAD_CLASSES,
    OVERLAY_ALPHA,
)


def apply_overlay(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int],
) -> Image.Image:
    """Surimprime un masque binaire coloré sur une image RGB."""
    layer = np.zeros((*mask.shape, 4), dtype=np.uint8)
    layer[mask == 1] = [*color, OVERLAY_ALPHA]
    layer_img = Image.fromarray(layer)
    overlay   = image.copy().convert("RGBA")
    overlay.paste(layer_img, mask=layer_img.split()[3])
    return overlay.convert("RGB")


def flair_colormap(class_map: np.ndarray) -> np.ndarray:
    """Carte de classes (H, W) → image RGB (H, W, 3)."""
    rgb = np.zeros((*class_map.shape, 3), dtype=np.uint8)
    for cls_id, color in FLAIR_CLASS_COLORS.items():
        rgb[class_map == cls_id] = color
    return rgb


def road_mask_from_flair(class_map: np.ndarray) -> np.ndarray:
    """Extrait un masque route binaire (imperméable + perméable)."""
    mask = np.zeros(class_map.shape, dtype=np.uint8)
    for cls in FLAIR_ROAD_CLASSES:
        mask |= (class_map == cls).astype(np.uint8)
    return mask


def flair_stats(class_map: np.ndarray) -> dict:
    """
    Retourne un dict de statistiques de couverture.

    Returns
    -------
    {
        "classes": { class_name: pct_float },
        "road_pct": float,
        "summary": str   # texte lisible
    }
    """
    classes = {}
    for cls_id, name in FLAIR_CLASS_NAMES.items():
        pct = float((class_map == cls_id).mean() * 100)
        classes[name] = round(pct, 2)

    road_pct = sum((class_map == c).mean() for c in FLAIR_ROAD_CLASSES) * 100

    lines = ["FLAIR — couverture :"]
    for name, pct in classes.items():
        if pct > 0.5:
            lines.append(f"  {name:22s}: {pct:.1f}%")
    lines.append(f"\n→ Surface route totale : {road_pct:.1f}%")

    return {
        "classes":  classes,
        "road_pct": round(float(road_pct), 2),
        "summary":  "\n".join(lines),
    }
