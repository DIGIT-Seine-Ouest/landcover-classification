"""Visualisation FLAIR : colorisation, overlays, statistiques."""

import numpy as np
from PIL import Image

from flair_inference.classes import FLAIR_CLASS_COLORS, FLAIR_CLASS_NAMES

OVERLAY_ALPHA: int = 180


def colorize(class_map: np.ndarray) -> np.ndarray:
    """Carte de classes (H, W) → image RGB (H, W, 3)."""
    rgb = np.zeros((*class_map.shape, 3), dtype=np.uint8)
    for cls_id, color in FLAIR_CLASS_COLORS.items():
        rgb[class_map == cls_id] = color
    return rgb


def apply_overlay(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: int = OVERLAY_ALPHA,
) -> Image.Image:
    """Surimprime un masque binaire coloré sur une image RGB."""
    layer = np.zeros((*mask.shape, 4), dtype=np.uint8)
    layer[mask == 1] = [*color, alpha]
    layer_img = Image.fromarray(layer)
    overlay   = image.copy().convert("RGBA")
    overlay.paste(layer_img, mask=layer_img.split()[3])
    return overlay.convert("RGB")


def mask_from_classes(class_map: np.ndarray, class_ids: set[int]) -> np.ndarray:
    """Extrait un masque binaire (0/1) pour un ensemble de classes."""
    mask = np.zeros(class_map.shape, dtype=np.uint8)
    for cls in class_ids:
        mask |= (class_map == cls).astype(np.uint8)
    return mask


def class_stats(
    class_map: np.ndarray,
    target_ids: set[int] | None = None,
) -> dict:
    """
    Statistiques de couverture par classe FLAIR.

    Parameters
    ----------
    class_map  : (H, W) uint8 — sortie de FlairModel.predict()
    target_ids : ensemble de classes d'intérêt (ex. {1, 2} pour les routes)

    Returns
    -------
    {
        "classes"   : { class_name: pct_float },
        "target_pct": float — couverture des classes cibles (0.0 si target_ids=None),
        "summary"   : str   — texte lisible
    }
    """
    classes: dict[str, float] = {}
    for cls_id, name in FLAIR_CLASS_NAMES.items():
        pct = float((class_map == cls_id).mean() * 100)
        classes[name] = round(pct, 2)

    target_pct = 0.0
    if target_ids is not None:
        target_pct = float(sum((class_map == c).mean() for c in target_ids) * 100)

    lines = ["FLAIR — couverture :"]
    for name, pct in classes.items():
        if pct > 0.5:
            lines.append(f"  {name:22s}: {pct:.1f}%")
    if target_ids is not None:
        lines.append(f"\n→ Surface cible : {target_pct:.1f}%")

    return {
        "classes":    classes,
        "target_pct": round(target_pct, 2),
        "summary":    "\n".join(lines),
    }
