"""Démo Gradio — détection de routes FLAIR-INC. Déployable sur HuggingFace Spaces."""

import numpy as np
import gradio as gr
from PIL import Image

from flair_inference import FlairModel, download_model, colorize, mask_from_classes, apply_overlay, class_stats
from flair_inference.classes import INPUT_SIZE
from use_cases.road_detection.config import ROAD_CLASSES, ROAD_COLOR

MODEL_PATH = download_model()
model      = FlairModel(MODEL_PATH)


def predict(image: Image.Image) -> tuple[Image.Image, Image.Image, str]:
    img    = image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
    arr255 = np.transpose(np.array(img, dtype=np.float32), (2, 0, 1))

    class_map  = model.predict(arr255)
    road_mask  = mask_from_classes(class_map, ROAD_CLASSES)
    stats      = class_stats(class_map, target_ids=ROAD_CLASSES)
    flair_img  = Image.fromarray(colorize(class_map))
    overlay    = apply_overlay(img, road_mask, ROAD_COLOR)

    return flair_img, overlay, stats["summary"]


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Ortho-photo IGN"),
    outputs=[
        gr.Image(label="Segmentation FLAIR — 12 classes"),
        gr.Image(label="Routes détectées"),
        gr.Textbox(label="Statistiques", lines=14),
    ],
    title="Détection de routes — FLAIR-INC (GPSO)",
    description=(
        "Segmentation sémantique sur ortho-photos IGN · "
        "Modèle FLAIR-INC ResNet34-UNet · 12 classes · "
        "Routes = surfaces imperméables + perméables"
    ),
    examples=[],
)

if __name__ == "__main__":
    demo.launch()
