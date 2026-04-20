"""
Téléchargement du modèle ONNX depuis HuggingFace Hub ou une URL directe.
"""

import os
import urllib.request
from pathlib import Path


def download_model(
    dest: str | os.PathLike = "flair_12cl_resnet34_unet.onnx",
    source: str = "hf",
    hf_repo: str = "mandresyandri/road-landcover-detection",
    hf_filename: str = "flair_12cl_resnet34_unet.onnx",
    url: str | None = None,
    force: bool = False,
) -> str:
    """
    Télécharge le modèle FLAIR-INC ONNX et retourne son chemin local.

    Parameters
    ----------
    dest        : chemin de destination (fichier ou dossier)
    source      : "hf" (HuggingFace Hub) ou "url" (téléchargement direct)
    hf_repo     : identifiant du dépôt HuggingFace (si source="hf")
    hf_filename : nom du fichier dans le dépôt HF
    url         : URL directe (si source="url")
    force       : re-télécharge même si le fichier existe déjà

    Returns
    -------
    Chemin absolu vers le fichier .onnx local

    Examples
    --------
    >>> from cv_routes import download_model
    >>> model_path = download_model()                      # HuggingFace (défaut)
    >>> model_path = download_model(dest="/tmp/flair.onnx")
    >>> model_path = download_model(source="url", url="https://…/flair.onnx")
    """
    dest = Path(dest)
    if dest.is_dir():
        dest = dest / hf_filename

    if dest.exists() and not force:
        print(f"Modèle déjà présent : {dest}")
        return str(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)

    if source == "hf":
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=hf_repo, filename=hf_filename, local_dir=str(dest.parent))
            # hf_hub_download peut placer le fichier dans un sous-dossier snapshots/
            if Path(path) != dest:
                import shutil
                shutil.copy2(path, dest)
            print(f"Modèle téléchargé (HF) : {dest}")
        except ImportError:
            raise ImportError(
                "huggingface_hub est requis pour source='hf'.\n"
                "  conda install -c conda-forge huggingface_hub\n"
                "  ou : pip install huggingface_hub"
            )

    elif source == "url":
        if not url:
            raise ValueError("url est requis quand source='url'")
        print(f"Téléchargement depuis {url} …")
        urllib.request.urlretrieve(url, dest)
        print(f"Modèle téléchargé (URL) : {dest}")

    else:
        raise ValueError(f"source doit être 'hf' ou 'url', reçu : {source!r}")

    return str(dest)
