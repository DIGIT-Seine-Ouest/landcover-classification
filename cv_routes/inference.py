"""
Modèle d'inférence ONNX — FLAIR-INC ResNet34-UNet.
"""

import numpy as np
import onnxruntime as ort


class FlairModel:
    """
    FLAIR-INC ResNet34-UNet — segmentation sémantique 12 classes.

    Parameters
    ----------
    model_path : str
        Chemin vers le fichier .onnx (flair_12cl_resnet34_unet.onnx).

    Example
    -------
    >>> model = FlairModel("models/flair_12cl_resnet34_unet.onnx")
    >>> class_map = model.predict(arr255)   # (3, 512, 512) float32
    """

    NUM_CLASSES = 12
    MEAN = np.array([105.08, 110.87, 101.82], dtype=np.float32).reshape(3, 1, 1)
    STD  = np.array([ 52.17,  45.38,  44.00], dtype=np.float32).reshape(3, 1, 1)

    def __init__(self, model_path: str) -> None:
        self._session    = ort.InferenceSession(model_path)
        self._input_name = self._session.get_inputs()[0].name

    def predict(self, arr255: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        arr255 : (3, H, W) float32 dans [0, 255]

        Returns
        -------
        class_map : (H, W) uint8 — indice de classe argmax (0-11)
        """
        normed = (arr255 - self.MEAN) / self.STD
        logits = self._session.run(None, {self._input_name: normed[np.newaxis]})[0][0]
        logits = logits[:self.NUM_CLASSES]
        logits -= logits.max(axis=0, keepdims=True)
        exp    = np.exp(logits)
        probs  = exp / exp.sum(axis=0, keepdims=True)
        return probs.argmax(axis=0).astype(np.uint8)
