"""CLIP-based encoder for unified text and image embeddings.

Uses openai/clip-vit-base-patch32 to produce 512-d vectors for both modalities.
These are projected to 1024-d to match BGE-M3's output dimension so both can be
stored in the same Milvus dense_vector field.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from app.utils.logger import get_logger

logger = get_logger(__name__)

_CLIP_DIM = 512
TARGET_DIM = 1024  # padded to match BGE-M3 output


class CLIPEncoder:
    """Singleton CLIP encoder. Thread-safe lazy initialisation."""

    _instance: CLIPEncoder | None = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> CLIPEncoder:
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
        return cls._instance

    def _initialize(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        if self._initialized:
            return
        logger.info("Loading CLIP model", model=model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        self._initialized = True
        logger.info("CLIP model loaded", device=self.device)

    def encode_image(self, image: Union[Image.Image, str, Path]) -> np.ndarray:
        """Return a 1024-d float32 vector for *image*."""
        self._initialize()
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.get_image_features(**inputs)
            features = output.image_embeds if hasattr(output, "image_embeds") else output
            features = features / features.norm(dim=-1, keepdim=True)
        vec = features.cpu().numpy()[0].astype(np.float32)
        return self._pad_to_target(vec)

    def encode_text(self, text: str) -> np.ndarray:
        """Return a 1024-d float32 vector for *text* (short captions / queries)."""
        self._initialize()
        inputs = self.processor(text=[text], return_tensors="pt", truncation=True).to(
            self.device
        )
        with torch.no_grad():
            output = self.model.get_text_features(**inputs)
            features = output.text_embeds if hasattr(output, "text_embeds") else output
            features = features / features.norm(dim=-1, keepdim=True)
        vec = features.cpu().numpy()[0].astype(np.float32)
        return self._pad_to_target(vec)

    @staticmethod
    def _pad_to_target(vec: np.ndarray) -> np.ndarray:
        """Zero-pad 512-d CLIP vector to TARGET_DIM so it fits the shared field."""
        if vec.shape[0] == TARGET_DIM:
            return vec
        padded = np.zeros(TARGET_DIM, dtype=np.float32)
        padded[: vec.shape[0]] = vec
        return padded


def get_clip_encoder() -> CLIPEncoder:
    """Return the application-wide CLIPEncoder singleton."""
    return CLIPEncoder()
