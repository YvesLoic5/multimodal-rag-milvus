"""BGE-M3 text encoder for dense (1024-d) and sparse embeddings.

BGE-M3 from FlagEmbedding produces:
- dense_vecs  : float32 numpy array of shape (1024,)
- lexical_weights : dict[token_id, float] usable as sparse vector
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from FlagEmbedding import BGEM3FlagModel

from app.utils.logger import get_logger

logger = get_logger(__name__)

DENSE_DIM = 1024


class BGEEncoder:
    """Singleton BGE-M3 encoder. Thread-safe lazy initialisation."""

    _instance: BGEEncoder | None = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> BGEEncoder:
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
        return cls._instance

    def _initialize(self, model_name: str = "BAAI/bge-m3") -> None:
        if self._initialized:
            return
        logger.info("Loading BGE-M3 model", model=model_name)
        self.model = BGEM3FlagModel(model_name, use_fp16=True)
        self._initialized = True
        logger.info("BGE-M3 model loaded")

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int = 12,
        max_length: int = 8192,
    ) -> list[dict[str, Any]]:
        """Encode *texts* and return dense + sparse representations.

        Each item in the returned list has:
            ``dense``  – float32 ndarray (1024,)
            ``sparse`` – dict[int, float] (token_id → weight)
        """
        self._initialize()
        outputs = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        results: list[dict[str, Any]] = []
        for i in range(len(texts)):
            dense = outputs["dense_vecs"][i].astype(np.float32)
            sparse = outputs["lexical_weights"][i]
            results.append({"dense": dense, "sparse": sparse})
        return results

    def encode_single(self, text: str) -> dict[str, Any]:
        """Convenience wrapper for a single text."""
        return self.encode([text])[0]


def get_bge_encoder() -> BGEEncoder:
    """Return the application-wide BGEEncoder singleton."""
    return BGEEncoder()
