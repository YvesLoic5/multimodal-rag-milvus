"""Cross-encoder reranker using ms-marco-MiniLM-L-6-v2.

Reranking happens after hybrid retrieval to push the most relevant chunks to
the top before passing them to the LLM.  Only text hits are reranked; image
hits retain their retrieval score.
"""

from __future__ import annotations

import threading
from typing import Any

from sentence_transformers import CrossEncoder

from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """Singleton cross-encoder reranker."""

    _instance: CrossEncoderReranker | None = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> CrossEncoderReranker:
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
        return cls._instance

    def _initialize(self) -> None:
        if self._initialized:
            return
        settings = get_settings()
        logger.info("Loading cross-encoder", model=settings.reranker_model)
        self._model = CrossEncoder(settings.reranker_model, max_length=512)
        self._initialized = True
        logger.info("Cross-encoder loaded")

    def rerank(
        self,
        query: str,
        hits: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank *hits* against *query*.  Returns at most *top_k* results.

        Image hits are passed through without reranking because the cross-encoder
        is text-only.
        """
        self._initialize()
        settings = get_settings()
        top_k = top_k or settings.top_k_rerank

        text_hits = [h for h in hits if h.get("modality") != "image"]
        image_hits = [h for h in hits if h.get("modality") == "image"]

        if text_hits:
            pairs = [(query, h["content_text"]) for h in text_hits]
            scores = self._model.predict(pairs)
            for hit, score in zip(text_hits, scores):
                hit["rerank_score"] = float(score)
            text_hits.sort(key=lambda h: h["rerank_score"], reverse=True)

        # Give image hits a neutral rerank score
        for h in image_hits:
            h["rerank_score"] = h.get("score", 0.0)

        merged = text_hits + image_hits
        merged.sort(key=lambda h: h["rerank_score"], reverse=True)
        result = merged[:top_k]

        logger.info(
            "Reranking complete",
            input_hits=len(hits),
            output_hits=len(result),
        )
        return result


def get_reranker() -> CrossEncoderReranker:
    """Return the application-wide CrossEncoderReranker singleton."""
    return CrossEncoderReranker()
