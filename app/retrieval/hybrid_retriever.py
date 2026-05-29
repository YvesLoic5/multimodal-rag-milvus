"""Hybrid retrieval using Milvus dense + sparse search with RRF fusion."""

from __future__ import annotations

from typing import Any

from app.embeddings.clip_encoder import get_clip_encoder
from app.embeddings.text_encoder import get_bge_encoder
from app.utils.config import get_settings
from app.utils.logger import get_logger
from app.vectorstore.milvus_client import get_vector_store

logger = get_logger(__name__)


class HybridRetriever:
    """Retrieve relevant chunks using dense + sparse hybrid search."""

    def __init__(self) -> None:
        self._settings = get_settings()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        modality_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the top-k most relevant chunks for *query*.

        Args:
            query: Natural language search query.
            top_k: Number of results (defaults to settings.top_k_retrieval).
            modality_filter: "text", "image", or None for both.

        Returns:
            List of hit dicts with keys: id, score, doc_id, modality,
            content_text, metadata.
        """
        top_k = top_k or self._settings.top_k_retrieval

        bge = get_bge_encoder()
        emb = bge.encode_single(query)
        dense_vec = emb["dense"].tolist()
        sparse_vec = emb["sparse"]

        store = get_vector_store()
        hits = store.hybrid_search(
            dense_vec=dense_vec,
            sparse_vec=sparse_vec,
            top_k=top_k,
        )

        if modality_filter:
            hits = [h for h in hits if h.get("modality") == modality_filter]

        logger.info(
            "Retrieval complete",
            query_preview=query[:80],
            hits=len(hits),
            top_k=top_k,
        )
        return hits

    def retrieve_with_image_query(
        self,
        image_path: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve using an image as the query (CLIP dense only)."""
        from PIL import Image

        top_k = top_k or self._settings.top_k_retrieval
        clip = get_clip_encoder()
        dense_vec = clip.encode_image(image_path).tolist()

        # Sparse vector is zeroed out — image queries have no BM25 component
        sparse_vec: dict[int, float] = {}

        store = get_vector_store()
        return store.hybrid_search(
            dense_vec=dense_vec,
            sparse_vec=sparse_vec,
            top_k=top_k,
        )


def get_retriever() -> HybridRetriever:
    """Return a HybridRetriever (lightweight, not a singleton)."""
    return HybridRetriever()
