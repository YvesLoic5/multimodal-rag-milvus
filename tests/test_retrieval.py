"""Tests for hybrid retriever and reranker (all external deps mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ── HybridRetriever ────────────────────────────────────────────────────────────

MOCK_HITS = [
    {
        "id": 1,
        "score": 0.95,
        "doc_id": "doc1",
        "modality": "text",
        "content_text": "Machine learning is a subset of AI.",
        "metadata": {"source": "ml.pdf", "page": 3},
    },
    {
        "id": 2,
        "score": 0.88,
        "doc_id": "doc1",
        "modality": "image",
        "content_text": "Image from page 5 of ml.pdf",
        "metadata": {"source": "ml.pdf", "page": 5},
    },
    {
        "id": 3,
        "score": 0.75,
        "doc_id": "doc2",
        "modality": "text",
        "content_text": "Deep learning uses neural networks.",
        "metadata": {"source": "dl.pdf", "page": 1},
    },
]


@patch("app.retrieval.hybrid_retriever.get_vector_store")
@patch("app.retrieval.hybrid_retriever.get_bge_encoder")
def test_retrieve_returns_hits(mock_bge: MagicMock, mock_store: MagicMock) -> None:
    from app.retrieval.hybrid_retriever import HybridRetriever

    mock_bge.return_value.encode_single.return_value = {
        "dense": np.zeros(1024, dtype=np.float32),
        "sparse": {0: 1.0, 1: 0.5},
    }
    store_mock = MagicMock()
    store_mock.hybrid_search.return_value = MOCK_HITS
    mock_store.return_value = store_mock

    retriever = HybridRetriever()
    hits = retriever.retrieve("What is machine learning?", top_k=5)

    assert len(hits) == 3
    assert hits[0]["doc_id"] == "doc1"
    store_mock.hybrid_search.assert_called_once()


@patch("app.retrieval.hybrid_retriever.get_vector_store")
@patch("app.retrieval.hybrid_retriever.get_bge_encoder")
def test_retrieve_modality_filter(mock_bge: MagicMock, mock_store: MagicMock) -> None:
    from app.retrieval.hybrid_retriever import HybridRetriever

    mock_bge.return_value.encode_single.return_value = {
        "dense": np.zeros(1024, dtype=np.float32),
        "sparse": {},
    }
    store_mock = MagicMock()
    store_mock.hybrid_search.return_value = MOCK_HITS
    mock_store.return_value = store_mock

    retriever = HybridRetriever()
    text_hits = retriever.retrieve("query", modality_filter="text")

    assert all(h["modality"] == "text" for h in text_hits)
    assert len(text_hits) == 2


# ── CrossEncoderReranker ───────────────────────────────────────────────────────

@patch("app.retrieval.reranker.CrossEncoder")
def test_reranker_orders_by_score(mock_ce_cls: MagicMock) -> None:
    from app.retrieval.reranker import CrossEncoderReranker

    # Reset singleton for this test
    CrossEncoderReranker._instance = None

    mock_model = MagicMock()
    mock_model.predict.return_value = [0.3, 0.9]  # second text hit scores higher
    mock_ce_cls.return_value = mock_model

    reranker = CrossEncoderReranker()
    text_hits = [h for h in MOCK_HITS if h["modality"] == "text"]
    result = reranker.rerank("What is deep learning?", text_hits, top_k=2)

    assert result[0]["rerank_score"] >= result[-1]["rerank_score"]
    assert len(result) <= 2


@patch("app.retrieval.reranker.CrossEncoder")
def test_reranker_image_hits_pass_through(mock_ce_cls: MagicMock) -> None:
    from app.retrieval.reranker import CrossEncoderReranker

    CrossEncoderReranker._instance = None
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.5]
    mock_ce_cls.return_value = mock_model

    reranker = CrossEncoderReranker()
    result = reranker.rerank("query", MOCK_HITS, top_k=10)

    image_results = [h for h in result if h["modality"] == "image"]
    assert len(image_results) == 1
    assert "rerank_score" in image_results[0]
