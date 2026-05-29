"""Tests for the Redis cache layer (all Redis calls mocked)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


SAMPLE_PAYLOAD = {
    "answer": "Machine learning is a subset of AI.",
    "sources_text": "📄 ml.pdf • Page 3 • Score: 0.95",
}


@patch("app.cache.redis_cache.SemanticCache")
@patch("app.cache.redis_cache.redis.from_url")
def test_exact_cache_miss_then_hit(mock_redis_cls: MagicMock, mock_sc_cls: MagicMock) -> None:
    from app.cache.redis_cache import RAGCache

    redis_mock = MagicMock()
    redis_mock.ping.return_value = True
    redis_mock.get.return_value = None  # miss on first call
    mock_redis_cls.return_value = redis_mock
    mock_sc_cls.return_value = MagicMock()

    cache = RAGCache.__new__(RAGCache)
    cache._settings = MagicMock()
    cache._settings.redis_url = "redis://localhost:6379"
    cache._settings.redis_ttl = 3600
    cache._settings.cache_similarity_threshold = 0.92
    cache._redis = redis_mock
    cache._semantic = None

    result = cache.get_exact("What is ML?")
    assert result is None

    redis_mock.get.return_value = json.dumps(SAMPLE_PAYLOAD)
    result2 = cache.get_exact("What is ML?")
    assert result2 == SAMPLE_PAYLOAD


@patch("app.cache.redis_cache.SemanticCache")
@patch("app.cache.redis_cache.redis.from_url")
def test_set_exact_calls_setex(mock_redis_cls: MagicMock, mock_sc_cls: MagicMock) -> None:
    from app.cache.redis_cache import RAGCache

    redis_mock = MagicMock()
    redis_mock.ping.return_value = True
    mock_redis_cls.return_value = redis_mock
    mock_sc_cls.return_value = MagicMock()

    cache = RAGCache.__new__(RAGCache)
    cache._settings = MagicMock()
    cache._settings.redis_ttl = 3600
    cache._redis = redis_mock
    cache._semantic = None

    cache.set_exact("What is ML?", SAMPLE_PAYLOAD)
    redis_mock.setex.assert_called_once()
    call_args = redis_mock.setex.call_args[0]
    assert call_args[1] == 3600
    assert json.loads(call_args[2]) == SAMPLE_PAYLOAD


@patch("app.cache.redis_cache.SemanticCache")
@patch("app.cache.redis_cache.redis.from_url")
def test_flush_deletes_exact_keys(mock_redis_cls: MagicMock, mock_sc_cls: MagicMock) -> None:
    from app.cache.redis_cache import RAGCache

    redis_mock = MagicMock()
    redis_mock.ping.return_value = True
    redis_mock.scan_iter.return_value = ["rag:exact:abc123", "rag:exact:def456"]
    mock_redis_cls.return_value = redis_mock
    mock_sc_cls.return_value = MagicMock()

    cache = RAGCache.__new__(RAGCache)
    cache._settings = MagicMock()
    cache._redis = redis_mock
    cache._semantic = None

    cache.flush()
    assert redis_mock.delete.call_count == 2


@patch("app.cache.redis_cache.SemanticCache")
@patch("app.cache.redis_cache.redis.from_url")
def test_get_returns_exact_before_semantic(
    mock_redis_cls: MagicMock, mock_sc_cls: MagicMock
) -> None:
    from app.cache.redis_cache import RAGCache

    redis_mock = MagicMock()
    redis_mock.ping.return_value = True
    redis_mock.get.return_value = json.dumps(SAMPLE_PAYLOAD)
    mock_redis_cls.return_value = redis_mock
    mock_sc_cls.return_value = MagicMock()

    cache = RAGCache.__new__(RAGCache)
    cache._settings = MagicMock()
    cache._settings.redis_ttl = 3600
    cache._redis = redis_mock
    semantic_mock = MagicMock()
    cache._semantic = semantic_mock

    result = cache.get("What is ML?")
    assert result == SAMPLE_PAYLOAD
    semantic_mock.check.assert_not_called()
