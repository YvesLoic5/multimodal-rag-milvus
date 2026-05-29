"""Semantic + exact cache backed by Redis.

Two-layer caching strategy:
  1. Exact cache: MD5 hash of the question → instant O(1) lookup.
  2. Semantic cache: BGE-M3 vector similarity search via RedisVL.
     If a past question is cosine-similar above the configured threshold,
     the cached answer is returned without running the full RAG pipeline.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import redis
from redisvl.extensions.llmcache import SemanticCache
from redisvl.utils.vectorize import HFTextVectorizer
from tenacity import retry, stop_after_attempt, wait_exponential

from app.embeddings.text_encoder import get_bge_encoder
from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_EXACT_PREFIX = "rag:exact:"
_SEMANTIC_INDEX = "rag_semantic_cache"


class RAGCache:
    """Unified exact + semantic cache for RAG queries."""

    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self._redis = self._connect_redis(settings.redis_url)
        self._semantic: SemanticCache | None = None
        self._init_semantic_cache()

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=15), reraise=True)
    def _connect_redis(self, url: str) -> redis.Redis:  # type: ignore[type-arg]
        client = redis.from_url(url, decode_responses=True)
        client.ping()
        logger.info("Redis connected", url=url)
        return client

    def _init_semantic_cache(self) -> None:
        try:
            # RedisVL SemanticCache wraps an index with cosine similarity search
            self._semantic = SemanticCache(
                name=_SEMANTIC_INDEX,
                redis_url=self._settings.redis_url,
                distance_threshold=1.0 - self._settings.cache_similarity_threshold,
                # We manage vectorisation ourselves — use a dummy vectorizer
                # and override the embedding at check/store time.
                vectorizer=HFTextVectorizer(model="BAAI/bge-m3"),
                ttl=self._settings.redis_ttl,
            )
            logger.info("Semantic cache initialised")
        except Exception as exc:
            logger.warning("Semantic cache init failed — falling back to exact only", error=str(exc))
            self._semantic = None

    # ── Exact cache ──────────────────────────────────────────────────────────

    def _exact_key(self, question: str) -> str:
        return _EXACT_PREFIX + hashlib.md5(question.encode()).hexdigest()

    def get_exact(self, question: str) -> dict[str, Any] | None:
        key = self._exact_key(question)
        raw = self._redis.get(key)
        if raw:
            logger.info("Exact cache hit", question_preview=question[:60])
            return json.loads(raw)  # type: ignore[arg-type]
        return None

    def set_exact(self, question: str, payload: dict[str, Any]) -> None:
        key = self._exact_key(question)
        self._redis.setex(key, self._settings.redis_ttl, json.dumps(payload))

    # ── Semantic cache ───────────────────────────────────────────────────────

    def get_semantic(self, question: str) -> dict[str, Any] | None:
        if self._semantic is None:
            return None
        try:
            results = self._semantic.check(prompt=question)
            if results:
                hit = results[0]
                logger.info(
                    "Semantic cache hit",
                    question_preview=question[:60],
                    score=hit.get("vector_distance"),
                )
                return json.loads(hit["response"])  # type: ignore[arg-type]
        except Exception as exc:
            logger.warning("Semantic cache lookup error", error=str(exc))
        return None

    def set_semantic(self, question: str, payload: dict[str, Any]) -> None:
        if self._semantic is None:
            return
        try:
            self._semantic.store(
                prompt=question,
                response=json.dumps(payload),
            )
        except Exception as exc:
            logger.warning("Semantic cache store error", error=str(exc))

    # ── Unified interface ─────────────────────────────────────────────────────

    def get(self, question: str) -> dict[str, Any] | None:
        """Check exact cache first, then semantic cache."""
        return self.get_exact(question) or self.get_semantic(question)

    def set(self, question: str, payload: dict[str, Any]) -> None:
        """Store in both exact and semantic caches."""
        self.set_exact(question, payload)
        self.set_semantic(question, payload)

    def invalidate(self, question: str) -> None:
        self._redis.delete(self._exact_key(question))

    def flush(self) -> None:
        """Clear all RAG cache entries (dangerous — use in tests only)."""
        for key in self._redis.scan_iter(f"{_EXACT_PREFIX}*"):
            self._redis.delete(key)
        logger.warning("Cache flushed")


_cache_instance: RAGCache | None = None


def get_cache() -> RAGCache:
    """Return the application-wide RAGCache singleton."""
    import threading

    global _cache_instance
    _cache_instance = _cache_instance or RAGCache()
    return _cache_instance
