"""Milvus vector store client.

Collection schema
─────────────────
  id              INT64            primary key, auto_id
  doc_id          VARCHAR(256)     source document identifier
  modality        VARCHAR(16)      "text" | "image"
  content_text    VARCHAR(65535)   chunk text or image caption
  dense_vector    FLOAT_VECTOR(1024)
  sparse_vector   SPARSE_FLOAT_VECTOR
  metadata        JSON             page, source, timestamp, …

Indexes
───────
  HNSW on dense_vector  (M=16, efConstruction=256, metric=COSINE)
  SPARSE_INVERTED_INDEX on sparse_vector (metric=IP)
"""

from __future__ import annotations

import time
from typing import Any

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
    utility,
)
from pymilvus.client.types import LoadState
from tenacity import retry, stop_after_attempt, wait_exponential

from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "multimodal_rag"
DENSE_DIM = 1024


def _build_schema() -> CollectionSchema:
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="modality", dtype=DataType.VARCHAR, max_length=16),
        FieldSchema(
            name="content_text", dtype=DataType.VARCHAR, max_length=65535, default_value=""
        ),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=DENSE_DIM),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="metadata", dtype=DataType.JSON),
    ]
    return CollectionSchema(fields=fields, description="Multimodal RAG collection", enable_dynamic_field=False)


class MilvusVectorStore:
    """Thin wrapper around pymilvus Collection for our RAG use case."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._collection: Collection | None = None
        self._connect()
        self._ensure_collection()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _connect(self) -> None:
        logger.info(
            "Connecting to Milvus",
            host=self._settings.milvus_host,
            port=self._settings.milvus_port,
        )
        connections.connect(
            alias="default",
            host=self._settings.milvus_host,
            port=self._settings.milvus_port,
        )
        logger.info("Milvus connection established")

    def _ensure_collection(self) -> None:
        name = self._settings.milvus_collection
        if not utility.has_collection(name):
            logger.info("Creating collection", collection=name)
            col = Collection(name=name, schema=_build_schema(), consistency_level="Strong")
            self._create_indexes(col)
        else:
            logger.info("Collection already exists", collection=name)

        self._collection = Collection(name)
        if utility.load_state(name) != LoadState.Loaded:
            self._collection.load()

    @staticmethod
    def _create_indexes(col: Collection) -> None:
        col.create_index(
            field_name="dense_vector",
            index_params={
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 256},
            },
        )
        col.create_index(
            field_name="sparse_vector",
            index_params={
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "IP",
                "params": {"drop_ratio_build": 0.2},
            },
        )
        logger.info("Indexes created")

    @property
    def collection(self) -> Collection:
        assert self._collection is not None, "Collection not initialised"
        return self._collection

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
    def insert_batch(self, records: list[dict[str, Any]]) -> list[int]:
        """Insert *records* and return auto-assigned primary keys.

        Each record must have keys: doc_id, modality, content_text,
        dense_vector, sparse_vector, metadata.
        """
        if not records:
            return []

        rows = []
        for r in records:
            rows.append({
                "doc_id": r["doc_id"],
                "modality": r["modality"],
                "content_text": r["content_text"][:65535],
                "dense_vector": r["dense_vector"].tolist(),
                "sparse_vector": r["sparse_vector"],
                "metadata": r["metadata"],
            })

        result = self.collection.insert(rows)
        self.collection.flush()
        ids: list[int] = result.primary_keys
        logger.info("Inserted records", count=len(ids))
        return ids

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
    def hybrid_search(
        self,
        dense_vec: list[float],
        sparse_vec: dict[int, float],
        top_k: int = 5,
        output_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Perform hybrid dense+sparse search with RRF fusion."""
        from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker

        if output_fields is None:
            output_fields = ["doc_id", "modality", "content_text", "metadata"]

        dense_req = AnnSearchRequest(
            data=[dense_vec],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k * 4,
        )
        sparse_req = AnnSearchRequest(
            data=[sparse_vec],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=top_k * 4,
        )

        results = self.collection.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=RRFRanker(k=60),
            limit=top_k,
            output_fields=output_fields,
        )

        hits = []
        for hit in results[0]:
            hits.append(
                {
                    "id": hit.id,
                    "score": hit.score,
                    "doc_id": hit.entity.get("doc_id"),
                    "modality": hit.entity.get("modality"),
                    "content_text": hit.entity.get("content_text"),
                    "metadata": hit.entity.get("metadata", {}),
                }
            )
        return hits

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
    def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all entries matching *doc_id*. Returns number of deleted records."""
        expr = f'doc_id == "{doc_id}"'
        result = self.collection.delete(expr)
        self.collection.flush()
        deleted = result.delete_count
        logger.info("Deleted records", doc_id=doc_id, count=deleted)
        return deleted

    def drop_collection(self) -> None:
        """Permanently drop the collection (used in tests / resets)."""
        name = self._settings.milvus_collection
        if utility.has_collection(name):
            utility.drop_collection(name)
            logger.warning("Collection dropped", collection=name)


_store_instance: MilvusVectorStore | None = None
_store_lock: Any = None


def get_vector_store() -> MilvusVectorStore:
    """Return the application-wide MilvusVectorStore singleton."""
    import threading

    global _store_instance, _store_lock
    if _store_lock is None:
        _store_lock = threading.Lock()
    with _store_lock:
        if _store_instance is None:
            _store_instance = MilvusVectorStore()
    return _store_instance
