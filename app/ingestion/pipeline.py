"""Ingestion pipeline: load → embed → (idempotent) upsert into Milvus.

Supports PDF, image files (PNG/JPG), and plain text files.
Idempotency: existing entries for the same doc_id are deleted before re-insertion.
"""

from __future__ import annotations

import datetime
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from app.embeddings.clip_encoder import get_clip_encoder
from app.embeddings.text_encoder import get_bge_encoder
from app.ingestion.image_loader import load_image
from app.ingestion.pdf_loader import ImageChunk, TextChunk, load_pdf
from app.utils.config import get_settings
from app.utils.logger import get_logger
from app.vectorstore.milvus_client import get_vector_store

logger = get_logger(__name__)


def ingest_file(
    file_path: str | Path,
    doc_id: str | None = None,
) -> dict[str, int]:
    """Ingest a single file (PDF or image) into Milvus.

    Returns a dict with keys ``text_chunks`` and ``image_chunks`` indicating
    how many records were indexed.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if doc_id is None:
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, file_path.name))

    logger.info("Starting ingestion", file=str(file_path), doc_id=doc_id)

    # Idempotency: remove previous entries for this doc
    store = get_vector_store()
    deleted = store.delete_by_doc_id(doc_id)
    if deleted:
        logger.info("Removed stale entries", doc_id=doc_id, count=deleted)

    counts: dict[str, int] = {"text_chunks": 0, "image_chunks": 0}

    if suffix == ".pdf":
        text_chunks, image_chunks = load_pdf(file_path, doc_id=doc_id)
        counts["text_chunks"] = _ingest_text_chunks(text_chunks)
        counts["image_chunks"] = _ingest_image_chunks(image_chunks)
    elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        loaded = load_image(file_path, doc_id=doc_id)
        counts["image_chunks"] = _ingest_image_chunks(
            [ImageChunk(doc_id=doc_id, image=loaded.image, metadata=loaded.metadata)]
        )
    elif suffix == ".txt":
        counts["text_chunks"] = _ingest_text_file(file_path, doc_id)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    logger.info("Ingestion complete", doc_id=doc_id, **counts)
    return counts


def _ingest_text_chunks(chunks: list[TextChunk]) -> int:
    if not chunks:
        return 0
    bge = get_bge_encoder()
    store = get_vector_store()
    settings = get_settings()

    records: list[dict[str, Any]] = []
    texts = [c.content for c in chunks]
    embeddings = bge.encode(texts, batch_size=8)

    for chunk, emb in zip(chunks, embeddings):
        records.append(
            {
                "doc_id": chunk.doc_id,
                "modality": "text",
                "content_text": chunk.content,
                "dense_vector": emb["dense"],
                "sparse_vector": emb["sparse"],
                "metadata": {
                    **chunk.metadata,
                    "ingested_at": datetime.datetime.utcnow().isoformat(),
                },
            }
        )

    store.insert_batch(records)
    return len(records)


def _describe_image(image: "Image.Image", source: str, page: str) -> str:
    """Generate a description of the image using GPT-4o Vision or fallback."""
    import base64
    import io

    settings = get_settings()
    fallback = f"Image from page {page} of {source}"

    if not settings.use_openai:
        return fallback

    try:
        import httpx

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            json={
                "model": "gpt-4o",
                "max_tokens": 300,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Décris cette image en détail en français. Mentionne tous les éléments visibles, le texte, les diagrammes, les schémas. Sois précis et factuel."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        ],
                    }
                ],
            },
            timeout=60,
        )
        resp.raise_for_status()
        description = resp.json()["choices"][0]["message"]["content"]
        caption = f"[Image: {source}, page {page}] {description}"
        logger.info("Image described by GPT-4o Vision", source=source, page=page, caption_len=len(caption))
        return caption
    except Exception as exc:
        logger.warning("Image captioning failed, using fallback", error=str(exc))
        return fallback


def _ingest_image_chunks(chunks: list[ImageChunk]) -> int:
    if not chunks:
        return 0
    clip = get_clip_encoder()
    bge = get_bge_encoder()
    store = get_vector_store()

    records: list[dict[str, Any]] = []
    for chunk in tqdm(chunks, desc="Encoding images"):
        dense_vec = clip.encode_image(chunk.image)
        caption = _describe_image(
            chunk.image,
            chunk.metadata.get("source", "unknown"),
            str(chunk.metadata.get("page", "?")),
        )
        sparse_emb = bge.encode_single(caption)["sparse"]

        records.append(
            {
                "doc_id": chunk.doc_id,
                "modality": "image",
                "content_text": caption,
                "dense_vector": dense_vec,
                "sparse_vector": sparse_emb,
                "metadata": {
                    **chunk.metadata,
                    "ingested_at": datetime.datetime.utcnow().isoformat(),
                },
            }
        )

    store.insert_batch(records)
    return len(records)


def _ingest_text_file(file_path: Path, doc_id: str) -> int:
    settings = get_settings()
    raw = file_path.read_text(encoding="utf-8", errors="ignore")
    from app.ingestion.pdf_loader import _split_text

    chunks_text = _split_text(raw, settings.chunk_size, settings.chunk_overlap)
    text_chunks = [
        TextChunk(
            doc_id=doc_id,
            content=t,
            metadata={"source": file_path.name, "page": 1},
        )
        for t in chunks_text
    ]
    return _ingest_text_chunks(text_chunks)


def ingest_directory(directory: str | Path) -> dict[str, int]:
    """Ingest all supported files in *directory* (non-recursive)."""
    directory = Path(directory)
    totals: dict[str, int] = {"text_chunks": 0, "image_chunks": 0}
    supported = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".txt"}
    files = [p for p in sorted(directory.iterdir()) if p.suffix.lower() in supported]

    logger.info("Ingesting directory", path=str(directory), file_count=len(files))
    for f in tqdm(files, desc="Ingesting files"):
        try:
            result = ingest_file(f)
            totals["text_chunks"] += result["text_chunks"]
            totals["image_chunks"] += result["image_chunks"]
        except Exception as exc:
            import traceback
            logger.error("Failed to ingest file", path=str(f), error=str(exc), trace=traceback.format_exc())

    logger.info("Directory ingestion complete", **totals)
    return totals
