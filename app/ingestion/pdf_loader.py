"""PDF ingestion: extract text chunks and embedded images using PyMuPDF."""

from __future__ import annotations

import io
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from PIL import Image

from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TextChunk:
    doc_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageChunk:
    doc_id: str
    image: Image.Image
    metadata: dict[str, Any] = field(default_factory=dict)


def load_pdf(
    pdf_path: str | Path,
    doc_id: str | None = None,
) -> tuple[list[TextChunk], list[ImageChunk]]:
    """Extract text chunks and images from *pdf_path*.

    Returns two lists: text chunks (already split) and image chunks.
    """
    settings = get_settings()
    pdf_path = Path(pdf_path)
    doc_id = doc_id or str(uuid.uuid5(uuid.NAMESPACE_URL, pdf_path.name))

    logger.info("Processing PDF", path=str(pdf_path), doc_id=doc_id)

    text_chunks: list[TextChunk] = []
    image_chunks: list[ImageChunk] = []

    doc = fitz.open(str(pdf_path))
    for page_num, page in enumerate(doc, start=1):
        # ── Text extraction ──────────────────────────────────────────────────
        raw_text = page.get_text("text")
        if raw_text.strip():
            for chunk in _split_text(
                raw_text,
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
            ):
                text_chunks.append(
                    TextChunk(
                        doc_id=doc_id,
                        content=chunk,
                        metadata={
                            "source": pdf_path.name,
                            "page": page_num,
                            "total_pages": len(doc),
                        },
                    )
                )

        # ── Image extraction ─────────────────────────────────────────────────
        for img_index, img_ref in enumerate(page.get_images(full=True)):
            xref = img_ref[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            try:
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                if pil_img.width < 64 or pil_img.height < 64:
                    continue  # skip tiny decorative images
                image_chunks.append(
                    ImageChunk(
                        doc_id=doc_id,
                        image=pil_img,
                        metadata={
                            "source": pdf_path.name,
                            "page": page_num,
                            "img_index": img_index,
                            "width": pil_img.width,
                            "height": pil_img.height,
                        },
                    )
                )
            except Exception as exc:
                logger.warning("Could not decode image", page=page_num, error=str(exc))

    doc.close()
    logger.info(
        "PDF processed",
        doc_id=doc_id,
        text_chunks=len(text_chunks),
        image_chunks=len(image_chunks),
    )
    return text_chunks, image_chunks


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Simple character-level sliding window chunker."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks
