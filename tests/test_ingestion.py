"""Tests for the ingestion pipeline.

All external dependencies (Milvus, BGE-M3, CLIP) are mocked so these tests
run without any infrastructure.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image


# ── Text splitting ─────────────────────────────────────────────────────────────

def test_split_text_basic() -> None:
    from app.ingestion.pdf_loader import _split_text

    text = "a" * 1000
    chunks = _split_text(text, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 200


def test_split_text_short_input() -> None:
    from app.ingestion.pdf_loader import _split_text

    text = "Hello world"
    chunks = _split_text(text, chunk_size=512, overlap=50)
    assert chunks == ["Hello world"]


def test_split_text_empty() -> None:
    from app.ingestion.pdf_loader import _split_text

    chunks = _split_text("", chunk_size=512, overlap=50)
    assert chunks == []


# ── Image loader ───────────────────────────────────────────────────────────────

def test_resize_if_needed_no_resize() -> None:
    from app.ingestion.image_loader import _resize_if_needed

    img = Image.new("RGB", (100, 100))
    result = _resize_if_needed(img, 1024)
    assert result.size == (100, 100)


def test_resize_if_needed_large_image() -> None:
    from app.ingestion.image_loader import _resize_if_needed

    img = Image.new("RGB", (2000, 1000))
    result = _resize_if_needed(img, 1024)
    assert max(result.size) == 1024


def test_load_image_unsupported_format(tmp_path: Path) -> None:
    from app.ingestion.image_loader import load_image

    fake_file = tmp_path / "test.svg"
    fake_file.write_text("<svg/>")
    with pytest.raises(ValueError, match="Unsupported"):
        load_image(fake_file)


def test_load_image_valid(tmp_path: Path) -> None:
    from app.ingestion.image_loader import load_image

    img = Image.new("RGB", (200, 200), color=(128, 0, 128))
    img_path = tmp_path / "test.png"
    img.save(img_path)

    loaded = load_image(img_path)
    assert loaded.image.size == (200, 200)
    assert loaded.metadata["source"] == "test.png"


# ── Pipeline (mocked) ─────────────────────────────────────────────────────────

@patch("app.ingestion.pipeline.get_vector_store")
@patch("app.ingestion.pipeline.get_bge_encoder")
@patch("app.ingestion.pipeline.get_clip_encoder")
def test_ingest_image_file(
    mock_clip: MagicMock,
    mock_bge: MagicMock,
    mock_store: MagicMock,
    tmp_path: Path,
) -> None:
    from app.ingestion.pipeline import ingest_file

    # Setup mocks
    mock_clip.return_value.encode_image.return_value = np.zeros(1024, dtype=np.float32)
    mock_bge.return_value.encode_single.return_value = {"dense": np.zeros(1024), "sparse": {0: 1.0}}
    store_mock = MagicMock()
    store_mock.delete_by_doc_id.return_value = 0
    store_mock.insert_batch.return_value = [1]
    mock_store.return_value = store_mock

    img = Image.new("RGB", (200, 200))
    img_path = tmp_path / "sample.png"
    img.save(img_path)

    counts = ingest_file(img_path)
    assert counts["image_chunks"] == 1
    assert counts["text_chunks"] == 0
    store_mock.insert_batch.assert_called_once()


@patch("app.ingestion.pipeline.get_vector_store")
@patch("app.ingestion.pipeline.get_bge_encoder")
def test_ingest_text_file(
    mock_bge: MagicMock,
    mock_store: MagicMock,
    tmp_path: Path,
) -> None:
    from app.ingestion.pipeline import ingest_file

    mock_bge.return_value.encode.return_value = [
        {"dense": np.zeros(1024), "sparse": {0: 1.0}}
    ]
    store_mock = MagicMock()
    store_mock.delete_by_doc_id.return_value = 0
    store_mock.insert_batch.return_value = [1]
    mock_store.return_value = store_mock

    txt_path = tmp_path / "sample.txt"
    txt_path.write_text("This is a test document " * 50)

    counts = ingest_file(txt_path)
    assert counts["text_chunks"] >= 1
    assert counts["image_chunks"] == 0
