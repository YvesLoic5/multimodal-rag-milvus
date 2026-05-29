"""Image file loader and preprocessor for PNG/JPG inputs."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from app.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
MAX_SIDE = 1024  # resize if either dimension exceeds this


@dataclass
class LoadedImage:
    doc_id: str
    image: Image.Image
    file_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)


def load_image(
    image_path: str | Path,
    doc_id: str | None = None,
) -> LoadedImage:
    """Load, validate, and preprocess a single image file."""
    image_path = Path(image_path)
    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {image_path.suffix}")

    doc_id = doc_id or str(uuid.uuid5(uuid.NAMESPACE_URL, image_path.name))
    img = Image.open(image_path).convert("RGB")
    img = _resize_if_needed(img, MAX_SIDE)

    logger.info("Image loaded", path=str(image_path), size=img.size, doc_id=doc_id)
    return LoadedImage(
        doc_id=doc_id,
        image=img,
        file_path=image_path,
        metadata={
            "source": image_path.name,
            "width": img.width,
            "height": img.height,
            "format": image_path.suffix.lower().lstrip("."),
        },
    )


def load_images_from_directory(directory: str | Path) -> list[LoadedImage]:
    """Recursively load all supported images from *directory*."""
    directory = Path(directory)
    images: list[LoadedImage] = []
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                images.append(load_image(path))
            except Exception as exc:
                logger.warning("Skipping image", path=str(path), error=str(exc))
    logger.info("Loaded images from directory", directory=str(directory), count=len(images))
    return images


def _resize_if_needed(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)
