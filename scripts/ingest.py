#!/usr/bin/env python3
"""CLI script for batch ingestion of documents into Milvus.

Usage:
    python scripts/ingest.py --path data/raw/
    python scripts/ingest.py --path data/raw/report.pdf
    python scripts/ingest.py --path data/raw/ --reset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.pipeline import ingest_directory, ingest_file
from app.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into Milvus")
    parser.add_argument("--path", required=True, help="File or directory to ingest")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the Milvus collection before ingesting",
    )
    args = parser.parse_args()

    target = Path(args.path)
    if not target.exists():
        print(f"Error: {target} does not exist")
        sys.exit(1)

    if args.reset:
        from app.vectorstore.milvus_client import get_vector_store
        store = get_vector_store()
        store.drop_collection()
        print("Collection dropped. Reinitialising…")
        # Re-trigger singleton recreation
        import app.vectorstore.milvus_client as mvc
        mvc._store_instance = None

    if target.is_dir():
        counts = ingest_directory(target)
    else:
        counts = ingest_file(target)

    print(f"\n✅ Ingestion complete:")
    print(f"   📄 Text chunks  : {counts['text_chunks']}")
    print(f"   🖼️  Image chunks : {counts['image_chunks']}")
    print(f"   Total          : {sum(counts.values())}")


if __name__ == "__main__":
    main()
