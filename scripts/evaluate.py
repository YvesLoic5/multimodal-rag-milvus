#!/usr/bin/env python3
"""CLI script to launch RAGAS evaluation.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --samples 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.logger import setup_logging

setup_logging()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of Q&A pairs to evaluate (max 20)",
    )
    args = parser.parse_args()

    from evaluation.ragas_eval import run_evaluation
    run_evaluation(num_samples=min(args.samples, 20))


if __name__ == "__main__":
    main()
