"""RAGAS evaluation script.

Builds a 20-pair (question, ground_truth) dataset, runs the full RAG pipeline
for each question, and evaluates with:
  - faithfulness
  - answer_relevancy
  - context_recall
  - context_precision

Results are saved to data/evaluation_results.json and printed as a table.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from tqdm import tqdm

from app.generation.llm_chain import generate_answer
from app.retrieval.hybrid_retriever import get_retriever
from app.retrieval.reranker import get_reranker
from app.utils.config import get_settings
from app.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

RESULTS_PATH = Path("data/evaluation_results.json")

# ── Sample evaluation dataset ─────────────────────────────────────────────────
# Replace these with domain-specific Q&A pairs from your actual documents.
EVAL_DATASET: list[dict[str, str]] = [
    {
        "question": "What is the main topic of the document?",
        "ground_truth": "The document covers machine learning fundamentals.",
    },
    {
        "question": "What datasets are mentioned in the paper?",
        "ground_truth": "The paper mentions CIFAR-10, ImageNet, and MNIST datasets.",
    },
    {
        "question": "What evaluation metrics are used?",
        "ground_truth": "The evaluation uses accuracy, F1-score, and AUC-ROC.",
    },
    {
        "question": "Who are the authors of the study?",
        "ground_truth": "The study was conducted by researchers at the AI lab.",
    },
    {
        "question": "What is the proposed architecture?",
        "ground_truth": "The paper proposes a transformer-based architecture.",
    },
    {
        "question": "What are the limitations described?",
        "ground_truth": "The main limitation is the computational cost of training.",
    },
    {
        "question": "What is the conclusion of the paper?",
        "ground_truth": "The method outperforms baselines on all benchmarks.",
    },
    {
        "question": "What hardware was used for experiments?",
        "ground_truth": "Experiments were run on NVIDIA A100 GPUs.",
    },
    {
        "question": "What is the training dataset size?",
        "ground_truth": "The model was trained on 1 million samples.",
    },
    {
        "question": "How is retrieval performed?",
        "ground_truth": "Retrieval uses dense and sparse hybrid search.",
    },
    {
        "question": "What embedding model is used?",
        "ground_truth": "BGE-M3 is used for generating text embeddings.",
    },
    {
        "question": "How are images processed?",
        "ground_truth": "Images are encoded using the CLIP model.",
    },
    {
        "question": "What caching strategy is used?",
        "ground_truth": "Semantic caching with Redis reduces redundant queries.",
    },
    {
        "question": "What vector database is used?",
        "ground_truth": "Milvus is used as the vector database.",
    },
    {
        "question": "What is the chunk size for text ingestion?",
        "ground_truth": "Text is split into chunks of 512 characters with 50 overlap.",
    },
    {
        "question": "What LLM is used for generation?",
        "ground_truth": "GPT-4o is used by default for answer generation.",
    },
    {
        "question": "How does hybrid retrieval work?",
        "ground_truth": "Hybrid retrieval combines dense cosine search with sparse BM25.",
    },
    {
        "question": "What reranking model is used?",
        "ground_truth": "A cross-encoder ms-marco-MiniLM-L-6-v2 reranks results.",
    },
    {
        "question": "What is RRF?",
        "ground_truth": "RRF stands for Reciprocal Rank Fusion for merging ranked lists.",
    },
    {
        "question": "What UI framework is used for the chatbot?",
        "ground_truth": "Chainlit is used to build the chatbot interface.",
    },
]


def run_pipeline_for_question(
    question: str,
    retriever: Any,
    reranker: Any,
) -> tuple[str, list[str]]:
    """Run the full RAG pipeline for one question.

    Returns (answer, list_of_context_strings).
    """
    hits = retriever.retrieve(question)
    hits = reranker.rerank(question, hits)
    contexts = [h["content_text"] for h in hits]
    answer = generate_answer(question, hits)
    return answer, contexts


def build_ragas_dataset(num_samples: int = 20) -> Dataset:
    """Build a HuggingFace Dataset for RAGAS evaluation."""
    retriever = get_retriever()
    reranker = get_reranker()
    settings = get_settings()

    samples = EVAL_DATASET[:num_samples]
    rows: dict[str, list[Any]] = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for item in tqdm(samples, desc="Running RAG pipeline"):
        try:
            answer, contexts = run_pipeline_for_question(
                item["question"], retriever, reranker
            )
        except Exception as exc:
            logger.error("Pipeline failed", question=item["question"], error=str(exc))
            answer = ""
            contexts = []

        rows["question"].append(item["question"])
        rows["answer"].append(answer)
        rows["contexts"].append(contexts if contexts else ["No context found."])
        rows["ground_truth"].append(item["ground_truth"])

    return Dataset.from_dict(rows)


def run_evaluation(num_samples: int = 20) -> dict[str, Any]:
    """Run RAGAS evaluation and return a results dict."""
    logger.info("Starting RAGAS evaluation", num_samples=num_samples)
    dataset = build_ragas_dataset(num_samples)

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )

    scores: dict[str, float] = {
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
        "context_recall": float(result["context_recall"]),
        "context_precision": float(result["context_precision"]),
    }
    avg = sum(scores.values()) / len(scores)

    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "num_samples": num_samples,
        "scores": scores,
        "average_score": avg,
    }

    # ── Save to disk ───────────────────────────────────────────────────────────
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(output, indent=2))
    logger.info("Results saved", path=str(RESULTS_PATH))

    # ── Print table ────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  RAGAS Evaluation Results  ({datetime.utcnow().strftime('%Y-%m-%d')})")
    print("=" * 55)
    print(f"  {'Metric':<30} {'Score':>10}")
    print("-" * 55)
    for metric, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {metric:<30} {score:>8.4f}  {bar}")
    print("-" * 55)
    print(f"  {'AVERAGE':<30} {avg:>8.4f}")
    print("=" * 55)
    print(f"\nResults saved to: {RESULTS_PATH}\n")

    return output


if __name__ == "__main__":
    run_evaluation()
