from dotenv import load_dotenv

load_dotenv()

from collections.abc import Sequence
from typing import Any

from src.evaluation.mlflow_tracking import log_retrieval_evaluation

def evaluate_retrieval(
    retriever: Any,
    dataset: Sequence[dict[str, str]],
    top_k: int = 5,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    results = []
    reciprocal_ranks = []

    for case in dataset:
        contexts = retriever.retrieve(
            query=case["question"],
            top_k=top_k,
            rerank_top_k=top_k,
        )

        expected_source = case["expected_source"]

        ranks = [
            index
            for index, context in enumerate(contexts, start=1)
            if context.get("metadata", {}).get("source") == expected_source
        ]

        rank = min(ranks) if ranks else None
        hit = rank is not None

        reciprocal_ranks.append(1 / rank if rank is not None else 0.0)

        results.append(
            {
                "question": case["question"],
                "expected_source": expected_source,
                "rank": rank,
                "hit": hit,
            }
        )

    total = len(results)

    return {
        "total_cases": total,
        "hit_rate": (
            sum(result["hit"] for result in results) / total
            if total
            else 0.0
        ),
        "mrr": (
            sum(reciprocal_ranks) / total
            if total
            else 0.0
        ),
        "results": results,
    }
    
def evaluate_and_log_retrieval(
    retriever: Any,
    dataset: Sequence[dict[str, str]],
    top_k: int = 5,
) -> dict[str, Any]:
    """Evaluate retrieval and log the results to MLflow."""
    metrics = evaluate_retrieval(
        retriever=retriever,
        dataset=dataset,
        top_k=top_k,
    )

    log_retrieval_evaluation(
        metrics=metrics,
        top_k=top_k,
    )

    return metrics