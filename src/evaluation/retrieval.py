from collections.abc import Sequence
from typing import Any


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