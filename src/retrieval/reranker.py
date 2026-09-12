from sentence_transformers import CrossEncoder

from src.config.settings import settings

class Reranker:
    """Rerank retrieved document chunks using a cross-encoder."""

    def __init__(
        self,
        model_name: str | None = None,
    ):
        self.model_name = model_name or settings.reranker_model
        self.model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 3,
    ) -> list[dict]:
        """Rerank documents according to query-document relevance."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not query.strip():
            raise ValueError("query cannot be empty")

        if not documents:
            return []

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        pairs = [(query, document["text"]) for document in documents]

        scores = self.model.predict(pairs)

        reranked = []

        for document, score in zip(documents, scores):
            result = document.copy()
            result["rerank_score"] = float(score)
            reranked.append(result)

        reranked.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return reranked[:top_k]