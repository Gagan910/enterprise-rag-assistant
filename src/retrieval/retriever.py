from typing import Any

from src.retrieval.embedder import TextEmbedder
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import VectorStore
from src.config.settings import settings


class Retriever:
    """Retrieve and rerank relevant document chunks."""

    def __init__(
        self,
        embedder: TextEmbedder,
        vector_store: VectorStore,
        reranker: Reranker | None = None,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        rerank_top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant chunks and optionally rerank them."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not query.strip():
            raise ValueError("query cannot be empty")

        top_k = (
            settings.retrieval_top_k
            if top_k is None
            else top_k
        )

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_embedding = self.embedder.embed_text(query)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            where=where,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        print(
            f"RETRIEVAL DEBUG: documents={len(documents)}, "
            f"metadatas={len(metadatas)}, "
            f"distances={len(distances)}, "
            f"ids={len(ids)}",
            flush=True,
        )

        retrieved_chunks = []

        for document, metadata, distance, chunk_id in zip(
            documents,
            metadatas,
            distances,
            ids,
        ):
            retrieved_chunks.append(
                {
                    "id": chunk_id,
                    "text": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

        if self.reranker and retrieved_chunks:
            rerank_k = (
                rerank_top_k or settings.rerank_top_k
            )

            return self.reranker.rerank(
                query=query,
                documents=retrieved_chunks,
                top_k=rerank_k,
            )

        return retrieved_chunks
    