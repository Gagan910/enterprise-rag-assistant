import logging
import time
from typing import Any

from src.config.settings import settings
from src.retrieval.embedder import TextEmbedder
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import VectorStore


logger = logging.getLogger(__name__)


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

        # 1. Query embedding
        embedding_start = time.perf_counter()

        query_embedding = self.embedder.embed_text(query)

        embedding_time = time.perf_counter() - embedding_start

        print(
            f"RETRIEVAL TIMING embedding={embedding_time:.2f}s",
            flush=True,
        )

        # 2. Vector search
        search_start = time.perf_counter()

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            where=where,
        )

        search_time = time.perf_counter() - search_start

        print(
            f"RETRIEVAL TIMING vector_search={search_time:.2f}s",
            flush=True,
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

        # 3. Reranking
        if self.reranker and retrieved_chunks:
            rerank_k = (
                rerank_top_k or settings.rerank_top_k
            )

            rerank_start = time.perf_counter()

            reranked = self.reranker.rerank(
                query=query,
                documents=retrieved_chunks,
                top_k=rerank_k,
            )

            rerank_time = time.perf_counter() - rerank_start

            print(
                f"RETRIEVAL TIMING rerank={rerank_time:.2f}s",
                flush=True,
            )

            return reranked

        return retrieved_chunks
    