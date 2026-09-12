from typing import Any
from src.config.settings import settings

import chromadb


class VectorStore:
    """Persistent vector store backed by ChromaDB."""

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str | None = None,
    ):
        persist_directory = persist_directory or settings.vector_store_path
        collection_name = collection_name or settings.vector_collection_name
        self.client = chromadb.PersistentClient(path=persist_directory)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> None:
        """Add document chunks and their embeddings to the vector store."""
        if not chunks:
            return

        if not (
            len(chunks)
            == len(embeddings)
            == len(metadatas)
            == len(ids)
        ):
            raise ValueError(
                "chunks, embeddings, metadatas, and ids must have the same length"
            )

        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Search for similar chunks with optional metadata filtering."""
        if not query_embedding:
            raise ValueError("query_embedding cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )

    def count(self) -> int:
        """Return the number of stored chunks."""
        return self.collection.count()