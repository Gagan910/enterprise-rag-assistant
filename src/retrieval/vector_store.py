import os
from typing import Any

import chromadb

from src.config.settings import settings


class VectorStore:
    """Vector store backed by ChromaDB."""

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str | None = None,
    ):
        persist_directory = persist_directory or settings.vector_store_path
        collection_name = collection_name or settings.vector_collection_name

        if os.getenv("CHROMA_API_KEY"):
            self.client = chromadb.CloudClient(
                api_key=os.environ["CHROMA_API_KEY"],
                tenant=os.environ["CHROMA_TENANT"],
                database=os.environ["CHROMA_DATABASE"],
            )
        else:
            self.client = chromadb.PersistentClient(
                path=persist_directory
            )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def exists(self, ids: list[str]) -> bool:
        """Return True if any of the given chunk IDs already exist."""
        if not ids:
            return False

        result = self.collection.get(ids=ids)

        return bool(result.get("ids"))

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

    def delete(self, ids: list[str]) -> None:
        """Delete chunks by their IDs."""
        if not ids:
            return

        existing_ids = self.collection.get(ids=ids).get("ids", [])

        if existing_ids:
            self.collection.delete(ids=existing_ids)

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

    def get_by_document_id(
        self,
        document_id: str,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """Return all chunks belonging to a document, optionally scoped to a workspace."""
        if not isinstance(document_id, str):
            raise TypeError("document_id must be a string")

        if not document_id.strip():
            raise ValueError("document_id cannot be empty")

        if workspace_id:
            where = {
                "$and": [
                    {"document_id": document_id},
                    {"workspace_id": workspace_id},
                ]
            }
        else:
            where = {"document_id": document_id}

        return self.collection.get(where=where)

    def get_all_documents(
        self,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """Return all stored chunks, optionally scoped to a workspace."""
        if workspace_id:
            return self.collection.get(
                where={"workspace_id": workspace_id}
            )

        return self.collection.get()

    def count(self) -> int:
        """Return the number of stored chunks."""
        return self.collection.count()