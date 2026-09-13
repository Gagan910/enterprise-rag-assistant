from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.generation.generator import RAGGenerator
from src.generation.llm import LLMClient
from src.retrieval.embedder import TextEmbedder
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore
from src.config.settings import settings


router = APIRouter()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: str | None = None


class Source(BaseModel):
    source: str
    chunk_id: int | str

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class DocumentInfo(BaseModel):
    document_id: str
    source: str
    chunk_count: int


embedder = TextEmbedder()
vector_store = VectorStore(
    collection_name=settings.vector_collection_name
)
reranker = Reranker()
retriever = Retriever(
    embedder=embedder,
    vector_store=vector_store,
    reranker=reranker,
)


@router.get("/documents", response_model=list[DocumentInfo])
def list_documents(
    skip: int = 0,
    limit: int = 100,
) -> list[DocumentInfo]:
    results = vector_store.get_all_documents()

    metadatas = results.get("metadatas", [])

    documents: dict[str, DocumentInfo] = {}

    for metadata in metadatas:
        document_id = metadata.get("document_id")
        source = metadata.get("source", "Unknown")

        if not document_id:
            continue

        if document_id not in documents:
            documents[document_id] = DocumentInfo(
                document_id=document_id,
                source=source,
                chunk_count=0,
            )

        documents[document_id].chunk_count += 1

    return list(documents.values())[skip : skip + limit]


@router.get("/documents/{document_id}", response_model=DocumentInfo)
def get_document(document_id: str) -> DocumentInfo:
    if not document_id.strip():
        raise HTTPException(
            status_code=400,
            detail="document_id cannot be empty.",
        )

    results = vector_store.get_by_document_id(document_id)

    metadatas = results.get("metadatas", [])

    if not metadatas:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    source = metadatas[0].get("source", "Unknown")

    return DocumentInfo(
        document_id=document_id,
        source=source,
        chunk_count=len(metadatas),
    )


@router.delete("/documents/{document_id}")
def delete_document(document_id: str) -> dict[str, str]:
    if not document_id.strip():
        raise HTTPException(
            status_code=400,
            detail="document_id cannot be empty.",
        )

    results = vector_store.get_by_document_id(document_id)

    if not results.get("ids"):
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    vector_store.delete(results["ids"])

    return {
        "message": "Document deleted successfully.",
        "document_id": document_id,
    }


@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    try:
        where = (
            {"document_id": request.document_id}
            if request.document_id
            else None
        )

        contexts = retriever.retrieve(
            query=request.question,
            top_k=request.top_k,
            rerank_top_k=request.top_k,
            where=where,
        )

        if not contexts:
            return QueryResponse(
                answer="I could not find this information in the provided documents.",
                sources=[],
            )

        sources = [
            Source(
                source=context["metadata"].get("source", "Unknown"),
                chunk_id=context["metadata"].get("chunk_id", context["id"]),
            )
            for context in contexts
        ]

        llm_client = LLMClient()
        generator = RAGGenerator(llm_client)

        answer = generator.generate(
            question=request.question,
            contexts=contexts,
        )

        return QueryResponse(
            answer=answer,
            sources=sources,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to process the query.",
        ) from exc
