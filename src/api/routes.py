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


class Source(BaseModel):
    source: str
    chunk_id: int | str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


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


@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    try:
        contexts = retriever.retrieve(
            query=request.question,
            top_k=request.top_k,
            rerank_top_k=request.top_k,
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