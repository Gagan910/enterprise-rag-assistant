import asyncio
from hashlib import sha256
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field


from src.generation.generator import RAGGenerator
from src.generation.llm import LLMClient
from src.config.settings import settings


router = APIRouter()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: str | None = Field(default=None, min_length=1)


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


class _LazyComponent:
    def __init__(self, factory):
        self._factory = factory
        self._instance = None

    def _get_instance(self):
        if self._instance is None:
            self._instance = self._factory()
        return self._instance

    def __getattr__(self, name):
        return getattr(self._get_instance(), name)


def _create_components():
    from src.ingestion.pipeline import IngestionPipeline
    from src.retrieval.embedder import TextEmbedder
    from src.retrieval.reranker import Reranker
    from src.retrieval.retriever import Retriever
    from src.retrieval.vector_store import VectorStore

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

    ingestion_pipeline = IngestionPipeline(
        embedder=embedder,
        vector_store=vector_store,
    )

    return (
        embedder,
        vector_store,
        reranker,
        retriever,
        ingestion_pipeline,
    )


class _Components:
    def __init__(self):
        self._initialized = False

        self.embedder = None
        self.vector_store = None
        self.reranker = None
        self.retriever = None
        self.ingestion_pipeline = None

    def _initialize(self):
        if not self._initialized:
            (
                self.embedder,
                self.vector_store,
                self.reranker,
                self.retriever,
                self.ingestion_pipeline,
            ) = _create_components()

            self._initialized = True


components = _Components()

embedder = _LazyComponent(lambda: components._initialize() or components.embedder)
vector_store = _LazyComponent(lambda: components._initialize() or components.vector_store)
reranker = _LazyComponent(lambda: components._initialize() or components.reranker)
retriever = _LazyComponent(lambda: components._initialize() or components.retriever)
ingestion_pipeline = _LazyComponent(
    lambda: components._initialize() or components.ingestion_pipeline
)

UPLOAD_DIR = Path("data/uploads")
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_UPLOAD_SIZE = 10 * 1024 * 1024


@router.post("/documents/upload", response_model=DocumentInfo)
async def upload_document(file: UploadFile = File(...)) -> DocumentInfo:
    filename = Path(file.filename or "").name

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required.",
        )

    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported document type. Allowed types: .pdf, .docx, .txt",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Uploaded file exceeds the 10 MB limit.",
        )

    document_id = sha256(content).hexdigest()[:16]

    try:
        existing = vector_store.get_by_document_id(document_id)
        existing_metadatas = existing.get("metadatas", [])

        if existing_metadatas:
            return DocumentInfo(
                document_id=document_id,
                source=existing_metadatas[0].get("source", filename),
                chunk_count=len(existing_metadatas),
            )

        document_directory = UPLOAD_DIR / document_id
        document_directory.mkdir(parents=True, exist_ok=True)

        file_path = document_directory / filename
        file_path.write_bytes(content)

        chunk_count = ingestion_pipeline.ingest(str(file_path))

        if chunk_count == 0:
            results = vector_store.get_by_document_id(document_id)
            metadatas = results.get("metadatas", [])

            if not metadatas:
                raise HTTPException(
                    status_code=422,
                    detail="Document could not be indexed.",
                )

            chunk_count = len(metadatas)

        return DocumentInfo(
            document_id=document_id,
            source=filename,
            chunk_count=chunk_count,
        )

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to upload and index document.",
        ) from exc


@router.get("/documents", response_model=list[DocumentInfo])
def list_documents(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=100),
) -> list[DocumentInfo]:
    try:
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

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to list documents.",
        ) from exc


@router.get("/documents/{document_id}", response_model=DocumentInfo)
def get_document(document_id: str) -> DocumentInfo:
    try:
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

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to get document.",
        ) from exc


@router.delete("/documents/{document_id}")
def delete_document(document_id: str) -> dict[str, str]:
    try:
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

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to delete document.",
        ) from exc


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

        contexts = await asyncio.to_thread(
            lambda: retriever.retrieve(
                query=request.question,
                top_k=request.top_k,
                rerank_top_k=request.top_k,
                where=where,
            )
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

        def generate_answer() -> str:
            llm_client = LLMClient()
            generator = RAGGenerator(llm_client)

            return generator.generate(
                question=request.question,
                contexts=contexts,
            )

        answer = await asyncio.to_thread(generate_answer)

        return QueryResponse(
            answer=answer,
            sources=sources,
        )

    except Exception as exc:
        print(f"QUERY ERROR: {type(exc).__name__}: {exc}", flush=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to process the query.",
        ) from exc
