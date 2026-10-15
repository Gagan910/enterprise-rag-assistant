from fastapi import Request, Response
from src.auth.session_service import (
    SessionAuthenticationError,
    create_session,
    get_session,
    refresh_session,
    revoke_session,
)
from src.auth.session_store import RedisSessionStore

import asyncio
import time
from hashlib import sha256
from pathlib import Path

import requests
from fastapi import (
    APIRouter,
    Depends,
    File,
    Header,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from src.config.settings import settings
from src.generation.generator import RAGGenerator
from src.generation.llm import LLMClient, LLMUnavailableError


router = APIRouter()


import secrets


class SessionCreateRequest(BaseModel):
    access_token: str = Field(..., min_length=1, max_length=8192)
    refresh_token: str = Field(..., min_length=1, max_length=8192)


def _get_session_store() -> RedisSessionStore:
    if not settings.redis_url:
        raise HTTPException(
            status_code=503,
            detail="Session persistence is not configured.",
        )

    try:
        return RedisSessionStore(settings.redis_url)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Session persistence is unavailable.",
        ) from exc


def _validate_frontend_origin(request: Request) -> None:
    origin = request.headers.get("origin", "").rstrip("/")
    allowed = {
        "https://enterprise-rag-frontend-g1bm.onrender.com",
    }

    if not origin or origin not in allowed:
        raise HTTPException(
            status_code=403,
            detail="Origin not allowed.",
        )


@router.post("/auth/session")
def create_auth_session(
    payload: SessionCreateRequest,
    request: Request,
):
    """Create a Redis-backed session and return its opaque identifier."""
    _validate_frontend_origin(request)
    store = _get_session_store()

    try:
        session_id, user = create_session(
            store,
            payload.access_token,
            payload.refresh_token,
        )
    except SessionAuthenticationError as exc:
        raise HTTPException(
            status_code=401,
            detail="Unable to establish an authenticated session.",
        ) from exc

    return {
        "session_id": session_id,
        "user": user,
    }


@router.get("/auth/session")
def restore_auth_session(
    request: Request,
    x_rag_session: str | None = Header(
        default=None,
        alias="X-RAG-Session",
    ),
):
    """Restore a session using its explicitly supplied identifier."""
    _validate_frontend_origin(request)

    if not x_rag_session:
        raise HTTPException(
            status_code=401,
            detail="Session not found.",
        )

    store = _get_session_store()

    try:
        user = get_session(store, x_rag_session)
    except SessionAuthenticationError as exc:
        raise HTTPException(
            status_code=503,
            detail="Session restoration is temporarily unavailable.",
        ) from exc

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Session expired or revoked.",
        )

    return {"user": user}


@router.delete("/auth/session")
def delete_auth_session(
    request: Request,
    x_rag_session: str | None = Header(
        default=None,
        alias="X-RAG-Session",
    ),
):
    """Revoke the supplied server-side session."""
    _validate_frontend_origin(request)

    if x_rag_session:
        store = _get_session_store()
        store.revoke(x_rag_session)

    return {"signed_out": True}

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


def _get_api_key_workspaces() -> dict[str, str]:
    """Build the API-key-to-workspace mapping from configuration."""

    workspaces: dict[str, str] = {}

    if settings.api_key:
        workspaces[settings.api_key] = (
            f"workspace-{sha256(settings.api_key.encode()).hexdigest()[:16]}"
        )

    if settings.api_keys:
        for entry in settings.api_keys.split(","):
            entry = entry.strip()

            if not entry:
                continue

            try:
                api_key, workspace_id = entry.split(":", 1)
            except ValueError as exc:
                raise HTTPException(
                    status_code=503,
                    detail="API key configuration is invalid.",
                ) from exc

            api_key = api_key.strip()
            workspace_id = workspace_id.strip()

            if not api_key or not workspace_id:
                raise HTTPException(
                    status_code=503,
                    detail="API key configuration is invalid.",
                )

            if api_key in workspaces:
                raise HTTPException(
                    status_code=503,
                    detail="API key configuration is invalid.",
                )

            workspaces[api_key] = workspace_id

    return workspaces


def _get_workspace_for_api_key(api_key: str | None) -> str:
    """Return the workspace associated with an authenticated API key."""

    workspaces = _get_api_key_workspaces()

    if not workspaces:
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured.",
        )

    if not api_key or api_key not in workspaces:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
        )

    return workspaces[api_key]


def _get_workspace_for_supabase_token(token: str) -> str:
    """Validate a Supabase access token and map the user to a workspace."""

    if not settings.supabase_url or not settings.supabase_anon_key:
        raise HTTPException(
            status_code=503,
            detail="Supabase authentication is not configured.",
        )

    url = f"{settings.supabase_url.rstrip('/')}/auth/v1/user"

    try:
        response = requests.get(
            url,
            headers={
                "apikey": settings.supabase_anon_key,
                "Authorization": f"Bearer {token}",
            },
            timeout=15,
        )
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail="Unable to validate Supabase authentication.",
        ) from exc

    if not response.ok:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired Supabase access token.",
        )

    try:
        user = response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=401,
            detail="Invalid Supabase authentication response.",
        ) from exc

    user_id = user.get("id")

    if not isinstance(user_id, str) or not user_id.strip():
        raise HTTPException(
            status_code=401,
            detail="Supabase user identity is missing.",
        )

    return f"user-{user_id}"


def get_workspace_id(
    api_key: str | None = Depends(api_key_header),
    authorization: str | None = Header(default=None),
    x_rag_session: str | None = Header(
        default=None,
        alias="X-RAG-Session",
    ),
) -> str:
    """Resolve a workspace from a Supabase token, persistent session, or API key."""

    if not isinstance(authorization, str):
        authorization = None

    if not isinstance(x_rag_session, str):
        x_rag_session = None

    # Preserve existing Supabase Bearer-token authentication.
    if authorization:
        scheme, _, token = authorization.partition(" ")

        if scheme.lower() == "bearer" and token.strip():
            return _get_workspace_for_supabase_token(token.strip())

        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header.",
        )

    # New Redis-backed persistent-session authentication.
    if x_rag_session:
        store = _get_session_store()

        try:
            user = get_session(store, x_rag_session)
        except SessionAuthenticationError as exc:
            raise HTTPException(
                status_code=503,
                detail="Session authentication is temporarily unavailable.",
            ) from exc

        if user is None:
            raise HTTPException(
                status_code=401,
                detail="Session expired or revoked. Please log in again.",
            )

        user_id = user.get("user_id")
        if not isinstance(user_id, str) or not user_id:
            raise HTTPException(
                status_code=401,
                detail="Session user identity is invalid.",
            )

        return f"user-{user_id}"

    # Preserve legacy API-key authentication.
    return _get_workspace_for_api_key(api_key)


def require_api_key(
    api_key: str | None = Depends(api_key_header),
) -> None:
    """Legacy compatibility dependency for API-key-only callers/tests."""

    _get_workspace_for_api_key(api_key)


def require_authentication(
    workspace_id: str = Depends(get_workspace_id),
) -> None:
    """Require either a valid Supabase session or a legacy API key."""

    return None




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
    provider: str | None = None
    fallback_used: bool = False
    provider_attempts: list[str] = []


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

embedder = _LazyComponent(
    lambda: components._initialize() or components.embedder
)

vector_store = _LazyComponent(
    lambda: components._initialize() or components.vector_store
)

reranker = _LazyComponent(
    lambda: components._initialize() or components.reranker
)

retriever = _LazyComponent(
    lambda: components._initialize() or components.retriever
)

ingestion_pipeline = _LazyComponent(
    lambda: components._initialize() or components.ingestion_pipeline
)


# Shared lazy LLM client.
# This preserves provider state and Gemini cooldown across requests.
llm_client = _LazyComponent(lambda: LLMClient())


UPLOAD_DIR = Path("data/uploads")
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_UPLOAD_SIZE = 10 * 1024 * 1024


@router.post(
    "/documents/upload",
    response_model=DocumentInfo,
    dependencies=[Depends(require_authentication)],
)
async def upload_document(
    file: UploadFile = File(...),
    workspace_id: str = Depends(get_workspace_id),
) -> DocumentInfo:
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
        existing = vector_store.get_by_document_id(
            document_id,
            workspace_id=workspace_id,
        )
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

        chunk_count = ingestion_pipeline.ingest(
            str(file_path),
            workspace_id=workspace_id,
        )

        if chunk_count == 0:
            results = vector_store.get_by_document_id(
                document_id,
                workspace_id=workspace_id,
            )
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


@router.get(
    "/documents",
    response_model=list[DocumentInfo],
    dependencies=[Depends(require_authentication)],
)
def list_documents(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=100),
    workspace_id: str = Depends(get_workspace_id),
) -> list[DocumentInfo]:
    try:
        results = vector_store.get_all_documents(
            workspace_id=workspace_id,
        )

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


@router.get(
    "/documents/{document_id}",
    response_model=DocumentInfo,
    dependencies=[Depends(require_authentication)],
)
def get_document(
    document_id: str,
    workspace_id: str = Depends(get_workspace_id),
) -> DocumentInfo:
    try:
        if not document_id.strip():
            raise HTTPException(
                status_code=400,
                detail="document_id cannot be empty.",
            )

        results = vector_store.get_by_document_id(
            document_id,
            workspace_id=workspace_id,
        )

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


@router.delete(
    "/documents/{document_id}",
    dependencies=[Depends(require_authentication)],
)
def delete_document(
    document_id: str,
    workspace_id: str = Depends(get_workspace_id),
) -> dict:
    try:
        if not document_id.strip():
            raise HTTPException(
                status_code=400,
                detail="document_id cannot be empty.",
            )

        results = vector_store.get_by_document_id(
            document_id,
            workspace_id=workspace_id,
        )

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


@router.post(
    "/query",
    response_model=QueryResponse,
    dependencies=[Depends(require_authentication)],
)
async def query(
    request: QueryRequest,
    workspace_id: str = Depends(get_workspace_id),
) -> QueryResponse:
    try:
        if request.document_id:
            where = {
                "$and": [
                    {"workspace_id": workspace_id},
                    {"document_id": request.document_id},
                ]
            }
        else:
            where = {
                "workspace_id": workspace_id,
            }

        retrieval_start = time.perf_counter()

        contexts = await asyncio.to_thread(
            lambda: retriever.retrieve(
                query=request.question,
                top_k=request.top_k,
                rerank_top_k=request.top_k,
                where=where,
            )
        )

        retrieval_time = time.perf_counter() - retrieval_start

        print(
            f"QUERY TIMING retrieval={retrieval_time:.2f}s",
            flush=True,
        )

        if not contexts:
            return QueryResponse(
                answer="I could not find this information in the provided documents.",
                sources=[],
            )

        sources = [
            Source(
                source=context["metadata"].get("source", "Unknown"),
                chunk_id=context["metadata"].get(
                    "chunk_id",
                    context["id"],
                ),
            )
            for context in contexts
        ]

        def generate_answer() -> tuple[str, str | None, list[str]]:
            generator = RAGGenerator(llm_client)

            answer = generator.generate(
                question=request.question,
                contexts=contexts,
            )

            return (
                answer,
                generator.provider_used,
                generator.provider_attempts,
            )

        llm_start = time.perf_counter()

        answer, provider, provider_attempts = await asyncio.to_thread(
            generate_answer
        )

        llm_time = time.perf_counter() - llm_start

        print(
            f"QUERY TIMING llm={llm_time:.2f}s provider={provider}",
            flush=True,
        )

        return QueryResponse(
            answer=answer,
            sources=sources,
            provider=provider,
            fallback_used=(
                provider is not None
                and provider != settings.llm_provider
            ),
            provider_attempts=provider_attempts,
        )

    except LLMUnavailableError as exc:
        print(
            f"QUERY ERROR: {type(exc).__name__}: {exc}",
            flush=True,
        )

        raise HTTPException(
            status_code=503,
            detail="The language model is temporarily unavailable. Please try again later.",
        ) from exc

    except Exception as exc:
        print(
            f"QUERY ERROR: {type(exc).__name__}: {exc}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to process the query.",
        ) from exc
        