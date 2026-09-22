import pytest
import sys
import httpx
from google.genai.errors import ServerError
from pathlib import Path
from unittest.mock import patch
from docx import Document
from fastapi import HTTPException

from fastapi.testclient import TestClient

from google.genai.errors import ClientError, ServerError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.parser import parse_document
from src.ingestion.chunker import chunk_text
from src.generation.prompt import build_rag_prompt
from src.generation.llm import LLMClient, LLMUnavailableError

from src.main import app
from src.config.settings import settings
from src.api.routes import get_workspace_id

settings.api_key = "test-api-key"
from hashlib import sha256

TEST_WORKSPACE_ID = (
    f"workspace-{sha256(settings.api_key.encode()).hexdigest()[:16]}"
)

def test_different_api_keys_produce_different_workspace_ids(monkeypatch):
    from src.api.routes import get_workspace_id

    api_key_a = "user-a-api-key"
    api_key_b = "user-b-api-key"

    monkeypatch.setattr(
        settings,
        "api_key",
        api_key_a,
    )

    workspace_a = get_workspace_id(api_key_a)

    monkeypatch.setattr(
        settings,
        "api_key",
        api_key_b,
    )

    workspace_b = get_workspace_id(api_key_b)

    assert workspace_a == (
        f"workspace-{sha256(api_key_a.encode()).hexdigest()[:16]}"
    )

    assert workspace_b == (
        f"workspace-{sha256(api_key_b.encode()).hexdigest()[:16]}"
    )

    assert workspace_a != workspace_b

def test_api_keys_map_to_configured_workspaces(monkeypatch):
    from src.api.routes import get_workspace_id

    monkeypatch.setattr(
        settings,
        "api_keys",
        "user-a-api-key:workspace-a,user-b-api-key:workspace-b",
    )

    workspace_a = get_workspace_id("user-a-api-key")
    workspace_b = get_workspace_id("user-b-api-key")

    assert workspace_a == "workspace-a"
    assert workspace_b == "workspace-b"
    assert workspace_a != workspace_b
    
def test_legacy_and_multi_api_keys_can_coexist(monkeypatch):
    from src.api.routes import get_workspace_id

    legacy_api_key = "legacy-api-key"

    monkeypatch.setattr(
        settings,
        "api_key",
        legacy_api_key,
    )

    monkeypatch.setattr(
        settings,
        "api_keys",
        "user-a-api-key:workspace-a,user-b-api-key:workspace-b",
    )

    legacy_workspace = get_workspace_id(legacy_api_key)
    workspace_a = get_workspace_id("user-a-api-key")
    workspace_b = get_workspace_id("user-b-api-key")

    assert legacy_workspace == (
        f"workspace-{sha256(legacy_api_key.encode()).hexdigest()[:16]}"
    )

    assert workspace_a == "workspace-a"
    assert workspace_b == "workspace-b"

    assert len(
        {legacy_workspace, workspace_a, workspace_b}
    ) == 3
    
def test_invalid_api_key_is_rejected_with_multi_key_configuration(monkeypatch):
    from src.api.routes import get_workspace_id

    monkeypatch.setattr(
        settings,
        "api_keys",
        "user-a-api-key:workspace-a,user-b-api-key:workspace-b",
    )

    with pytest.raises(HTTPException) as exc_info:
        get_workspace_id("invalid-api-key")

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid or missing API key."
    
def test_malformed_api_keys_configuration_returns_503(monkeypatch):
    from src.api.routes import get_workspace_id

    monkeypatch.setattr(
        settings,
        "api_keys",
        "user-a-api-key:workspace-a,INVALID_ENTRY",
    )

    with pytest.raises(HTTPException) as exc_info:
        get_workspace_id("user-a-api-key")

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "API key configuration is invalid."

def test_duplicate_api_keys_configuration_returns_503(monkeypatch):
    from src.api.routes import get_workspace_id

    monkeypatch.setattr(
        settings,
        "api_keys",
        "user-a-api-key:workspace-a,user-a-api-key:workspace-b",
    )

    with pytest.raises(HTTPException) as exc_info:
        get_workspace_id("user-a-api-key")

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "API key configuration is invalid."

from src.ingestion.cleaner import clean_text
from src.retrieval.embedder import TextEmbedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.retrieval.reranker import Reranker
from src.evaluation.dataset import EVALUATION_DATASET
from src.evaluation.retrieval import evaluate_retrieval


client = TestClient(
    app,
    headers={"X-API-Key": "test-api-key"},
)

def test_protected_endpoint_rejects_missing_api_key():
    response = client.get(
        "/documents",
        headers={"X-API-Key": ""},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key."


def test_protected_endpoint_rejects_invalid_api_key():
    response = client.get(
        "/documents",
        headers={"X-API-Key": "wrong-api-key"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key."


def test_health_endpoint_is_public():
    response = client.get(
        "/health",
        headers={"X-API-Key": "wrong-api-key"},
    )

    assert response.status_code == 200

def test_upload_document():
    from src.api import routes

    with (
        patch.object(
            routes.vector_store,
            "get_by_document_id",
            return_value={"metadatas": []},
        ),
        patch.object(
            routes.ingestion_pipeline,
            "ingest",
            return_value=1,
        ) as mock_ingest,
    ):
        response = client.post(
            "/documents/upload",
            files={
                "file": (
                    "test-upload.txt",
                    b"Employees receive 20 days of paid annual leave.",
                    "text/plain",
                )
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["source"] == "test-upload.txt"
    assert data["chunk_count"] == 1
    assert len(data["document_id"]) == 16

    mock_ingest.assert_called_once()
    
def test_upload_document_rejects_unsupported_type():
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "malicious.exe",
                b"not a supported document",
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Unsupported document type. Allowed types: .pdf, .docx, .txt"
    }

def mock_llm_response(*args, **kwargs):
    return "Employees receive 20 days of paid annual leave each calendar year [Source 1]."

def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_query(monkeypatch):
    class FakeRetriever:
        def retrieve(self, query, top_k, rerank_top_k, where=None):
            assert where == {
                "workspace_id": TEST_WORKSPACE_ID,
            }

            return [
                {
                    "id": "test-document-id-0",
                    "text": "Employees receive 20 days of paid annual leave.",
                    "metadata": {
                        "source": "sample.txt",
                        "chunk_id": 0,
                        "document_id": "test-document-id",
                        "workspace_id": TEST_WORKSPACE_ID,
                    },
                }
            ]

    monkeypatch.setattr(
        "src.api.routes.retriever",
        FakeRetriever(),
    )

    with patch(
        "src.api.routes.LLMClient.generate",
        side_effect=mock_llm_response,
    ):
        response = client.post(
            "/query",
            json={
                "question": "How many days of paid annual leave do employees receive?",
                "top_k": 1,
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert "answer" in data
    assert "sources" in data

    assert data["sources"]
    assert data["sources"][0]["source"] == "sample.txt"

def test_query_rejects_empty_question():
    response = client.post(
        "/query",
        json={
            "question": "",
            "top_k": 1,
        },
    )

    assert response.status_code == 422

def test_query_rejects_invalid_top_k():
    response = client.post(
        "/query",
        json={
            "question": "What is the leave policy?",
            "top_k": 0,
        },
    )

    assert response.status_code == 422

def test_query_accepts_max_top_k():
    with patch("src.api.routes.LLMClient.generate", side_effect=mock_llm_response):
        response = client.post(
            "/query",
            json={
                "question": "What is the work week?",
                "top_k": 20,
            },
        )

    assert response.status_code == 200

def test_query_rejects_top_k_above_limit():
    response = client.post(
        "/query",
        json={
            "question": "What is the work week?",
            "top_k": 21,
        },
    )

    assert response.status_code == 422

def test_query_rejects_missing_question():
    response = client.post(
        "/query",
        json={
            "top_k": 1,
        },
    )

    assert response.status_code == 422

def test_llm_client_generate():
    client = LLMClient()

    mock_response = type(
        "MockResponse",
        (),
        {"text": "Test response"},
    )()

    with patch.object(
        client.client.models,
        "generate_content",
        return_value=mock_response,
    ) as mock_generate:
        result = client.generate("Test prompt")

    assert result == "Test response"

    mock_generate.assert_called_once_with(
        model=client.model_name,
        contents="Test prompt",
    )
    
def test_llm_client_retries_on_server_error():
    client = LLMClient()

    mock_response = type(
        "Response",
        (),
        {"text": "successful response"},
    )()

    server_error = ServerError(503, {})

    with patch.object(
        client.client.models,
        "generate_content",
        side_effect=[server_error, mock_response],
    ) as mock_generate:
        result = client.generate("test prompt")

    assert result == "successful response"
    assert mock_generate.call_count == 2


def test_llm_client_retries_on_read_timeout():
    client = LLMClient()

    mock_response = type(
        "Response",
        (),
        {"text": "successful response"},
    )()

    timeout_error = httpx.ReadTimeout("request timed out")

    with patch.object(
        client.client.models,
        "generate_content",
        side_effect=[timeout_error, mock_response],
    ) as mock_generate:
        result = client.generate("test prompt")

    assert result == "successful response"
    assert mock_generate.call_count == 2

def test_llm_client_does_not_retry_permanent_error():
    client = LLMClient()

    with patch.object(
        client.client.models,
        "generate_content",
        side_effect=ValueError("invalid request"),
    ) as mock_generate:
        with pytest.raises(ValueError, match="invalid request"):
            client.generate("test prompt")

    assert mock_generate.call_count == 1

def test_llm_client_rejects_empty_prompt():
    client = LLMClient()

    with pytest.raises(ValueError, match="prompt cannot be empty"):
        client.generate("")

def test_llm_client_rejects_non_string_prompt():
    client = LLMClient()

    with pytest.raises(TypeError, match="prompt must be a string"):
        client.generate(None)

def test_build_rag_prompt():
    contexts = [
        {
            "text": "Employees receive 20 days of paid annual leave.",
            "metadata": {"source": "sample.txt"},
        }
    ]

    prompt = build_rag_prompt(
        question="How many days of annual leave do employees receive?",
        contexts=contexts,
    )

    assert "ONLY the provided context" in prompt
    assert "Employees receive 20 days of paid annual leave." in prompt
    assert "[Source 1: sample.txt]" in prompt
    assert "How many days of annual leave do employees receive?" in prompt
    assert "Do not invent facts." in prompt

def test_build_rag_prompt_rejects_empty_contexts():
    with pytest.raises(ValueError, match="contexts cannot be empty"):
        build_rag_prompt(
            question="What is the leave policy?",
            contexts=[],
        )

def test_chunk_text_creates_overlapping_chunks():
    text = " ".join(f"word{i}" for i in range(1200))

    chunks = chunk_text(
        text,
        chunk_size=500,
        chunk_overlap=100,
    )

    assert len(chunks) == 3
    assert chunks[0].chunk_id == 0
    assert chunks[1].chunk_id == 1
    assert chunks[2].chunk_id == 2

    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()

    assert len(first_words) == 500
    assert len(second_words) == 500
    assert first_words[-100:] == second_words[:100]

def test_chunk_text_returns_empty_for_blank_text():
    chunks = chunk_text("   ")

    assert chunks == []

def test_chunk_text_rejects_invalid_chunk_size():
    with pytest.raises(ValueError, match="chunk_size must be greater than zero"):
        chunk_text(
            "some document text",
            chunk_size=0,
            chunk_overlap=0,
        )

def test_chunk_text_rejects_negative_overlap():
    with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
        chunk_text(
            "some document text",
            chunk_size=100,
            chunk_overlap=-1,
        )

def test_chunk_text_rejects_overlap_equal_to_chunk_size():
    with pytest.raises(
        ValueError,
        match="chunk_overlap must be smaller than chunk_size",
    ):
        chunk_text(
            "some document text",
            chunk_size=100,
            chunk_overlap=100,
        )

def test_parse_document_txt(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text(
        "Employees receive 20 days of paid annual leave.",
        encoding="utf-8",
    )

    result = parse_document(str(file_path))

    assert result == "Employees receive 20 days of paid annual leave."

def test_parse_document_rejects_missing_file():
    with pytest.raises(FileNotFoundError, match="Document not found"):
        parse_document("does_not_exist.txt")

def test_parse_document_rejects_unsupported_file_type(tmp_path):
    file_path = tmp_path / "sample.csv"
    file_path.write_text("name,age\nGagan,25", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported document type"):
        parse_document(str(file_path))

def test_parse_document_docx(tmp_path):
    file_path = tmp_path / "sample.docx"

    document = Document()
    document.add_paragraph("Employees receive 20 days of paid annual leave.")
    document.save(file_path)

    result = parse_document(str(file_path))

    assert result == "Employees receive 20 days of paid annual leave."

def test_parse_document_pdf(tmp_path):
    from reportlab.pdfgen import canvas

    file_path = tmp_path / "sample.pdf"

    pdf = canvas.Canvas(str(file_path))
    pdf.drawString(
        100,
        750,
        "Employees receive 20 days of paid annual leave.",
    )
    pdf.save()

    result = parse_document(str(file_path))

    assert "Employees receive 20 days of paid annual leave." in result

def test_clean_text_normalizes_whitespace():
    raw_text = "  Hello   world.  \r\n\r\n\r\nThis is a test.   "

    result = clean_text(raw_text)

    assert result == "Hello world.\n\nThis is a test."

def test_clean_text_rejects_non_string():
    with pytest.raises(TypeError, match="text must be a string"):
        clean_text(None)

def test_embed_text():
    embedder = TextEmbedder()

    embedding = embedder.embed_text("This is a test document.")

    assert isinstance(embedding, list)
    assert len(embedding) == embedder.dimension
    assert all(isinstance(value, float) for value in embedding)

def test_embed_text_rejects_empty_text():
    embedder = TextEmbedder()

    with pytest.raises(ValueError, match="text cannot be empty"):
        embedder.embed_text("   ")

def test_embed_text_rejects_non_string():
    embedder = TextEmbedder()

    with pytest.raises(TypeError, match="text must be a string"):
        embedder.embed_text(None)

def test_embed_text_rejects_non_string():
    embedder = TextEmbedder()

    with pytest.raises(TypeError, match="text must be a string"):
        embedder.embed_text(None)

def test_embed_texts():
    embedder = TextEmbedder()

    texts = [
        "First document.",
        "Second document.",
    ]

    embeddings = embedder.embed_texts(texts)

    assert len(embeddings) == 2
    assert all(len(embedding) == embedder.dimension for embedding in embeddings)

def test_embed_texts_returns_empty_for_empty_list():
    embedder = TextEmbedder()

    embeddings = embedder.embed_texts([])

    assert embeddings == []

def test_embed_texts_rejects_non_string_items():
    embedder = TextEmbedder()

    with pytest.raises(TypeError, match="all texts must be strings"):
        embedder.embed_texts(
            ["valid text", None]
        )

def test_embed_texts_rejects_empty_strings():
    embedder = TextEmbedder()

    with pytest.raises(
        ValueError,
        match="texts cannot contain empty strings",
    ):
        embedder.embed_texts(
            ["valid text", "   "]
        )

def test_vector_store_add_and_count(tmp_path):
    store = VectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="test_collection",
    )

    store.add_chunks(
        chunks=["Python is a programming language."],
        embeddings=[[1.0, 0.0, 0.0]],
        metadatas=[{"source": "test.txt", "chunk_id": 0}],
        ids=["test-0"],
    )

    assert store.count() == 1

def test_vector_store_search(tmp_path):
    store = VectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="search_collection",
    )

    store.add_chunks(
        chunks=[
            "Python is a programming language.",
            "The company provides health insurance.",
        ],
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        metadatas=[
            {"source": "python.txt", "chunk_id": 0},
            {"source": "insurance.txt", "chunk_id": 0},
        ],
        ids=["python-0", "insurance-0"],
    )

    results = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=1,
    )

    assert results["documents"][0][0] == "Python is a programming language."
    assert results["metadatas"][0][0]["source"] == "python.txt"

def test_vector_store_rejects_empty_query_embedding(tmp_path):
    store = VectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="empty_query_collection",
    )

    with pytest.raises(
        ValueError,
        match="query_embedding cannot be empty",
    ):
        store.search([])

def test_vector_store_rejects_invalid_top_k(tmp_path):
    store = VectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="invalid_top_k_collection",
    )

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        store.search(
            query_embedding=[1.0, 0.0, 0.0],
            top_k=0,
        )

def test_retriever_returns_retrieved_chunks():
    embedder = TextEmbedder()
    vector_store = VectorStore(
        persist_directory="data/test_retriever_chroma",
        collection_name="retriever_collection",
    )

    vector_store.add_chunks(
        chunks=[
            "Python is a programming language.",
            "The company provides health insurance.",
        ],
        embeddings=[
            embedder.embed_text("Python is a programming language."),
            embedder.embed_text("The company provides health insurance."),
        ],
        metadatas=[
            {"source": "python.txt", "chunk_id": 0},
            {"source": "insurance.txt", "chunk_id": 0},
        ],
        ids=["retriever-python-0", "retriever-insurance-0"],
    )

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    results = retriever.retrieve(
        query="What is Python?",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["metadata"]["source"] == "python.txt"

def test_retriever_rejects_empty_query():
    embedder = TextEmbedder()
    vector_store = VectorStore(
        persist_directory="data/test_retriever_chroma",
        collection_name="validation_collection",
    )

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    with pytest.raises(ValueError, match="query cannot be empty"):
        retriever.retrieve("   ")

def test_reranker_sorts_by_relevance():
    reranker = Reranker()

    documents = [
        {"text": "The weather is sunny.", "metadata": {"source": "weather.txt"}},
        {
            "text": "Python is a programming language.",
            "metadata": {"source": "python.txt"},
        },
    ]

    results = reranker.rerank(
        query="What is Python?",
        documents=documents,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["metadata"]["source"] == "python.txt"
    assert "rerank_score" in results[0]
    assert isinstance(results[0]["rerank_score"], float)

def test_reranker_returns_empty_for_no_documents():
    reranker = Reranker()

    results = reranker.rerank(
        query="What is Python?",
        documents=[],
        top_k=3,
    )

    assert results == []

def test_reranker_rejects_empty_query():
    reranker = Reranker()

    with pytest.raises(ValueError, match="query cannot be empty"):
        reranker.rerank(
            query="   ",
            documents=[
                {"text": "Python is a programming language."}
            ],
            top_k=1,
        )

def test_reranker_rejects_invalid_top_k():
    reranker = Reranker()

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        reranker.rerank(
            query="What is Python?",
            documents=[
                {"text": "Python is a programming language."}
            ],
            top_k=0,
        )

def test_ingestion_pipeline(tmp_path):
    from src.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline(
        embedder=TextEmbedder(),
        vector_store=VectorStore(
            persist_directory=str(tmp_path / "chroma"),
            collection_name="pipeline_collection",
        ),
    )

    file_path = Path("data/sample.txt")

    chunks_ingested = pipeline.ingest(str(file_path))

    assert chunks_ingested > 0
    results = pipeline.vector_store.collection.get()

    assert len(results["metadatas"]) == chunks_ingested
    assert results["metadatas"][0]["document_id"]
    assert results["ids"][0].startswith(
        results["metadatas"][0]["document_id"]
    )
    assert pipeline.vector_store.count() == chunks_ingested

def test_ingestion_pipeline_returns_zero_for_empty_document(tmp_path):
    from src.ingestion.pipeline import IngestionPipeline

    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    pipeline = IngestionPipeline(
        embedder=TextEmbedder(),
        vector_store=VectorStore(
            persist_directory="data/test_empty_pipeline_chroma",
            collection_name="empty_pipeline_collection",
        ),
    )

    chunks_ingested = pipeline.ingest(str(file_path))

    assert chunks_ingested == 0
    assert pipeline.vector_store.count() == 0

def test_ingestion_pipeline_missing_file():
    from src.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline(
        embedder=TextEmbedder(),
        vector_store=VectorStore(
            persist_directory="data/test_missing_pipeline_chroma",
            collection_name="missing_pipeline_collection",
        ),
    )

    with pytest.raises(FileNotFoundError, match="Document not found"):
        pipeline.ingest("data/does_not_exist.txt")

def test_ingestion_pipeline_unsupported_file():
    from src.ingestion.pipeline import IngestionPipeline

    file_path = Path("data/sample.csv")
    file_path.write_text("name,value\nA,10\n", encoding="utf-8")

    pipeline = IngestionPipeline(
        embedder=TextEmbedder(),
        vector_store=VectorStore(
            persist_directory="data/test_unsupported_pipeline_chroma",
            collection_name="unsupported_pipeline_collection",
        ),
    )

    with pytest.raises(ValueError, match="Unsupported document type"):
        pipeline.ingest(str(file_path))

def test_document_id_is_deterministic():
    from src.ingestion.pipeline import IngestionPipeline

    path = Path("data/sample.txt")

    first_id = IngestionPipeline._document_id(path)
    second_id = IngestionPipeline._document_id(path)

    assert first_id == second_id
    assert len(first_id) == 16

def test_vector_store_exists():
    store = VectorStore(
        persist_directory="data/test_exists_chroma",
        collection_name="exists_collection",
    )

    store.add_chunks(
        chunks=["Python is a programming language."],
        embeddings=[[1.0, 0.0, 0.0]],
        metadatas=[{"source": "test.txt", "chunk_id": 0}],
        ids=["test-id-1"],
    )

    assert store.exists(["test-id-1"])
    assert not store.exists(["missing-id"])

def test_delete_document(tmp_path):
    from src.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline(
        embedder=TextEmbedder(),
        vector_store=VectorStore(
            persist_directory=str(tmp_path / "chroma"),
            collection_name="delete_collection",
        ),
    )

    file_path = Path("data/sample.txt")

    chunks_ingested = pipeline.ingest(str(file_path))

    assert chunks_ingested > 0
    assert pipeline.vector_store.count() == chunks_ingested

    pipeline.delete_document(str(file_path))

    assert pipeline.vector_store.count() == 0

def test_delete_document_with_no_chunks(tmp_path):
    from src.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline(
        embedder=TextEmbedder(),
        vector_store=VectorStore(
            persist_directory=str(tmp_path / "chroma"),
            collection_name="empty_delete_collection",
        ),
    )

    pipeline.delete_document("data/sample.txt")

    assert pipeline.vector_store.count() == 0

def test_vector_store_delete():
    store = VectorStore(
        persist_directory="data/test_delete_chroma",
        collection_name="delete_collection",
    )

    store.add_chunks(
        chunks=["Python is a programming language."],
        embeddings=[[1.0, 0.0, 0.0]],
        metadatas=[{"source": "test.txt", "chunk_id": 0}],
        ids=["delete-id-1"],
    )

    assert store.count() == 1

    store.delete(["delete-id-1"])

    assert store.count() == 0

def test_vector_store_delete_multiple():
    store = VectorStore(
        persist_directory="data/test_delete_multiple_chroma",
        collection_name="delete_multiple_collection",
    )

    store.add_chunks(
        chunks=[
            "Python is a programming language.",
            "Machine learning uses data to learn patterns.",
        ],
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        metadatas=[
            {"source": "test.txt", "chunk_id": 0},
            {"source": "test.txt", "chunk_id": 1},
        ],
        ids=["delete-id-1", "delete-id-2"],
    )

    assert store.count() == 2

    store.delete(["delete-id-1", "delete-id-2"])

    assert store.count() == 0

def test_vector_store_delete_empty_ids():
    store = VectorStore(
        persist_directory="data/test_delete_empty_chroma",
        collection_name="delete_empty_collection",
    )

    store.delete([])

    assert store.count() == 0

def test_query_accepts_document_id():
    with patch(
        "src.api.routes.LLMClient.generate",
        side_effect=mock_llm_response,
    ):
        response = client.post(
            "/query",
            json={
                "question": "What benefits do employees receive?",
                "document_id": "test-document-id",
            },
        )

    assert response.status_code == 200

def test_query_rejects_empty_document_id():
    response = client.post(
        "/query",
        json={
            "question": "What benefits do employees receive?",
            "document_id": "",
        },
    )

    assert response.status_code == 422

def test_query_passes_document_id_filter():
    with patch(
        "src.api.routes.retriever.retrieve",
        return_value=[],
    ) as mock_retrieve:
        response = client.post(
            "/query",
            json={
                "question": "What benefits do employees receive?",
                "document_id": "test-document-id",
            },
        )

    assert response.status_code == 200

    mock_retrieve.assert_called_once()

    assert mock_retrieve.call_args.kwargs["where"] == {
        "$and": [
            {"workspace_id": TEST_WORKSPACE_ID},
            {"document_id": "test-document-id"},
        ]
    }

def test_retriever_rejects_zero_top_k():
    retriever = Retriever(
        embedder=TextEmbedder(),
        vector_store=VectorStore(
            persist_directory="data/test_zero_top_k_chroma",
            collection_name="zero_top_k_collection",
        ),
    )

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        retriever.retrieve(
            query="What is Python?",
            top_k=0,
        )

def test_retriever_rejects_zero_top_k():
    retriever = Retriever(
        embedder=TextEmbedder(),
        vector_store=VectorStore(
            persist_directory="data/test_zero_top_k_chroma",
            collection_name="zero_top_k_collection",
        ),
    )

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        retriever.retrieve(
            query="What is Python?",
            top_k=0,
        )

def test_vector_store_get_by_document_id_rejects_non_string():
    store = VectorStore(
        persist_directory="data/test_document_id_type_chroma",
        collection_name="document_id_type_collection",
    )

    with pytest.raises(
        TypeError,
        match="document_id must be a string",
    ):
        store.get_by_document_id(123)

def test_vector_store_get_by_document_id_rejects_empty():
    store = VectorStore(
        persist_directory="data/test_document_id_empty_chroma",
        collection_name="document_id_empty_collection",
    )

    with pytest.raises(
        ValueError,
        match="document_id cannot be empty",
    ):
        store.get_by_document_id("")

def test_vector_store_get_all_documents():
    store = VectorStore(
        persist_directory="data/test_get_all_chroma",
        collection_name="get_all_collection",
    )

    store.add_chunks(
        chunks=["Python is a programming language."],
        embeddings=[[1.0, 0.0, 0.0]],
        metadatas=[{"source": "test.txt", "chunk_id": 0}],
        ids=["all-id-1"],
    )

    results = store.get_all_documents()

    assert results["ids"] == ["all-id-1"]
    assert results["documents"] == [
        "Python is a programming language."
    ]
    assert results["metadatas"][0]["source"] == "test.txt"

def test_list_documents():
    response = client.get("/documents")

    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_list_documents_rejects_negative_skip():
    response = client.get("/documents?skip=-1")

    assert response.status_code == 422


def test_list_documents_rejects_invalid_limit():
    response = client.get("/documents?limit=0")

    assert response.status_code == 422


def test_list_documents_rejects_excessive_limit():
    response = client.get("/documents?limit=101")

    assert response.status_code == 422

def test_list_documents_returns_generic_error_on_exception():
    with patch(
        "src.api.routes.vector_store.get_all_documents",
        side_effect=RuntimeError("internal database failure"),
    ):
        response = client.get("/documents")

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Failed to list documents."
    }

def test_list_documents_contains_sample_document(tmp_path):
    from src.api import routes

    original_store = routes.vector_store

    test_store = VectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="documents_test",
    )

    routes.vector_store = test_store

    routes.vector_store.collection.upsert(
        documents=["Employees receive 20 days of paid annual leave."],
        embeddings=[TextEmbedder().embed_text(
            "Employees receive 20 days of paid annual leave."
        )],
        metadatas=[
            {
                "source": "sample.txt",
                "document_id": "test-document-id",
                "chunk_id": 0,
                "workspace_id": TEST_WORKSPACE_ID
            }
        ],
        ids=["test-document-id-0"],
    )

    response = client.get("/documents")

    assert response.status_code == 200

    documents = response.json()

    assert any(
        document["source"] == "sample.txt"
        for document in documents
    )

def test_list_documents_groups_chunks(tmp_path):
    from src.api import routes

    original_store = routes.vector_store

    test_store = VectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="documents_test",
    )

    routes.vector_store = test_store

    routes.vector_store.collection.upsert(
        documents=[
            "First chunk.",
            "Second chunk.",
        ],
        embeddings=[
            TextEmbedder().embed_text("First chunk."),
            TextEmbedder().embed_text("Second chunk."),
        ],
        metadatas=[
            {
                "source": "multi.txt",
                "document_id": "multi-document-id",
                "chunk_id": 0,
                "workspace_id": TEST_WORKSPACE_ID,
            },
            {
                "source": "multi.txt",
                "document_id": "multi-document-id",
                "chunk_id": 1,
                "workspace_id": TEST_WORKSPACE_ID,
            },
        ],
        ids=[
            "multi-document-id-0",
            "multi-document-id-1",
        ],
    )

    response = client.get("/documents")

    routes.vector_store = original_store

    assert response.status_code == 200

    documents = response.json()

    document = next(
        item
        for item in documents
        if item["document_id"] == "multi-document-id"
    )

    assert document["source"] == "multi.txt"
    assert document["chunk_count"] == 2
    routes.vector_store = original_store
    
def test_list_documents_isolated_between_workspaces(tmp_path, monkeypatch):
    from src.api import routes

    monkeypatch.setattr(
        settings,
        "api_keys",
        "user-a-api-key:workspace-a,user-b-api-key:workspace-b",
    )

    original_store = routes.vector_store

    test_store = VectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="workspace_isolation_collection",
    )

    routes.vector_store = test_store

    embedder = TextEmbedder()

    test_store.collection.upsert(
        documents=[
            "Workspace A document.",
            "Workspace B document.",
        ],
        embeddings=[
            embedder.embed_text("Workspace A document."),
            embedder.embed_text("Workspace B document."),
        ],
        metadatas=[
            {
                "source": "workspace-a.txt",
                "document_id": "workspace-a-document",
                "chunk_id": 0,
                "workspace_id": "workspace-a",
            },
            {
                "source": "workspace-b.txt",
                "document_id": "workspace-b-document",
                "chunk_id": 0,
                "workspace_id": "workspace-b",
            },
        ],
        ids=[
            "workspace-a-document-0",
            "workspace-b-document-0",
        ],
    )

    try:
        response_a = client.get(
            "/documents",
            headers={"X-API-Key": "user-a-api-key"},
        )

        response_b = client.get(
            "/documents",
            headers={"X-API-Key": "user-b-api-key"},
        )
    finally:
        routes.vector_store = original_store

    assert response_a.status_code == 200
    assert response_b.status_code == 200

    documents_a = response_a.json()
    documents_b = response_b.json()

    assert [document["source"] for document in documents_a] == [
        "workspace-a.txt"
    ]

    assert [document["source"] for document in documents_b] == [
        "workspace-b.txt"
    ]

def test_get_document_isolated_between_workspaces(tmp_path, monkeypatch):
    from src.api import routes

    monkeypatch.setattr(
        settings,
        "api_keys",
        "user-a-api-key:workspace-a,user-b-api-key:workspace-b",
    )

    original_store = routes.vector_store

    test_store = VectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="get_workspace_isolation_collection",
    )

    routes.vector_store = test_store

    embedder = TextEmbedder()

    test_store.collection.upsert(
        documents=["Workspace B private document."],
        embeddings=[
            embedder.embed_text("Workspace B private document.")
        ],
        metadatas=[
            {
                "source": "workspace-b-private.txt",
                "document_id": "workspace-b-private-document",
                "chunk_id": 0,
                "workspace_id": "workspace-b",
            }
        ],
        ids=["workspace-b-private-document-0"],
    )

    try:
        response = client.get(
            "/documents/workspace-b-private-document",
            headers={"X-API-Key": "user-a-api-key"},
        )
    finally:
        routes.vector_store = original_store

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."

def test_delete_document_isolated_between_workspaces(tmp_path, monkeypatch):
    from src.api import routes

    monkeypatch.setattr(
        settings,
        "api_keys",
        "user-a-api-key:workspace-a,user-b-api-key:workspace-b",
    )

    original_store = routes.vector_store

    test_store = VectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="delete_workspace_isolation_collection",
    )

    routes.vector_store = test_store

    embedder = TextEmbedder()

    test_store.collection.upsert(
        documents=["Workspace B private document."],
        embeddings=[
            embedder.embed_text("Workspace B private document.")
        ],
        metadatas=[
            {
                "source": "workspace-b-private.txt",
                "document_id": "workspace-b-private-document",
                "chunk_id": 0,
                "workspace_id": "workspace-b",
            }
        ],
        ids=["workspace-b-private-document-0"],
    )

    try:
        response = client.delete(
            "/documents/workspace-b-private-document",
            headers={"X-API-Key": "user-a-api-key"},
        )

        remaining = test_store.get_by_document_id(
            "workspace-b-private-document",
            workspace_id="workspace-b",
        )
    finally:
        routes.vector_store = original_store

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."

    assert remaining["ids"] == ["workspace-b-private-document-0"]

def test_query_isolated_between_workspaces(monkeypatch):
    from src.api import routes

    monkeypatch.setattr(
        settings,
        "api_keys",
        "user-a-api-key:workspace-a,user-b-api-key:workspace-b",
    )

    captured_where = {}

    class FakeRetriever:
        def retrieve(self, query, top_k, rerank_top_k, where=None):
            captured_where["where"] = where

            return [
                {
                    "id": "workspace-a-document-0",
                    "text": "Workspace A private information.",
                    "metadata": {
                        "source": "workspace-a-private.txt",
                        "document_id": "workspace-a-document",
                        "chunk_id": 0,
                        "workspace_id": "workspace-a",
                    },
                }
            ]

    monkeypatch.setattr(
        "src.api.routes.retriever",
        FakeRetriever(),
    )

    with patch(
        "src.api.routes.LLMClient.generate",
        side_effect=lambda *args, **kwargs: (
            "Workspace A private information [Source 1]."
        ),
    ):
        response = client.post(
            "/query",
            headers={"X-API-Key": "user-a-api-key"},
            json={
                "question": "What is the private information?",
                "top_k": 1,
            },
        )

    assert response.status_code == 200

    assert captured_where["where"] == {
        "workspace_id": "workspace-a"
    }

    data = response.json()

    assert data["sources"]
    assert data["sources"][0]["source"] == "workspace-a-private.txt"

def test_get_document():
    from src.api import routes

    document_id = "lookup-document-id"

    routes.vector_store.collection.upsert(
        documents=["Document content."],
        embeddings=[
            TextEmbedder().embed_text("Document content.")
        ],
        metadatas=[
            {
                "source": "lookup.txt",
                "document_id": document_id,
                "chunk_id": 0,
                "workspace_id": TEST_WORKSPACE_ID,
            }
        ],
        ids=[f"{document_id}-0"],
    )

    response = client.get(f"/documents/{document_id}")

    assert response.status_code == 200

    document = response.json()

    assert document["document_id"] == document_id
    assert document["source"] == "lookup.txt"
    assert document["chunk_count"] == 1

def test_get_document_not_found():
    response = client.get("/documents/non-existent-document-id")

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."

def test_get_document_rejects_empty_document_id():
    response = client.get("/documents/%20")

    assert response.status_code == 400
    assert response.json()["detail"] == "document_id cannot be empty."

def test_get_document_returns_generic_error_on_exception():
    with patch(
        "src.api.routes.vector_store.get_by_document_id",
        side_effect=RuntimeError("internal database failure"),
    ):
        response = client.get("/documents/test-document-id")

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Failed to get document."
    }

def test_delete_document():
    from src.api import routes

    document_id = "api-delete-document"

    routes.vector_store.collection.upsert(
        documents=["Document content."],
        embeddings=[
            TextEmbedder().embed_text("Document content.")
        ],
        metadatas=[
            {
                "source": "delete.txt",
                "document_id": document_id,
                "chunk_id": 0,
                "workspace_id": TEST_WORKSPACE_ID,
            }
        ],
        ids=[f"{document_id}-0"],
    )

    response = client.delete(f"/documents/{document_id}")

    assert response.status_code == 200

    result = response.json()

    assert result["document_id"] == document_id
    assert result["message"] == "Document deleted successfully."

    assert not routes.vector_store.get_by_document_id(document_id).get("ids")

def test_delete_document_not_found():
    response = client.delete("/documents/non-existent-document-id")

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."

def test_delete_document_returns_generic_error_on_exception():
    with patch(
        "src.api.routes.vector_store.get_by_document_id",
        side_effect=RuntimeError("internal database failure"),
    ):
        response = client.delete("/documents/test-document-id")

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Failed to delete document."
    }

def test_settings_rejects_invalid_chunk_size():
    from src.config.settings import Settings

    with pytest.raises(ValueError):
        Settings(
            gemini_api_key="test-key",
            chunk_size=0,
        )


def test_settings_rejects_negative_chunk_overlap():
    from src.config.settings import Settings

    with pytest.raises(ValueError):
        Settings(
            gemini_api_key="test-key",
            chunk_overlap=-1,
        )


def test_settings_rejects_invalid_retrieval_top_k():
    from src.config.settings import Settings

    with pytest.raises(ValueError):
        Settings(
            gemini_api_key="test-key",
            retrieval_top_k=0,
        )


def test_settings_rejects_invalid_rerank_top_k():
    from src.config.settings import Settings

    with pytest.raises(ValueError):
        Settings(
            gemini_api_key="test-key",
            rerank_top_k=0,
        )

def test_settings_rejects_chunk_overlap_equal_to_chunk_size():
    from src.config.settings import Settings

    with pytest.raises(
        ValueError,
        match="chunk_overlap must be smaller than chunk_size",
    ):
        Settings(
            gemini_api_key="test-key",
            chunk_size=100,
            chunk_overlap=100,
        )


def test_settings_rejects_chunk_overlap_greater_than_chunk_size():
    from src.config.settings import Settings

    with pytest.raises(
        ValueError,
        match="chunk_overlap must be smaller than chunk_size",
    ):
        Settings(
            gemini_api_key="test-key",
            chunk_size=100,
            chunk_overlap=101,
        )

def test_query_returns_generic_error_on_exception():
    with patch(
        "src.api.routes.retriever.retrieve",
        side_effect=RuntimeError("internal database failure"),
    ):
        response = client.post(
            "/query",
            json={
                "question": "What is the leave policy?",
                "top_k": 1,
            },
        )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Failed to process the query."
    }

def test_evaluate_retrieval_calculates_hit_rate_and_mrr():
    class FakeRetriever:
        def retrieve(self, query, top_k, rerank_top_k, where=None):
            return [
                {"metadata": {"source": "other.txt"}},
                {"metadata": {"source": "sample.txt"}},
            ]

    dataset = [
        {
            "question": "test question",
            "expected_source": "sample.txt",
        }
    ]

    result = evaluate_retrieval(
        retriever=FakeRetriever(),
        dataset=dataset,
        top_k=2,
    )

    assert result["total_cases"] == 1
    assert result["hit_rate"] == 1.0
    assert result["mrr"] == 0.5
    assert result["results"][0]["rank"] == 2
    assert result["results"][0]["hit"] is True


def test_evaluate_retrieval_handles_missing_source():
    class FakeRetriever:
        def retrieve(self, query, top_k, rerank_top_k, where=None):
            return [
                {"metadata": {"source": "other.txt"}},
            ]

    dataset = [
        {
            "question": "test question",
            "expected_source": "sample.txt",
        }
    ]

    result = evaluate_retrieval(
        retriever=FakeRetriever(),
        dataset=dataset,
        top_k=1,
    )

    assert result["total_cases"] == 1
    assert result["hit_rate"] == 0.0
    assert result["mrr"] == 0.0
    assert result["results"][0]["rank"] is None
    assert result["results"][0]["hit"] is False


def test_evaluate_retrieval_rejects_invalid_top_k():
    with pytest.raises(
        ValueError,
        match="top_k must be greater than 0",
    ):
        evaluate_retrieval(
            retriever=None,
            dataset=[],
            top_k=0,
        )


def test_evaluate_retrieval_handles_empty_dataset():
    class FakeRetriever:
        def retrieve(self, query, top_k, rerank_top_k, where=None):
            raise AssertionError("Retriever should not be called.")

    result = evaluate_retrieval(
        retriever=FakeRetriever(),
        dataset=[],
        top_k=1,
    )

    assert result["total_cases"] == 0
    assert result["hit_rate"] == 0.0
    assert result["mrr"] == 0.0
    assert result["results"] == []
    
def test_llm_client_uses_gemini_when_available():
    client = LLMClient()

    with patch.object(
        client,
        "_generate_with_gemini",
        return_value="Gemini response",
    ) as mock_gemini, patch.object(
        client,
        "_generate_with_groq",
    ) as mock_groq:
        result = client.generate("test prompt")

    assert result == "Gemini response"
    assert client.provider_used == "gemini"
    mock_gemini.assert_called_once_with("test prompt")
    mock_groq.assert_not_called()


def test_llm_client_falls_back_to_groq_on_server_error():
    client = LLMClient()

    server_error = ServerError(503, {})

    with patch.object(
        client,
        "_generate_with_gemini",
        side_effect=server_error,
    ) as mock_gemini, patch.object(
        client,
        "_generate_with_groq",
        return_value="Groq fallback response",
    ) as mock_groq:
        result = client.generate("test prompt")

    assert result == "Groq fallback response"
    assert client.provider_used == "groq"
    mock_gemini.assert_called_once_with("test prompt")
    mock_groq.assert_called_once_with("test prompt")
    

def test_llm_client_falls_back_to_groq_on_gemini_quota_error():
    client = LLMClient()

    quota_error = ClientError(429, {})

    with patch.object(
        client,
        "_generate_with_gemini",
        side_effect=quota_error,
    ) as mock_gemini, patch.object(
        client,
        "_generate_with_groq",
        return_value="Groq quota fallback response",
    ) as mock_groq:
        result = client.generate("test prompt")

    assert result == "Groq quota fallback response"
    assert client.provider_used == "groq"
    assert client.provider_attempts == ["gemini", "groq"]
    mock_gemini.assert_called_once_with("test prompt")
    mock_groq.assert_called_once_with("test prompt")


def test_llm_client_falls_back_to_groq_on_timeout():
    client = LLMClient()

    timeout_error = httpx.ReadTimeout("request timed out")

    with patch.object(
        client,
        "_generate_with_gemini",
        side_effect=timeout_error,
    ) as mock_gemini, patch.object(
        client,
        "_generate_with_groq",
        return_value="Groq timeout fallback",
    ) as mock_groq:
        result = client.generate("test prompt")

    assert result == "Groq timeout fallback"
    assert client.provider_used == "groq"
    mock_gemini.assert_called_once_with("test prompt")
    mock_groq.assert_called_once_with("test prompt")


def test_llm_client_does_not_fallback_on_permanent_error():
    client = LLMClient()

    with patch.object(
        client,
        "_generate_with_gemini",
        side_effect=ValueError("invalid request"),
    ) as mock_gemini, patch.object(
        client,
        "_generate_with_groq",
    ) as mock_groq:
        with pytest.raises(ValueError, match="invalid request"):
            client.generate("test prompt")

    assert client.provider_used is None
    mock_gemini.assert_called_once_with("test prompt")
    mock_groq.assert_not_called()
    
def test_query_response_includes_gemini_provider(monkeypatch):
    class FakeRetriever:
        def retrieve(self, query, top_k, rerank_top_k, where=None):
            return [
                {
                    "id": "chunk-1",
                    "text": "Employees receive 20 days of paid annual leave.",
                    "metadata": {
                        "source": "sample.txt",
                        "chunk_id": 1,
                    },
                }
            ]

    class FakeGenerator:
        provider_used = "gemini"
        provider_attempts = ["gemini"]

        def __init__(self, llm_client):
            pass

        def generate(self, question, contexts):
            return "Employees receive 20 days of paid annual leave."

    monkeypatch.setattr(
        "src.api.routes.retriever",
        FakeRetriever(),
    )
    monkeypatch.setattr(
        "src.api.routes.RAGGenerator",
        FakeGenerator,
    )

    from src.api.routes import QueryRequest, query

    response = __import__("asyncio").run(
        query(QueryRequest(question="How many annual leave days?"))
    )

    assert response.provider == "gemini"
    assert response.fallback_used is False


def test_query_response_includes_groq_fallback_status(monkeypatch):
    class FakeRetriever:
        def retrieve(self, query, top_k, rerank_top_k, where=None):
            return [
                {
                    "id": "chunk-1",
                    "text": "Employees receive 20 days of paid annual leave.",
                    "metadata": {
                        "source": "sample.txt",
                        "chunk_id": 1,
                    },
                }
            ]

    class FakeGenerator:
        provider_used = "groq"
        provider_attempts = ["gemini", "groq"]


        def __init__(self, llm_client):
            pass

        def generate(self, question, contexts):
            return "Employees receive 20 days of paid annual leave."

    monkeypatch.setattr(
        "src.api.routes.retriever",
        FakeRetriever(),
    )
    monkeypatch.setattr(
        "src.api.routes.RAGGenerator",
        FakeGenerator,
    )

    from src.api.routes import QueryRequest, query

    response = __import__("asyncio").run(
        query(QueryRequest(question="How many annual leave days?"))
    )

    assert response.provider == "groq"
    assert response.fallback_used is True
    
def test_query_returns_503_when_all_llm_providers_fail(monkeypatch):
    class FakeRetriever:
        def retrieve(self, query, top_k, rerank_top_k, where=None):
            return [
                {
                    "id": "chunk-1",
                    "text": "Test context",
                    "metadata": {
                        "source": "sample.txt",
                        "chunk_id": 1,
                    },
                }
            ]

    class FakeGenerator:
        def __init__(self, llm_client):
            pass

        def generate(self, question, contexts):
            raise LLMUnavailableError(
                "All configured LLM providers are unavailable."
            )

    monkeypatch.setattr(
        "src.api.routes.retriever",
        FakeRetriever(),
    )
    monkeypatch.setattr(
        "src.api.routes.RAGGenerator",
        FakeGenerator,
    )

    from src.api.routes import QueryRequest, query

    with pytest.raises(HTTPException) as exc_info:
        __import__("asyncio").run(
            query(QueryRequest(question="Test question"))
        )

    assert exc_info.value.status_code == 503
    assert (
        exc_info.value.detail
        == "The language model is temporarily unavailable. Please try again later."
    )


def test_query_returns_503_for_llm_unavailable_error(monkeypatch):
    class FakeRetriever:
        def retrieve(self, query, top_k, rerank_top_k, where=None):
            return [
                {
                    "id": "chunk-1",
                    "text": "Test context",
                    "metadata": {
                        "source": "sample.txt",
                        "chunk_id": 1,
                    },
                }
            ]

    class FakeGenerator:
        def __init__(self, llm_client):
            pass

        def generate(self, question, contexts):
            raise LLMUnavailableError("Provider unavailable")

    monkeypatch.setattr(
        "src.api.routes.retriever",
        FakeRetriever(),
    )
    monkeypatch.setattr(
        "src.api.routes.RAGGenerator",
        FakeGenerator,
    )

    from src.api.routes import QueryRequest, query

    with pytest.raises(HTTPException) as exc_info:
        __import__("asyncio").run(
            query(QueryRequest(question="Another test question"))
        )

    assert exc_info.value.status_code == 503
    
