import pytest
import sys
from pathlib import Path
from unittest.mock import patch
from docx import Document

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.parser import parse_document
from src.ingestion.chunker import chunk_text
from src.generation.prompt import build_rag_prompt
from src.generation.llm import LLMClient
from src.main import app
from src.ingestion.cleaner import clean_text
from src.retrieval.embedder import TextEmbedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.retrieval.reranker import Reranker


client = TestClient(app)
def mock_llm_response(*args, **kwargs):
    return "Employees receive 20 days of paid annual leave each calendar year [Source 1]."

def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    
def test_query():
    with patch("src.api.routes.LLMClient.generate", side_effect=mock_llm_response):
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
        
def test_ingestion_pipeline():
    from src.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline(
        embedder=TextEmbedder(),
        vector_store=VectorStore(
            persist_directory="data/test_pipeline_chroma",
            collection_name="pipeline_collection",
        ),
    )

    file_path = Path("data/sample.txt")

    chunks_ingested = pipeline.ingest(str(file_path))

    assert chunks_ingested > 0
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