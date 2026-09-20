# Enterprise Document Intelligence & RAG Assistant

An end-to-end Retrieval-Augmented Generation (RAG) application for asking grounded questions over enterprise documents.

The system ingests PDF, DOCX, and TXT files, converts them into searchable chunks, retrieves relevant context using semantic search, reranks the results with a cross-encoder, and generates an answer with source citations. It also includes API authentication, multi-LLM fallback, DVC data versioning, MLflow retrieval evaluation, Docker, GitHub Actions CI, FastAPI deployment, and a Streamlit frontend.

## Live Demo

- **Frontend:** https://enterprise-rag-frontend-g1bm.onrender.com
- **Backend API:** https://enterprise-rag-assistant-ik30.onrender.com
- **Swagger API Docs:** https://enterprise-rag-assistant-ik30.onrender.com/docs
- **Health Check:** https://enterprise-rag-assistant-ik30.onrender.com/health
- **GitHub:** https://github.com/Gagan910/enterprise-rag-assistant
- **DagsHub:** https://dagshub.com/Gagan910/enterprise-rag-assistant

## Architecture

```text
                         ┌──────────────────────┐
                         │   Streamlit Frontend │
                         └──────────┬───────────┘
                                    │ X-API-Key
                                    ▼
                         ┌──────────────────────┐
                         │    FastAPI Backend   │
                         └──────────┬───────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  │                                   │
                  ▼                                   ▼
        ┌──────────────────┐                ┌──────────────────┐
        │ Document Upload  │                │      /query      │
        └────────┬─────────┘                └────────┬─────────┘
                 │                                   │
                 ▼                                   ▼
        Parse → Clean → Chunk                 Query Embedding
                 │                                   │
                 ▼                                   ▼
          Sentence Embeddings                 ChromaDB Search
                 │                                   │
                 ▼                                   ▼
              ChromaDB                    Cross-Encoder Reranking
                                                     │
                                                     ▼
                                           Grounded Prompt
                                                     │
                                  ┌──────────────────┴──────────────────┐
                                  │                                     │
                                  ▼                                     ▼
                           Gemini (Primary)                    Groq (Fallback)
                                  │                                     │
                                  └──────────────────┬──────────────────┘
                                                     ▼
                                           Answer + Source Citations
```

## RAG Pipeline

```text
PDF / DOCX / TXT
      ↓
Document Parser
      ↓
Text Cleaning
      ↓
Chunking
      ↓
Sentence-Transformer Embeddings
      ↓
ChromaDB Vector Store
      ↓
Semantic Retrieval
      ↓
Cross-Encoder Reranking
      ↓
Grounded Prompt
      ↓
Gemini / Groq
      ↓
Answer + Sources
```

## Key Features

### Document Intelligence
- PDF, DOCX, and TXT ingestion
- Text cleaning and configurable chunking
- Deterministic content-based document IDs
- Duplicate document detection
- Document listing and pagination
- Single-document lookup
- Document deletion
- Metadata-based document filtering
- File type and upload-size validation

### Retrieval
- `all-MiniLM-L6-v2` sentence embeddings
- ChromaDB vector database
- Semantic similarity retrieval
- Cross-encoder reranking using `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Configurable retrieval and reranking parameters
- Document-specific retrieval

### Generation
- Gemini as the primary LLM
- Groq as the fallback LLM
- Provider attempt tracking
- Fallback status exposed through the API
- Grounded prompts using retrieved document context
- Source citations in generated answers
- Gemini rate-limit cooldown handling

### API Security
- `X-API-Key` authentication
- `/health` remains public
- Document and query endpoints are protected
- Invalid or missing API keys return `401`
- API secrets are stored through environment variables / deployment secrets
- Local Streamlit secrets are excluded from Git

### MLOps
- DVC for dataset/versioned document tracking
- DagsHub as the DVC remote
- MLflow retrieval evaluation
- DagsHub MLflow tracking
- GitHub Actions CI
- Dockerized backend
- Render deployment

### Frontend
- Streamlit interface
- Single-screen document management and Q&A workflow
- Backend health indicator
- Document listing
- Document upload
- Retrieval `top_k` control
- Source display
- LLM provider display
- Fallback trail display

## Tech Stack

| Area | Technology |
|---|---|
| Backend | FastAPI |
| API Server | Uvicorn |
| Validation / Config | Pydantic v2 / Pydantic Settings |
| Document Parsing | pypdf, python-docx |
| Embeddings | Sentence Transformers |
| Vector Database | ChromaDB |
| Reranking | Cross-Encoder |
| Primary LLM | Google Gemini |
| Fallback LLM | Groq |
| Frontend | Streamlit |
| Testing | Pytest |
| Containerization | Docker |
| Data Versioning | DVC |
| Data / Experiment Platform | DagsHub |
| Experiment Tracking | MLflow |
| CI/CD | GitHub Actions |
| Deployment | Render |
| Language | Python 3.12 |

## Project Structure

```text
enterprise-rag-assistant/
│
├── app.py
├── Dockerfile
├── requirements.txt
├── requirements-frontend.txt
├── .python-version
├── .gitignore
├── README.md
│
├── data/
│   ├── .gitkeep
│   ├── sample.txt.dvc
│   └── ...
│
├── models/
│
├── src/
│   ├── api/
│   │   └── routes.py
│   │
│   ├── config/
│   │   └── settings.py
│   │
│   ├── generation/
│   │   ├── generator.py
│   │   ├── llm.py
│   │   └── prompt.py
│   │
│   ├── ingestion/
│   │   ├── parser.py
│   │   ├── cleaner.py
│   │   ├── chunker.py
│   │   └── pipeline.py
│   │
│   ├── retrieval/
│   │   ├── embedder.py
│   │   ├── retriever.py
│   │   ├── reranker.py
│   │   └── vector_store.py
│   │
│   ├── evaluation/
│   │   └── mlflow_tracking.py
│   │
│   └── main.py
│
├── tests/
│   └── test_health.py
│
└── .github/
    └── workflows/
        └── ci.yml
```

## Configuration

Centralized configuration is handled by:

```text
src/config/settings.py
```

Important configuration values include:

```text
GEMINI_API_KEY
GROQ_API_KEY
API_KEY

LLM_PROVIDER
LLM_FALLBACK_PROVIDER

LLM_MODEL
GROQ_MODEL

EMBEDDING_MODEL
RERANKER_MODEL

CHUNK_SIZE
CHUNK_OVERLAP

RETRIEVAL_TOP_K
RERANK_TOP_K

MLFLOW_TRACKING_URI
```

**Never commit `.env`, API keys, tokens, passwords, or Streamlit secrets.**

## Local Backend Setup

### 1. Clone the repository

```bash
git clone https://github.com/Gagan910/enterprise-rag-assistant.git
cd enterprise-rag-assistant
```

### 2. Create the backend environment

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a local `.env` file with the required secrets and configuration.

Do not commit the file.

### 5. Start the API

```powershell
uvicorn src.main:app --reload
```

API:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

## Local Streamlit Frontend

The frontend uses a separate environment because Streamlit and the backend use different Starlette compatibility requirements.

### Create frontend environment

```powershell
python -m venv .streamlit-venv
.\.streamlit-venv\Scripts\Activate.ps1
```

### Install frontend dependencies

```powershell
pip install -r requirements-frontend.txt
```

### Configure local Streamlit secrets

Create:

```text
.streamlit/secrets.toml
```

Example:

```toml
RAG_API_URL = "http://127.0.0.1:8000"
RAG_API_KEY = "your-api-key"
```

The file is ignored by Git.

### Start Streamlit

```powershell
streamlit run app.py
```

## API Endpoints

| Method | Endpoint | Authentication | Purpose |
|---|---|---|---|
| GET | `/health` | Public | Health check |
| POST | `/documents/upload` | API key | Upload and index a document |
| GET | `/documents` | API key | List indexed documents |
| GET | `/documents/{document_id}` | API key | Get one document |
| DELETE | `/documents/{document_id}` | API key | Delete a document |
| POST | `/query` | API key | Query the RAG system |

### Query example

Request:

```json
{
  "question": "How many days of paid annual leave do employees receive?",
  "top_k": 5
}
```

Response structure:

```json
{
  "answer": "Employees receive 20 days of paid annual leave each calendar year [Source 1].",
  "sources": [
    {
      "source": "sample.txt",
      "chunk_id": 0
    }
  ],
  "provider": "gemini",
  "fallback_used": false,
  "provider_attempts": [
    "gemini"
  ]
}
```

If Gemini is unavailable and Groq handles the request, the response exposes the fallback trail through `provider`, `fallback_used`, and `provider_attempts`.

## Testing

The project has a comprehensive automated test suite covering:

- API endpoints
- authentication
- document management
- ingestion
- parsing
- cleaning
- chunking
- embeddings
- vector storage
- retrieval
- reranking
- RAG generation
- LLM fallback behavior
- configuration validation
- response validation

Run:

```powershell
pytest -q
```

The latest validated project state had **100/100 tests passing**.

## Retrieval Evaluation

Retrieval quality is evaluated with MLflow and tracked through DagsHub.

Current verified evaluation metrics:

```text
Hit Rate: 1.0
MRR:      1.0
Top-K:    5
```

Tracked configuration includes:

```text
Embedding Model:
all-MiniLM-L6-v2

Reranker:
cross-encoder/ms-marco-MiniLM-L-6-v2

Chunk Size:
500

Chunk Overlap:
100

Retrieval Top-K:
5
```

## DVC

The project uses DVC to version document data.

DagsHub DVC remote:

```text
https://dagshub.com/Gagan910/enterprise-rag-assistant.dvc
```

Useful commands:

```powershell
dvc status
dvc pull
dvc push
```

The sample document is tracked through:

```text
data/sample.txt.dvc
```

## MLflow

MLflow tracking is configured with the DagsHub MLflow endpoint.

```text
https://dagshub.com/Gagan910/enterprise-rag-assistant.mlflow
```

The retrieval evaluation experiment is:

```text
enterprise-rag-retrieval-evaluation
```

## Docker

Build the backend image:

```powershell
docker build -t enterprise-rag-assistant:latest .
```

Run:

```powershell
docker run --rm -p 8000:8000 `
  -e GEMINI_API_KEY="your-key" `
  -e API_KEY="your-api-key" `
  enterprise-rag-assistant:latest
```

The Docker image includes the embedding and reranker model assets required by the backend.

## CI/CD

GitHub Actions runs the CI workflow on pushes and pull requests to `master`.

The pipeline includes:

1. Python environment setup
2. Dependency installation
3. DVC authentication
4. DVC data pull
5. Test vector-store preparation
6. Automated test execution

Workflow:

```text
.github/workflows/ci.yml
```

The latest verified CI workflow is passing.

## Deployment

### Backend

The FastAPI backend is deployed on Render:

```text
https://enterprise-rag-assistant-ik30.onrender.com
```

The production backend uses:

- Render environment variables
- API-key authentication
- Dynamic `$PORT`
- Docker-compatible model assets
- Gemini → Groq fallback

### Frontend

The Streamlit frontend is deployed separately on Render:

```text
https://enterprise-rag-frontend-g1bm.onrender.com
```

Production frontend configuration uses Render environment variables:

```text
RAG_API_URL
RAG_API_KEY
```

The API key is never entered manually in the deployed UI.

## Security Considerations

- Secrets are loaded from environment variables or Streamlit secrets.
- `.env` and `.streamlit/secrets.toml` are ignored by Git.
- Protected API endpoints require `X-API-Key`.
- Health checks remain publicly accessible.
- Production logs avoid printing prompts, generated answers, API keys, or other secret values.
- API credentials should be rotated immediately if accidentally exposed.

## Example Workflow

```text
1. Upload enterprise document
          ↓
2. Validate file
          ↓
3. Parse document
          ↓
4. Clean text
          ↓
5. Split into chunks
          ↓
6. Generate embeddings
          ↓
7. Store in ChromaDB
          ↓
8. User asks a question
          ↓
9. Retrieve relevant chunks
          ↓
10. Rerank chunks
          ↓
11. Build grounded prompt
          ↓
12. Generate with Gemini
          ↓
13. Fall back to Groq if required
          ↓
14. Return answer + sources + provider trail
```

## Engineering Highlights

This project demonstrates practical implementation of:

- Production-style FastAPI API design
- RAG architecture
- Semantic retrieval
- Cross-encoder reranking
- Multi-LLM fallback
- Centralized configuration
- API authentication
- Document lifecycle management
- Automated testing
- Dockerization
- DVC data versioning
- MLflow experiment tracking
- DagsHub integration
- GitHub Actions CI
- Cloud deployment
- Streamlit frontend integration

## License

This project is intended as a portfolio and learning project.
