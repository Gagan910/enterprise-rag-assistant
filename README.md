# Enterprise Document Intelligence & RAG Assistant

An end-to-end, production-style Retrieval-Augmented Generation (RAG) application for asking grounded questions over enterprise documents.

The system supports PDF, DOCX, and TXT documents, converts them into searchable chunks, retrieves relevant context with semantic search, reranks results with a cross-encoder, and generates grounded answers with source citations.

It also includes:

- Multi-user authentication with Supabase
- User/workspace isolation
- FastAPI backend
- Streamlit frontend
- Gemini primary LLM with Groq fallback
- ChromaDB vector storage / Chroma Cloud production support
- Cross-encoder reranking
- Docker deployment
- GitHub Actions CI
- DVC data versioning
- MLflow retrieval evaluation
- DagsHub integration
- Automated testing
- Render deployment

---

## Live Demo

- **Frontend:** https://enterprise-rag-frontend-g1bm.onrender.com
- **Backend API:** https://enterprise-rag-assistant-ik30.onrender.com
- **Swagger API Docs:** https://enterprise-rag-assistant-ik30.onrender.com/docs
- **Health Check:** https://enterprise-rag-assistant-ik30.onrender.com/health
- **GitHub:** https://github.com/Gagan910/enterprise-rag-assistant
- **DagsHub:** https://dagshub.com/Gagan910/enterprise-rag-assistant

> **Note:** The live application uses Supabase authentication. Each authenticated user is mapped to an isolated workspace so documents and queries are scoped to that user.

---

## Architecture

```text
                         ┌──────────────────────────┐
                         │   Streamlit Frontend     │
                         │  Login / Register / RAG   │
                         └────────────┬─────────────┘
                                      │
                                      │ Supabase Auth
                                      ▼
                         ┌──────────────────────────┐
                         │     Supabase Auth        │
                         │   User Session / JWT     │
                         └────────────┬─────────────┘
                                      │
                                      │ Bearer Token
                                      ▼
                         ┌──────────────────────────┐
                         │      FastAPI Backend     │
                         │ Authentication + RAG API │
                         └────────────┬─────────────┘
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
          Workspace-scoped                    Cross-Encoder
          Vector Storage                      Reranking
                                                       │
                                                       ▼
                                                Grounded Prompt
                                                       │
                                      ┌────────────────┴───────────────┐
                                      │                                │
                                      ▼                                ▼
                              Gemini (Primary)                  Groq (Fallback)
                                      │                                │
                                      └────────────────┬───────────────┘
                                                       ▼
                                             Answer + Source Citations
```

### Multi-user isolation

```text
User A
  │
  ├── Supabase account
  ├── Workspace A
  └── Documents A
          │
          └── Queries only retrieve Workspace A data


User B
  │
  ├── Supabase account
  ├── Workspace B
  └── Documents B
          │
          └── Queries only retrieve Workspace B data
```

Workspace identifiers are derived from authenticated user identity, and document ingestion/retrieval applies workspace-scoped filtering.

---

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
Workspace-scoped Vector Storage
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

---

## Key Features

### Authentication & Multi-user Support

- Supabase email/password authentication
- User registration and login
- Password-reset workflow
- Bearer-token authentication between frontend and backend
- User-specific workspace mapping
- Workspace-scoped document ingestion
- Workspace-scoped retrieval
- Workspace-isolated document listing
- Workspace-isolated document lookup
- Workspace-isolated document deletion
- Workspace-isolated RAG queries
- Legacy API-key compatibility retained by the backend

> Custom SMTP / custom-domain email delivery is intentionally not required for the current deployment. Supabase's built-in email service is currently used.

### Document Intelligence

- PDF, DOCX, and TXT ingestion
- Text cleaning
- Configurable chunking
- Deterministic content-based document IDs
- Duplicate document detection
- Workspace-aware duplicate detection
- Document listing and pagination
- Single-document lookup
- Document deletion
- Metadata-based document filtering
- File-type validation
- Upload-size validation

### Retrieval

- `all-MiniLM-L6-v2` sentence embeddings
- ChromaDB vector database
- Chroma Cloud support for production
- Semantic similarity retrieval
- Cross-encoder reranking with `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Configurable retrieval and reranking parameters
- Document-specific retrieval
- Workspace-specific retrieval

### Generation

- Google Gemini as the primary LLM
- Groq as the fallback LLM
- Provider attempt tracking
- Fallback status exposed through the API
- Grounded prompts using retrieved document context
- Source citations in generated answers
- Gemini rate-limit cooldown handling

### API & Security

- FastAPI backend
- Supabase Bearer-token authentication
- Legacy `X-API-Key` authentication compatibility
- Public `/health` endpoint
- Protected document and query endpoints
- Workspace-level data isolation
- Secrets loaded from environment variables / deployment secrets
- Local Streamlit secrets excluded from Git
- Production logs avoid exposing prompts, generated answers, API keys, or other secret values

### MLOps

- DVC for document/data versioning
- DagsHub as the DVC remote
- MLflow retrieval evaluation
- DagsHub MLflow tracking
- GitHub Actions CI
- Dockerized backend
- Render deployment

### Frontend

- Streamlit interface
- Supabase Login / Register experience
- Password recovery UI
- Document management
- Document upload and indexing
- Document listing
- RAG question answering
- Backend health check
- Retrieval `top_k` control
- Source display
- LLM provider display
- Fallback trail display

---

## Tech Stack

| Area | Technology |
|---|---|
| Language | Python 3.12 |
| Backend | FastAPI |
| API Server | Uvicorn |
| Authentication | Supabase Auth |
| Configuration | Pydantic v2 / Pydantic Settings |
| Document Parsing | pypdf, python-docx |
| Embeddings | Sentence Transformers |
| Vector Database | ChromaDB / Chroma Cloud |
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

---

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
│   └── ...
│
└── .github/
    └── workflows/
        └── ci.yml
```

---

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
API_KEYS

SUPABASE_URL
SUPABASE_ANON_KEY

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

VECTOR_STORE_PATH
VECTOR_COLLECTION_NAME

CHROMA_CLOUD_HOST
CHROMA_CLOUD_API_KEY
CHROMA_CLOUD_TENANT
CHROMA_CLOUD_DATABASE

MLFLOW_TRACKING_URI
```

### Secrets

Never commit:

```text
.env
.streamlit/secrets.toml
API keys
LLM credentials
Supabase credentials
Chroma Cloud credentials
tokens
passwords
```

Use environment variables or the deployment platform's secret/environment-variable configuration.

---

## Local Backend Setup

### 1. Clone the repository

```bash
git clone https://github.com/Gagan910/enterprise-rag-assistant.git
cd enterprise-rag-assistant
```

### 2. Create the backend environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a local `.env` file with the required configuration and secrets.

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

---

## Local Streamlit Frontend

The frontend uses a separate environment.

### 1. Create the frontend environment

```powershell
python -m venv .streamlit-venv
.\.streamlit-venv\Scripts\Activate.ps1
```

### 2. Install frontend dependencies

```powershell
pip install -r requirements-frontend.txt
```

### 3. Configure local Streamlit secrets

Create:

```text
.streamlit/secrets.toml
```

Example:

```toml
RAG_API_URL = "http://127.0.0.1:8000"

SUPABASE_URL = "your-supabase-url"
SUPABASE_ANON_KEY = "your-supabase-anon-key"
```

The file is ignored by Git.

### 4. Start Streamlit

```powershell
streamlit run app.py
```

---

## API Endpoints

| Method | Endpoint | Authentication | Purpose |
|---|---|---|---|
| GET | `/health` | Public | Health check |
| POST | `/documents/upload` | Bearer / API key | Upload and index a document |
| GET | `/documents` | Bearer / API key | List workspace documents |
| GET | `/documents/{document_id}` | Bearer / API key | Get one workspace document |
| DELETE | `/documents/{document_id}` | Bearer / API key | Delete a workspace document |
| POST | `/query` | Bearer / API key | Query the RAG system |

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

---

## Testing

The project has a comprehensive automated test suite covering:

- API endpoints
- Authentication
- Multi-workspace isolation
- Document management
- Ingestion
- Parsing
- Cleaning
- Chunking
- Embeddings
- Vector storage
- Retrieval
- Reranking
- RAG generation
- LLM fallback behavior
- Configuration validation
- Response validation

Run:

```powershell
pytest -q
```

### Current validated state

```text
114 passed
```

The current backend test suite has been fully validated with **114/114 tests passing**.

---

## Retrieval Evaluation

Retrieval quality is evaluated with MLflow and tracked through DagsHub.

### Verified evaluation metrics

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

---

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

---

## MLflow

MLflow tracking is configured with the DagsHub MLflow endpoint.

```text
https://dagshub.com/Gagan910/enterprise-rag-assistant.mlflow
```

Retrieval evaluation experiment:

```text
enterprise-rag-retrieval-evaluation
```

---

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

For production, configure secrets through the deployment platform rather than placing real credentials directly in shell history or source files.

The Docker image includes the embedding and reranker model assets required by the backend.

---

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

---

## Deployment

### Backend

The FastAPI backend is deployed on Render:

```text
https://enterprise-rag-assistant-ik30.onrender.com
```

Production backend capabilities include:

- Render environment variables
- Supabase Bearer-token authentication
- Legacy API-key compatibility
- Workspace isolation
- Dynamic `$PORT`
- Docker-compatible model assets
- Chroma Cloud support
- Gemini → Groq fallback

### Frontend

The Streamlit frontend is deployed separately on Render:

```text
https://enterprise-rag-frontend-g1bm.onrender.com
```

Production frontend configuration uses Render environment variables for:

```text
RAG_API_URL
SUPABASE_URL
SUPABASE_ANON_KEY
RAG_FRONTEND_URL
```

The deployed UI does not require users to manually enter an API key.

---

## Security Considerations

- Authentication is handled through Supabase Auth.
- Backend requests are authenticated with Supabase Bearer tokens.
- Workspace identifiers scope document and query operations.
- Legacy API-key authentication remains available for compatibility.
- Secrets are loaded from environment variables or Streamlit secrets.
- `.env` and `.streamlit/secrets.toml` are ignored by Git.
- Health checks remain publicly accessible.
- Protected API endpoints require authentication.
- Production logs avoid printing prompts, generated answers, API keys, tokens, or other secret values.
- Never commit credentials to Git.
- Rotate credentials immediately if they are accidentally exposed.

---

## Example End-to-End Workflow

```text
1. User registers / logs in
              ↓
2. Supabase authenticates the user
              ↓
3. Backend maps the authenticated user to a workspace
              ↓
4. User uploads an enterprise document
              ↓
5. Validate file
              ↓
6. Parse document
              ↓
7. Clean text
              ↓
8. Split into chunks
              ↓
9. Generate embeddings
              ↓
10. Store workspace-scoped vectors
              ↓
11. User asks a question
              ↓
12. Generate query embedding
              ↓
13. Retrieve relevant workspace-scoped chunks
              ↓
14. Rerank chunks
              ↓
15. Build grounded prompt
              ↓
16. Generate with Gemini
              ↓
17. Fall back to Groq if required
              ↓
18. Return answer + sources + provider trail
```

---

## Engineering Highlights

This project demonstrates practical implementation of:

- Production-style FastAPI API design
- Multi-user authentication
- Workspace-level data isolation
- Retrieval-Augmented Generation
- Semantic retrieval
- Sentence-transformer embeddings
- Cross-encoder reranking
- Grounded generation
- Multi-LLM fallback
- Centralized configuration
- Document lifecycle management
- Duplicate detection
- Automated testing
- Dockerization
- DVC data versioning
- MLflow retrieval evaluation
- DagsHub integration
- GitHub Actions CI
- Cloud deployment
- Streamlit frontend integration

---

## Project Status

**Production-style portfolio project — deployed and operational.**

Current validated state:

```text
Backend tests:       114/114 passing
Frontend:            Deployed
Backend:             Deployed
Authentication:      Supabase
Multi-user isolation: Enabled
RAG pipeline:        Operational
Vector storage:      ChromaDB / Chroma Cloud
Primary LLM:         Gemini
Fallback LLM:        Groq
CI/CD:               GitHub Actions
Deployment:          Render
```

Custom SMTP/domain email delivery is intentionally postponed; the current application uses Supabase's built-in authentication email service.

---

## License

This project is intended as a portfolio and learning project.
