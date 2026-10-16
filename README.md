
# Enterprise Document Intelligence & RAG Assistant

An end-to-end Retrieval-Augmented Generation (RAG) application that enables users to query enterprise documents and receive grounded answers with source citations. The platform combines semantic retrieval, cross-encoder reranking, LLM-based generation, authentication, and document management.

**Live Demo:** [Streamlit Frontend](https://enterprise-rag-frontend-g1bm.onrender.com)

**Backend API:** [FastAPI](https://enterprise-rag-assistant-ik30.onrender.com) · [Swagger API Docs](https://enterprise-rag-assistant-ik30.onrender.com/docs) · [Health Check](https://enterprise-rag-assistant-ik30.onrender.com/health)

**GitHub:** [Gagan910/enterprise-rag-assistant](https://github.com/Gagan910/enterprise-rag-assistant)

## Features

- **Document ingestion:** Upload PDF, DOCX, and TXT files.
- **Cancellable batch uploads:** Upload multiple documents with live progress and cancel a batch without starting the remaining files. An in-progress request may finish.
- **RAG pipeline:** Document parsing, text cleaning, chunking, embedding generation, retrieval, reranking, and answer generation.
- **Grounded responses:** Generate answers using retrieved document context and source citations.
- **Hybrid LLM configuration:** Groq primary provider with Gemini fallback, according to the application configuration.
- **Semantic retrieval:** Sentence Transformers embeddings with ChromaDB / Chroma Cloud vector storage.
- **Cross-encoder reranking:** Re-rank retrieved passages to improve context relevance.
- **Document management:** List, retrieve, upload, and delete indexed documents, with duplicate detection and metadata filtering.
- **Authentication:** Supabase-based authentication and user/workspace-aware access controls.
- **Password recovery:** Password-reset workflow with a dedicated recovery page.
- **Session management:** Redis-backed session storage using opaque session identifiers and a seven-day TTL.
- **API security:** Session-aware API requests and legacy API-key compatibility.
- **Evaluation and experiment tracking:** Retrieval evaluation with MLflow, with DVC and DagsHub integration.
- **Testing and deployment:** Pytest, Docker, GitHub Actions, and Render.

## Architecture

```text
                    DOCUMENT INGESTION
                             |
                   PDF / DOCX / TXT
                             |
                             v
                    Parse and Clean
                             |
                             v
                     Text Chunking
                             |
                             v
                    Embedding Model
                             |
                             v
                       ChromaDB
                             |
                             |
USER QUESTION ---------------+
       |
       v
Retrieve Relevant Passages
       |
       v
Cross-Encoder Reranking
       |
       v
Build Context and Prompt
       |
       v
     Groq / Gemini
       |
       v
Grounded Answer + Citations
```

### Application and session architecture

```text
User
 |
 v
Streamlit Frontend
 |
 +---- Supabase Authentication
 |
 +---- Session Management
 |          |
 |          v
 |        Redis
 |    Session IDs / TTL
 |
 v
FastAPI Backend
 |
 +---- Document Processing
 |
 +---- Retrieval and Reranking
 |
 +---- LLM Providers
 |
 v
Grounded Response
```

## Technology Stack

| Layer | Technologies |
|---|---|
| Programming language | Python 3.12 |
| Frontend | Streamlit |
| Backend API | FastAPI, Uvicorn, Pydantic |
| Authentication | Supabase Auth |
| Session storage | Redis |
| Embeddings | Sentence Transformers |
| Embedding model | `all-MiniLM-L6-v2` |
| Vector database | ChromaDB / Chroma Cloud |
| Reranking | Cross-encoder |
| Reranker model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM providers | Groq, Google Gemini |
| Evaluation | MLflow |
| Data and experiment versioning | DVC, DagsHub |
| Testing | Pytest |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Deployment | Render |

## RAG Pipeline

1. **Ingestion:** Accept supported document formats.
2. **Parsing and cleaning:** Extract and normalize document text.
3. **Chunking:** Split content into smaller, overlapping passages.
4. **Embedding:** Convert passages into vector representations.
5. **Retrieval:** Retrieve the top five candidate passages.
6. **Reranking:** Re-rank candidates and select the top three passages.
7. **Prompt construction:** Build context from the selected passages.
8. **Generation:** Use the configured LLM provider to generate an answer.
9. **Citations:** Return an answer grounded in retrieved document sources.

## Document Management

The application supports:

- Multi-file document uploads.
- Upload progress reporting.
- Cooperative batch cancellation.
- Indexed document listing and refresh.
- Document deletion.
- Document metadata and chunk-count display.
- Duplicate detection and metadata filtering.

Cancellation prevents subsequent files from starting. It does not roll back a file that has already been accepted or processed by the backend.

## Authentication and Session Management

The application integrates Supabase authentication with backend session handling.

- Authenticated access to application features.
- Password-reset and recovery flow.
- Opaque session identifiers.
- Redis-backed session storage.
- Seven-day session TTL.
- Session-aware frontend-to-backend requests.
- User/workspace-aware access controls.

Redis is used for session infrastructure; it is not the vector database. ChromaDB is used for vector storage and semantic retrieval.

## Configuration

The project uses environment-based configuration. Documented defaults include:

| Setting | Default |
|---|---|
| Primary LLM provider | `groq` |
| Fallback LLM provider | `gemini` |
| Groq model | `llama-3.3-70b-versatile` |
| Gemini model | `gemini-3.7-flash` |
| Embedding model | `all-MiniLM-L6-v2` |
| Reranker model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Chunk size | `500` |
| Chunk overlap | `100` |
| Retrieval top-k | `5` |
| Rerank top-k | `3` |

Verify these values against the current configuration before deployment. The application settings model has required credentials and configuration constraints; follow `src/config/settings.py` and the deployment environment for the authoritative list of required variables.

Never commit `.env`, frontend secrets, Redis credentials, Supabase secrets, or LLM API keys.

## Run Locally

### Prerequisites

- Python 3.12
- Git
- Credentials for the configured LLM providers
- Supabase configuration for authentication
- Redis configuration for session storage, as required by the deployment settings

### 1. Clone the repository

```powershell
git clone https://github.com/Gagan910/enterprise-rag-assistant.git
cd enterprise-rag-assistant
```

### 2. Set up the backend environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Configure the required environment variables using the names defined in the application settings.

### 3. Start the backend

```powershell
uvicorn src.main:app --reload
```

Open the API documentation at:

http://127.0.0.1:8000/docs

### 4. Set up the frontend

Open a second terminal in the repository root:

```powershell
python -m venv .streamlit-venv
.\.streamlit-venv\Scripts\Activate.ps1

pip install -r requirements-frontend.txt
streamlit run app.py
```

Configure frontend secrets in `.streamlit/secrets.toml` when required. These may include `RAG_API_URL`, `SUPABASE_URL`, and `SUPABASE_ANON_KEY`.

Use the actual configuration names and requirements from the current codebase.

## Testing

Run the test suite from the repository root:

```powershell
python -m pytest -q
```

The project includes automated tests. Run the command above to obtain the current test count and results; do not treat test collection as proof that all tests pass.

## Docker

Build the backend image:

```powershell
docker build -t enterprise-rag-assistant:latest .
```

Run the application with the required environment variables:

```powershell
docker run --rm -p 8000:8000 --env-file .env enterprise-rag-assistant:latest
```

Configure external services such as Redis, Supabase, the vector store, and LLM providers according to the deployment setup. Ensure that required service URLs and credentials are available to the container.

## Evaluation

The project includes retrieval evaluation and MLflow tracking.

A previously recorded evaluation reported:

- **Hit Rate@5:** 1.00
- **MRR@5:** 1.00

These results apply to the specific evaluation dataset and setup used. They should not be interpreted as guaranteed performance on arbitrary documents or user queries.

## Deployment

- **Frontend:** [Render](https://enterprise-rag-frontend-g1bm.onrender.com)
- **Backend:** [Render](https://enterprise-rag-assistant-ik30.onrender.com)
- **API documentation:** [Swagger UI](https://enterprise-rag-assistant-ik30.onrender.com/docs)
- **Experiment tracking:** [DagsHub repository](https://dagshub.com/Gagan910/enterprise-rag-assistant)

The repository includes deployment and CI configuration. Consult the current workflow and deployment settings for the exact build steps, environment variables, and triggers.

## Security Considerations

- Do not commit API keys, passwords, access tokens, or production credentials.
- Use environment variables or the platform's secret-management features.
- Restrict access to authenticated users and enforce workspace-level isolation.
- Avoid uploading sensitive enterprise documents to a public demo unless the deployment has been approved for that data.
- Rotate any credential that has been exposed accidentally.
- Configure Redis access securely and use the appropriate connection and authentication settings for the environment.

## Project Structure

```text
enterprise-rag-assistant/
├── app.py
├── src/
│   ├── main.py
│   └── config/
│       └── settings.py
├── tests/
├── requirements.txt
├── requirements-frontend.txt
├── Dockerfile
└── README.md
```

Additional modules and folders support document ingestion, retrieval, evaluation, authentication, session management, scripts, and deployment.

## License

No license is specified in this README. Contact the repository owner before reusing or redistributing the project.
