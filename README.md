# Enterprise Document Intelligence & RAG Assistant

An end-to-end Retrieval-Augmented Generation (RAG) application for asking questions about enterprise documents and receiving grounded answers with source citations.

**Live demo:** [Frontend](https://enterprise-rag-frontend-g1bm.onrender.com) · [API](https://enterprise-rag-assistant-ik30.onrender.com) · [API docs](https://enterprise-rag-assistant-ik30.onrender.com/docs) · [Health](https://enterprise-rag-assistant-ik30.onrender.com/health)

**Repository:** [Gagan910/enterprise-rag-assistant](https://github.com/Gagan910/enterprise-rag-assistant)

---

## Features

- **Document ingestion:** PDF, DOCX, and TXT.
- **RAG pipeline:** document parsing, text cleaning, chunking, embeddings, retrieval, reranking, and answer generation.
- **Grounded answers:** relevant source citations accompany responses.
- **Hybrid LLM configuration:** Groq primary provider with Gemini fallback.
- **Retrieval quality:** Sentence Transformers embeddings and cross-encoder reranking.
- **Document management:** upload, list, retrieve, and delete documents, with duplicate detection and metadata filtering.
- **Authentication and isolation:** Supabase authentication and user/workspace-aware access controls, plus legacy API-key compatibility.
- **Evaluation and tracking:** retrieval evaluation with MLflow; DVC and DagsHub integration.
- **Deployment:** FastAPI backend, Streamlit frontend, Docker, and GitHub Actions CI.

## Architecture

```text
PDF / DOCX / TXT
       |
       v
Parse -> Clean -> Chunk -> Embed
                         |
                         v
                    ChromaDB
                         |
Question -> Retrieve top 5 -> Rerank top 3
                                      |
                                      v
                            Context + Prompt
                                      |
                                      v
                              Groq / Gemini
                                      |
                                      v
                       Grounded answer + citations
```

## Tech stack

| Layer | Tools |
|---|---|
| API | FastAPI, Uvicorn, Pydantic |
| UI | Streamlit |
| Embeddings | Sentence Transformers, `all-MiniLM-L6-v2` |
| Vector store | ChromaDB / Chroma Cloud |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM providers | Groq, Google Gemini |
| Authentication | Supabase |
| Evaluation and versioning | MLflow, DVC, DagsHub |
| Testing and deployment | Pytest, Docker, GitHub Actions, Render |

## RAG configuration

Current defaults from `src/config/settings.py`:

| Setting | Default |
|---|---|
| Primary LLM provider | `groq` |
| Fallback provider | `gemini` |
| Groq model | `llama-3.3-70b-versatile` |
| Gemini model | `gemini-3.7-flash` |
| Embedding model | `all-MiniLM-L6-v2` |
| Reranker model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Chunk size | `500` |
| Chunk overlap | `100` |
| Retrieval top-k | `5` |
| Rerank top-k | `3` |

Settings are loaded from `.env` as UTF-8. `GEMINI_API_KEY` is required by the current settings model, even when Groq is configured as the primary provider. Keep API keys and credentials out of source control.

## Run locally

### 1. Prerequisites

- Python 3.12
- Git
- API credentials for the configured LLM providers
- Supabase credentials if using authenticated/workspace features

### 2. Clone and install backend dependencies

```powershell
git clone https://github.com/Gagan910/enterprise-rag-assistant.git
cd enterprise-rag-assistant

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` file in the project root and set the required values. Use the variable names defined in `src/config/settings.py`; do not commit `.env`.

### 3. Start the API

```powershell
uvicorn src.main:app --reload
```

Open `http://127.0.0.1:8000/docs` to explore the API.

### 4. Start the Streamlit frontend

In a second PowerShell terminal:

```powershell
cd C:\MLProjects\enterprise-rag-assistant

python -m venv .streamlit-venv
.\.streamlit-venv\Scripts\Activate.ps1

pip install -r requirements-frontend.txt
streamlit run app.py
```

Configure frontend secrets in `.streamlit/secrets.toml` as required by the application, including `RAG_API_URL`, `SUPABASE_URL`, and `SUPABASE_ANON_KEY`. Do not commit the secrets file.

## Testing

Run the test suite from the repository root:

```powershell
python -m pytest -q
```

The suite was observed to **collect 114 tests on 14 October 2026**. Collection is not the same as a successful test run; run the command above to verify the current checkout.

## Docker

Build the backend image:

```powershell
docker build -t enterprise-rag-assistant:latest .
```

Run it with the required environment variables configured locally:

```powershell
docker run --rm -p 8000:8000 --env-file .env enterprise-rag-assistant:latest
```

Never place production credentials in the image or commit them to the repository.

## Evaluation

The project includes retrieval evaluation and MLflow tracking. A previously recorded evaluation reported **Hit Rate@5 = 1.00** and **MRR@5 = 1.00** on its evaluation set. These are dataset-specific results, not a guarantee of performance on arbitrary documents or queries.

## Deployment and experiment tracking

- **Frontend:** [Render deployment](https://enterprise-rag-frontend-g1bm.onrender.com)
- **Backend:** [Render deployment](https://enterprise-rag-assistant-ik30.onrender.com)
- **API documentation:** [Swagger UI](https://enterprise-rag-assistant-ik30.onrender.com/docs)
- **DagsHub repository:** [enterprise-rag-assistant](https://dagshub.com/Gagan910/enterprise-rag-assistant)

The repository also includes Docker and CI configuration. Check the workflow file for the exact branch and event triggers configured in the current checkout.

## Security notes

- Use synthetic or non-sensitive documents for demos unless your deployment has been reviewed and approved for the data involved.
- Keep `.env`, `.streamlit/secrets.toml`, access tokens, and API keys out of Git.
- Use authentication and workspace isolation when exposing the application to multiple users.
- Rotate any credential that has been exposed accidentally.

## Project structure

```text
enterprise-rag-assistant/
├── app.py                    # Streamlit frontend
├── src/
│   ├── main.py               # FastAPI application
│   └── config/
│       └── settings.py       # Application settings
├── tests/                    # Automated tests
├── requirements.txt          # Backend dependencies
├── requirements-frontend.txt # Frontend dependencies
├── Dockerfile
└── README.md
```

Additional folders support ingestion, retrieval, evaluation, scripts, and deployment. See the source tree for the current layout.

## License

No license is specified here. Contact the repository owner before reusing or redistributing the project.
