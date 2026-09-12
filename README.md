\# Enterprise Document Intelligence \& RAG Assistant



An end-to-end Retrieval-Augmented Generation (RAG) system for enterprise document intelligence.



The system allows users to query information across documents and receive grounded answers with source citations.



\## Architecture



```text

Documents

(PDF / DOCX / TXT)

&#x20;       ↓

Document Parser

&#x20;       ↓

Cleaning \& Normalization

&#x20;       ↓

Chunking

&#x20;       ↓

Embedding Model

&#x20;       ↓

Vector Database

&#x20;       ↓

Retrieval

&#x20;       ↓

Reranking

&#x20;       ↓

Prompt + Context

&#x20;       ↓

LLM

&#x20;       ↓

Answer + Source Citations

