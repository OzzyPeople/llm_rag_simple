A small Retrieval-Augmented Generation (RAG) service built with FastAPI, LangChain, and Gemini, that answers questions about stock market reports (e.g., Stock_Market_Performance_2024.pdf).

Project structure:

POC_RAG/
├─ data/                        # PDFs or other documents
│   └─ Stock_Market_Performance_2024.pdf
├─ chroma_db/                   # Vector store (persisted embeddings)
├─ src/
│   ├─ api/                     # FastAPI application
│   │   ├─ main.py              # App entrypoint
│   │   ├─ deps.py              # Loads RAG pipeline (vectorstore, retriever, LLM)
│   │   └─ routes/
│   │       └─ stock.py         # /stock/ask endpoint
│   ├─ llm_poc/                 # RAG core logic
│   │   ├─ pipeline.py          # Build and persist vectorstore
│   │   └─ llm.py               # LLM configuration
│   ├─ retrievers/              # Retriever definitions
│   └─ utils/                   # Utility modules
├─ pyproject.toml
└─ README.md

SET UP

# 1. Activate Poetry environment
cd POC_RAG
poetry install
poetry shell

# 2. Run FastAPI
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8080

Tech Stack

FastAPI — lightweight API framework
LangChain — document loaders, retrievers, chains
ChromaDB — local vector database
Google Gemini / Anthropic Claude — LLMs for QA
PyMuPDF / pdfplumber — PDF parsing

Testing

Swagger UI → http://127.0.0.1:8080/docs
ReDoc → http://127.0.0.1:8080/redoc

📌 Next Steps

Add multiple document ingestion.
Return page references in answers.
Integrate re-ranking (MMR / cross-encoder).
Deploy to Cloud Run or AWS Lambda.