# RAG-Powered Document Assistant

A Core Track Retrieval-Augmented Generation (RAG) application built with a local Ollama LLM, Chroma vector database, FastAPI backend, and Streamlit frontend.

## Overview

The application indexes a PDF document, splits its text into overlapping chunks, creates local embeddings, persists them in Chroma, retrieves relevant chunks for a question, and generates a grounded answer using Ollama.

The Core Track is text/PDF based. The Extended Track YOLO/computer-vision component is not included.

## Architecture

```text
PDF
 │
 ▼
Load & Inspect
 │
 ▼
Recursive Character Chunking
(1000 chars / 150 overlap)
 │
 ▼
Ollama Embeddings
(nomic-embed-text)
 │
 ▼
Persistent Chroma
 │
 ▼
FastAPI /query
 │
 ├── Retrieve top-k chunks
 │
 └── Ollama LLM (llama3.2:3b)
          │
          ▼
   Grounded answer + sources
          │
          ▼
      Streamlit UI
```

## Tech Stack

- Python 3.10+
- Ollama
- `llama3.2:3b`
- `nomic-embed-text`
- Chroma
- LangChain
- FastAPI
- Streamlit
- Pytest

## Project Structure

```text
rag-assistant-project/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/routes/query.py
│   │   ├── core/config.py
│   │   ├── schemas/query.py
│   │   ├── services/retrieval.py
│   │   ├── services/generation.py
│   │   ├── services/rag.py
│   │   └── utils/logging_config.py
│   ├── tests/test_query.py
│   ├── requirements.txt
│   ├── .env.example
│   └── Dockerfile
├── data/
│   ├── raw/
│   │   └── Attention.pdf
│   └── vector_store/
├── frontend/
│   ├── app.py
│   ├── api_client.py
│   ├── requirements.txt
│   └── .env.example
└── README.md
```

## Data

The knowledge base is the PDF used in Phase 2. The notebook extracts text, preserves page metadata, creates 1000-character chunks with 150-character overlap, and persists the resulting Chroma vector store.

Do not commit a large raw corpus or large vector store to GitHub. Keep the source-data acquisition instructions in the README if those files are excluded.

## Prerequisites

Install:

- Python 3.10+
- Ollama
- Git

Pull the local models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

Make sure Ollama is running.

## Phase 2

Run the Phase 2 notebook from the project environment.

It must create:

```text
data/vector_store/
data/vector_store/config.json
```

The backend does not rebuild embeddings at request time.

## Backend Setup

From the project root:

```bash
cd backend
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Copy:

```text
.env.example -> .env
```

Start the backend:

```bash
uvicorn app.main:app --reload
```

The API is available at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

## Environment Variables

### Backend

| Variable | Example | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `LLM_MODEL` | `llama3.2:3b` | Generation model |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `TOP_K` | `5` | Number of retrieved chunks |
| `FRONTEND_ORIGIN` | `http://localhost:8501` | Allowed frontend origin |

### Frontend

| Variable | Example | Purpose |
|---|---|---|
| `API_BASE_URL` | `http://localhost:8000` | FastAPI base URL |

## API Reference

### GET /health

Returns backend health information.

Example:

```bash
curl http://localhost:8000/health
```

### POST /query

Request:

```json
{
  "question": "What is transformer architecture?"
}
```

Example:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What is transformer architecture?\"}"
```

Response:

```json
{
  "answer": ".... [Source: Attention.pdf, p. 3]",
  "sources": [
    "Attention.pdf p.3"
  ]
}
```

## Tests

From `backend/`:

```bash
pytest -q
```

The tests cover:

1. a valid query / health happy path;
2. invalid empty input returning HTTP 422.

## Frontend Setup

In another terminal:

```bash
cd frontend
python -m venv .venv
```

Activate the environment and install:

```bash
pip install -r requirements.txt
```

Copy:

```text
.env.example -> .env
```

Run:

```bash
streamlit run app.py
```

Open the Streamlit URL shown by the terminal.

The frontend reads `API_BASE_URL` from the environment rather than hard-coding the backend address into the application logic.

## End-to-End Verification

1. Start Ollama.
2. Start FastAPI on port 8000.
3. Open `/docs` and test `/health`.
4. Test `/query` with a real document question.
5. Start Streamlit.
6. Ask the same question from the UI.
7. Confirm that the answer and cited sources are displayed.

## Evaluation

Phase 2 contains the 10-question retrieval/evaluation table required by the project guide.

Before submitting, copy the actual evaluation results from the completed notebook into this section. Do not invent evaluation scores.

## Screenshots

After the end-to-end demo is running, add screenshots here:

```text
docs/screenshots/backend-swagger.png
docs/screenshots/streamlit-chat.png
docs/screenshots/grounded-answer.png
```

Recommended screenshots:
1. FastAPI Swagger `/docs` showing `/health` and `/query`.
2. Streamlit chat interface with a real question.
3. A final answer showing the page/source citations.

## GitHub

Create `.gitignore` before the first commit.

Recommended commit:

```bash
git init
git add .
git commit -m "RAG assistant: notebook, FastAPI backend, frontend"
```

Then create a GitHub repository and push:

```bash
git remote add origin https://github.com/<your-username>/rag-assistant-app.git
git branch -M main
git push -u origin main
```

## Submission Checklist

- [ ] Phase 2 notebook runs top-to-bottom
- [ ] Persistent vector store exists
- [ ] Backend `/health` works
- [ ] Backend `/query` works
- [ ] At least 2 backend tests pass
- [ ] Frontend displays answers and sources
- [ ] Frontend uses `API_BASE_URL`
- [ ] README contains setup/API/evaluation information
- [ ] `.env` and `.venv` are not committed
- [ ] Large raw corpus/vector store is not committed
- [ ] Full question → API → retrieval → LLM → UI flow works
