# 📄 Smart Contract Summary & Q&A Assistant

A session-aware RAG (Retrieval Augmented Generation) application for uploading PDF/DOCX contracts, generating structured summaries, and chatting with them using a grounded conversational interface. Now featuring a Unified Backend/Frontend architecture and Docker support.

**Tech Stack:** LangChain · FastAPI · LangServe · FAISS · Gradio · OpenAI / Ollama

---

## 🚀 Features

- 📁 Upload PDF or DOCX contracts
- 🧠 Semantic search with FAISS vector store
- 💬 Conversational Q&A with memory (session-based)
- 📜 Structured contract summarization
- 📚 Source citation (page + chunk references)
- 🔌 Dual backend support:
  - OpenAI (if API key provided)
  - Ollama local fallback (no API key required)
- 🧪 LangServe playground for chain inspection

---

## 🚀 Key Updates (Latest Version)

- Unified Port Architecture: Both API (FastAPI) and UI (Gradio) now run on a single port (8000) for easier deployment.
- Full Dockerization: Run the entire stack (except Ollama) with one command.
- Enhanced Local Support: Optimized for `host.docker.internal` to connect seamlessly with local Ollama instances on Windows/Mac.

---

## 📂 Project Structure

```
smart_contract_assistant/
├── server.py          # Unified FastAPI + Gradio App (RAG pipeline + UI)
├── client_ui.py       # Gradio frontend (file upload + chat UI)
├── Dockerfile         # Optimized Python environment
├── compose.yaml       # Docker orchestration
├── requirements.txt   # Backend & Frontend dependencies
├── .env               # Environment variables (NOT committed)
├── .env.example       # Template file for evaluators
└── README.md
```

---

## 🐳 Quick Start with Docker (The Easiest Way)

1. Clone the repo
2. Start the application:
```bash
docker compose up --build
```
3. - 🎨 Web UI: http://localhost:8000
   - ⚙️ API Docs: http://localhost:8000/docs
   - 🧪 Playground: http://localhost:8000/chain/playground

## ⚙️ Ollama Setup (Local LLM)

To use the local fallback, ensure Ollama is running on your host machine:

1. Install Ollama and pull the models:
```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```
2. Windows Users: Set Environment Variables to allow Docker access:

- `OLLAMA_HOST=0.0.0.0`

- `OLLAMA_ORIGINS=*`
(Restart Ollama after setting these)

## 🧠 Architecture Flow

```bash
User (Browser) → Port 8000 (FastAPI)
                  │
                  ├── / (Gradio UI Route)
                  ├── /ingest (RAG Pipeline)
                  ├── /ask (Conversational Chain)
                  └── Connection → host.docker.internal:11434 (Ollama Host)
```

## 🔎 Why this Architecture?

- Zero-Config UI: No need to manage two terminal windows.
- Scalability: The FastAPI backend remains accessible for external API calls while serving the UI.
- Environment Isolation: Docker ensures that PDF parsing and FAISS indexing work consistently across all operating systems.

## 🧰 Prerequisites

- Python 3.10+
- (Optional) OpenAI API key
- (Optional) Ollama installed locally (`ollama serve`)

---

## ⚙️ Environment Configuration

Create a `.env` file in the root directory:

```env
# If you are the evaluator:
# Paste your OpenAI API key below.
# If left empty, the system will automatically fallback to Ollama (local model).

OPENAI_API_KEY=
SERVER_URL=http://localhost:8000
```

> ⚠️ Do **NOT** commit `.env` to version control.

---

## 🛠 Installation

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

---

## ▶️ Running the Application

You need two terminals.

**Terminal 1 — Start Backend**
```bash
python server.py
```

Backend runs on: `http://localhost:8000`

You can also inspect:
- Swagger Docs → `http://localhost:8000/docs`
- LangServe Playground → `http://localhost:8000/chain/playground`

**Terminal 2 — Start Frontend**
```bash
python client_ui.py
```

UI runs on: `http://localhost:7860`

---

## 🧠 How It Works (Architecture Overview)

```
User → Gradio UI (7860)
        │
        ├── POST /ingest → FastAPI (8000)
        │       ├─ Load document (PDF/DOCX)
        │       ├─ Chunk text
        │       ├─ Embed chunks
        │       ├─ Store in FAISS
        │       └─ Return session_id
        │
        ├── POST /ask
        │       ├─ Retrieve top-k chunks
        │       ├─ Format context with citations
        │       ├─ Apply grounded RAG prompt
        │       └─ Return answer
        │
        └── POST /summarize
                └─ Generate structured contract analysis
```

---

## 💬 Example Questions

- What is the termination clause?
- What are the payment terms?
- Who are the parties involved?
- What risks does this contract contain?
- Summarize the financial obligations.

---

## 📜 Structured Summary Output

The `/summarize` endpoint produces:

- Executive Summary
- Key Clauses
- Potential Risks
- Important Dates / Financial Terms

---

## 🔎 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/ingest` | POST | Upload document and create session |
| `/ask` | POST | Ask question (requires `session_id`) |
| `/summarize` | POST | Generate structured contract summary |
| `/chain/playground` | GET | LangServe interactive testing |
| `/docs` | GET | Swagger API documentation |
| `/health` | GET | Check active backend |

---

## 🧪 Evaluation Considerations

| Metric | Implementation |
|---|---|
| Faithfulness | System prompt restricts answers to provided context only |
| Citations | Each chunk labeled with filename + page |
| Retrieval | Top-4 similarity search via FAISS |
| Memory | Session-based retriever mapping |
| Modularity | Strict separation between UI and backend |
| Deployment Flexibility | OpenAI or Ollama backend |

---

## ⚠️ Notes

- Large contracts may take 10–30 seconds to embed.
- Ollama backend requires:

```bash
ollama serve
ollama pull llama3.1
ollama pull nomic-embed-text
```

---

## 📊 Evaluation Report

### Evaluation Setup

The system was evaluated using a sample legal contract (10–20 pages). Three representative contract-related queries were tested:

1. What is the termination clause?
2. What are the payment terms?
3. What is the governing law?

Each query was tested for retrieval relevance, faithfulness to context, citation accuracy, and response latency.

### Metrics

| Metric | Description | Observation |
|---|---|---|
| Retrieval Quality | Top-4 semantic similarity via FAISS | Relevant clauses retrieved in most cases |
| Faithfulness | Answers restricted to retrieved context | No hallucinated external facts observed |
| Citation Accuracy | Each chunk includes filename + page | Correct page references returned |
| Latency | End-to-end response time | ~2–5 seconds (OpenAI backend) |
| Summarization Quality | Structured output format | Clear clause extraction and risk identification |

### Limitations

- Large contracts (>100 pages) may increase embedding time.
- Retrieval depends on chunk size and embedding quality.
- Similarity threshold is not dynamically optimized.
- Legal reasoning is extractive, not interpretive.
- Not a substitute for professional legal advice.

---

## 📌 Disclaimer

This tool is for educational purposes only and does not constitute legal advice.
