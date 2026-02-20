# 📄 Smart Contract Summary & Q&A Assistant

A session-aware RAG (Retrieval Augmented Generation) application for uploading PDF/DOCX contracts, generating structured summaries, and chatting with them using a grounded conversational interface.

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

## 📂 Project Structure

```
smart_contract_assistant/
├── server.py          # FastAPI + LangServe backend (RAG pipeline)
├── client_ui.py       # Gradio frontend (file upload + chat UI)
├── requirements.txt   # Python dependencies
├── .env               # Environment variables (NOT committed)
├── .env.example       # Template file for evaluators
└── README.md
```

---

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

## 📌 Disclaimer

This tool is for educational purposes only and does not constitute legal advice.
