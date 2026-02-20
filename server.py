"""
server.py - FastAPI + LangServe Backend for Smart Contract Assistant

This module handles:
  1. Document ingestion (PDF/DOCX → text → chunks → embeddings → FAISS)
  2. RAG chain definition (retrieval + LLM answer generation with citations)
  3. Exposing the chain as a REST microservice via LangServe

LLM / Embeddings backend:
  - If OPENAI_API_KEY is set in the environment → OpenAI (gpt-4o-mini + text-embedding-3-small)
  - Otherwise → Ollama (llama3.1 + nomic-embed-text)  ← requires `ollama serve` running locally
"""

import os
import uuid
import tempfile
from pathlib import Path
from typing import List

# ── Environment ──────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()  # Loads OPENAI_API_KEY (and others) from .env file

# ── FastAPI & LangServe ───────────────────────────────────────────────────────
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

# ── LangChain core ────────────────────────────────────────────────────────────
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage

# ── LangChain community / OpenAI ──────────────────────────────────────────────
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─────────────────────────────────────────────────────────────────────────────
# Backend selection: OpenAI (if key present) or Ollama (local fallback)
# ─────────────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = "" #os.getenv("OPENAI_API_KEY", "").strip()
USE_OPENAI = bool(OPENAI_API_KEY)

# Ollama connection settings (override via env vars if needed)
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:latest")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")

if USE_OPENAI:
    print("✅ OPENAI_API_KEY found — using OpenAI backend.")
else:
    print(
        f"⚠️  No OPENAI_API_KEY found — falling back to Ollama "
        f"({OLLAMA_LLM_MODEL} @ {OLLAMA_BASE_URL}).\n"
        f"   Make sure `ollama serve` is running and the models are pulled:\n"
        f"     ollama pull {OLLAMA_LLM_MODEL}\n"
        f"     ollama pull {OLLAMA_EMBED_MODEL}"
    )


def get_llm(temperature: int = 0):
    """
    Return the appropriate LLM based on available credentials.
    - OpenAI: gpt-4o-mini  (fast, cost-effective)
    - Ollama: llama3.1      (local, free, requires ollama serve)
    """
    if USE_OPENAI:
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
            openai_api_key=OPENAI_API_KEY,
        )
    else:
        # Lazy import so the package is only required when actually used
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise RuntimeError(
                "langchain-ollama is not installed. "
                "Run: pip install langchain-ollama"
            )
        return ChatOllama(
            model=OLLAMA_LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=temperature,
        )


def get_embeddings():
    """
    Return the appropriate embeddings model based on available credentials.
    - OpenAI: text-embedding-3-small
    - Ollama: nomic-embed-text (pull with: ollama pull nomic-embed-text)
    """
    if USE_OPENAI:
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=OPENAI_API_KEY,
        )
    else:
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError:
            raise RuntimeError(
                "langchain-ollama is not installed. "
                "Run: pip install langchain-ollama"
            )
        return OllamaEmbeddings(
            model=OLLAMA_EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Global in-memory store: maps session_id → FAISS retriever
# In production you'd persist these, but local workshop scope keeps it simple.
# ─────────────────────────────────────────────────────────────────────────────
SESSION_RETRIEVERS: dict = {}

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Smart Contract Assistant API",
    description="RAG pipeline for PDF/DOCX Q&A with source citations",
    version="1.0.0",
)

# Allow the Gradio UI (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# GET /health  –  Show which backend is active
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Quick sanity-check endpoint that reveals the active backend."""
    if USE_OPENAI:
        return {"status": "ok", "backend": "openai", "llm": "gpt-4o-mini"}
    return {
        "status": "ok",
        "backend": "ollama",
        "llm": OLLAMA_LLM_MODEL,
        "embeddings": OLLAMA_EMBED_MODEL,
        "ollama_url": OLLAMA_BASE_URL,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load a document (PDF or DOCX) and return LangChain Document objects
# ─────────────────────────────────────────────────────────────────────────────
def load_document(file_path: str) -> List:
    """
    Choose the correct LangChain loader based on file extension.
    - PDF  → PyMuPDFLoader  (fast, preserves layout metadata)
    - DOCX → Docx2txtLoader (simple text extraction)
    """
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        loader = PyMuPDFLoader(file_path)
    elif ext in (".docx", ".doc"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader.load()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: chunk documents into overlapping pieces for embedding
# ─────────────────────────────────────────────────────────────────────────────
def chunk_documents(docs: List) -> List:
    """
    Split long documents into manageable chunks.
    - chunk_size=1000  → roughly a paragraph; fits in embedding context window
    - chunk_overlap=200 → 20 % overlap preserves context across chunk boundaries
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],  # Try paragraph → line → word
    )
    return splitter.split_documents(docs)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a FAISS vector store from a list of chunks
# ─────────────────────────────────────────────────────────────────────────────
def build_vector_store(chunks: List) -> FAISS:
    """
    Embed every chunk and store in FAISS.
    Uses OpenAI embeddings when an API key is available, otherwise Ollama.
    FAISS is an in-process library (no server needed), ideal for local workshops.
    """
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# ─────────────────────────────────────────────────────────────────────────────
# POST /ingest  –  Upload a file, embed it, store retriever in session map
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """
    Endpoint called by the Gradio UI when the user uploads a document.

    Flow:
      1. Save the upload to a temp file.
      2. Load → chunk → embed → FAISS.
      3. Store the retriever under a new session_id.
      4. Return the session_id so subsequent /ask calls know which index to use.
    """
    # Validate file type before doing any work
    if not file.filename.endswith((".pdf", ".docx", ".doc")):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")

    # Write upload to a temporary file (avoids holding bytes in memory)
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        docs   = load_document(tmp_path)          # Step 1: parse
        chunks = chunk_documents(docs)             # Step 2: chunk
        vs     = build_vector_store(chunks)        # Step 3: embed + index
    finally:
        os.unlink(tmp_path)  # Always clean up the temp file

    # Store vectorstore + retriever per session
    session_id = str(uuid.uuid4())
    SESSION_RETRIEVERS[session_id] = {
        "vectorstore": vs,
        "retriever": vs.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )
    }


    return {
        "session_id": session_id,
        "chunks_indexed": len(chunks),
        "backend": "openai" if USE_OPENAI else f"ollama/{OLLAMA_LLM_MODEL}",
        "message": "Document ingested successfully.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# RAG prompt template
# ─────────────────────────────────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful legal/contract assistant. 
Answer ONLY using the context provided below — never hallucinate.
Always end your answer with a "Sources:" section listing the page numbers 
or chunk excerpts you used. If the context doesn't contain the answer, say so.

Context:
{context}
""",
    ),
    MessagesPlaceholder(variable_name="chat_history"),  # Inject conversation history
    ("human", "{question}"),
])


# ─────────────────────────────────────────────────────────────────────────────
# Helper: format retrieved docs into a single context string with source labels
# ─────────────────────────────────────────────────────────────────────────────
def format_docs(docs) -> str:
    """
    Combine retrieved chunks into one block.
    Each chunk is labeled with its source file and page number (when available)
    so the LLM can cite them accurately.
    """
    parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "uploaded document")
        page   = doc.metadata.get("page", "?")
        parts.append(f"[Chunk {i+1} | {source} | Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# POST /ask  –  Answer a question given a session_id and optional chat history
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/ask")
async def ask(payload: dict):
    """
    Main Q&A endpoint.

    Expected JSON body:
    {
        "session_id":   "<uuid returned by /ingest>",
        "question":     "What is the termination clause?",
        "chat_history": [{"role": "human", "content": "..."}, ...]   # optional
    }

    Flow:
      1. Retrieve top-k relevant chunks from the session's FAISS index.
      2. Format them as context.
      3. Build chat history as LangChain message objects.
      4. Run LLM with the RAG prompt.
      5. Return the answer string.
    """
    session_id   = payload.get("session_id")
    question     = payload.get("question", "")
    raw_history  = payload.get("chat_history", [])

    if not session_id or session_id not in SESSION_RETRIEVERS:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a document first.")

    retriever = SESSION_RETRIEVERS[session_id]["retriever"]


    # Convert raw history dicts → LangChain message objects
    chat_history = []
    for msg in raw_history:
        if msg["role"] == "human":
            chat_history.append(HumanMessage(content=msg["content"]))
        else:
            chat_history.append(AIMessage(content=msg["content"]))

    # Retrieve relevant chunks
    docs    = retriever.invoke(question)
    context = format_docs(docs)

    # Build and invoke the RAG chain using the appropriate backend
    llm    = get_llm(temperature=0)
    chain  = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({
        "context":      context,
        "question":     question,
        "chat_history": chat_history,
    })

    return {"answer": answer, "session_id": session_id}

# ─────────────────────────────────────────────────────────────────────────────
# POST /summarize – Generate structured summary of the entire document
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/summarize")
async def summarize(payload: dict):
    """
    Generate a structured summary of the uploaded contract.

    Expected JSON body:
    {
        "session_id": "<uuid returned by /ingest>"
    }
    """

    session_id = payload.get("session_id")

    if not session_id or session_id not in SESSION_RETRIEVERS:
        raise HTTPException(status_code=404, detail="Session not found.")

    vectorstore = SESSION_RETRIEVERS[session_id]["vectorstore"]

    # Retrieve many chunks to approximate full-document summary
    docs = vectorstore.similarity_search("", k=20)

    full_text = "\n\n".join([doc.page_content for doc in docs])

    summary_prompt = ChatPromptTemplate.from_template("""
You are a legal contract analyst.

Provide a structured summary with:

1. Executive Summary (short overview)
2. Key Clauses (bullet points)
3. Potential Risks
4. Important Dates / Financial Terms

Document:
{document_text}
""")

    llm = get_llm(temperature=0)
    chain = summary_prompt | llm | StrOutputParser()

    summary = chain.invoke({"document_text": full_text})

    return {"summary": summary}

# ─────────────────────────────────────────────────────────────────────────────
# LangServe route (optional but required by spec)
# Mounts the RAG chain at /chain/playground for interactive testing
# ─────────────────────────────────────────────────────────────────────────────
# We expose a simple stateless chain here for LangServe compatibility.
# For full session-aware RAG, use the /ask endpoint above.
_demo_prompt = ChatPromptTemplate.from_template(
    "Answer the following contract question concisely:\n\n{question}"
)
_demo_chain  = _demo_prompt | get_llm(temperature=0) | StrOutputParser()

add_routes(
    app,
    _demo_chain,
    path="/chain",  # Browse to http://localhost:8000/chain/playground
)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)