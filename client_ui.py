"""
client_ui.py - Gradio Frontend for Smart Contract Assistant

This module is intentionally separate from server.py (as required by the spec).
It only knows:
  - Where the FastAPI server is running (SERVER_URL)
  - How to call /ingest and /ask via HTTP
  - How to render the UI with Gradio

It does NOT contain any LangChain, embedding, or LLM logic.
"""

import os
import requests
import gradio as gr
from dotenv import load_dotenv

load_dotenv()  # Load SERVER_URL from .env if overridden there

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — change SERVER_URL if running the server on a different host
# ─────────────────────────────────────────────────────────────────────────────
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")


# ─────────────────────────────────────────────────────────────────────────────
# State helpers
# ─────────────────────────────────────────────────────────────────────────────

def reset_state():
    """Return fresh defaults for all session state variables."""
    return None, []   # session_id=None, chat_history=[]


# ─────────────────────────────────────────────────────────────────────────────
# Action: Upload a document → call /ingest on the backend
# ─────────────────────────────────────────────────────────────────────────────
def upload_document(file_obj, session_id, chat_history):
    """
    Called when the user clicks 'Upload & Process'.

    Args:
        file_obj:     Gradio file object (has .name attribute with temp path)
        session_id:   Current session ID (None if no doc uploaded yet)
        chat_history: Current conversation history list

    Returns:
        status_text  – A message shown in the UI status box
        session_id   – New session ID (or unchanged on failure)
        chat_history – Unchanged (reset only on explicit new upload)
    """
    if file_obj is None:
        return "⚠️ Please select a file first.", session_id, chat_history

    # POST the file to the /ingest endpoint as multipart form data
    try:
        with open(file_obj.name, "rb") as f:
            filename = os.path.basename(file_obj.name)
            response = requests.post(
                f"{SERVER_URL}/ingest",
                files={"file": (filename, f)},
                timeout=120,  # Large docs may take a while to embed
            )
        response.raise_for_status()
        data       = response.json()
        session_id = data["session_id"]
        chunks     = data["chunks_indexed"]
        status     = f"✅ '{filename}' ingested successfully — {chunks} chunks indexed.\nReady to chat!"
        # Reset chat history when a new document is loaded
        chat_history = []
    except requests.exceptions.ConnectionError:
        status = "❌ Cannot connect to server. Make sure server.py is running on port 8000."
    except Exception as e:
        status = f"❌ Ingestion failed: {str(e)}"

    return status, session_id, chat_history


# ─────────────────────────────────────────────────────────────────────────────
# Action: Send a question → call /ask on the backend → update chatbot
# ─────────────────────────────────────────────────────────────────────────────
def ask_question(user_message, chat_history, session_id):
    """
    Called when the user submits a question in the chat box.

    Args:
        user_message: The question typed by the user
        chat_history: List of {"role": ..., "content": ...} dicts (conversation so far)
        session_id:   Active session ID from /ingest

    Returns:
        "":           Clears the input textbox after submission
        chat_history: Updated conversation with the new Q&A appended
        session_id:   Unchanged
    """
    if not user_message.strip():
        return "", chat_history, session_id

    if session_id is None:
        # Guard-rail: prevent questions without an uploaded document
        chat_history.append({"role": "assistant", "content": "⚠️ Please upload a document first using the Upload tab."})
        return "", chat_history, session_id

    # Append user turn to history (shown immediately in UI)
    chat_history.append({"role": "user", "content": user_message})

    # Call the /ask endpoint with the current question and full history
    try:
        response = requests.post(
            f"{SERVER_URL}/ask",
            json={
                "session_id":   session_id,
                "question":     user_message,
                "chat_history": chat_history[:-1],  # Exclude the just-added user turn (server re-adds it)
            },
            timeout=60,
        )
        response.raise_for_status()
        answer = response.json().get("answer", "No answer returned.")
    except requests.exceptions.ConnectionError:
        answer = "❌ Cannot connect to server. Make sure server.py is running."
    except Exception as e:
        answer = f"❌ Error: {str(e)}"

    # Append assistant turn to history
    chat_history.append({"role": "assistant", "content": answer})

    return "", chat_history, session_id  # "" clears the input textbox

def get_summary(session_id):
    if not session_id:
        return "⚠️ Please upload a document first!"
    try:
        # إرسال طلب للـ endpoint الجديد اللي في server.py
        resp = requests.post(f"{SERVER_URL}/summarize", json={"session_id": session_id})
        if resp.status_code == 200:
            return resp.json().get("summary")
        return f"❌ Error: {resp.text}"
    except Exception as e:
        return f"🔌 Connection Error: {str(e)}"

# ─────────────────────────────────────────────────────────────────────────────
# Build the Gradio UI
# ─────────────────────────────────────────────────────────────────────────────
def build_ui():
    # امسح theme و css من هنا
    with gr.Blocks(title="Smart Contract Assistant") as demo:

        # ── Page header ──────────────────────────────────────────────────────
        gr.Markdown(
            """
            # 📄 Smart Contract Summary & Q&A Assistant
            Upload a PDF or DOCX contract, then ask questions about it.
            The assistant will answer using only the document content and cite its sources.
            """
        )

        # ── Session state (persisted across tab switches) ────────────────────
        # Gradio `gr.State` keeps Python values alive on the server per browser session.
        session_id_state   = gr.State(value=None)
        chat_history_state = gr.State(value=[])

        # ── Tab 1: Upload ────────────────────────────────────────────────────
        with gr.Tab("📁 Upload Document"):
            gr.Markdown("### Step 1: Upload your contract (PDF or DOCX)")

            file_input = gr.File(
                label="Select File",
                file_types=[".pdf", ".docx", ".doc"],
                type="filepath",  # Gradio saves to a temp path; we read from there
            )

            upload_btn = gr.Button("⚙️ Process Document", variant="primary")

            status_box = gr.Textbox(
                label="Status",
                interactive=False,   # Read-only
                lines=3,
                placeholder="Upload a document to get started…",
            )

            # Wire up the button
            upload_btn.click(
                fn=upload_document,
                inputs=[file_input, session_id_state, chat_history_state],
                outputs=[status_box, session_id_state, chat_history_state],
            )

        # ── Tab 2: summarize ─────────────────────────────────────────────────
        with gr.Tab("📜 Contract Summary"):
            summary_btn = gr.Button("✨ Generate Full Analysis", variant="primary")
            # الـ Markdown أفضل لعرض الملخص لأنه بيدعم الـ formatting اللي الموديل بيطلعه
            summary_display = gr.Markdown("Click the button to summarize the contract...")

            # ربط الزرار بالدالة
            summary_btn.click(
                get_summary,
                inputs=[session_id_state], # اتأكد إن ده اسم المتغير اللي شايل الـ uuid
                outputs=summary_display
            )

        # ── Tab 3: Chat ──────────────────────────────────────────────────────
        with gr.Tab("💬 Ask Questions"):
            gr.Markdown("### Step 2: Chat with your document")

            chatbot = gr.Chatbot(
                label="Conversation",
                height=450,
                show_label=True,
                avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=contract"),
            )

            with gr.Row():
                question_input = gr.Textbox(
                    placeholder="e.g. What is the termination clause? What are the payment terms?",
                    label="Your Question",
                    scale=4,
                    lines=1,
                )
                send_btn = gr.Button("Send ➤", variant="primary", scale=1)

            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

            # Send on button click or Enter key
            send_btn.click(
                fn=ask_question,
                inputs=[question_input, chat_history_state, session_id_state],
                outputs=[question_input, chatbot, session_id_state],
            )
            question_input.submit(
                fn=ask_question,
                inputs=[question_input, chat_history_state, session_id_state],
                outputs=[question_input, chatbot, session_id_state],
            )

            # Clear chat clears both the display and the stored history
            clear_btn.click(
                fn=lambda: ([], []),
                outputs=[chatbot, chat_history_state],
            )

        # ── Footer ───────────────────────────────────────────────────────────
        gr.Markdown(
            """
            ---
            *This tool is for educational purposes only. Not legal advice.*
            """
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
# http://127.0.0.1:7860
if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",  # التغيير الجوهري هنا
        server_port=7860,
        share=False,
    )
