from __future__ import annotations

import os
import tempfile
import uuid
import base64
from typing import Dict, Any, List
from threading import Thread
from io import BytesIO

from flask import Flask, render_template, request, jsonify, session as flask_session
from dotenv import load_dotenv

from Utils.pdf_utils import load_pdf, clean_documents
from Utils.chunking import chunk_documents
from Utils.embedding import EmbedData
from Utils.vector_db import FAISSVectorStore
from Utils.retriever import Retriever
from Utils.rag import RAG
from Utils.image_embedding import ImageEmbedder
from Utils.web_search import search_to_documents
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from History.history import History
from Utils.logger import logging
from PIL import Image


load_dotenv()
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY")


"""In-memory user session mapping.

# We now delegate per-chat (conversation) storage to History. Each chat == one
# History session. For each browser/user (flask session "sid"), we keep only a
# list of their History session ids plus the active one.

Structure:
SESSIONS[sid] = { "chat_ids": [history_session_ids...], "active_chat_id": str }
"""

SESSIONS: Dict[str, Dict[str, Any]] = {}
HISTORY = History()
NORMAL_CHAT_LLM: ChatGoogleGenerativeAI | None = None
IMAGE_EMBEDDER: ImageEmbedder | None = None


# ---------------- Helper Utilities (Deduplicate common patterns) ---------------- #
def _json_error(message: str, code: int = 400):
    logging.warning(f"error: {message} status={code}")
    return jsonify({"error": message}), code


def _session_state():
    """Return (sid, session_state_dict, active_chat_id)."""
    sid = get_session_id()
    state = SESSIONS[sid]
    return sid, state, state["active_chat_id"]


def _validate_upload(part_name: str, expected_exts=None):
    """Retrieve a file from request.files and validate extension.

    expected_exts: iterable of lowercase extensions including dot (e.g., ['.pdf']). If None skip ext check.
    Returns file object or a tuple (json_response, status) to be returned directly.
    """
    if part_name not in request.files:
        return _json_error("No file part")
    file = request.files[part_name]
    if not file.filename or file.filename == "":
        return _json_error("Empty filename")
    if expected_exts:
        fname_lower = file.filename.lower()
        if not any(fname_lower.endswith(ext) for ext in expected_exts):
            return _json_error("Unsupported format" if len(expected_exts) > 1 else f"Only {expected_exts[0]} supported")
    return file


def _summarize(text: str, max_len: int = 50) -> str:
    t = " ".join(text.strip().split())
    return t if len(t) <= max_len else t[: max_len - 3] + "..."

def get_session_id() -> str:
    sid = flask_session.get("sid")
    if not sid or sid not in SESSIONS:
        sid = str(uuid.uuid4())
        flask_session["sid"] = sid
        # create first History session
        chat_id = HISTORY.new_chat()
        SESSIONS[sid] = {"chat_ids": [chat_id], "active_chat_id": chat_id}
        logging.info(f"new user session sid={sid} chat_id={chat_id}")
    return sid

def _serialize_messages(chat_id: str) -> List[Dict[str, str]]:
    msgs = HISTORY.get_messages(chat_id)
    out: List[Dict[str, str]] = []
    for m in msgs:
        # LangChain HumanMessage typically has .type == 'human'
        mtype = getattr(m, "type", "")
        if mtype == "human":
            role = "user"
        elif mtype in ("ai", "assistant"):  # ai message type is usually 'ai'
            role = "assistant"
        else:
            # fallback: look for 'Human'/'AI' in class name
            name = m.__class__.__name__.lower()
            role = "user" if "human" in name else "assistant"
        out.append({"role": role, "content": getattr(m, "content", str(m))})
    return out


def _get_normal_chat_llm() -> ChatGoogleGenerativeAI | None:
    """Return a singleton Gemini chat LLM for normal (non-RAG) chat."""
    global NORMAL_CHAT_LLM
    if NORMAL_CHAT_LLM is not None:
        return NORMAL_CHAT_LLM
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        api_key = api_key.strip().strip('"').strip("'")
    if not api_key:
        logging.warning("GOOGLE_API_KEY not found for normal chat LLM")
        return None
    model_name = os.getenv("LLM_MODEL", "gemini-2.0-flash-exp")
    NORMAL_CHAT_LLM = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.6)
    logging.info(f"initialized normal chat LLM model={model_name}")
    return NORMAL_CHAT_LLM


def _get_image_embedder() -> ImageEmbedder:
    global IMAGE_EMBEDDER
    if IMAGE_EMBEDDER is not None:
        return IMAGE_EMBEDDER
    model_name = os.getenv("IMAGE_EMBEDDER_MODEL", "clip-ViT-B-32")
    IMAGE_EMBEDDER = ImageEmbedder(model_name=model_name)
    logging.info(f"initialized image embedder model={model_name}")
    return IMAGE_EMBEDDER


def _answer_with_docs_via_llm(llm: ChatGoogleGenerativeAI, question: str, docs: List[Any]) -> str:
    """Stuff docs into a default prompt and invoke the LLM."""
    context = "\n\n".join([(d.page_content or "") for d in (docs or [])])
    prompt = (
        "Answer the question concisely. If the provided context contains useful information, incorporate it.\n"
        "If the context is irrelevant or insufficient, answer from your general knowledge.\n"
        "Do not fabricate citations. Only cite sources if you actually used the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    try:
        resp = llm.invoke(prompt)
        answer = getattr(resp, "content", str(resp))
    except Exception:
        return _normal_chat_answer(get_session_id(), question)
    # Append minimal sources
    srcs = []
    for d in (docs or [])[:3]:
        title = d.metadata.get("title") if isinstance(getattr(d, "metadata", None), dict) else None
        url = d.metadata.get("url") if isinstance(getattr(d, "metadata", None), dict) else None
        src = title or (os.path.basename(str(url)) if url else None) or (str(url) if url else None)
        if src:
            srcs.append(src)
    if srcs:
        answer += "\n\n---\nSources:\n" + "\n".join(f"- {s}" for s in srcs)
    return answer


def _warmup_image_embedder(sync: bool = False):
    """Warm up the image embedder by encoding a tiny RGB image.

    This triggers model downloads (text + image towers) so first user upload
    doesn't block on large downloads.
    """
    def _do_warm():
        try:
            embedder = _get_image_embedder()
            # Create a tiny 8x8 red image
            img = Image.new("RGB", (8, 8), color=(255, 0, 0))
            buf = BytesIO()
            img.save(buf, format="PNG")
            raw = buf.getvalue()
            # Call predict_labels to exercise both text+image encoders
            labels = embedder.predict_labels(raw, top_k=3)
            logging.info(f"image embedder warm-up complete labels={labels}")
        except Exception as e:
            logging.warning(f"image embedder warm-up failed: {e}")

    if sync:
        logging.info("starting synchronous image embedder warm-up")
        _do_warm()
    else:
        logging.info("starting background image embedder warm-up thread")
        t = Thread(target=_do_warm, daemon=True)
        t.start()


def _normal_chat_answer(chat_id: str, user_message: str) -> str:
    """Generate a normal chat response using Gemini with chat history context."""
    llm = _get_normal_chat_llm()
    if llm is None:
        return f"(Normal Chat) {user_message}"

    # Pull prior messages from History to maintain context
    try:
        history = HISTORY.existing_chat_access(chat_id)
        past_msgs = getattr(history, "messages", []) or []
    except Exception:
        past_msgs = []

    # Build message list for the LLM
    messages = [
        SystemMessage(content=(
            "You are a helpful, concise assistant. Be clear and factual. "
            "If the user refers to earlier context, use the prior messages."
        ))
    ]
    messages.extend(past_msgs)
    messages.append(HumanMessage(content=user_message))

    try:
        resp = llm.invoke(messages)
        return getattr(resp, "content", str(resp))
    except Exception as e:
        logging.exception("normal chat generation failed")
        return f"(Normal Chat) {user_message}"


def build_rag_pipeline_from_pdf(file_storage) -> RAG:
    """Create RAG pipeline from uploaded PDF using LangChain components."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, file_storage.filename)
        file_storage.save(path)
        
        # 1. Load PDF with metadata
        documents = load_pdf(path)
        logging.info(f"loaded pdf={file_storage.filename} pages={len(documents)}")
        # Clean text to improve chunking/embeddings
        documents = clean_documents(documents)
        total_chars = sum(len(d.page_content or "") for d in documents)
        logging.info(f"pdf text total_chars={total_chars}")
        
        # 2. Chunk documents with overlap
        chunks = chunk_documents(documents, chunk_size=1500, chunk_overlap=250)
        # Chunk stats
        clens = [len(c.page_content or "") for c in chunks]
        if clens:
            logging.info(
                f"chunked into {len(chunks)} chunks len_min={min(clens)} len_p50={sorted(clens)[len(clens)//2]} len_max={max(clens)}"
            )
        else:
            logging.warning("no chunks produced from PDF")
        
        # 3. Initialize embedding model (toggle via USE_GOOGLE_EMBEDDINGS)
        use_google = os.getenv("USE_GOOGLE_EMBEDDINGS", "0").strip().lower() in {"1", "true", "yes"}
        embed_model = EmbedData(use_fallback=not use_google)
        logging.info(f"embedding backend={'google' if use_google else 'huggingface'}")
        
        # 4. Create FAISS vector store
        vector_store = FAISSVectorStore(embed_model.embedding_model)
        vector_store.create_from_documents(chunks)
        logging.info(f"created FAISS index")
        
        # 5. Create retriever (use MMR to improve diversity/recall)
        retriever = Retriever(
            vector_store=vector_store,
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.5},
        )
        
        # 6. Build RAG chain
        rag = RAG(retriever)
        logging.info(f"RAG pipeline built for pdf={file_storage.filename}")
        
    return rag


# ---------------- Routes ---------------- #
@app.route("/")
def index():
    sid, state, active_id = _session_state()
    chats_meta = []
    for cid in state["chat_ids"]:
        summary = HISTORY.get_summary(cid) or "New Chat"
        chats_meta.append({"id": cid, "summary": summary, "active": (cid == active_id)})
    active_chat = {
        "id": active_id,
        "messages": _serialize_messages(active_id),
        "pdf_name": HISTORY.get_pdf(active_id),
        "images": HISTORY.get_images(active_id),
        "summary": HISTORY.get_summary(active_id) or "New Chat",
    }
    logging.debug(f"render index sid={sid} active_chat={active_id} chats={len(chats_meta)}")
    return render_template(
        "chat.html",
        chats=chats_meta,
        active_chat=active_chat,
    )


@app.route("/api/chat", methods=["POST"])
def api_chat():
    _, state, chat_id = _session_state()
    payload = request.get_json(force=True)
    message = (payload or {}).get("message", "").strip()
    if not message:
        return _json_error("Empty message")
    HISTORY.add_message(chat_id, "user", message)
    logging.info(f"user message chat_id={chat_id} len={len(message)}")
    rag: RAG | None = HISTORY.get_rag(chat_id)
    labels = HISTORY.get_image_labels(chat_id)
    web_docs = []
    try:
        if labels:
            # Use a focused query combining user message and top labels
            q = message
            if labels:
                q = f"{message} {' '.join(labels[:3])}"
            web_docs = search_to_documents(q, max_results=5)
    except Exception:
        web_docs = []

    if rag and HISTORY.get_pdf(chat_id):
        try:
            if web_docs:
                answer = rag.answer_augmented(message, web_docs)
            else:
                answer = rag.answer(message)
        except Exception as e:
            logging.exception(f"rag answer failed chat_id={chat_id}")
            answer = f"Error generating RAG answer: {e}"
    else:
        # No PDF attached → normal chat with optional web augmentation
        if web_docs:
            # Answer from web docs only using LLM with stuffed context
            llm = _get_normal_chat_llm()
            if llm is None:
                answer = _normal_chat_answer(chat_id, message)
            else:
                answer = _answer_with_docs_via_llm(llm, message, web_docs)
        else:
            answer = _normal_chat_answer(chat_id, message)
    HISTORY.add_message(chat_id, "assistant", answer)
    logging.info(f"assistant answer chat_id={chat_id} chars={len(answer)}")
    return jsonify({"answer": answer, "messages": _serialize_messages(chat_id)})


@app.route("/api/upload_pdf", methods=["POST"])
def api_upload_pdf():
    _, state, chat_id = _session_state()
    file = _validate_upload("file", [".pdf"])
    if not hasattr(file, "filename"):
        return file
    try:
        rag = build_rag_pipeline_from_pdf(file)
        HISTORY.set_pdf(chat_id, file.filename, rag)
        logging.info(f"pdf attached chat_id={chat_id} pdf={file.filename}")
        # Provide a friendly summary including chunk count if available from logs is not feasible here.
        return jsonify({
            "status": "ok",
            "pdf_name": file.filename,
            "message": f"Successfully processed {file.filename}. You can start asking questions."
        })
    except ValueError as e:
        # User-friendly errors (e.g., quota exceeded)
        error_msg = str(e)
        logging.error(f"pdf upload failed chat_id={chat_id}: {error_msg}")
        return jsonify({"error": error_msg}), 400
    except Exception as e:
        logging.exception("pdf processing failed")
        error_msg = "Failed to process PDF. Please try again or check the logs."
        return jsonify({"error": error_msg}), 500


@app.route("/api/upload_image", methods=["POST"])
def api_upload_image():
    _, state, chat_id = _session_state()
    img = _validate_upload("file", [".png", ".jpg", ".jpeg"])
    if not hasattr(img, "filename"):
        return img
    raw = img.read()
    b64 = base64.b64encode(raw).decode("utf-8")
    mime = "image/png" if img.filename.lower().endswith(".png") else "image/jpeg"
    HISTORY.add_image(chat_id, img.filename, f"data:{mime};base64,{b64}")
    # Predict labels and store them at session level
    try:
        embedder = _get_image_embedder()
        labels = embedder.predict_labels(raw, top_k=5)
        if labels:
            HISTORY.set_image_labels(chat_id, labels)
            logging.info(f"image added chat_id={chat_id} name={img.filename} labels={labels}")
        else:
            logging.info(f"image added chat_id={chat_id} name={img.filename} labels=none")
    except Exception as e:
        logging.warning(f"image labeling failed: {e}")
    return jsonify({
        "status": "ok",
        "images": HISTORY.get_images(chat_id),
        "image_labels": HISTORY.get_image_labels(chat_id),
        "message": f"Successfully uploaded {img.filename}"
    })


@app.route("/api/new_chat", methods=["POST"])
def api_new_chat():
    _, state, _ = _session_state()
    chat_id = HISTORY.new_chat()
    state["chat_ids"].append(chat_id)
    state["active_chat_id"] = chat_id
    logging.info(f"new chat created chat_id={chat_id}")
    return jsonify({"active_chat_id": chat_id})


@app.route("/api/delete_chat", methods=["POST"])
def api_delete_chat():
    _, state, active_id = _session_state()
    payload = request.get_json(force=True)
    cid = (payload or {}).get("chat_id")
    if not cid or cid not in state["chat_ids"]:
        return _json_error("Invalid chat id")
    try:
        HISTORY.delete_session(cid)
        logging.info(f"chat deleted chat_id={cid}")
    except KeyError:
        logging.warning(f"delete non-existent chat chat_id={cid}")
    state["chat_ids"] = [x for x in state["chat_ids"] if x != cid]
    if not state["chat_ids"]:
        new_id = HISTORY.new_chat()
        state["chat_ids"].append(new_id)
        state["active_chat_id"] = new_id
        logging.info(f"auto new chat after delete chat_id={new_id}")
    elif state["active_chat_id"] == cid:
        state["active_chat_id"] = state["chat_ids"][0]
    return jsonify({"active_chat_id": state["active_chat_id"]})


@app.route("/api/set_active_chat", methods=["POST"])
def api_set_active_chat():
    _, state, _ = _session_state()
    payload = request.get_json(force=True)
    cid = (payload or {}).get("chat_id")
    if not cid or cid not in state["chat_ids"]:
        return _json_error("Invalid chat id")
    state["active_chat_id"] = cid
    logging.debug(f"active chat switched chat_id={cid}")
    return jsonify({"active_chat_id": cid})


@app.route("/api/rename_chat", methods=["POST"])
def api_rename_chat():
    _, state, _ = _session_state()
    payload = request.get_json(force=True)
    cid = (payload or {}).get("chat_id")
    name = (payload or {}).get("name", "").strip()
    if not cid or cid not in state["chat_ids"] or not name:
        return _json_error("Invalid inputs")
    HISTORY.rename_session(cid, _summarize(name, 60))
    logging.info(f"chat renamed chat_id={cid} name={name}")
    return jsonify({"status": "ok"})


@app.route("/api/reset_active", methods=["POST"])
def api_reset_active():
    _, state, chat_id = _session_state()
    HISTORY.reset_session(chat_id)
    logging.info(f"chat reset chat_id={chat_id}")
    return jsonify({"status": "ok"})


@app.route("/api/remove_pdf", methods=["POST"])
def api_remove_pdf():
    _, state, chat_id = _session_state()
    HISTORY.clear_pdf(chat_id)
    logging.info(f"pdf removed chat_id={chat_id}")
    return jsonify({"status": "ok"})


@app.route("/api/remove_image", methods=["POST"])
def api_remove_image():
    _, state, chat_id = _session_state()
    payload = request.get_json(force=True)
    name = (payload or {}).get("name")
    if not name:
        return _json_error("Missing image name")
    removed = HISTORY.remove_image(chat_id, name)
    if not removed:
        return _json_error("Image not found", 404)
    logging.info(f"image removed chat_id={chat_id} name={name}")
    return jsonify({"status": "ok", "images": HISTORY.get_images(chat_id)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Optional warm-up on boot to pre-download image model weights
    warm = os.getenv("IMAGE_EMBED_WARMUP", "1").strip().lower() in {"1", "true", "yes"}
    warm_sync = os.getenv("IMAGE_EMBED_WARMUP_SYNC", "0").strip().lower() in {"1", "true", "yes"}
    # Avoid double warmup in Flask debug reloader; only warm in main process
    is_main_proc = os.getenv("WERKZEUG_RUN_MAIN") == "true" or not os.getenv("WERKZEUG_RUN_MAIN")
    if warm and is_main_proc:
        _warmup_image_embedder(sync=warm_sync)
    app.run(host="0.0.0.0", port=port, debug=True)
