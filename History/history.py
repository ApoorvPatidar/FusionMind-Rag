"""Session-based chat history & summarization utilities (in-memory only).

Features:
 - Creates UUID chat sessions and stores LangChain ChatMessageHistory objects.
 - First user message auto-summarized (LLM if available, else fallback heuristic).
 - Utility methods: list, delete, rename, add message (with auto summary).
 - Vector/embedding awareness for attached RAG (FAISS) pipelines: track backend, dim, index size.
 - No external persistence (MongoDB removed) for simpler setup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import uuid
import os
import datetime

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from Utils.logger import logging


def _build_summary_chain():
    if ChatGoogleGenerativeAI is None:
        return None
    # Will read key from env; if unavailable, invocation will fallback to heuristic
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent chat session naming assistant. Your task is to analyze the conversation and generate a concise, descriptive title that captures the main topic or purpose of the chat.

    Guidelines:
    - Create titles that are 2-6 words long
    - Focus on the primary topic, question, or task discussed
    - Use clear, specific language that users can easily understand
    - Avoid generic phrases like "Chat about" or "Discussion on"
    - For technical topics, include relevant keywords
    - For coding questions, mention the programming language or technology
    - For academic topics, include the subject area
    - Make titles that help users quickly identify and find this conversation later

    Examples:
    - "Python Data Structures Help" (for coding questions)
    - "Machine Learning Career Advice" (for career guidance) 
    - "React Component Debugging" (for technical troubleshooting)
    - "Spanish Grammar Practice" (for language learning)
    - "College Application Essay Tips" (for academic help)

    Return only the title, no additional text or formatting."""),
        ("human", "Conversation to summarize and name: {input}")
    ])

    return prompt | model | StrOutputParser()

SUMMARY_CHAIN = _build_summary_chain()

def _fallback_summary(text: str, max_len: int = 70) -> str:
    cleaned = " ".join(text.strip().split())
    return cleaned if len(cleaned) <= max_len else cleaned[: max_len - 3] + "..."


@dataclass
class SessionMeta:
    session_id: str
    summary: Optional[str] = None
    pdf_name: Optional[str] = None
    rag: Any | None = None
    images: List[Dict[str, Any]] = field(default_factory=list)
    # Vector/embedding introspection (best-effort)
    embedding_backend: Optional[str] = None  # e.g., 'google', 'huggingface'
    embedding_dim: Optional[int] = None
    index_size: Optional[int] = None
    # Image labels extracted from uploaded images (session-level aggregate)
    image_labels: List[str] = field(default_factory=list)


@dataclass
class SimpleMessage:
    """Lightweight message object mimicking LangChain .type/.content for UI serialization."""
    type: str  # 'human' or 'ai'
    content: str


class History:
    """Manage multiple chat sessions in-memory with auto summaries.

    RAG pipeline objects remain in-memory (not serialized) via an internal cache
    keyed by session_id. MongoDB support removed for simplicity.
    """

    def __init__(self):
        self._rag_cache: Dict[str, Any] = {}
        self._storage: Dict[str, ChatMessageHistory] = {}
        self._meta: Dict[str, SessionMeta] = {}
        # Summarizer (LLM or fallback)
        if SUMMARY_CHAIN is not None:
            self._summarizer = lambda msg: SUMMARY_CHAIN.invoke({"input": msg})  # type: ignore
        else:
            self._summarizer = _fallback_summary

    # ------------------ Helpers ------------------ #
    def _ensure_session(self, session_id: str) -> None:
        if session_id not in self._storage:
            raise KeyError("Session does not exist")
        
    # ------------------ Session Management ------------------ #
    def new_chat(self) -> str:
        session_id = str(uuid.uuid4())
        self._storage[session_id] = ChatMessageHistory()
        self._meta[session_id] = SessionMeta(session_id=session_id)
        return session_id

    def existing_chat_access(self, session_id: str):
        self._ensure_session(session_id)
        return self._storage[session_id]

    def delete_session(self, session_id: str) -> None:
        self._ensure_session(session_id)
        del self._storage[session_id]
        del self._meta[session_id]
        self._rag_cache.pop(session_id, None)

    def rename_session(self, session_id: str, new_summary: str) -> None:
        self._ensure_session(session_id)
        summary = new_summary.strip()
        self._meta[session_id].summary = summary

    # ------------------ Messages ------------------ #
    def add_message(self, session_id: str, role: str, content: str) -> None:
        self._ensure_session(session_id)
        if role not in {"user", "assistant", "ai"}:
            raise ValueError("role must be 'user' or 'assistant'")
        history = self._storage[session_id]
        if role == "user":
            history.add_user_message(content)
            if self._meta[session_id].summary is None:
                self._meta[session_id].summary = self._summarizer(content)
        else:  # assistant
            history.add_ai_message(content)

    def chat_summary(self, session_id: str, message: str) -> None:
        self._ensure_session(session_id)
        if self._meta[session_id].summary is None:
            self._meta[session_id].summary = self._summarizer(message)

    # Attachment & RAG helpers
    def set_pdf(self, session_id: str, pdf_name: str, rag: Any):
        self._ensure_session(session_id)
        self._rag_cache[session_id] = rag
        meta = self._meta[session_id]
        meta.pdf_name = pdf_name
        meta.rag = rag
        # Store vector stats (best-effort)
        info = self._introspect_vector_info(rag)
        if info:
            meta.embedding_backend = info.get("embedding_backend")
            meta.embedding_dim = info.get("embedding_dim")
            meta.index_size = info.get("index_size")

    def clear_pdf(self, session_id: str):
        self._ensure_session(session_id)
        self._rag_cache.pop(session_id, None)
        meta = self._meta[session_id]
        meta.pdf_name = None
        meta.rag = None
        meta.embedding_backend = None
        meta.embedding_dim = None
        meta.index_size = None

    def add_image(self, session_id: str, name: str, data_url: str):
        self._ensure_session(session_id)
        self._meta[session_id].images.append({"name": name, "data_url": data_url})

    def remove_image(self, session_id: str, name: str) -> bool:
        self._ensure_session(session_id)
        imgs = self._meta[session_id].images
        before = len(imgs)
        self._meta[session_id].images = [im for im in imgs if im.get("name") != name]
        return len(self._meta[session_id].images) != before

    def set_image_labels(self, session_id: str, labels: List[str]):
        """Store aggregated image labels for the session (deduplicated, limited)."""
        self._ensure_session(session_id)
        current = set(self._meta[session_id].image_labels)
        for lb in labels:
            if lb:
                current.add(lb)
        # Keep top-N unique labels (order not guaranteed)
        self._meta[session_id].image_labels = list(sorted(current))[:20]

    def get_image_labels(self, session_id: str) -> List[str]:
        self._ensure_session(session_id)
        return list(self._meta[session_id].image_labels)

    def reset_session(self, session_id: str):
        self._ensure_session(session_id)
        self._rag_cache.pop(session_id, None)
        self._storage[session_id] = ChatMessageHistory()
        meta = self._meta[session_id]
        meta.summary = "New Chat"
        meta.pdf_name = None
        meta.rag = None
        meta.images = []
        meta.embedding_backend = None
        meta.embedding_dim = None
        meta.index_size = None
        meta.image_labels = []

    # Accessors
    def get_pdf(self, session_id: str) -> Optional[str]:
        self._ensure_session(session_id)
        return self._meta[session_id].pdf_name

    def get_rag(self, session_id: str) -> Any:
        self._ensure_session(session_id)
        return self._meta[session_id].rag

    def get_vector_info(self, session_id: str) -> Dict[str, Any]:
        """Return vector/embedding details for the session (best-effort).

        Keys: embedding_backend, embedding_dim, index_size
        """
        self._ensure_session(session_id)
        meta = self._meta[session_id]
        return {
            "embedding_backend": meta.embedding_backend,
            "embedding_dim": meta.embedding_dim,
            "index_size": meta.index_size,
        }

    # ------------------ Internal: Vector/Embedding Introspection ------------------ #
    def _introspect_vector_info(self, rag: Any) -> Optional[Dict[str, Any]]:
        """Try to extract vector index size and embedding backend/dimension from RAG.

        Works with our RAG + Retriever + FAISSVectorStore stack. Returns None if
        introspection fails anywhere; avoids throwing.
        """
        try:
            # Our RAG stores the original Retriever wrapper at rag.retriever
            retr = getattr(rag, "retriever", None)
            if retr is None:
                return None
            # Our Retriever holds the FAISSVectorStore wrapper as .vector_store
            vs_wrapper = getattr(retr, "vector_store", None)
            if vs_wrapper is None:
                return None
            lc_faiss = getattr(vs_wrapper, "vector_store", None)
            emb_model = getattr(vs_wrapper, "embedding_model", None)

            index_size = None
            emb_dim = None
            backend = None
            # Try FAISS index stats
            if lc_faiss is not None:
                idx = getattr(lc_faiss, "index", None)
                if idx is not None:
                    index_size = getattr(idx, "ntotal", None)
                    emb_dim = getattr(idx, "d", None)

            # Detect embedding backend type
            if emb_model is not None:
                cname = emb_model.__class__.__name__.lower()
                # SmartEmbeddings wrapper: prefer .primary if present
                primary = getattr(emb_model, "primary", None)
                if primary is not None:
                    cname = primary.__class__.__name__.lower()
                if "google" in cname or "generativeai" in cname:
                    backend = "google"
                elif "huggingface" in cname or "sentence" in cname:
                    backend = "huggingface"

            return {
                "embedding_backend": backend,
                "embedding_dim": emb_dim,
                "index_size": index_size,
            }
        except Exception:
            return None

    def get_images(self, session_id: str) -> List[Dict[str, Any]]:
        self._ensure_session(session_id)
        return list(self._meta[session_id].images)

    # ------------------ Introspection ------------------ #
    def list_sessions(self) -> List[Tuple[str, Optional[str]]]:
        return [(sid, meta.summary) for sid, meta in self._meta.items()]
    

    def get_summary(self, session_id: str) -> Optional[str]:
        """Return the summary (first user message synopsis) for a session."""
        self._ensure_session(session_id)
        return self._meta[session_id].summary

    def has_summary(self, session_id: str) -> bool:
        """Whether the session already has a summary."""
        return self.get_summary(session_id) is not None

    def session_exists(self, session_id: str) -> bool:
        return session_id in self._storage

    def get_session_count(self) -> int:
        return len(self._storage)

    def get_messages(self, session_id: str) -> Any:
        self._ensure_session(session_id)
        history = self._storage[session_id]
        return getattr(history, "messages", [])

__all__ = ["History", "SessionMeta", "SimpleMessage"]