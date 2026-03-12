"""Embedding generation with Google AI and HuggingFace fallback."""

import os
# Ensure Transformers does not try to import TensorFlow/Keras when using sentence-transformers
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from typing import List, Optional
from Utils.logger import logging


class SmartEmbeddings(Embeddings):
    """Embeddings wrapper that tries a primary backend and falls back on error.

    This ensures FAISS.from_documents can benefit from fallback automatically.
    """

    def __init__(self, primary: Embeddings, fallback: Optional[Embeddings] = None):
        self.primary = primary
        self.fallback = fallback

    def _should_fallback(self, err: Exception) -> bool:
        msg = str(err).lower()
        return any(x in msg for x in ["quota", "429", "exceeded your current quota"])

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            return self.primary.embed_documents(texts)
        except Exception as e:
            if self.fallback and self._should_fallback(e):
                logging.warning("primary embeddings failed; using fallback for documents")
                return self.fallback.embed_documents(texts)
            logging.exception("embed_documents failed")
            raise

    def embed_query(self, text: str) -> List[float]:
        try:
            return self.primary.embed_query(text)
        except Exception as e:
            if self.fallback and self._should_fallback(e):
                logging.warning("primary embeddings failed; using fallback for query")
                return self.fallback.embed_query(text)
            logging.exception("embed_query failed")
            raise


class EmbedData:
    """Wrapper for embeddings with Google AI and HuggingFace fallback."""
    
    def __init__(self, model_name: str = None, batch_size: int = 8, use_fallback: bool = False):
        """Initialize embedding model with automatic fallback.
        
        Args:
            model_name: Google embedding model name or HuggingFace model.
            batch_size: Batch size for embedding generation.
            use_fallback: Force use of HuggingFace embeddings.
        """
        self.batch_size = batch_size
        # Prefer a working Google embedding model name by default
        self.model_name = model_name or os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")
        self.embeddings: List[List[float]] = []
        self.texts: List[str] = []
        self.use_google = not use_fallback
        
        # Try Google first, fallback to HuggingFace if quota exceeded or no API key
        if not use_fallback:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                api_key = api_key.strip().strip('"').strip("'")
            if api_key:
                try:
                    primary = GoogleGenerativeAIEmbeddings(
                        model=self.model_name,
                        google_api_key=api_key
                    )
                    fallback = None
                    try:
                        from langchain_huggingface import HuggingFaceEmbeddings
                        fallback = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-MiniLM-L6-v2",
                            model_kwargs={'device': 'cpu'},
                            encode_kwargs={'normalize_embeddings': True}
                        )
                    except Exception as hf_err:
                        logging.warning(f"HuggingFace fallback unavailable: {hf_err}")
                    self.embedding_model = SmartEmbeddings(primary=primary, fallback=fallback)
                    logging.info(f"initialized Google embeddings model={self.model_name}")
                    return
                except Exception as e:
                    logging.warning(f"Google embeddings init failed: {e}, falling back to HuggingFace")
                    self.use_google = False
            else:
                logging.warning("GOOGLE_API_KEY not found, using HuggingFace fallback")
                self.use_google = False
        
        # Fallback to HuggingFace
        from langchain_huggingface import HuggingFaceEmbeddings
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logging.info("initialized HuggingFace embeddings model=all-MiniLM-L6-v2")

    def embed(self, texts: List[str]):
        """Generate embeddings for list of texts with automatic fallback.
        
        Args:
            texts: List of text strings to embed.
        """
        self.texts = texts
        try:
            # Batch embedding for efficiency
            self.embeddings = self.embedding_model.embed_documents(texts)
            logging.info(f"generated embeddings count={len(self.embeddings)} dim={len(self.embeddings[0]) if self.embeddings else 0}")
        except Exception as e:
            # If Google fails due to quota, try HuggingFace fallback
            if self.use_google and "quota" in str(e).lower():
                logging.warning(f"Google quota exceeded, switching to HuggingFace fallback")
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                self.use_google = False
                # Retry with fallback
                try:
                    self.embeddings = self.embedding_model.embed_documents(texts)
                    logging.info(f"generated embeddings with fallback count={len(self.embeddings)} dim={len(self.embeddings[0]) if self.embeddings else 0}")
                except Exception as fallback_error:
                    logging.exception("fallback embedding also failed")
                    raise
            else:
                logging.exception("embedding generation failed")
                raise
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query with automatic fallback.
        
        Args:
            query: Query text to embed.
            
        Returns:
            Embedding vector.
        """
        try:
            embedding = self.embedding_model.embed_query(query)
            return embedding
        except Exception as e:
            # If Google fails due to quota, try HuggingFace fallback
            if self.use_google and "quota" in str(e).lower():
                logging.warning(f"Google quota exceeded for query, switching to HuggingFace fallback")
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                self.use_google = False
                # Retry with fallback
                try:
                    embedding = self.embedding_model.embed_query(query)
                    return embedding
                except Exception as fallback_error:
                    logging.exception(f"fallback query embedding failed query={query[:50]}")
                    raise
            else:
                logging.exception(f"query embedding failed query={query[:50]}")
                raise


def save_embeddings(embeddata: 'EmbedData', filename: str):
    """Legacy stub for saving embeddings (deprecated with vector store usage)."""
    pass
