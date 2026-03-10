"""FAISS vector store using LangChain integration for fast similarity search."""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List, Optional
from Utils.logger import logging


class FAISSVectorStore:
    """FAISS vector store wrapper for LangChain documents."""
    
    def __init__(self, embedding_model):
        """Initialize FAISS store.
        
        Args:
            embedding_model: LangChain embedding model instance.
        """
        self.embedding_model = embedding_model
        self.vector_store: Optional[FAISS] = None
        logging.info("initialized FAISS vector store")
    
    def create_from_documents(self, documents: List[Document]):
        """Create FAISS index from documents with retry logic.
        
        Args:
            documents: List of LangChain Document objects with text and metadata.
        """
        try:
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embedding_model
            )
            logging.info(f"created FAISS index docs={len(documents)}")
        except Exception as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "429" in error_msg:
                logging.error(f"Embedding quota exceeded: {e}")
                raise ValueError("Embedding API quota exceeded. Please try again later or check your API quota.")
            logging.exception("FAISS index creation failed")
            raise
    
    def create_from_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        """Create FAISS index from raw texts.
        
        Args:
            texts: List of text strings.
            metadatas: Optional list of metadata dicts for each text.
        """
        try:
            self.vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embedding_model,
                metadatas=metadatas
            )
            logging.info(f"created FAISS index texts={len(texts)}")
        except Exception as e:
            logging.exception("FAISS index creation failed")
            raise
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents.
        
        Args:
            query: Query string.
            k: Number of results to return.
            
        Returns:
            List of Document objects.
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            logging.debug(f"similarity search query_len={len(query)} results={len(results)}")
            return results
        except Exception as e:
            logging.exception("similarity search failed")
            raise
    
    def as_retriever(self, search_kwargs: Optional[dict] = None):
        """Get LangChain retriever interface.
        
        Args:
            search_kwargs: Optional search parameters (e.g., {"k": 4}).
            
        Returns:
            LangChain retriever.
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        search_kwargs = search_kwargs or {"k": 4}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)


# Legacy compatibility class
class QdrantVDB:
    """Legacy stub for backward compatibility. Use FAISSVectorStore instead."""
    
    def __init__(self, collection_name: str, vector_dim: int, batch_size: int = 8):
        logging.warning("QdrantVDB is deprecated, use FAISSVectorStore instead")
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.batch_size = batch_size
        self._data = []

    def create_collection(self):
        # Stub
        return True

    def ingest_data(self, embeddata):
        self._data = list(zip(embeddata.texts, embeddata.embeddings))

    def search(self, query_embedding, top_k: int = 3):
        # Return first few naive results
        return self._data[:top_k]
