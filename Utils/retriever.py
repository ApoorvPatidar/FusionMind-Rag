"""Retriever using LangChain's retriever interface."""

from langchain_core.documents import Document
from typing import List
from Utils.logger import logging


class Retriever:
    """Wrapper for LangChain retriever with legacy compatibility."""
    
    def __init__(self, vector_store=None, embeddata=None, langchain_retriever=None, *, search_type: str = "similarity", search_kwargs: dict | None = None):
        """Initialize retriever.

        Args:
            vector_store: FAISSVectorStore or legacy store.
            embeddata: Legacy EmbedData (deprecated).
            langchain_retriever: Direct LangChain retriever instance.
            search_type: Retrieval mode — 'similarity' for pure cosine/dot-product
                nearest-neighbour, or 'mmr' for Maximal Marginal Relevance.
            search_kwargs: Extra kwargs forwarded to the LangChain retriever.
                For MMR the meaningful keys are:
                  k          – number of docs to return to the caller.
                  fetch_k    – candidate pool size fetched from the index before
                               MMR re-ranking (must be >= k; ideally 3-4x k).
                  lambda_mult – controls the relevance/diversity trade-off.

        MMR scoring (what interviewers ask about):
            For each candidate document d not yet selected, FAISS scores it as:

                score(d) = lambda_mult * sim(d, query)
                         - (1 - lambda_mult) * max_{s in Selected} sim(d, s)

            The first term rewards relevance to the query; the second penalises
            redundancy with already-chosen documents.  At lambda_mult=1.0 MMR
            degenerates to pure similarity search.  At lambda_mult=0.0 it returns
            maximally diverse documents regardless of query relevance.

            Practical guidance:
              0.5  – equal weight; good when the corpus is large and repetitive.
              0.7  – 70 % relevance / 30 % diversity; recommended for focused
                     document QA where factual accuracy matters more than breadth.
              0.9  – near-similarity mode; useful for narrow factual retrieval.
        """
        self.vector_store = vector_store
        self.embeddata = embeddata
        self.search_type = search_type
        self.search_kwargs = search_kwargs or {"k": 4}
        
        # Prefer native LangChain retriever if provided
        if langchain_retriever:
            self.retriever = langchain_retriever
        elif hasattr(vector_store, 'as_retriever'):
            try:
                # Newer LangChain FAISS retriever supports search_type
                self.retriever = vector_store.as_retriever(search_kwargs=self.search_kwargs)
                # Try to set search_type if supported
                if hasattr(self.retriever, 'search_type'):
                    self.retriever.search_type = self.search_type
            except TypeError:
                # Fallback if search_type not supported in this version
                self.retriever = vector_store.as_retriever(search_kwargs=self.search_kwargs)
        else:
            self.retriever = None
            logging.warning("no LangChain retriever available, using legacy mode")
    
    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve relevant documents for query.
        
        Args:
            query: Query string.
            k: Number of documents to retrieve.
            
        Returns:
            List of Document objects.
        """
        if self.retriever:
            # Use LangChain retriever
            # DESIGN NOTE: invoke() is the stable API from LangChain ≥0.2.
            # get_relevant_documents() was deprecated in 0.1.46 and removed in
            # 0.2.0 — calling it on newer installs raises AttributeError.
            try:
                results = self.retriever.invoke(query)
                logging.debug(f"retrieved docs={len(results)} query_len={len(query)}")
                return results
            except Exception as e:
                logging.exception("retrieval failed")
                raise
        else:
            # Legacy fallback
            if hasattr(self.vector_store, 'similarity_search'):
                return self.vector_store.similarity_search(query, k=k)
            elif hasattr(self.vector_store, 'search') and self.embeddata:
                # Old Qdrant stub
                query_embedding = self.embeddata.embed_query(query) if hasattr(self.embeddata, 'embed_query') else []
                results = self.vector_store.search(query_embedding, top_k=k)
                # Convert tuples to Documents
                return [Document(page_content=text, metadata={}) for text, _ in results]
            else:
                raise ValueError("No valid retrieval method available")

