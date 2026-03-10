"""RAG (Retrieval-Augmented Generation) using LangChain chains."""

from langchain_classic.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from typing import List, Optional
from Utils.logger import logging
from Utils.telemetry import TELEMETRY, RetrievalRecord
import os
import re


class RAG:
    """RAG pipeline using LangChain's RetrievalQA chain."""
    
    def __init__(self, retriever, llm=None, prompt_template: Optional[str] = None):
        """Initialize RAG chain.
        
        Args:
            retriever: LangChain retriever or Retriever wrapper.
            llm: LangChain LLM instance (defaults to Gemini).
            prompt_template: Optional custom prompt template.
        """
        self.retriever = retriever
        
        # Initialize LLM if not provided
        if llm is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                api_key = api_key.strip().strip('"').strip("'")
            if not api_key:
                logging.warning("GOOGLE_API_KEY not found")
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                google_api_key=api_key,
                temperature=0.3
            )
            logging.info("initialized Gemini LLM for RAG")
        else:
            self.llm = llm
        
        # Setup prompt template
        if prompt_template:
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
        else:
            # Default prompt (allows general knowledge fallback, careful with citations)
            prompt = PromptTemplate(
                template=(
                    "Answer the question concisely. If the provided context contains useful information, incorporate it.\n"
                    "If the context is irrelevant or insufficient, answer from your general knowledge.\n"
                    "Do not fabricate citations. Only cite sources if you actually used the context.\n\n"
                    "Context:\n{context}\n\n"
                    "Question: {question}\n\n"
                    "Answer:"
                ),
                input_variables=["context", "question"]
            )
        # Save prompt for reuse in augmented answering paths
        self.prompt = prompt
        
        # Get LangChain retriever interface
        if hasattr(retriever, 'retriever'):
            # Wrapped retriever
            lc_retriever = retriever.retriever
        elif hasattr(retriever, 'get_relevant_documents'):
            # Already a LangChain retriever
            lc_retriever = retriever
        else:
            raise ValueError("Invalid retriever type")
        
        # Create RetrievalQA chain
        try:
            self.chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=lc_retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.prompt}
            )
            logging.info("initialized RAG chain")
        except Exception as e:
            logging.exception("RAG chain initialization failed")
            raise
    
    def answer(self, question: str) -> str:
        """Generate answer for question using retrieval and LLM.
        
        Args:
            question: User question.
            
        Returns:
            Generated answer string.
        """
        try:
            result = self.chain({"query": question})
            answer = result.get("result", "No answer generated")
            source_docs = result.get("source_documents", [])
            
            logging.info(f"generated answer question_len={len(question)} answer_len={len(answer)} sources={len(source_docs)}")
            
            # Optional: append compact sources with clearer snippets
            if source_docs:
                def best_sentence(text: str, query: str) -> str:
                    # Simple heuristic: choose sentence with most query keyword overlap
                    sents = re.split(r"(?<=[.!?])\s+|\n+", (text or "").strip())
                    qwords = {w for w in re.findall(r"[a-zA-Z]{4,}", query.lower())}
                    if not sents:
                        return (text or "")[:160]
                    best = sents[0]
                    best_score = -1
                    for s in sents:
                        words = set(re.findall(r"[a-zA-Z]{4,}", s.lower()))
                        score = len(words & qwords)
                        if score > best_score:
                            best_score = score
                            best = s
                    return best[:240]

                sources_text = "\n\n---\nSources:\n"
                for i, doc in enumerate(source_docs[:3], 1):  # Show top 3
                    page = doc.metadata.get("page", "?")
                    src = os.path.basename(str(doc.metadata.get("source", "")))
                    snippet = best_sentence(doc.page_content, question)
                    prefix = f"{src}, page {page}" if src else f"Page {page}"
                    sources_text += f"{i}. {prefix}: {snippet}\n"
                answer += sources_text
            
            # Fire-and-forget telemetry: never let logging errors surface to the caller.
            try:
                used_mmr = getattr(
                    getattr(self.retriever, "search_type", None), "__str__", lambda: ""
                )() == "mmr" or getattr(self.retriever, "search_type", "") == "mmr"
                TELEMETRY.log(RetrievalRecord(
                    query=question,
                    retrieved_chunk_sources=[
                        os.path.basename(str(d.metadata.get("source", ""))) for d in source_docs
                    ],
                    retrieved_chunk_lengths=[len(d.page_content or "") for d in source_docs],
                    avg_chunk_length=(
                        sum(len(d.page_content or "") for d in source_docs) / len(source_docs)
                        if source_docs else 0.0
                    ),
                    answer_length=len(answer),
                    used_web_augmentation=False,
                    used_mmr=used_mmr,
                ))
            except Exception as _tel_err:
                logging.warning(f"telemetry log failed in answer() (non-fatal): {_tel_err}")

            return answer
        except Exception as e:
            logging.exception("RAG answer generation failed")
            return f"Error generating answer: {str(e)}"

    # ---------------- Augmented Answering with Extra Docs ---------------- #
    def _format_context(self, docs: List[Document]) -> str:
        parts: List[str] = []
        for d in docs:
            parts.append(d.page_content or "")
        return "\n\n".join(parts)

    def _invoke_llm_with_context(self, question: str, docs: List[Document]) -> str:
        """Call LLM directly by stuffing combined docs into the prompt."""
        ctx = self._format_context(docs)
        try:
            prompt_str = self.prompt.format(context=ctx, question=question)  # type: ignore[attr-defined]
        except Exception:
            # Fallback if LangChain Prompt not compatible with str.format
            prompt_str = f"Context:\n{ctx}\n\nQuestion: {question}\n\nAnswer:"
        resp = self.llm.invoke(prompt_str)
        return getattr(resp, "content", str(resp))

    def answer_augmented(self, question: str, extra_docs: List[Document], cite_limit: int = 3) -> str:
        """Combine retriever docs with extra docs (e.g., web search) and answer.

        Returns an answer string with compact sources appended (mix of PDF and web).
        """
        try:
            pdf_docs: List[Document] = []
            try:
                pdf_docs = self.retriever.retriever.invoke(question)  # type: ignore
            except Exception:
                pdf_docs = []

            # DESIGN NOTE: URL-domain deduplication before merging.
            # When a PDF is attached the user may have uploaded a paper or report
            # that is also indexed on the web.  Without deduplication,
            # answer_augmented() can fill the context window with near-duplicate
            # content from the PDF chunks AND a web snippet of the same document,
            # wasting tokens and confusing the LLM's citation logic.
            # Strategy: collect base-names of PDF source paths, then drop any
            # web doc whose URL or title contains that base-name as a substring.
            pdf_source_hints: set = set()
            for d in pdf_docs:
                src = str(d.metadata.get("source", ""))
                if src:
                    pdf_source_hints.add(os.path.basename(src).lower().replace(".pdf", ""))

            def _is_duplicate_web_doc(doc: Document) -> bool:
                if not pdf_source_hints:
                    return False
                url = str(doc.metadata.get("url", "") or doc.metadata.get("source", "")).lower()
                title = str(doc.metadata.get("title", "")).lower()
                return any(hint and (hint in url or hint in title) for hint in pdf_source_hints)

            deduped_extra = [d for d in (extra_docs or []) if not _is_duplicate_web_doc(d)]
            removed = len(extra_docs or []) - len(deduped_extra)
            if removed:
                logging.info(f"deduped {removed} web doc(s) overlapping with PDF source")

            # Merge and truncate to a manageable number for context stuffing
            all_docs = (pdf_docs[:5] if pdf_docs else []) + (deduped_extra[:5] if deduped_extra else [])
            if not all_docs:
                # No docs available; just defer to standard answer
                return self.answer(question)

            answer = self._invoke_llm_with_context(question, all_docs)

            # Build sources list from both sets
            def mk_source(doc: Document) -> str:
                src = doc.metadata.get("source") or doc.metadata.get("url") or ""
                src = str(src)
                base = os.path.basename(src) if src else ""
                title = doc.metadata.get("title")
                page = doc.metadata.get("page")
                if title and src:
                    label = f"{title} ({base or src})"
                elif src:
                    label = base or src
                else:
                    label = title or ""
                if page is not None:
                    label += f", page {page}"
                return label or "(context)"

            sources = []
            for d in (pdf_docs[:cite_limit] if pdf_docs else []):
                sources.append(mk_source(d))
            for d in (extra_docs[:cite_limit] if extra_docs else []):
                sources.append(mk_source(d))
            if sources:
                answer += "\n\n---\nSources:\n" + "\n".join(f"- {s}" for s in sources[:cite_limit * 2])

            # Fire-and-forget telemetry for augmented answers.
            try:
                all_retrieved = pdf_docs + deduped_extra
                used_mmr = getattr(self.retriever, "search_type", "") == "mmr"
                TELEMETRY.log(RetrievalRecord(
                    query=question,
                    retrieved_chunk_sources=[
                        os.path.basename(str(d.metadata.get("source", "") or d.metadata.get("url", ""))) 
                        for d in all_retrieved
                    ],
                    retrieved_chunk_lengths=[len(d.page_content or "") for d in all_retrieved],
                    avg_chunk_length=(
                        sum(len(d.page_content or "") for d in all_retrieved) / len(all_retrieved)
                        if all_retrieved else 0.0
                    ),
                    answer_length=len(answer),
                    used_web_augmentation=bool(deduped_extra),
                    used_mmr=used_mmr,
                ))
            except Exception as _tel_err:
                logging.warning(f"telemetry log failed in answer_augmented() (non-fatal): {_tel_err}")

            return answer
        except Exception as e:
            logging.exception("augmented answer failed")
            return self.answer(question)

    def direct_answer_from_docs(self, question: str, docs: List[Document]) -> str:
        """Answer using only provided docs (no retriever)."""
        try:
            if not docs:
                return self.answer(question)
            answer = self._invoke_llm_with_context(question, docs[:8])
            # Minimal sources
            srcs = []
            for d in docs[:3]:
                src = d.metadata.get("source") or d.metadata.get("url") or ""
                title = d.metadata.get("title") or ""
                page = d.metadata.get("page")
                label = title or os.path.basename(str(src)) or str(src)
                if page is not None:
                    label += f", page {page}"
                srcs.append(label or "(context)")
            if srcs:
                answer += "\n\n---\nSources:\n" + "\n".join(f"- {s}" for s in srcs)
            return answer
        except Exception:
            return self.answer(question)

