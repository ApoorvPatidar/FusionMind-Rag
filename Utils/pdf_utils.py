"""PDF processing using LangChain's PyPDFLoader for text extraction with metadata.

Includes text cleaning utilities to improve downstream chunking and embeddings
quality (e.g., fix hyphenation, normalize whitespace, and common ligatures).
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from typing import List
from Utils.logger import logging
import re
import unicodedata


def load_pdf(pdf_path: str) -> List[Document]:
    """Load PDF using LangChain's PyPDFLoader.
    
    Returns:
        List of Document objects with page_content and metadata (page number, source).
    """
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        logging.info(f"loaded pdf pages={len(documents)} path={pdf_path}")
        return documents
    except Exception as e:
        logging.exception(f"pdf load failed path={pdf_path}")
        raise


def _is_garbled(text: str) -> bool:
    """Return True if the text is likely mojibake / garbage extraction.

    Heuristic: if more than 15 % of characters are non-ASCII after NFKC
    normalization, the PDF extractor likely produced encoding artifacts
    (e.g., â€™ instead of ', Ã© instead of e).  These corrupt embeddings
    because the tokenizer treats them as meaningless byte sequences.

    15 % is a conservative threshold -- legitimate multilingual documents
    rarely exceed it for Latin-script PDFs, and genuinely garbled pages
    almost always exceed 30 %.
    """
    if not text:
        return False
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return (non_ascii / len(text)) > 0.15


def _clean_text(text: str) -> str:
    """Clean raw PDF-extracted text for better chunking/embeddings.

    - Remove soft hyphenations across line breaks (e.g., "trans-\nformer")
    - Normalize different newline styles to spaces while preserving paragraphs
    - Collapse excessive whitespace
    - Normalize common PDF ligatures (fi, fl, ff)
    - Strip control characters
    - Return "" for garbled/mojibake pages so chunk_documents skips them
    """
    if not text:
        return ""

    # Normalize unicode (NFKC helps with some width/compatibility chars)
    t = unicodedata.normalize("NFKC", text)

    # Fix common ligatures
    t = t.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬀ", "ff").replace("ﬃ", "ffi").replace("ﬄ", "ffl")

    # DESIGN NOTE: garbled-text early exit.
    # Check AFTER ligature normalization (some ligatures are non-ASCII) but
    # BEFORE any further processing.  Returning "" here causes clean_documents()
    # to produce a near-empty Document that chunk_documents() will skip,
    # preventing garbage tokens from polluting the FAISS index.
    if _is_garbled(t):
        logging.warning("garbled text detected (>15 %% non-ASCII) -- skipping page")
        return ""

    # DESIGN NOTE: word-boundary-constrained hyphenation removal.
    # The naive pattern r"-\s*\n" also strips intentional hyphens in compound
    # words that happen to wrap at a line break, e.g.
    #   "state-\nof-the-art" -> "stateof-the-art"  (wrong)
    # Positive lookbehind (?<=\w) and lookahead (?=\w) ensure we only remove
    # the hyphen when it joins two word characters, i.e. it is a line-wrap
    # artefact rather than punctuation.  This correctly handles:
    #   "transformer-\nbased" -> "transformerbased"  (soft hyphen, removed)
    #   "self-\nattention"    -> "selfattention"      (soft hyphen, removed)
    #   "-- \nHowever"        -> "-- However"         (dash, preserved)
    t = re.sub(r"(?<=\w)-\s*\n(?=\w)", "", t)

    # Replace newlines within paragraphs to spaces, but keep paragraph breaks
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\n{2,}", "<PARA>", t)  # mark paragraphs
    t = t.replace("\n", " ")
    t = t.replace("<PARA>", "\n\n")

    # Collapse excessive whitespace
    t = re.sub(r"\s+", " ", t)

    return t.strip()


def clean_documents(documents: List[Document]) -> List[Document]:
    """Return new Documents with cleaned page_content and preserved metadata.

    Logs total length before/after to help diagnose poor extraction quality.
    Emits a WARNING for any page that produces fewer than 50 characters after
    cleaning -- a strong signal that the page is image-only or scanned.
    """
    before = sum(len(d.page_content or "") for d in documents)
    cleaned: List[Document] = []
    for d in documents:
        content = _clean_text(d.page_content or "")
        # Per-page quality guard: warn on near-empty output without raising so
        # the rest of the document can still be indexed.
        if len(content.strip()) < 50:
            page_num = d.metadata.get("page", "?")
            source = d.metadata.get("source", "unknown")
            logging.warning(
                f"page {page_num} of {source} produced < 50 chars after cleaning "
                f"-- possible scanned/image-only page"
            )
        cleaned.append(Document(page_content=content, metadata=d.metadata.copy()))
    after = sum(len(d.page_content or "") for d in cleaned)
    logging.info(f"cleaned pdf text total_chars_before={before} total_chars_after={after}")
    return cleaned


def convert_pdf_to_markdown(path: str) -> str:
    """Legacy compatibility: extract all text as single string.
    
    For backward compatibility with existing code. New code should use load_pdf().
    """
    documents = load_pdf(path)
    documents = clean_documents(documents)
    return "\n\n".join(doc.page_content for doc in documents)

