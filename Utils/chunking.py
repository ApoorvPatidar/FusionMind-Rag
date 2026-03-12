"""Text chunking using LangChain's RecursiveCharacterTextSplitter."""

from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Tuple
from Utils.logger import logging


# DESIGN NOTE: Adaptive chunk sizing matters because retrieval quality is
# sensitive to the ratio of chunk size to document length.
# On a tiny doc (< 5 k chars) a 1 500-char chunk leaves only 3 chunks total,
# so MMR has nothing to diversify over and precision collapses.  On a
# book-length doc (> 200 k chars) small chunks produce thousands of index
# entries, slow retrieval, and force the LLM to stitch micro-fragments
# together.  Matching chunk size to document scale keeps the index at a
# practical size (20–200 chunks) across the full range of document lengths.
def select_chunk_size(total_chars: int) -> Tuple[int, int]:
    """Return (chunk_size, chunk_overlap) tuned to document length.

    Thresholds
    ----------
    < 5 000 chars   — tiny doc (1–3 pages): chunk_size=500,  overlap=50
        Small chunks maximise precision; with so few tokens the model can
        read the whole document in one pass anyway.
    5 000–50 000    — normal doc (3–30 pages): chunk_size=1000, overlap=150
        Balanced default.  Produces 5–50 retrievable chunks with enough
        context per chunk for faithful answers.
    50 000–200 000  — long doc (30–120 pages): chunk_size=1500, overlap=250
        Larger chunks reduce index size while keeping ~16 % overlap to
        preserve cross-boundary context.
    > 200 000 chars — book-length (> 120 pages): chunk_size=2000, overlap=300
        Reduces FAISS index to a manageable size; prevents the retriever
        from drowning in thousands of micro-fragments.

    The overlap ratio (~15–16 %) is the LangChain community standard: enough
    to capture a sentence that straddles a boundary without doubling index size.
    """
    if total_chars < 5_000:
        return 500, 50
    elif total_chars < 50_000:
        return 1_000, 150
    elif total_chars < 200_000:
        return 1_500, 250
    else:
        return 2_000, 300


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """Split documents into overlapping chunks using RecursiveCharacterTextSplitter.

    Splitter choice — RecursiveCharacterTextSplitter vs TokenTextSplitter
    -----------------------------------------------------------------------
    RecursiveCharacterTextSplitter is character-based: it measures size with
    ``len()`` and requires no tokenizer.  This makes it deterministic,
    fast, and dependency-free.  TokenTextSplitter counts BPE tokens, which
    gives a more accurate picture of LLM context-window consumption, but
    requires loading a tiktoken / HuggingFace tokenizer at import time,
    adding hundreds of milliseconds of cold-start latency.  For document QA
    the character approximation is close enough: at ~4 chars/token a
    1 500-char chunk ≈ 375 tokens, well within any modern LLM window.

    Separator order
    ---------------
    ``["\\n\\n", "\\n", " ", ""]`` is tried left-to-right as a cascade:
    1. ``\\n\\n`` — paragraph boundary; preserves the widest semantic unit.
    2. ``\\n``     — line break; used when a paragraph exceeds chunk_size.
    3. ``" "``     — word boundary; last resort before hard character splits.
    4. ``""``      — hard split; guarantees the size limit is honoured even
                     for unbroken strings (e.g. URLs, code tokens).
    Trying coarser splits first means chunks stay semantically coherent;
    the hard split is a safety escape hatch, not the common path.

    Chunk size / overlap tradeoff
    -----------------------------
    Larger chunks deliver more context per retrieval hit, which helps the
    LLM answer multi-sentence questions.  But they also reduce the total
    number of retrievable units, so a query may pull in an entire section
    when only one paragraph is relevant (low precision).  Smaller chunks
    improve precision but may omit the surrounding context needed to
    interpret a key sentence.

    Overlap at ~16 % of chunk_size (e.g. 250 / 1 500) is the standard
    community recommendation: it ensures that a sentence split across a
    boundary appears in full in at least one chunk, without doubling the
    index size the way a 50 % overlap would.

    Args:
        documents: List of LangChain Document objects.
        chunk_size: Target size for each chunk in characters.
        chunk_overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        List of Document objects with chunked content and preserved metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    logging.info(f"split documents={len(documents)} into chunks={len(chunks)} size={chunk_size} overlap={chunk_overlap}")
    return chunks


def chunk_markdown(markdown_text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Legacy compatibility: chunk plain text string.
    
    For backward compatibility. New code should use chunk_documents().
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(markdown_text)
    logging.info(f"chunked text into chunks={len(chunks)} size={chunk_size} overlap={chunk_overlap}")
    return chunks

