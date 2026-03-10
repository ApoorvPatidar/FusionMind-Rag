"""Web search utilities using DuckDuckGo to retrieve context as LangChain Documents."""
from __future__ import annotations

from typing import List
from urllib.parse import urlparse

from duckduckgo_search import DDGS
from langchain_core.documents import Document
from Utils.logger import logging


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc
    except Exception:
        return ""


# DESIGN NOTE: COCO-80 is an object-detection vocabulary. Its labels describe
# *what is visually present in the frame* (people, furniture, everyday items),
# not the *semantic topic* the user is asking about. Appending raw COCO labels
# to a search query almost always makes the query worse:
#   - Scene-descriptors like "person", "chair", "bench", "couch" tell a search
#     engine nothing about the domain of the question.
#   - Only a small subset of COCO labels are specific enough to carry topical
#     signal (e.g., "laptop" -> technology, "book" -> literature/education,
#     "cell phone" -> mobile/telecom).
# The filter below keeps only those specific labels and discards the noisy
# scene-description ones before building the augmented query.

# Scene-descriptor labels that add no search value when appended to a query.
_SCENE_NOISE_LABELS = {
    "person", "chair", "bench", "couch", "dining table", "table", "bed",
    "toilet", "sink", "floor", "wall", "ceiling", "window", "door",
    "potted plant", "vase", "bowl", "cup", "bottle", "fork", "knife",
    "spoon", "umbrella", "backpack", "handbag", "tie", "suitcase",
    "sports ball", "frisbee", "kite",
}

# Question-intent words that suggest the user is seeking information.
_QUESTION_WORDS = {
    "what", "how", "why", "when", "where", "who", "which",
    "explain", "describe", "tell", "show", "is", "are", "does",
}


def build_augmented_query(user_message: str, image_labels: List[str]) -> str:
    """Build a web-search query that merges the user message with informative image labels.

    Args:
        user_message: The raw user question.
        image_labels: CLIP-predicted COCO-80 labels for the uploaded image(s).

    Returns:
        A query string. If no labels survive the noise filter, the original
        user_message is returned unchanged so the caller always gets a usable query.
    """
    informative = [lb for lb in image_labels if lb.lower() not in _SCENE_NOISE_LABELS]
    if not informative:
        return user_message
    # Append at most 3 informative labels to keep the query focused.
    return f"{user_message} {' '.join(informative[:3])}"


def search_to_documents(query: str, max_results: int = 5) -> List[Document]:
    """Run a web search and convert results to LangChain Documents.

    Each Document.page_content contains a short summary/snippet; metadata includes
    title, url, and source (domain).
    """
    docs: List[Document] = []
    try:
        with DDGS(timeout=10) as ddg:
            results = ddg.text(query, max_results=max_results)
        for r in results or []:
            title = r.get("title") or ""
            href = r.get("href") or r.get("url") or ""
            body = r.get("body") or ""
            content = (title + ": " if title else "") + body
            meta = {
                "title": title,
                "url": href,
                "source": _domain(href),
            }
            docs.append(Document(page_content=content, metadata=meta))
    except Exception as e:
        logging.warning(f"web search failed: {e}")
    logging.info(f"web search query='{query}' docs={len(docs)}")
    return docs
