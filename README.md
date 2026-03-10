# FusionMind — Multimodal RAG over Documents and Images

## The Problem

Researchers and analysts who work with dense technical PDFs — papers, reports, contracts — spend significant time re-reading documents to answer questions they have already encountered. Before this existed, they had to manually locate relevant sections, switch context between an uploaded image and a search engine, and synthesise the result themselves: a slow, error-prone process when documents exceed 50 pages and images provide visual context that text alone cannot supply.

---

## System Architecture

```
                         +---------------------------------------------+
                         |               PDF Upload Path                |
                         +---------------------------------------------+
                                              |
                         PyPDFLoader (LangChain)
                                              |
                         clean_documents()     <- _is_garbled() filter,
                         (pdf_utils.py)           ligature normalization,
                                              |   hyphenation repair
                                              |
                         select_chunk_size()   <- adaptive: 4 tiers based on
                         (chunking.py)            total_chars (500->2000 chars)
                                              |
                         RecursiveCharacterTextSplitter
                         chunk_overlap ~16% of chunk_size
                                              |
                         EmbedData / SmartEmbeddings
                         (embedding.py)        <- primary: gemini-embedding-001
                                              |   fallback: all-MiniLM-L6-v2
                                              |   trigger: HTTP 429 / quota error
                                              |
                         FAISSVectorStore      <- in-process, no server required
                         (vector_db.py)
                                              |
                         MMR Retriever         <- lambda=0.7, k=6, fetch_k=20
                         (retriever.py)           fetch_k clamped to index.ntotal


                         +---------------------------------------------+
                         |               Image Upload Path              |
                         +---------------------------------------------+
                                              |
                         ImageEmbedder         <- CLIP ViT-B/32 via
                         (image_embedding.py)     sentence-transformers
                                              |
                         zero-shot cosine sim  <- pre-encoded COCO-80 prompts
                         top-5 labels             "a photo of a {class}"
                                              |
                         _SCENE_NOISE_LABELS   <- filter: person, chair, bench...
                         build_augmented_query()  appends <=3 informative labels
                                              |
                         DuckDuckGo DDGS       <- max_results=5, 10s timeout
                         search_to_documents()    gated by question-intent heuristic


                         +---------------------------------------------+
                         |               Query Path                     |
                         +---------------------------------------------+
                                              |
                         User query
                                  +----------------------------------+
                                  | PDF context                      | Web context
                                  | (MMR retriever)                  | (DDG docs)
                                  +----------+-----------------------+
                                             | URL-domain deduplication
                                             | (answer_augmented in rag.py)
                                             |
                         RetrievalQA chain   <- LangChain "stuff" chain
                         Gemini 2.0 Flash       temperature=0.3
                                              |
                         answer + sources     -> RetrievalTelemetry.log()


                         +---------------------------------------------+
                         |           Session / History Layer            |
                         +---------------------------------------------+

                         flask_session cookie  -> sid (UUID)
                         TTLDict[sid]          <- evicts entries idle > 2h
                                              |   eviction runs every 100 requests
                         History[chat_id]      <- ChatMessageHistory (LangChain)
                                              |   + attached RAG pipeline
                                              |   + uploaded image metadata
                                              |   + CLIP labels
```

---

## Key Engineering Decisions

### 1. MMR over cosine similarity for retrieval

**Decision:** Use Maximal Marginal Relevance (MMR) with `lambda_mult=0.7`.

**Alternative considered:** Pure cosine similarity nearest-neighbour search (`search_type="similarity"`).

**Reason:** Cosine similarity returns the `k` most similar vectors to the query. When a PDF repeats the same concept across multiple pages — common in papers with abstract, introduction, and conclusion sections — all `k` slots fill with near-duplicate chunks. MMR re-scores each candidate document against those already selected:

```
score(d) = 0.7 * sim(d, query) - 0.3 * max_{s in Selected} sim(d, s)
```

The second term penalises documents too similar to an already-chosen one, pushing the retriever to cover different sections of the document. `lambda_mult=0.7` was chosen over the LangChain default of `0.5` because focused document QA needs factual accuracy more than breadth — at `0.5` the diversity penalty is large enough to discard the second-most-relevant passage in short PDFs where there is no true redundancy.

---

### 2. SmartEmbeddings quota-aware fallback

**Decision:** Wrap `gemini-embedding-001` in a `SmartEmbeddings` class that catches HTTP 429 / quota errors and re-calls `all-MiniLM-L6-v2` (HuggingFace, local inference).

**Alternative considered:** Failing hard when Google quota is exhausted; letting the user see an error and retry later.

**Reason:** Quota exhaustion is predictable (free-tier Google API resets daily) and recoverable. Returning a 500 error on every embed call once quota expires makes the application unusable for hours. The fallback (`all-MiniLM-L6-v2`, 384-dimensional) produces lower-quality embeddings than `gemini-embedding-001` (768-dimensional) but the FAISS index is rebuilt per upload, so dimension mismatch is never an issue — the fallback simply produces a different manifold. The `_should_fallback()` predicate checks for `"quota"`, `"429"`, and `"exceeded your current quota"` specifically; other errors (network timeouts, invalid API key) are not swallowed.

---

### 3. CLIP zero-shot label prediction over a trained classifier

**Decision:** Use `clip-ViT-B-32` via `sentence-transformers` with the COCO-80 vocabulary encoded as text prompts (`"a photo of a {class}"`), compared to the image embedding by cosine similarity.

**Alternative considered:** Training a lightweight multi-label image classifier (e.g., ResNet-18 fine-tuned on a labelled dataset).

**Reason:** The application has no labelled training dataset, and the use of labels is narrow: filtering web search queries, not accurate scene understanding. CLIP's zero-shot capability is sufficient — recognising that an image contains a "laptop" or "dog" to steer a DuckDuckGo query costs no labelling effort. A fine-tuned classifier would require ongoing maintenance as the label vocabulary changes. The trade-off is that CLIP performs poorly on domain-specific objects (microscopy slides, circuit diagrams) where COCO-80 has no matching class — which is why `_SCENE_NOISE_LABELS` removes generic labels (person, chair, bench) that carry no search signal regardless of accuracy.

---

### 4. In-memory History with TTL over database persistence

**Decision:** Store all chat sessions (`ChatMessageHistory`, attached RAG pipelines, images) in a `TTLDict` in process memory with a 2-hour eviction window. No database.

**Alternative considered:** MongoDB (the original codebase had a MongoDB import that was removed) or SQLite for persistence across restarts.

**Reason:** For a portfolio demo, eliminating the database dependency reduces setup to zero — no connection string, no schema migration, no docker-compose. The TTL prevents the dict from growing unbounded; the eviction hook fires every 100 requests to amortise the O(n) scan cost. The honest limitation is that a server restart loses all chat history and document indexes. Moving to production would require: serialising `ChatMessageHistory` to a document store (Redis, MongoDB), persisting FAISS indexes to disk with `faiss.write_index`, and replacing `TTLDict` with a distributed TTL cache across gunicorn workers.

---

### 5. Adaptive chunk size based on document length

**Decision:** `select_chunk_size(total_chars)` returns a `(chunk_size, chunk_overlap)` pair from four tiers: `(500, 50)` for < 5k chars, `(1000, 150)` for 5-50k, `(1500, 250)` for 50-200k, `(2000, 300)` for > 200k.

**Alternative considered:** A single hardcoded value (the original was `chunk_size=1500, chunk_overlap=250` for every document).

**Reason:** A fixed chunk size performs poorly at the extremes. A 3-page document (~4k chars) with `chunk_size=1500` produces 2-3 chunks: MMR has nothing to compare, `fetch_k=20` is clamped to 3, and the retriever degenerates to returning the whole document. A 200-page book (~300k chars) with `chunk_size=500` produces ~600 chunks: FAISS retrieval slows, the `fetch_k=20` pool covers a tiny fraction of the index, and the LLM receives micro-fragments lacking surrounding context. Adaptive sizing keeps the practical chunk count in the 20-200 range for any input. The overlap ratio is held near 16% across all tiers — enough to preserve a sentence that straddles a boundary without doubling index size.

---

### 6. `lambda_mult=0.7` for MMR

**Decision:** Set `MMR_LAMBDA=0.7` as the default, overridable via environment variable.

**Alternative considered:** The LangChain default of `lambda_mult=0.5`.

**Reason:** At `lambda_mult=0.5` the MMR score weights relevance and diversity equally:

```
score(d) = 0.5 * sim(d, query) - 0.5 * max_{s} sim(d, s)
```

In a large, repetitive corpus this is appropriate. For focused document QA the diversity penalty at `0.5` is strong enough to discard the second-most-relevant passage when it discusses the same topic — which is expected in any well-structured document. At `0.7`:

```
score(d) = 0.7 * sim(d, query) - 0.3 * max_{s} sim(d, s)
```

The retriever still avoids identical duplicates but does not penalise thematically related passages that provide complementary evidence. The value is exposed as `MMR_LAMBDA` in `.env` so it can be tuned per deployment without modifying source code.

---

## Retrieval Quality

A lightweight telemetry endpoint is available at `GET /api/telemetry` (localhost-only, or remotely via `?secret=TELEMETRY_SECRET`). It tracks:

- Total queries processed since last restart
- Average number of retrieved chunks per query
- Web augmentation rate (fraction of queries that triggered a DuckDuckGo search)
- Unique source filenames seen across all retrievals

`RetrievalTelemetry` stores the last 500 `RetrievalRecord` entries in a `collections.deque(maxlen=500)` — capped at roughly 200 KB of memory — and exposes both a summary dict and a JSONL export of the last 50 records.

**Honest note:** This is proxy telemetry, not ground-truth evaluation. Chunk count and source diversity are structural metrics, not measures of answer quality. For production, I would add RAGAS-based faithfulness and answer relevance scoring: RAGAS uses an LLM judge to score whether the generated answer is grounded in the retrieved context (`faithfulness`) and whether the context is relevant to the question (`answer_relevance`). These metrics require a reference question set, which this demo does not have.

---

## Limitations and Honest Tradeoffs

**1. Scanned (image-only) PDFs are rejected.**
`PyPDFLoader` extracts the text layer. A scanned PDF has no text layer — `clean_documents()` returns near-empty pages, `chunk_documents()` produces zero chunks, and the application raises a `ValueError` with an HTTP 400. The fix is an OCR pre-processing step (Tesseract via `pytesseract`, or a cloud Vision API) before calling `load_pdf()`. This was not added because OCR requires system-level installation (`tesseract-ocr`) and significantly increases processing time per page.

**2. DuckDuckGo web search has no rate-limit protection.**
`search_to_documents()` calls `DDGS().text()` with a 10-second timeout but no retry backoff and no per-user request budget. DuckDuckGo throttles aggressive scrapers; under moderate concurrent load this will start returning empty results silently. The production fix is a proper search API (Bing Search API, SerpAPI) with a server-side request queue and per-session search budgets.

**3. All session data is in-memory.**
A server restart loses every chat history, every document index, and every session association. Users returning after a restart see an empty chat. Moving to multi-user production requires FAISS index serialisation to disk or object storage, a persistent `ChatMessageHistory` store (Redis, PostgreSQL), and a distributed TTL cache for sessions across gunicorn workers.

**4. CLIP COCO-80 vocabulary does not cover domain-specific content.**
The zero-shot classifier predicts from 80 COCO object categories. A medical scan, a financial chart, an architectural floor plan, or a chromatography plot will receive low-confidence predictions from irrelevant categories. In those cases `_SCENE_NOISE_LABELS` filtering removes the noise labels and `build_augmented_query()` falls back to the user message alone — web search augmentation provides no value. A domain-specific CLIP fine-tune or a vision-language model with a broader vocabulary would be required for non-photographic domains.

---

## Setup and Usage

**Requirements:** Python 3.10+, a Google API key with Gemini access.

```bash
# 1. Clone and install
git clone <repo-url>
cd MultiModal_Rag
pip install -r requirements.txt

# 2. Configure environment
# Set at minimum:
#   GOOGLE_API_KEY=<your key>
#   FLASK_SECRET_KEY=<random string>

# 3. Run
python app.py
# Server starts on http://0.0.0.0:5000
```

**Environment variables (all optional, tunable without code changes):**

| Variable | Default | Effect |
|---|---|---|
| `USE_GOOGLE_EMBEDDINGS` | `0` | Set to `1` to use `gemini-embedding-001`; `0` uses `all-MiniLM-L6-v2` locally |
| `LLM_MODEL` | `gemini-2.0-flash-exp` | Gemini model name for chat and RAG |
| `MMR_K` | `6` | Number of chunks returned to the LLM |
| `MMR_FETCH_K` | `20` | Candidate pool size for MMR re-ranking |
| `MMR_LAMBDA` | `0.7` | Relevance/diversity balance (0.0-1.0) |
| `TELEMETRY_SECRET` | _(unset)_ | Enables remote access to `/api/telemetry` |
| `IMAGE_EMBED_WARMUP` | `1` | Pre-download CLIP weights on startup |
| `PORT` | `5000` | Flask listen port |
