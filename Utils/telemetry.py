"""Lightweight in-memory retrieval telemetry for the RAG pipeline.

# DESIGN NOTE: Why deque(maxlen=500) instead of a plain list?
# A plain list grows without bound — in a long-running server session that
# processes thousands of queries this would be another silent memory leak
# (the same class of problem we fixed in SESSIONS with TTLDict).
# deque(maxlen=500) is O(1) append and O(1) removal of the oldest entry;
# once capacity is reached old records are evicted automatically with no
# extra bookkeeping.  500 records covers ≈8–10 hours of moderate usage and
# is more than enough for a demo or portfolio inspection.
#
# Why NOT persist to disk?
# For a single-process demo app, in-process memory is sufficient and keeps
# the dependency surface minimal (no SQLite, no file-locking, no rotation).
# In a production deployment you would replace this class with a Prometheus
# Counter/Histogram + Grafana, or push events to OpenTelemetry (OTEL).
# The interface here (log / summary / to_jsonl) is easy to swap out: just
# replace the deque writes with metric increments.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Optional

from Utils.logger import logging


@dataclass
class RetrievalRecord:
    """Snapshot of a single RAG retrieval + generation event."""
    query: str
    retrieved_chunk_sources: List[str]   # source filenames of each chunk
    retrieved_chunk_lengths: List[int]   # character lengths of each chunk
    avg_chunk_length: float
    answer_length: int
    used_web_augmentation: bool
    used_mmr: bool
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RetrievalTelemetry:
    """Collect and summarise in-memory retrieval quality metrics.

    All public methods are safe to call from the answer-generation path:
    they never raise — any internal error is caught and only a WARNING is
    logged so telemetry failures never degrade user-visible functionality.
    """

    def __init__(self) -> None:
        # DESIGN NOTE: maxlen=500 caps memory at ≈500 * ~400 bytes ≈ 200 KB.
        self._records: deque[RetrievalRecord] = deque(maxlen=500)

    def log(self, record: RetrievalRecord) -> None:
        """Append a record; silently drops on any error."""
        try:
            self._records.append(record)
        except Exception as e:
            logging.warning(f"telemetry.log failed (non-fatal): {e}")

    def summary(self) -> dict:
        """Return aggregate statistics over all held records."""
        try:
            records = list(self._records)
            if not records:
                return {
                    "total_queries": 0,
                    "avg_chunks_per_query": 0.0,
                    "avg_answer_length": 0.0,
                    "web_augmentation_rate": 0.0,
                    "unique_sources_seen": 0,
                }
            total = len(records)
            avg_chunks = sum(len(r.retrieved_chunk_sources) for r in records) / total
            avg_answer = sum(r.answer_length for r in records) / total
            web_rate = sum(1 for r in records if r.used_web_augmentation) / total
            all_sources: set = set()
            for r in records:
                all_sources.update(r.retrieved_chunk_sources)
            return {
                "total_queries": total,
                "avg_chunks_per_query": round(avg_chunks, 2),
                "avg_answer_length": round(avg_answer, 1),
                "web_augmentation_rate": round(web_rate, 3),
                "unique_sources_seen": len(all_sources),
            }
        except Exception as e:
            logging.warning(f"telemetry.summary failed (non-fatal): {e}")
            return {}

    def to_jsonl(self) -> str:
        """Serialise the most recent 50 records as newline-delimited JSON."""
        try:
            recent = list(self._records)[-50:]
            return "\n".join(json.dumps(asdict(r)) for r in recent)
        except Exception as e:
            logging.warning(f"telemetry.to_jsonl failed (non-fatal): {e}")
            return ""


# Module-level singleton shared across all RAG instances in the process.
TELEMETRY = RetrievalTelemetry()
