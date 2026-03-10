"""Lightweight TTL-aware dict for capping in-process session memory growth.

# DESIGN NOTE: Why not threading.Timer per entry?
# Each Timer spawns a thread that sleeps until expiry. With N concurrent sessions
# you'd have N sleeping threads competing for the GIL and OS scheduler slots.
# For a web server with hundreds of idle sessions this is wasteful and can cause
# thundering-herd wakeups. Instead we evict lazily:
#   - On every __getitem__ / __contains__ / get() we check the individual entry age.
#   - A bulk evict() sweeps the whole dict and is called periodically from a
#     before_request hook (every Nth request), amortizing the O(n) scan cost
#     across incoming traffic rather than paying it at deletion time.
# This pattern is called "lazy + periodic eviction" and is used by Redis, Guava
# Cache, and Python's functools.lru_cache internals for similar reasons.
"""

from __future__ import annotations

import time
from typing import Any, Iterator, Optional


class TTLDict:
    """Dict-like container that expires entries after a configurable idle period.

    An entry's TTL clock resets on every read or write (access-based expiry,
    not creation-based).  This mirrors typical web session semantics where an
    active user should never be evicted mid-session.

    Thread safety: NOT thread-safe by default. Flask's development server is
    single-threaded; for production (gunicorn with workers) each worker process
    gets its own TTLDict so no cross-process locking is needed.
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._ts: dict[str, float] = {}  # last-access timestamps

    # ------------------------------------------------------------------ #
    # Core interface
    # ------------------------------------------------------------------ #

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value
        self._ts[key] = time.monotonic()

    def __getitem__(self, key: str) -> Any:
        # Raises KeyError if absent — intentional, matches plain dict contract
        val = self._data[key]
        self._ts[key] = time.monotonic()  # touch on read
        return val

    def __delitem__(self, key: str) -> None:
        del self._data[key]
        self._ts.pop(key, None)

    def __contains__(self, key: object) -> bool:
        if key not in self._data:
            return False
        # DESIGN NOTE: We update the timestamp on __contains__ as well because
        # app.py's get_session_id() calls `sid not in SESSIONS` as its freshness
        # check — treating that read as an access prevents a session from being
        # evicted between the `in` check and the subsequent `__getitem__`.
        self._ts[key] = time.monotonic()  # type: ignore[arg-type]
        return True

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._data:
            return default
        return self[key]  # goes through __getitem__ → updates timestamp

    def __iter__(self) -> Iterator[str]:
        return iter(list(self._data))

    def __len__(self) -> int:
        return len(self._data)

    # ------------------------------------------------------------------ #
    # TTL management
    # ------------------------------------------------------------------ #

    def touch(self, key: str) -> None:
        """Explicitly reset the idle timer for *key* without reading its value."""
        if key in self._data:
            self._ts[key] = time.monotonic()

    def evict(self, max_age_seconds: float = 3600) -> int:
        """Remove all entries that have not been accessed in *max_age_seconds*.

        Returns the number of entries removed.

        # DESIGN NOTE: We iterate over a snapshot (list) of keys so we can
        # safely delete from _data while iterating.  Collecting expired keys
        # first and batch-deleting avoids repeated dict resizing mid-loop.
        """
        cutoff = time.monotonic() - max_age_seconds
        expired = [k for k, ts in self._ts.items() if ts < cutoff]
        for k in expired:
            self._data.pop(k, None)
            self._ts.pop(k, None)
        return len(expired)
