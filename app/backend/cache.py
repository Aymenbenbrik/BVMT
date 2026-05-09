"""
In-memory cache for agent pipeline results.

Keys are ticker names (str), values are the full AgentState.to_dict()
output plus a timestamp.  TTL is 30 minutes by default.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CacheEntry:
    """Single cached result for one ticker."""
    ticker: str
    data: Dict[str, Any]
    computed_at: float  # time.time() epoch
    computation_seconds: float = 0.0


class TickerCache:
    """Thread-safe-ish in-memory dict cache with TTL."""

    def __init__(self, ttl_seconds: int = 7200):
        self._store: Dict[str, CacheEntry] = {}
        self.ttl = ttl_seconds  # default 2 hours

    # ── read ─────────────────────────────────────────────────────────
    def get(self, ticker: str) -> Optional[CacheEntry]:
        entry = self._store.get(ticker)
        if entry is None:
            return None
        if self._is_expired(entry):
            return None
        return entry

    def is_fresh(self, ticker: str) -> bool:
        return self.get(ticker) is not None

    # ── write ────────────────────────────────────────────────────────
    def set(
        self,
        ticker: str,
        data: Dict[str, Any],
        computation_seconds: float = 0.0,
    ) -> None:
        self._store[ticker] = CacheEntry(
            ticker=ticker,
            data=data,
            computed_at=time.time(),
            computation_seconds=computation_seconds,
        )

    # ── bulk helpers ─────────────────────────────────────────────────
    def status(self) -> Dict[str, Any]:
        """Return cache summary for the /cache/status endpoint."""
        now = time.time()
        entries = {}
        for ticker, entry in self._store.items():
            entries[ticker] = {
                "computed_at": entry.computed_at,
                "computation_seconds": round(entry.computation_seconds, 2),
                "age_seconds": round(now - entry.computed_at, 1),
                "fresh": not self._is_expired(entry),
            }
        return {
            "cached_count": len(self._store),
            "ttl_seconds": self.ttl,
            "entries": entries,
        }

    def all_cached(self) -> Dict[str, CacheEntry]:
        """Return all non-expired entries."""
        return {
            ticker: entry
            for ticker, entry in self._store.items()
            if not self._is_expired(entry)
        }

    def clear(self, ticker: Optional[str] = None) -> None:
        if ticker:
            self._store.pop(ticker, None)
        else:
            self._store.clear()

    # ── internals ────────────────────────────────────────────────────
    def _is_expired(self, entry: CacheEntry) -> bool:
        return (time.time() - entry.computed_at) > self.ttl
