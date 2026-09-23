"""Experimental iteration-level speculative decoding trace (HeteroSpec research).

Records, once per decode verify, the batch composition and the accepted draft
count per request:

    {"iteration": 182, "batch_size": 8, "active_k": 5, "ts": 1234.5,
     "requests": [{"rid": "r1", "accepted": 5}, {"rid": "r2", "accepted": 1}]}

Why this exists
---------------
SGLang already exposes per-request speculative aggregates in response
``meta_info`` (``spec_verify_ct``, ``spec_num_correct_drafts``,
``spec_correct_drafts_histogram``). What those cannot express is *when* and
*alongside whom* each observation happened. Without that, per-position acceptance
derived from a histogram is unreliable whenever ``speculative_num_steps`` changes
during a request's life: a round with active K=2 can never report
``accept >= 5``, so deep positions get scored as rejected although they were never
proposed. The bias cannot be corrected after the fact, because the histogram does
not record K.

This module supplies exactly the missing join:
``(request identity, accepted drafts, batch composition, active K)``.

Design constraints
------------------
* **Stdlib only.** Imported from the scheduler hot path, so it must not pull in
  torch or anything else. It also makes the logic unit-testable without a GPU.
* **Zero cost when disabled.** ``enabled()`` is a module-level constant, so a
  disabled server pays one global lookup per verify -- no list building, no
  formatting. The caller guards on ``enabled()`` before constructing arguments.
* **Bounded.** ``SGLANG_HETEROSPEC_TRACE_MAX_RECORDS`` caps the file so a long run
  cannot fill a disk; once the cap is hit a single warning is logged.
* **Follows the in-repo convention.** Modelled on the existing DSpark dump
  (``SGLANG_DSPARK_DEBUG_DUMP``, bounded records, periodic flush) rather than
  inventing a new shape, so this stays easy to review if it is ever upstreamed.

This is a research facility, deliberately not a general logging API. It is
env-gated and off by default.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import time
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "enabled",
    "record_iteration",
    "flush",
    "close",
]


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default
    return value if value > 0 else default


_PATH: str | None = os.environ.get("SGLANG_HETEROSPEC_TRACE") or None
_MAX_RECORDS: int = _env_int("SGLANG_HETEROSPEC_TRACE_MAX_RECORDS", 200_000)
_FLUSH_EVERY: int = _env_int("SGLANG_HETEROSPEC_TRACE_FLUSH_EVERY", 64)

#: Module-level constant so the disabled path is a single global lookup.
_ENABLED: bool = _PATH is not None


class _Tracer:
    """Append-only JSONL writer with a record cap and periodic flush."""

    def __init__(self, path: str, max_records: int, flush_every: int) -> None:
        self._max_records = max_records
        self._flush_every = flush_every
        self._lock = threading.Lock()
        self._count = 0
        self._iteration = 0
        self._capped = False
        self._since_flush = 0
        self._path = path
        # Line-buffered: a crash mid-run still leaves a usable prefix.
        self._file = open(path, "w", buffering=1)
        logger.info(
            "HeteroSpec iteration trace enabled: %s (max_records=%d)",
            path,
            max_records,
        )

    # -- writing -------------------------------------------------------------

    def record(
        self,
        *,
        active_k: int | None,
        rids: Sequence[str],
        accepted: Sequence[int],
    ) -> None:
        if self._capped:
            return
        with self._lock:
            if self._count >= self._max_records:
                self._capped = True
                logger.warning(
                    "HeteroSpec iteration trace reached its %d-record cap at %s; "
                    "further iterations are not recorded. Raise "
                    "SGLANG_HETEROSPEC_TRACE_MAX_RECORDS to capture more.",
                    self._max_records,
                    self._path,
                )
                return

            n = min(len(rids), len(accepted))
            entry = {
                "iteration": self._iteration,
                "batch_size": n,
                "active_k": active_k,
                "ts": time.monotonic(),
                "requests": [
                    {"rid": rids[i], "accepted": accepted[i]} for i in range(n)
                ],
            }
            self._iteration += 1
            self._count += 1
            self._since_flush += 1
            self._file.write(json.dumps(entry) + "\n")
            if self._since_flush >= self._flush_every:
                self._file.flush()
                self._since_flush = 0

    def flush(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.flush()

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.flush()
                self._file.close()
                logger.info(
                    "HeteroSpec iteration trace closed: %d records -> %s",
                    self._count,
                    self._path,
                )

    @property
    def count(self) -> int:
        return self._count


_TRACER: _Tracer | None = _Tracer(_PATH, _MAX_RECORDS, _FLUSH_EVERY) if _PATH else None

if _TRACER is not None:
    atexit.register(_TRACER.close)


def enabled() -> bool:
    """Whether iteration tracing is on.

    Callers must guard on this **before** building argument lists, so a disabled
    server never pays for ``[r.rid for r in batch.reqs]``.
    """
    return _ENABLED


def record_iteration(
    *,
    active_k: int | None,
    rids: Iterable[str],
    accepted: Iterable[int],
) -> None:
    """Record one decode verify. No-op when tracing is disabled."""
    tracer = _TRACER
    if tracer is None:
        return
    tracer.record(active_k=active_k, rids=list(rids), accepted=list(accepted))


def flush() -> None:
    if _TRACER is not None:
        _TRACER.flush()


def close() -> None:
    if _TRACER is not None:
        _TRACER.close()
