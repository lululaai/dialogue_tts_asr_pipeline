from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

_LIMIT_LOCK = threading.Lock()
_REQUEST_SEMAPHORE: threading.BoundedSemaphore | None = None
_REQUEST_LIMIT: int | None = None


def configure_google_request_limit(limit: int) -> None:
    global _REQUEST_LIMIT, _REQUEST_SEMAPHORE

    normalized = max(1, int(limit))
    with _LIMIT_LOCK:
        if _REQUEST_LIMIT == normalized and _REQUEST_SEMAPHORE is not None:
            return
        _REQUEST_LIMIT = normalized
        _REQUEST_SEMAPHORE = threading.BoundedSemaphore(normalized)


@contextmanager
def google_request_slot() -> Iterator[None]:
    semaphore = _REQUEST_SEMAPHORE
    if semaphore is None:
        yield
        return
    semaphore.acquire()
    try:
        yield
    finally:
        semaphore.release()
