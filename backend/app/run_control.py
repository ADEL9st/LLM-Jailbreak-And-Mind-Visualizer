"""Per-run cooperative cancellation shared by the WebSocket and adapters."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event
from typing import Any, Iterator


_CANCEL_EVENT: ContextVar[Event | None] = ContextVar("lmv_cancel_event", default=None)


@contextmanager
def use(cancel_event: Event) -> Iterator[None]:
    token = _CANCEL_EVENT.set(cancel_event)
    try:
        yield
    finally:
        _CANCEL_EVENT.reset(token)


def cancelled() -> bool:
    current = _CANCEL_EVENT.get()
    return bool(current and current.is_set())


def stopping_criteria() -> Any | None:
    """Build a Transformers criterion without making it a core dependency."""

    current = _CANCEL_EVENT.get()
    if current is None:
        return None
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList
    except ImportError:
        return None

    class Cancelled(StoppingCriteria):
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            return current.is_set()

    return StoppingCriteriaList([Cancelled()])
