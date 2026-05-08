"""Helpers for forcing uncached OpenAI-compatible calls in OSR-only flows."""

from __future__ import annotations

from typing import Any, Callable

NO_CACHE_OPTIONS: dict[str, bool] = {
    "no-cache": True,
    "no-store": True,
}


def merge_no_cache_extra_body(extra_body: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(extra_body or {})
    cache_body = dict(merged.get("cache") or {})
    cache_body.update(NO_CACHE_OPTIONS)
    merged["cache"] = cache_body
    return merged


def wrap_llm_chat(
    base_chat: Callable[..., Any],
) -> Callable[..., Any]:
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        patched = dict(kwargs)
        patched["cache_ttl"] = None
        cache_options = dict(patched.get("cache_options") or {})
        cache_options.update(NO_CACHE_OPTIONS)
        patched["cache_options"] = cache_options
        return base_chat(*args, **patched)

    return _wrapped
