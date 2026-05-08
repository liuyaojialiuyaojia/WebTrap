"""Helpers for forcing uncached model API calls in file OSR-only flows."""

from __future__ import annotations

from typing import Any, Callable

NO_CACHE_OPTIONS: dict[str, bool] = {
    "no-cache": True,
    "no-store": True,
}


def wrap_generate(base_generate: Callable[..., Any]) -> Callable[..., Any]:
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        patched = dict(kwargs)
        patched["cache_ttl"] = None
        cache_options = dict(patched.get("cache_options") or {})
        cache_options.update(NO_CACHE_OPTIONS)
        patched["cache_options"] = cache_options
        return base_generate(*args, **patched)

    return _wrapped
