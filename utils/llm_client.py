"""Minimal OpenAI-compatible chat client for WebTrap experiment scripts.

The repository should not prescribe a model-serving stack. Configure the client
with ``OPENAI_API_KEY`` and, for non-OpenAI providers or local gateways, either
``OPENAI_BASE_URL`` or ``OPENAI_API_BASE``.
"""

from __future__ import annotations

import os
from typing import Any, Optional
from urllib.parse import urlparse

from openai import OpenAI


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _uses_localhost(base_url: str) -> bool:
    hostname = (urlparse(base_url).hostname or "").strip("[]").lower()
    return hostname in {"localhost", "127.0.0.1", "::1"}


def _ensure_local_no_proxy(base_url: str) -> None:
    if not _uses_localhost(base_url):
        return

    for key in ("NO_PROXY", "no_proxy"):
        current = os.getenv(key, "")
        entries = [entry.strip() for entry in current.split(",") if entry.strip()]
        merged = ["localhost", "127.0.0.1", "::1"]
        for entry in entries:
            if entry not in merged:
                merged.append(entry)
        os.environ[key] = ",".join(merged)


def resolve_base_url(base_url: Optional[str] = None) -> Optional[str]:
    resolved = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    return resolved.strip() if resolved and resolved.strip() else None


def get_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs: Any,
) -> OpenAI:
    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_api_key:
        raise RuntimeError(
            "Set OPENAI_API_KEY before calling an OpenAI-compatible model API."
        )

    resolved_base_url = resolve_base_url(base_url)
    if resolved_base_url:
        _ensure_local_no_proxy(resolved_base_url)
        kwargs.setdefault("base_url", resolved_base_url)

    return OpenAI(api_key=resolved_api_key, **kwargs)


def _cache_body_enabled() -> bool:
    return _truthy_env("WEBTRAP_ENABLE_CACHE_BODY") or _truthy_env(
        "WEBTRAP_CACHE_BODY"
    )


def chat(
    messages: list[dict[str, Any]],
    model: str,
    cache_ttl: Optional[int] = None,
    namespace: Optional[str] = None,
    cache_options: Optional[dict[str, Any]] = None,
    **kwargs: Any,
):
    """Create a chat completion using the configured OpenAI-compatible API.

    ``cache_ttl`` and related cache fields are retained as optional experiment
    metadata. They are only forwarded through ``extra_body.cache`` when
    ``WEBTRAP_ENABLE_CACHE_BODY=1`` is set, because cache extensions are not part
    of the standard OpenAI API.
    """

    extra_body = dict(kwargs.pop("extra_body", {}) or {})
    cache_body = dict(extra_body.pop("cache", {}) or {})
    if cache_ttl is not None:
        cache_body["ttl"] = int(cache_ttl)
    if namespace:
        cache_body["namespace"] = namespace
    if cache_options:
        cache_body.update(cache_options)
    if cache_body and _cache_body_enabled():
        extra_body["cache"] = cache_body

    client = get_client()
    request_kwargs = dict(kwargs)
    if extra_body:
        request_kwargs["extra_body"] = extra_body
    return client.chat.completions.create(
        model=model,
        messages=messages,
        **request_kwargs,
    )
