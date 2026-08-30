import os
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional

from openai import OpenAI

LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1")
LITELLM_MASTER_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-yaojia-get-ccfa")


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


def get_client(api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs) -> OpenAI:
    resolved_base_url = base_url or LITELLM_BASE_URL
    _ensure_local_no_proxy(resolved_base_url)
    return OpenAI(
        api_key=api_key or LITELLM_MASTER_KEY,
        base_url=resolved_base_url,
        **kwargs,
    )


def chat(
    messages: List[Dict[str, Any]],
    model: str,
    cache_ttl: Optional[int] = None,
    namespace: Optional[str] = None,
    cache_options: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    extra_body = kwargs.pop("extra_body", {}) or {}
    cache_body = extra_body.get("cache", {})
    if cache_ttl is not None:
        cache_body["ttl"] = int(cache_ttl)
    if namespace:
        cache_body["namespace"] = namespace
    if cache_options:
        cache_body.update(cache_options)
    if cache_body:
        extra_body["cache"] = cache_body
    client = get_client()
    return client.chat.completions.create(
        model=model,
        messages=messages,
        extra_body=extra_body,
        **kwargs,
    )
