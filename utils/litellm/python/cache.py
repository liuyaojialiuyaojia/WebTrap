from typing import Any, Dict, Optional


def cache_control(
    ttl: Optional[int] = None,
    namespace: Optional[str] = None,
    no_cache: bool = False,
    no_store: bool = False,
    s_maxage: Optional[int] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {}
    if ttl is not None:
        body["ttl"] = int(ttl)
    if namespace:
        body["namespace"] = namespace
    if no_cache:
        body["no-cache"] = True
    if no_store:
        body["no-store"] = True
    if s_maxage is not None:
        body["s-maxage"] = int(s_maxage)
    return body
