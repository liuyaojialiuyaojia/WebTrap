import os
import urllib.request

LITELLM_MASTER_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-yaojia-get-ccfa")


def _ensure_local_no_proxy():
    local_hosts = "localhost,127.0.0.1"
    for key in ("NO_PROXY", "no_proxy"):
        current = os.getenv(key, "")
        os.environ[key] = f"{local_hosts},{current}" if current else local_hosts


def _get(url: str):
    _ensure_local_no_proxy()
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {LITELLM_MASTER_KEY}"},
    )
    with urllib.request.urlopen(req, timeout=8) as response:
        return response.status, response.read().decode()


def ping_cache():
    base = os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1").rstrip("/")
    url = base[:-3] + "/cache/ping" if base.endswith("/v1") else base + "/cache/ping"
    return _get(url)


def list_models():
    base = os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1").rstrip("/")
    return _get(base + "/models")


if __name__ == "__main__":
    print("cache:", ping_cache())
    print("models:", list_models())
