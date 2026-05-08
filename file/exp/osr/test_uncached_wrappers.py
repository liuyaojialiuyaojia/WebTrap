import importlib.util
from pathlib import Path


def _load_module(module_name: str, relative_path: str):
    module_path = Path(__file__).resolve().parent / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


no_cache = _load_module("file_osr_no_cache", "no_cache.py")


def test_wrap_generate_forces_no_cache() -> None:
    seen: dict[str, object] = {}

    def _fake_generate(*args, **kwargs):
        seen["kwargs"] = kwargs
        return {"ok": True}

    wrapped = no_cache.wrap_generate(_fake_generate)
    wrapped(messages=[], model="test-model", cache_ttl=86400, namespace="file_exp1_user_question")

    kwargs = dict(seen["kwargs"])
    assert kwargs["cache_ttl"] is None
    assert kwargs["cache_options"]["no-cache"] is True
    assert kwargs["cache_options"]["no-store"] is True
