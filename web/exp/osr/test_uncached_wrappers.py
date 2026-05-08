import importlib.util
import sys
from pathlib import Path


def _load_module(module_name: str, relative_path: str):
    module_path = Path(__file__).resolve().parent / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


no_cache = _load_module("web_osr_no_cache", "no_cache.py")
run_agent_uncached = _load_module("web_osr_run_agent_uncached", "run_agent_uncached.py")
tool_agent_uncached = _load_module(
    "web_osr_tool_calling_agent_uncached", "tool_calling_agent_uncached.py"
)


def test_wrap_llm_chat_forces_no_cache() -> None:
    seen: dict[str, object] = {}

    def _fake_chat(*args, **kwargs):
        seen["kwargs"] = kwargs
        return {"ok": True}

    wrapped = no_cache.wrap_llm_chat(_fake_chat)
    wrapped(model="test-model", messages=[], cache_ttl=86400, namespace="ns")

    kwargs = dict(seen["kwargs"])
    assert kwargs["cache_ttl"] is None
    assert kwargs["cache_options"]["no-cache"] is True
    assert kwargs["cache_options"]["no-store"] is True


def test_merge_no_cache_extra_body_preserves_other_fields() -> None:
    merged = no_cache.merge_no_cache_extra_body({"foo": "bar", "cache": {"ttl": 10}})
    assert merged["foo"] == "bar"
    assert merged["cache"]["ttl"] == 10
    assert merged["cache"]["no-cache"] is True
    assert merged["cache"]["no-store"] is True


def test_build_command_uncached_uses_osr_agent_wrapper() -> None:
    command = run_agent_uncached._build_command_uncached(
        task_path=Path("/tmp/task.json"),
        model="test-model",
        temperature=0.8,
        seed=None,
        defense_mode="default_attack",
        max_actions=20,
        max_observations=3,
        log_path=Path("/tmp/trace.jsonl"),
    )
    assert command[:2] == ["python", "web/exp/osr/tool_calling_agent_uncached.py"]
    assert "--seed" not in command


def test_with_uncached_request_kwargs_adds_cache_controls() -> None:
    patched = tool_agent_uncached._with_uncached_request_kwargs(
        {"model": "test-model", "extra_body": {"cache": {"ttl": 5}}}
    )
    assert patched["extra_body"]["cache"]["ttl"] == 5
    assert patched["extra_body"]["cache"]["no-cache"] is True
    assert patched["extra_body"]["cache"]["no-store"] is True
