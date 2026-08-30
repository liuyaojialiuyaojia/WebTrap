import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent / "tool_calling_agent_defended.py"
SPEC = importlib.util.spec_from_file_location(
    "tool_calling_agent_defended_under_test",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
agent_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = agent_module
SPEC.loader.exec_module(agent_module)


def test_completion_kwargs_are_claude_tool_call_compatible() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "click",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    kwargs = agent_module._build_completion_kwargs(
        model="anthropic/claude-haiku-4.5",
        messages=[{"role": "user", "content": "Use one tool."}],
        tools=tools,
        temperature=1.0,
        seed=42,
    )

    assert kwargs["model"] == "anthropic/claude-haiku-4.5"
    assert kwargs["tools"] == tools
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["parallel_tool_calls"] is False
    assert kwargs["temperature"] == 1.0
    assert kwargs["seed"] == 42
    assert kwargs["extra_body"] == {
        "cache": {
            "no-cache": True,
            "no-store": True,
        }
    }


def test_auxiliary_completion_is_uncached_without_tool_settings() -> None:
    kwargs = agent_module._build_completion_kwargs(
        model="anthropic/claude-haiku-4.5",
        messages=[{"role": "user", "content": "Classify this observation."}],
        temperature=0.0,
        seed=None,
        max_tokens=128,
    )

    assert kwargs["max_tokens"] == 128
    assert kwargs["extra_body"]["cache"]["no-cache"] is True
    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs
    assert "parallel_tool_calls" not in kwargs


def test_retryable_openai_error_is_retried(monkeypatch) -> None:
    class RetryableError(Exception):
        status_code = 502

    class FakeCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls < 3:
                raise RetryableError("bad gateway")
            return SimpleNamespace(
                choices=[SimpleNamespace(message="ok")],
                error=None,
                kwargs=kwargs,
            )

    class FakeClient:
        def __init__(self) -> None:
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    client = FakeClient()
    monkeypatch.setenv("WEBTRAP_OPENAI_RETRY_DELAYS", "0,0")

    result = agent_module._create_chat_completion_with_retries(
        client,
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert result.choices[0].message == "ok"
    assert result.kwargs["model"] == "gpt-4.1-mini"
    assert client.chat.completions.calls == 3


def test_retryable_completion_error_payload_is_retried(monkeypatch) -> None:
    class FakeCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls < 2:
                return SimpleNamespace(
                    choices=None,
                    error={
                        "message": "Upstream connection pool is saturated; retry shortly.",
                        "code": "503",
                    },
                )
            return SimpleNamespace(choices=[SimpleNamespace(message="ok")], error=None)

    class FakeClient:
        def __init__(self) -> None:
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    client = FakeClient()
    monkeypatch.setenv("WEBTRAP_OPENAI_RETRY_DELAYS", "0,0")

    result = agent_module._create_chat_completion_with_retries(
        client,
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert result.choices[0].message == "ok"
    assert client.chat.completions.calls == 2


def test_initial_jitter_runs_once_per_process(monkeypatch) -> None:
    class FakeCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(choices=[SimpleNamespace(message="ok")], error=None)

    class FakeClient:
        def __init__(self) -> None:
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    sleeps: list[float] = []
    monkeypatch.setenv("WEBTRAP_OPENAI_INITIAL_JITTER_SECONDS", "10")
    monkeypatch.setattr(agent_module.random, "uniform", lambda low, high: 3.0)
    monkeypatch.setattr(agent_module.time, "sleep", sleeps.append)
    monkeypatch.setattr(agent_module, "_INITIAL_OPENAI_JITTER_DONE", False)
    client = FakeClient()

    agent_module._create_chat_completion_with_retries(
        client,
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "first"}],
    )
    agent_module._create_chat_completion_with_retries(
        client,
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "second"}],
    )

    assert sleeps == [3.0]
    assert client.chat.completions.calls == 2


def test_non_retryable_openai_error_is_not_retried(monkeypatch) -> None:
    class NonRetryableError(Exception):
        status_code = 400

    class FakeCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            raise NonRetryableError("bad request")

    class FakeClient:
        def __init__(self) -> None:
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    client = FakeClient()
    monkeypatch.setenv("WEBTRAP_OPENAI_RETRY_DELAYS", "0,0")

    try:
        agent_module._create_chat_completion_with_retries(
            client,
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "hi"}],
        )
    except NonRetryableError:
        pass
    else:
        raise AssertionError("expected NonRetryableError")

    assert client.chat.completions.calls == 1


def test_openai_request_timeout_env(monkeypatch) -> None:
    monkeypatch.delenv("WEBTRAP_OPENAI_TIMEOUT", raising=False)
    assert agent_module._openai_request_timeout() is None

    monkeypatch.setenv("WEBTRAP_OPENAI_TIMEOUT", "0.5")
    assert agent_module._openai_request_timeout() == 1.0

    monkeypatch.setenv("WEBTRAP_OPENAI_TIMEOUT", "75")
    assert agent_module._openai_request_timeout() == 75.0


def test_openai_retry_jitter_env(monkeypatch) -> None:
    monkeypatch.delenv("WEBTRAP_OPENAI_RETRY_JITTER_SECONDS", raising=False)
    assert agent_module._openai_retry_jitter_seconds() == 0.0

    monkeypatch.setenv("WEBTRAP_OPENAI_RETRY_JITTER_SECONDS", "45")
    assert agent_module._openai_retry_jitter_seconds() == 45.0


def test_openai_initial_jitter_env(monkeypatch) -> None:
    monkeypatch.delenv("WEBTRAP_OPENAI_INITIAL_JITTER_SECONDS", raising=False)
    assert agent_module._openai_initial_jitter_seconds() == 0.0

    monkeypatch.setenv("WEBTRAP_OPENAI_INITIAL_JITTER_SECONDS", "30")
    assert agent_module._openai_initial_jitter_seconds() == 30.0


def test_completion_usage_tokens_tolerates_missing_usage() -> None:
    completion = SimpleNamespace(usage=None)

    assert agent_module._completion_usage_tokens(completion) == ("unknown", "unknown")


def test_completion_usage_tokens_reads_present_usage() -> None:
    completion = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=34)
    )

    assert agent_module._completion_usage_tokens(completion) == (12, 34)
