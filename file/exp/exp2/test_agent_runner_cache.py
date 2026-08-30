from __future__ import annotations

from file.exp.exp2 import agent_runner


def test_target_agent_sampling_forces_no_cache(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_generate(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(agent_runner, "generate", fake_generate)

    agent_runner._generate_target_agent_response(
        messages=[],
        model="demo",
        temperature=1.0,
        tools=[],
        max_tokens=512,
    )

    assert seen["cache_ttl"] is None
    assert seen["cache_options"] == {
        "no-cache": True,
        "no-store": True,
    }
    assert seen["temperature"] == 1.0
    assert seen["max_tokens"] == 512
