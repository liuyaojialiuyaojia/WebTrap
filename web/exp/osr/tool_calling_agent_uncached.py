#!/usr/bin/env python3
"""OSR-only wrapper for the web tool-calling agent with request caching disabled."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from web.exp.osr.no_cache import merge_no_cache_extra_body


def _load_shared_module():
    module_path = Path(__file__).resolve().parents[1] / "05_evaluate" / "tool_calling_agent_defended.py"
    spec = importlib.util.spec_from_file_location("web_osr_tool_agent_shared", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


shared = _load_shared_module()


def _with_uncached_request_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    patched = dict(kwargs)
    patched["extra_body"] = merge_no_cache_extra_body(
        patched.get("extra_body") if isinstance(patched.get("extra_body"), dict) else None
    )
    return patched


def _call_model_uncached(self: Any, messages: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": self.tools_definitions,
        }
        if self.temperature is not None:
            kwargs["temperature"] = float(self.temperature)
        if self.seed is not None:
            kwargs["seed"] = int(self.seed)
        completion = self.client.chat.completions.create(
            **_with_uncached_request_kwargs(kwargs)
        )
        print(
            f"Received model response. Used {completion.usage.prompt_tokens} prompt tokens and {completion.usage.completion_tokens} completion tokens"
        )
        return shared._parse_response_to_json(completion.choices[0].message)
    except Exception as exc:
        print(f"Error occurred while requesting OpenAI API: {exc}")
        return {
            "role": "assistant",
            "content": f"Error occurred while requesting OpenAI API: {exc}",
            "tool_calls": [],
        }


def _ask_defense_model_uncached(
    self: Any, messages: list[dict[str, str]], max_tokens: int
) -> str | None:
    try:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        if self.seed is not None:
            kwargs["seed"] = int(self.seed)
        completion = self.client.chat.completions.create(
            **_with_uncached_request_kwargs(kwargs)
        )
        response = completion.choices[0].message.content
        return str(response or "").strip()
    except Exception as exc:
        print(f"[defense] auxiliary model call failed: {exc}")
        return None


def install_patches() -> None:
    shared.GPTWebAgent._call_model = _call_model_uncached
    shared.GPTWebAgent._ask_defense_model = _ask_defense_model_uncached


def main(argv: list[str] | None = None) -> None:
    install_patches()
    shared.main.main(
        args=list(sys.argv[1:] if argv is None else argv),
        prog_name="tool_calling_agent_uncached.py",
        standalone_mode=False,
    )


if __name__ == "__main__":
    main()
