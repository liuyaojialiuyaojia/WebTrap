#!/usr/bin/env python3
"""Utilities for parsing exp2 trace logs for exp3 evaluation.

exp2 currently writes per-step JSONL rows that look like:
  {"step": 1, "type": "action", "model_output": "...", "observation": "Observation: {...json...}"}

Some older/newer variants may instead include structured fields like:
  tool_call: {"name": "...", "arguments": {...}}
  tool_observation: {...}

This module normalizes those rows into TraceEvent objects.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional


_ACTION_RE = re.compile(r"\bAction\s*:\s*([A-Za-z0-9_]+)\b")
_OBS_PREFIX_RE = re.compile(r"^\s*Observation\s*:\s*", flags=re.IGNORECASE)


@dataclass(frozen=True)
class TraceEvent:
    step: int
    row_type: str
    tool_name: Optional[str]
    tool_args: dict[str, Any]
    model_output: str
    observation: dict[str, Any]
    raw_row: dict[str, Any]


def infer_root_logical(logical_path: str) -> str:
    raw = str(logical_path or "").strip()
    if raw in {"", "/"}:
        return "/"
    if not raw.startswith("/"):
        raise ValueError(f"Logical path must be absolute: {logical_path!r}")
    parts = [p for p in raw.split("/") if p]
    if not parts:
        return "/"
    return "/" + parts[0]


def logical_to_virtual(logical_path: str, *, root_logical: Optional[str] = None) -> str:
    raw = str(logical_path or "").strip()
    if raw in {"", "/"}:
        return "/"
    if not raw.startswith("/"):
        # Best-effort fallback: treat as virtual-ish relative path.
        return "/" + raw.lstrip("/")

    root = root_logical or infer_root_logical(raw)
    if root not in {"/", ""} and raw == root:
        return "/"
    if root not in {"/", ""} and raw.startswith(root + "/"):
        return "/" + raw[len(root) :].lstrip("/")
    # Fallback: keep as-is.
    return raw


def _parse_tool_name_from_model_output(text: str) -> Optional[str]:
    match = _ACTION_RE.search(str(text or ""))
    if not match:
        return None
    name = match.group(1).strip()
    return name or None


def _parse_action_input_json(text: str) -> tuple[dict[str, Any], Optional[str], Optional[str]]:
    """Parse JSON object following 'Action Input:' from model_output.

    Returns: (parsed_args, error, raw_json_text)
    """
    raw_text = str(text or "")
    idx = raw_text.lower().find("action input:")
    if idx < 0:
        return {}, "missing_action_input", None
    tail = raw_text[idx:]
    start = tail.find("{")
    if start < 0:
        return {}, "action_input_not_json_object", None

    json_text: Optional[str] = None
    depth = 0
    in_string = False
    escape = False
    for offset, ch in enumerate(tail[start:], start=start):
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                json_text = tail[start : offset + 1]
                break

    if json_text is None:
        return {}, "action_input_unclosed_brace", None

    try:
        parsed = json.loads(json_text)
    except Exception as exc:  # pragma: no cover
        return {}, f"invalid_json_arguments:{exc}", json_text
    if not isinstance(parsed, dict):
        return {}, "tool_arguments_not_object", json_text
    return parsed, None, json_text


def _parse_observation_payload(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    text = str(raw)
    text = _OBS_PREFIX_RE.sub("", text).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Failed to parse observation JSON: {exc}: {text[:200]!r}") from exc
    if isinstance(parsed, dict):
        return parsed
    return {"result": parsed}


def parse_trace_rows(trace_rows: list[dict[str, Any]]) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    for row in trace_rows:
        if not isinstance(row, dict):
            continue
        step = int(row.get("step") or 0)
        row_type = str(row.get("type") or "").strip() or "unknown"
        model_output = str(row.get("model_output") or "")

        tool_name: Optional[str] = None
        tool_args: dict[str, Any] = {}

        tool_call = row.get("tool_call")
        if isinstance(tool_call, dict):
            tool_name = str(tool_call.get("name") or "").strip() or None
            args = tool_call.get("arguments")
            if isinstance(args, dict):
                tool_args = dict(args)
        if tool_name is None:
            tool_name = _parse_tool_name_from_model_output(model_output)
        if not tool_args and tool_name is not None:
            parsed_args, _err, _raw_json = _parse_action_input_json(model_output)
            tool_args = parsed_args

        raw_obs = row.get("tool_observation") if "tool_observation" in row else row.get("observation")
        observation = _parse_observation_payload(raw_obs)

        events.append(
            TraceEvent(
                step=step,
                row_type=row_type,
                tool_name=tool_name,
                tool_args=tool_args,
                model_output=model_output,
                observation=observation,
                raw_row=row,
            )
        )
    return events


def clip_for_prompt(
    value: Any,
    *,
    max_str_len: int = 2000,
    max_list_items: int = 50,
    max_depth: int = 4,
) -> Any:
    """Clip potentially large observations for LLM prompts."""

    if max_depth <= 0:
        return "[truncated]"
    if isinstance(value, str):
        if len(value) <= max_str_len:
            return value
        return value[: max_str_len].rstrip() + "\n...[truncated]"
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        clipped = [
            clip_for_prompt(
                v,
                max_str_len=max_str_len,
                max_list_items=max_list_items,
                max_depth=max_depth - 1,
            )
            for v in value[:max_list_items]
        ]
        if len(value) > max_list_items:
            clipped.append(f"...[{len(value) - max_list_items} more items truncated]")
        return clipped
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in list(value.items())[:200]:
            out[str(k)] = clip_for_prompt(
                v,
                max_str_len=max_str_len,
                max_list_items=max_list_items,
                max_depth=max_depth - 1,
            )
        if len(value) > 200:
            out["..."] = f"...[{len(value) - 200} more keys truncated]"
        return out
    return str(value)
