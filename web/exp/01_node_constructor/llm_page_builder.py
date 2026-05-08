"""Generate Stage 1 page definitions with an LLM."""

from __future__ import annotations

import json
import textwrap
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Tuple

from utils.llm_client import chat

from . import primitives
from .page_builder import (
    ActionSpec,
    PageDraft,
    PageMetadata,
    PageRequest,
    SiblingInfo,
    format_breadcrumb,
)

SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are an expert in web page structure generation. Please strictly follow the format constraints provided by the user, and always return valid JSON.
    """
)
DIRECT_PROMPT_TEMPLATE = textwrap.dedent(
    """\
    You are a web content generation model responsible for directly generating a root webpage with a clear theme and strong extensibility from a given primitive pool.

    Input:
    - Primitive pool: {primitive_pool}

    Task:
    1. Based on the concepts, entities, or functions contained in the primitive pool, generate a semantically broad and extensible webpage.
    2. The webpage content should reflect a comprehensive and sufficiently broad theme.
    3. The output must include the following fields:
    - title: The webpage title, reflecting the core theme and serving as the unique identifier of the page.
    - body: The textual content of the webpage, introducing the core content of the theme.
    - action: Actions for operations/interactions on this webpage, extracted from the primitive pool.
    - summary: An overall summary of the webpage, summarizing its main content.

    Output format:
    {{
    "title": "...",
    "body": "...",
    "action": ["..."],
    "summary": "..."
    }}
    """
).strip()

DEPTH_PROMPT_TEMPLATE = textwrap.dedent(
    """\
    You are a webpage hierarchy generation model responsible for expanding downward in a webpage hierarchical structure to produce a child node with a deeper, more specific theme.

    Input:
    - Primitive pool: {primitive_pool}
    - Parent path (sequence of titles from root to parent node): {title_path}
    - Parent node summary: {parent_summary}

    Task:
    1. Based on the parent node’s thematic semantics, vertically extend from the parent theme to generate a more specific and in-depth child page.
    2. The child node should semantically belong to a subordinate concept or sub-scenario of the parent node.
    3. The output must include the following fields:
    - title: Child node title, which should reflect the refinement direction of the parent node’s theme.
    - body: The textual content of the webpage, introducing the content of this child-theme page.
    - action: Actions for operations/interactions on this child page, extracted from the primitive pool.
    - summary: An overall summary of the child webpage, summarizing its main content.

    Output format:
    {{
    "title": "...",
    "body": "...",
    "action": ["..."],
    "summary": "..."
    }}
    """
).strip()

BREADTH_PROMPT_TEMPLATE = textwrap.dedent(
    """\
    You are a webpage hierarchy generation model responsible for generating sibling nodes at the same level for a specified node within a webpage hierarchical structure.

    Input:
    - Primitive pool: {primitive_pool}
    - Parent path (sequence of titles from root to parent node): {title_path}
    - Existing sibling node information:
    {siblings_info}

    Task:
    1. Read the existing sibling node information and understand their thematic scope and semantic boundaries.
    2. While preserving semantic coherence, generate {batch_size} new sibling nodes such that they are at the same thematic level and non-duplicative.
    3. Each node must include the following fields:
    - title: Title of the new sibling node page.
    - body: The textual content of the webpage, introducing the content of this sibling node page.
    - action: Actions for operations/interactions on this sibling node page, extracted from the primitive pool.
    - summary: An overall summary of the webpage, summarizing its main content.

    Output format (generate a list of length {batch_size}):
    [
    {{
        "title": "...",
        "body": "...",
        "action": ["..."],
        "summary": "..."
    }}
    ]
    """
).strip()

_PRIMITIVE_POOL = tuple(primitives.get_concepts())
_VALID_PRIMITIVES = frozenset(_PRIMITIVE_POOL)


def _format_primitive_pool() -> str:
    return json.dumps(_PRIMITIVE_POOL, ensure_ascii=False)


def _format_title_path(path: Sequence[str]) -> str:
    return json.dumps(list(path), ensure_ascii=False)


def _format_parent_summary(summary: str) -> str:
    return json.dumps(summary, ensure_ascii=False)


def _format_siblings_info(siblings: Sequence[SiblingInfo]) -> str:
    payload = []
    for sibling in siblings:
        entry: Dict[str, Any] = {"title": sibling.title}
        if sibling.summary:
            entry["summary"] = sibling.summary
        payload.append(entry)
    json_text = json.dumps(payload, ensure_ascii=False, indent=2)
    return textwrap.indent(json_text, "  ")


def _build_prompt(request: PageRequest) -> str:
    primitive_pool = _format_primitive_pool()
    mode = request.gen_mode or "direct"

    if mode == "direct":
        return DIRECT_PROMPT_TEMPLATE.format(primitive_pool=primitive_pool)

    if mode == "extend_depth":
        if not request.title_path:
            raise ValueError("Depth expansion requires a non-empty title_path.")
        if request.parent_summary is None:
            raise ValueError("Depth expansion requires parent_summary text.")
        return DEPTH_PROMPT_TEMPLATE.format(
            primitive_pool=primitive_pool,
            title_path=_format_title_path(request.title_path),
            parent_summary=_format_parent_summary(request.parent_summary),
        )

    if mode == "extend_breadth":
        if not request.title_path:
            raise ValueError("Breadth expansion requires a non-empty title_path.")
        siblings_block = _format_siblings_info(request.siblings)
        return BREADTH_PROMPT_TEMPLATE.format(
            primitive_pool=primitive_pool,
            title_path=_format_title_path(request.title_path),
            siblings_info=siblings_block,
            batch_size=request.batch_size,
        )

    raise ValueError(f"Unsupported generation mode: {mode}")


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped[3:]
        stripped = stripped.lstrip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    return stripped.strip()


def _parse_llm_json(raw_text: str) -> Any:
    stripped = _strip_code_fence(raw_text)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:  # noqa: PERF203
        raise ValueError(f"LLM response is not valid JSON: {raw_text!r}") from exc


def _ensure_text(payload: Dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if value is None:
        raise ValueError(f"LLM response is missing field {key!r}")
    text = str(value).strip()
    if not text:
        raise ValueError(f"LLM response field {key!r} must not be empty.")
    return text


def _normalize_action_entry(entry: Any) -> tuple[str, Dict[str, object]]:
    if isinstance(entry, str):
        concept = entry.strip()
        props: Dict[str, object] = {}
    elif isinstance(entry, Mapping):
        raw_concept = entry.get("concept", entry.get("name", ""))
        concept = str(raw_concept).strip()
        if not concept:
            raise ValueError("LLM action object is missing a concept field.")
        props_obj = entry.get("props")
        if props_obj is None:
            props = {str(k): v for k, v in entry.items() if k not in {"concept"}}
        elif isinstance(props_obj, Mapping):
            props = {str(k): v for k, v in props_obj.items()}
        else:
            raise ValueError("action.props must be an object.")
    else:
        concept = str(entry).strip()
        props = {}
    if not concept:
        raise ValueError("Action entries must not be empty.")
    return concept, props


def _ensure_actions(payload: Dict[str, Any]) -> Tuple[ActionSpec, ...]:
    raw_actions = payload.get("action", [])
    if isinstance(raw_actions, (str, Mapping)):
        entries = [raw_actions]
    elif isinstance(raw_actions, Sequence):
        entries = [item for item in raw_actions if item is not None]
    else:
        raise ValueError("LLM response field 'action' must be a string, object, or list.")

    normalized: list[ActionSpec] = []
    for entry in entries:
        concept, props = _normalize_action_entry(entry)
        normalized.append(ActionSpec(concept=concept, props=props))

    if not normalized:
        raise ValueError("LLM response field 'action' must contain at least one primitive.")

    filtered_specs = tuple(spec for spec in normalized if spec.concept in _VALID_PRIMITIVES)
    if len(filtered_specs) != len(normalized):
        missing = [spec.concept for spec in normalized if spec.concept not in _VALID_PRIMITIVES]
        warnings.warn(
            f"LLM response contains unknown primitives and they were dropped: {missing!r}.",
            stacklevel=2,
        )

    if filtered_specs:
        return filtered_specs

    fallback_action = _PRIMITIVE_POOL[0] if _PRIMITIVE_POOL else None
    if fallback_action is None:
        raise ValueError("Primitive pool is empty; cannot generate a valid page action.")
    warnings.warn(
        "All LLM actions were invalid; using fallback primitive "
        f"{fallback_action!r} to keep the page structure valid.",
        stacklevel=2,
    )
    return (ActionSpec(concept=fallback_action),)


def _parse_page_object(payload: Dict[str, Any]) -> PageDraft:
    title = _ensure_text(payload, "title")
    body = _ensure_text(payload, "body")
    summary = _ensure_text(payload, "summary")
    actions = _ensure_actions(payload)
    return PageDraft(title=title, body=body, action=actions, summary=summary)


def _parse_breadth_payload(payload: Any, request: PageRequest) -> Tuple[PageDraft, ...]:
    if isinstance(payload, dict):
        if "pages" in payload:
            payload = payload["pages"]
        elif "items" in payload:
            payload = payload["items"]
        else:
            raise ValueError("Breadth expansion expected a list but received an object.")

    if not isinstance(payload, list):
        raise ValueError("Breadth expansion must return a list of page objects.")
    if not payload:
        raise ValueError("Breadth expansion returned an empty list.")

    drafts = tuple(_parse_page_object(item) for item in payload)
    if request.batch_size and len(drafts) != request.batch_size:
        raise ValueError(
            f"Breadth expansion expected {request.batch_size} nodes but received {len(drafts)}."
        )
    return drafts


def generate_page_drafts(
    request: PageRequest,
    *,
    model: str,
    max_tokens: int = 768,
    temperature: float = 0.6,
    top_p: float | None = None,
    seed: int | None = None,
) -> Tuple[PageDraft, ...]:
    """Build the prompt, call the model, and parse page drafts."""

    user_prompt = _build_prompt(request)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    chat_kwargs: Dict[str, Any] = {
        "messages": messages,
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if top_p is not None:
        chat_kwargs["top_p"] = top_p
    if seed is not None:
        chat_kwargs["seed"] = seed

    response = chat(**chat_kwargs)
    content = response.choices[0].message.content  # type: ignore[assignment]
    if content is None:
        raise ValueError("LLM returned empty content.")

    payload = _parse_llm_json(content)

    if request.gen_mode == "extend_breadth":
        return _parse_breadth_payload(payload, request)

    if not isinstance(payload, dict):
        raise ValueError("Direct generation and depth expansion must return one page object.")
    return (_parse_page_object(payload),)


def _draft_to_metadata(request: PageRequest, draft: PageDraft) -> PageMetadata:
    return PageMetadata(
        page_id=request.page_id,
        page_index=request.page_index,
        breadcrumb=format_breadcrumb(request.breadcrumb),
        title=draft.title,
        summary=draft.summary,
        body=draft.body,
        action=draft.action,
        action_page=tuple(request.page_index for _ in draft.action),
    )


def build_page(
    request: PageRequest,
    *,
    model: str,
    max_tokens: int = 768,
    temperature: float = 0.6,
    top_p: float | None = None,
    seed: int | None = None,
) -> PageMetadata:
    """Generate one page definition and fill interface-maintained fields.

    For `extend_breadth`, use this only when `batch_size == 1`. Use
    `generate_page_drafts` when multiple sibling pages are expected.
    """

    drafts = generate_page_drafts(
        request,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )

    if not drafts:
        raise ValueError("Model did not return any pages.")
    if request.gen_mode == "extend_breadth" and len(drafts) != 1:
        raise ValueError(
            "Breadth expansion produced multiple pages; use `generate_page_drafts` for batched results."
        )
    return _draft_to_metadata(request, drafts[0])
