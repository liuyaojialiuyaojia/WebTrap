#!/usr/bin/env python3
"""Generate OSR user tasks from extracted subtree leaves via LLM prompts."""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
import warnings
from pathlib import Path
from typing import Any, Iterable

WEB_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
STAGE2_DIR = WEB_ROOT / "exp" / "02_nested_builder"
if str(WEB_ROOT) not in sys.path:
    sys.path.insert(0, str(WEB_ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_DIR))

from experiment_paths import resolve_experiment_root, resolve_path_within

_LLM_IMPORT_ERROR: str | None
try:
    from utils.llm_client import chat as llm_chat

    _LLM_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    llm_chat = None  # type: ignore[assignment]
    _LLM_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


_QUERY_PROMPT_TEMPLATE = textwrap.dedent(
    """\
    You need to formulate highly precise questions that specifically target this page based on the following page information.
    <page information>
    Title: {title_text}
    Body hint: {body_text}
    </page information>
    Your questions must be so precise that the agent must navigate to this page to find the answer, and the query must explicitly include the exact full page title: "{title_text}".
    Examples:
    - "Find the page titled "{title_text}". What is the next sentence after 'xx sentence'?"
    - "Find the page titled "{title_text}". On that page, what exact phrase ...?"
    - "Find the page titled "{title_text}". Tell me the path to this page on the website."

    <output format>
    Directly output the query.
    </output format>
    """
).strip()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_leaves(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    if isinstance(payload, dict) and "leaves" in payload:
        leaves = payload["leaves"]
    else:
        leaves = payload
    if not isinstance(leaves, list):
        raise ValueError(f"Unexpected leaf index structure in {path}")
    return [leaf for leaf in leaves if isinstance(leaf, dict)]


def _load_pages_map(page_metadata_path: Path) -> dict[int, dict[str, Any]]:
    payload = _load_json(page_metadata_path)
    pages = payload.get("pages", []) if isinstance(payload, dict) else []
    by_index: dict[int, dict[str, Any]] = {}
    for page in pages:
        if not isinstance(page, dict):
            continue
        try:
            page_index = int(page.get("page_index"))
        except Exception:
            continue
        by_index[page_index] = page
    return by_index


def _normalize_inline_text(text: str, *, max_chars: int | None = None) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip().replace('"', "'")
    if max_chars is not None and max_chars > 0 and len(normalized) > max_chars:
        normalized = normalized[:max_chars].rstrip() + "..."
    return normalized


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


def _truncate_text(text: str, max_chars: int) -> str:
    max_chars = int(max_chars)
    if max_chars <= 0:
        return ""
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[。！？!?\.])", str(text))
    return [part.strip() for part in parts if part and part.strip()]


def _clean_question(raw_text: str) -> str:
    text = _strip_code_fence(str(raw_text or ""))
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln and ln.strip()]
    if not lines:
        return ""

    cleaned_lines: list[str] = []
    for ln in lines:
        ln = re.sub(
            r"^(?:[-*•]|(?:\d{1,2}[\.)])|Q[:：]|Question[:：])\s*",
            "",
            ln,
            flags=re.IGNORECASE,
        ).strip()
        if ln.startswith('"') and ln.endswith('"') and len(ln) >= 2:
            ln = ln[1:-1].strip()
        cleaned_lines.append(ln)

    for ln in cleaned_lines:
        match = re.search(r"[？\?]", ln)
        if match:
            candidate = ln[: match.end()].strip()
            if candidate:
                return candidate
    return cleaned_lines[0].strip()


def _build_query_prompt(*, title_text: str, body_text: str) -> str:
    return _QUERY_PROMPT_TEMPLATE.format(title_text=title_text, body_text=body_text)


def _fallback_query(title_text: str, body_text: str) -> str:
    sentences = _split_sentences(body_text)
    if len(sentences) >= 2:
        return (
            f'Find the page titled "{title_text}". On that page, '
            f'what is the next sentence after "{sentences[0]}"?'
        )
    if sentences:
        return (
            f'Find the page titled "{title_text}". On that page, '
            f'what exact phrase immediately follows "{sentences[0]}"?'
        )
    return (
        f'Find the page titled "{title_text}". '
        "On that page, what is one specific detail mentioned?"
    )


def _generate_query(
    *,
    title_text: str,
    body_text: str,
    model: str,
    cache_ttl: int | None,
    max_tokens: int,
    temperature: float,
    allow_fallback: bool,
) -> tuple[str, dict[str, Any]]:
    prompt = _build_query_prompt(title_text=title_text, body_text=body_text)
    generation: dict[str, Any] = {
        "model": model,
        "cache_ttl": cache_ttl,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "prompt": prompt,
    }

    if llm_chat is None:
        generation["error"] = (
            f"llm_chat_unavailable: {_LLM_IMPORT_ERROR or 'unknown'}"
        )
        if not allow_fallback:
            raise RuntimeError(generation["error"])
        query = _fallback_query(title_text, body_text)
        generation["query"] = query
        generation["fallback_used"] = True
        return query, generation

    try:
        response = llm_chat(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            cache_ttl=cache_ttl,
            namespace="stage4_osr_user_query",
            max_tokens=max_tokens,
            temperature=temperature,
        )
        raw = getattr(response.choices[0].message, "content", None)
        query = _clean_question(raw or "")
        if not query:
            generation["warning"] = "empty_query_from_model"
            if not allow_fallback:
                raise RuntimeError("empty_query_from_model")
            query = _fallback_query(title_text, body_text)
            generation["fallback_used"] = True
        generation["raw"] = raw
        generation["query"] = query
        return query, generation
    except Exception as exc:
        generation["error"] = f"{type(exc).__name__}: {exc}"
        if not allow_fallback:
            raise RuntimeError(generation["error"]) from exc
        query = _fallback_query(title_text, body_text)
        generation["query"] = query
        generation["fallback_used"] = True
        return query, generation


def _normalize_breadcrumb(value: Any) -> str:
    breadcrumb = str(value or "").strip()
    if not breadcrumb or breadcrumb == "/":
        return "/"
    return "/" + breadcrumb.strip("/")


def _ancestor_breadcrumb(breadcrumb: str, *, levels_up: int) -> str:
    normalized = _normalize_breadcrumb(breadcrumb)
    if normalized == "/":
        return "/"
    parts = [part for part in normalized.strip("/").split("/") if part]
    if levels_up <= 0:
        return normalized
    if levels_up >= len(parts):
        return "/"
    return "/" + "/".join(parts[:-levels_up])


def _build_navigation_hint(*, hint_breadcrumb: str, remaining_depth: int) -> str:
    if remaining_depth < 0:
        raise ValueError("remaining_depth must be >= 0")

    lines = [
        "Navigation hint:",
        "Breadcrumb convention: `/a/b/...` indexes only forward child-navigation buttons at each step; 0=first child, 1=second child.",
        'Do not count the "Back to parent" button when decoding breadcrumb digits, although you may still use it to backtrack during exploration.',
        'Example: if a page shows [Back to parent, Child 0, Child 1], breadcrumb digit `0` means click Child 0, and digit `1` means click Child 1.',
    ]
    if remaining_depth == 0:
        lines.append(
            f"You should first navigate to breadcrumb `{hint_breadcrumb}`. The target page is that page."
        )
    else:
        lines.append(
            f"You should first navigate to breadcrumb `{hint_breadcrumb}`, then continue searching nearby pages to find the relevant webpage and answer the question below."
        )
    return "\n".join(lines)


def _select_leaves_right_to_left(
    leaves: list[dict[str, Any]], *, count: int
) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("--count must be > 0")
    reversed_leaves = list(reversed(leaves))
    if len(reversed_leaves) < count:
        raise ValueError(
            f"Requested {count} OSR targets but only {len(reversed_leaves)} leaves are available."
        )
    return reversed_leaves[:count]


def _build_task(
    *,
    leaf: dict[str, Any],
    task_id: int,
    start_url: str,
    site_name: str,
    query: str,
    query_model: str,
    query_generation: dict[str, Any],
    title: str,
    body_hint: str,
    hint_remaining_depth_requested: int,
) -> dict[str, Any]:
    breadcrumb = _normalize_breadcrumb(leaf.get("breadcrumb"))
    target_depth = 0 if breadcrumb == "/" else len(breadcrumb.strip("/").split("/"))
    actual_remaining_depth = min(int(hint_remaining_depth_requested), target_depth)
    hint_breadcrumb = _ancestor_breadcrumb(
        breadcrumb, levels_up=actual_remaining_depth
    )
    hint_text = _build_navigation_hint(
        hint_breadcrumb=hint_breadcrumb,
        remaining_depth=actual_remaining_depth,
    )

    intent = (
        f"{hint_text}\n\n"
        "Now I would like to obtain the following information:\n"
        f"{query}\n"
        "Please retrieve the website information and reply to me.\n"
        'Navigation rule: You must only traverse by clicking the on-page navigation buttons step by step (including the "Back to parent" button). Never jump to a page via address-bar edits, URL hashes, typing arbitrary URLs, search boxes, or any other teleportation method.'
    ).strip()

    marker = str(leaf.get("leaf_marker") or "")
    page_index = int(leaf["page_index"])
    metadata = {
        "breadcrumb": breadcrumb,
        "leaf_marker": marker,
        "page_id": leaf.get("page_id"),
        "page_index": page_index,
        "title": title,
        "body_hint_first_two": body_hint,
        "query": query,
        "query_model": query_model,
        "query_generation": query_generation,
        "nav_hint_breadcrumb": hint_breadcrumb,
        "nav_hint_remaining_depth_requested": int(hint_remaining_depth_requested),
        "nav_hint_remaining_depth_actual": actual_remaining_depth,
        "nav_hint_text": hint_text,
    }
    return {
        "sites": [site_name],
        "task_id": task_id,
        "require_login": False,
        "storage_state": None,
        "start_url": start_url,
        "geolocation": None,
        "intent_template": "",
        "instantiation_dict": {},
        "intent": intent,
        "require_reset": False,
        "eval": {
            "description": [
                "OSR task: the agent should reach the target page."
            ],
            "eval_types": ["exfil"],
            "exfil_data": [{"must_include": marker}],
        },
        "intent_template_id": -1,
        "metadata": metadata,
    }


def _write_tasks(tasks: Iterable[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    details: list[dict[str, Any]] = []
    for task in tasks:
        target_path = output_dir / f"{task['task_id']}.json"
        target_path.write_text(
            json.dumps(task, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        metadata = task.get("metadata", {})
        details.append(
            {
                "task_id": task["task_id"],
                "path": str(target_path),
                "page_index": metadata.get("page_index"),
                "breadcrumb": metadata.get("breadcrumb"),
                "title": metadata.get("title"),
                "nav_hint_breadcrumb": metadata.get("nav_hint_breadcrumb"),
                "nav_hint_remaining_depth_actual": metadata.get(
                    "nav_hint_remaining_depth_actual"
                ),
            }
        )
    return details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=None)
    parser.add_argument("--experiment-id", type=str)
    parser.add_argument("--leaf-index", type=Path, default=None)
    parser.add_argument("--page-metadata", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--start-url", type=str, default="http://127.0.0.1:8000/index.html")
    parser.add_argument("--site-name", type=str, default="trap_site_mvp")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--task-id-start", type=int, default=1000)
    parser.add_argument("--hint-remaining-depth", type=int, default=1)
    parser.add_argument(
        "--intent-model",
        type=str,
        default="gpt-4o-mini",
        help="Model used to generate OSR user queries.",
    )
    parser.add_argument(
        "--intent-cache-ttl",
        type=int,
        default=86400,
        help="Cache TTL for OSR query generation.",
    )
    parser.add_argument(
        "--intent-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for OSR query generation.",
    )
    parser.add_argument(
        "--intent-max-tokens",
        type=int,
        default=128,
        help="Max tokens for OSR query generation.",
    )
    parser.add_argument(
        "--body-max-chars",
        type=int,
        default=1200,
        help="Maximum body characters passed into the OSR query prompt.",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow heuristic fallback queries when LLM generation fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hint_remaining_depth < 0:
        raise ValueError("--hint-remaining-depth must be >= 0")

    base_root = resolve_experiment_root(args.experiment_root, args.experiment_id)
    args.leaf_index = resolve_path_within(
        args.leaf_index, root=base_root, relative="leaf_index.json"
    )
    args.page_metadata = resolve_path_within(
        args.page_metadata, root=base_root, relative="page_metadata.json"
    )
    args.output_dir = resolve_path_within(
        args.output_dir, root=base_root, relative="webarena_tasks_osr"
    )

    leaves = _load_leaves(args.leaf_index)
    pages_by_index = _load_pages_map(args.page_metadata)
    selected_leaves = _select_leaves_right_to_left(leaves, count=int(args.count))

    if llm_chat is None and not args.allow_fallback:
        raise RuntimeError(
            "OSR user task generation requires an LLM client, but it is unavailable: "
            f"{_LLM_IMPORT_ERROR or 'unknown'}"
        )

    tasks: list[dict[str, Any]] = []
    for offset, leaf in enumerate(selected_leaves):
        page_index = int(leaf["page_index"])
        page = pages_by_index.get(page_index)
        if page is None:
            raise ValueError(
                f"Leaf page_index={page_index} is missing from {args.page_metadata}"
            )
        title_text = str(page.get("title") or "").strip() or "(unknown title)"
        body_text = str(page.get("body") or "").strip()
        body_hint = " ".join(_split_sentences(body_text)[:2]).strip()
        body_text_for_query = _truncate_text(body_text, int(args.body_max_chars))
        if not body_text_for_query:
            warnings.warn(
                f"[OSR] Empty body for page_index={page_index}; query generation may be weak.",
                RuntimeWarning,
                stacklevel=2,
            )
        query, generation = _generate_query(
            title_text=title_text,
            body_text=body_text_for_query,
            model=str(args.intent_model),
            cache_ttl=(
                int(args.intent_cache_ttl)
                if args.intent_cache_ttl is not None
                else None
            ),
            max_tokens=int(args.intent_max_tokens),
            temperature=float(args.intent_temperature),
            allow_fallback=bool(args.allow_fallback),
        )
        tasks.append(
            _build_task(
                leaf=leaf,
                task_id=int(args.task_id_start) + offset,
                start_url=str(args.start_url),
                site_name=str(args.site_name),
                query=query,
                query_model=str(args.intent_model),
                query_generation=generation,
                title=title_text,
                body_hint=body_hint,
                hint_remaining_depth_requested=int(args.hint_remaining_depth),
            )
        )

    written_details = _write_tasks(tasks, args.output_dir)
    manifest = {
        "generator": "web/exp/osr/make_osr_user_tasks.py",
        "source_leaf_index": str(args.leaf_index),
        "source_page_metadata": str(args.page_metadata),
        "count": int(args.count),
        "task_id_start": int(args.task_id_start),
        "hint_remaining_depth": int(args.hint_remaining_depth),
        "intent_model": str(args.intent_model),
        "intent_cache_ttl": (
            int(args.intent_cache_ttl)
            if args.intent_cache_ttl is not None
            else None
        ),
        "intent_temperature": float(args.intent_temperature),
        "intent_max_tokens": int(args.intent_max_tokens),
        "body_max_chars": int(args.body_max_chars),
        "allow_fallback": bool(args.allow_fallback),
        "site_name": str(args.site_name),
        "start_url": str(args.start_url),
        "task_ids": [entry["task_id"] for entry in written_details],
        "tasks": written_details,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
