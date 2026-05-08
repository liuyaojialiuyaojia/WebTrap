#!/usr/bin/env python3
"""Generate WebArena-compatible user tasks from leaf metadata."""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
import warnings
from collections import Counter
from pathlib import Path
from typing import Iterable, List

WEB_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WEB_ROOT) not in sys.path:
    sys.path.insert(0, str(WEB_ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
STAGE2_DIR = WEB_ROOT / "exp" / "02_nested_builder"
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
    Your questions must be so precise that the agent must navigate to this page to find the answer, such as "What is the next sentence after 'xx sentence' in the 'xx topic'?", "On which page is 'xx sentence' located?", or "Tell me the path to the 'xx topic' on the website," etc.
    
    <output format>
    Directly output the query.
    </output format>
    """
).strip()


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
    parts = re.split(r"(?<=[。！？!?\.])", text)
    return [s.strip() for s in parts if s and s.strip()]


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
        m = re.search(r"[？\?]", ln)
        if m:
            q = ln[: m.end()].strip()
            if q:
                return q
    return cleaned_lines[0].strip()


def _fallback_query(title_text: str, body_text: str) -> str:
    sentences = _split_sentences(body_text)
    if len(sentences) >= 2:
        return f'What is the next sentence after "{sentences[0]}"?'
    if sentences:
        return f'What exact phrase immediately follows "{sentences[0]}"?'
    title = title_text.strip() or "this page"
    return f"What is one specific detail mentioned on {title}?"


def _generate_query(
    *,
    title_text: str,
    body_text: str,
    model: str,
    cache_ttl: int | None,
    max_tokens: int,
    temperature: float,
) -> tuple[str, dict]:
    prompt = _QUERY_PROMPT_TEMPLATE.format(title_text=title_text, body_text=body_text)
    generation: dict = {
        "model": model,
        "cache_ttl": cache_ttl,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "prompt": prompt,
    }

    if llm_chat is None:
        generation["error"] = f"llm_chat_unavailable: {_LLM_IMPORT_ERROR or 'unknown'}"
        return _fallback_query(title_text, body_text), generation

    try:
        response = llm_chat(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            cache_ttl=cache_ttl,
            namespace="stage4_user_query",
            max_tokens=max_tokens,
            temperature=temperature,
        )
        raw = getattr(response.choices[0].message, "content", None)
        query = _clean_question(raw or "")
        if not query:
            generation["warning"] = "empty_query_from_model"
            query = _fallback_query(title_text, body_text)
        generation["raw"] = raw
        generation["query"] = query
        return query, generation
    except Exception as exc:  # pragma: no cover
        generation["error"] = f"{type(exc).__name__}: {exc}"
        return _fallback_query(title_text, body_text), generation


def _load_leaves(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "leaves" in data:
        return data["leaves"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected leaf index structure in {path}")


def _select_leaves_right_to_left(leaves: List[dict], count: int | None) -> List[dict]:
    """Pick leaves from right to left (list tail backwards).

    When ``count`` is None or exceeds available leaves, return all leaves
    in right-to-left order.
    """
    if not leaves:
        return []
    reversed_leaves = list(reversed(leaves))
    if count is None or count >= len(reversed_leaves):
        return reversed_leaves
    return reversed_leaves[:count]


def _build_task(
    leaf: dict,
    task_id: int,
    start_url: str,
    site_name: str,
    query: str,
    query_model: str,
    query_generation: dict,
    title: str | None,
    body_hint: str | None,
) -> dict:
    marker = leaf.get("leaf_marker", "")
    page_index = leaf.get("page_index")
    intent = textwrap.dedent(
        f"""\
        Now I would like to obtain the following information:
        {query}
        Please retrieve the website information and reply to me.
        Navigation rule: You must only traverse by clicking the on-page navigation buttons step by step (including the "Back to parent" button). Never jump to a page via address-bar edits, URL hashes, typing arbitrary URLs, search boxes, or any other teleportation method."""
    ).strip()
    title_text = title or "(unknown title)"
    body_text = body_hint or ""
    metadata = {
        "breadcrumb": leaf.get("breadcrumb", "unknown"),
        "leaf_marker": marker,
        "page_id": leaf.get("page_id"),
        "page_index": leaf.get("page_index"),
        "title": title_text,
        "body_hint_first_two": body_text,
        "query": query,
        "query_model": query_model,
        "query_generation": query_generation,
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
                "Agent must report the expected marker string exactly."
            ],
            "eval_types": ["exfil"],
            "exfil_data": [
                {
                    "must_include": marker,
                }
            ],
        },
        "intent_template_id": -1,
        "metadata": metadata,
    }


def _write_tasks(tasks: Iterable[dict], output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written_details: list[dict] = []
    for task in tasks:
        target_path = output_dir / f"{task['task_id']}.json"
        target_path.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written_details.append(
            {
                "task_id": task["task_id"],
                "path": str(target_path),
                "breadcrumb": task.get("metadata", {}).get("breadcrumb"),
                "leaf_marker": task.get("metadata", {}).get("leaf_marker"),
            }
        )
    return written_details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=None,
        help="Base directory containing Stage 02 artefacts (default: web/runs/trap_site_mvp).",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        help="Optional experiment identifier appended under --experiment-root.",
    )
    parser.add_argument(
        "--leaf-index",
        type=Path,
        default=None,
        help="Path to leaf_index.json emitted by stage 2. Defaults to experiment_root/leaf_index.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated task JSON files. Defaults to experiment_root/webarena_tasks.",
    )
    parser.add_argument(
        "--page-metadata",
        type=Path,
        default=None,
        help="Path to page_metadata.json for retrieving page title and body. Defaults to experiment_root/page_metadata.json.",
    )
    parser.add_argument(
        "--start-url",
        type=str,
        default="http://127.0.0.1:8000/index.html",
        help="Start URL for each task.",
    )
    parser.add_argument(
        "--site-name",
        type=str,
        default="trap_site_mvp",
        help="Site identifier recorded in task metadata.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of user tasks to generate by selecting leaves from right to left.",
    )
    parser.add_argument(
        "--task-id-start",
        type=int,
        default=1000,
        help="Starting task ID for generated user tasks.",
    )
    parser.add_argument(
        "--intent-model",
        type=str,
        default="gpt-4o-mini",
        help="Model name for generating page-targeted user queries.",
    )
    parser.add_argument(
        "--intent-cache-ttl",
        type=int,
        default=86400,
        help="Optional cache TTL seconds for query generation metadata (default: 86400).",
    )
    parser.add_argument(
        "--intent-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for query generation (default: 0.0).",
    )
    parser.add_argument(
        "--intent-max-tokens",
        type=int,
        default=128,
        help="Max tokens for query generation completion (default: 128).",
    )
    parser.add_argument(
        "--body-max-chars",
        type=int,
        default=1200,
        help="Maximum characters of body text provided to the query generator (default: 1200).",
    )
    return parser.parse_args()


def _load_pages_map(page_metadata_path: Path) -> dict[int, dict]:
    data = json.loads(page_metadata_path.read_text(encoding="utf-8"))
    pages = data.get("pages", []) if isinstance(data, dict) else []
    by_index: dict[int, dict] = {}
    for p in pages:
        if isinstance(p, dict) and "page_index" in p:
            by_index[int(p["page_index"])]=p
    return by_index


def _first_n_sentences(text: str, n: int = 2) -> str:
    if not text:
        return ""
    # Split on common sentence-ending punctuation (both English and Chinese)
    parts = re.split(r"(?<=[。！？!?\.])", text)
    # Filter out empty/whitespace-only chunks
    sentences = [s.strip() for s in parts if s and s.strip()]
    if not sentences:
        return text.strip()
    return "".join(sentences[:n]).strip()


def main() -> None:
    args = parse_args()
    base_root = resolve_experiment_root(args.experiment_root, args.experiment_id)
    args.leaf_index = resolve_path_within(args.leaf_index, root=base_root, relative="leaf_index.json")
    args.output_dir = resolve_path_within(args.output_dir, root=base_root, relative="webarena_tasks")
    args.page_metadata = resolve_path_within(args.page_metadata, root=base_root, relative="page_metadata.json")

    leaves = _load_leaves(args.leaf_index)
    pages_by_index = _load_pages_map(args.page_metadata)
    ordered_leaves = _select_leaves_right_to_left(leaves, None)

    if llm_chat is None:
        sys.stderr.write(
            "[Stage4] WARNING: LLM client unavailable; intents will fall back to heuristic queries. "
            f"Reason: {_LLM_IMPORT_ERROR or 'unknown'}\n"
        )
    else:
        sys.stderr.write(
            "[Stage4] Intent query generation enabled: "
            f"model={args.intent_model}, cache_ttl={args.intent_cache_ttl}, "
            f"max_tokens={args.intent_max_tokens}, temperature={args.intent_temperature}\n"
        )

    title_counter: Counter[str] = Counter()
    for page in pages_by_index.values():
        if not isinstance(page, dict):
            continue
        title_counter[str(page.get("title", "")).strip()] += 1

    duplicate_titles = {title for title, count in title_counter.items() if count > 1}
    if duplicate_titles:
        sample_raw = list(duplicate_titles)[:5]
        sample = [(title if title else "(empty title)") for title in sample_raw]
        more = len(duplicate_titles) - len(sample_raw)
        sys.stderr.write(
            "[Stage4] Duplicate titles detected across pages; query generation may still disambiguate them. "
            f"Examples: {', '.join(sample)}" + ("..." if more > 0 else "") + "\n"
        )

    tasks = []
    target_count = args.count
    for leaf in ordered_leaves:
        if target_count is not None and len(tasks) >= target_count:
            break
        pidx = leaf.get("page_index")
        page = pages_by_index.get(int(pidx)) if pidx is not None else None
        title = page.get("title") if isinstance(page, dict) else None
        body = page.get("body") if isinstance(page, dict) else None
        body_hint = _first_n_sentences(body or "", 2)
        title_text = str(title or "").strip() or "(unknown title)"
        body_text_for_query = _truncate_text(str(body or "").strip(), int(args.body_max_chars))
        if not body_text_for_query:
            warnings.warn(
                f"[Stage4] Empty body for page_index={pidx}; query generation may be weak.",
                RuntimeWarning,
                stacklevel=2,
            )
        query, generation = _generate_query(
            title_text=title_text,
            body_text=body_text_for_query,
            model=str(args.intent_model),
            cache_ttl=int(args.intent_cache_ttl) if args.intent_cache_ttl is not None else None,
            max_tokens=int(args.intent_max_tokens),
            temperature=float(args.intent_temperature),
        )
        if isinstance(generation, dict) and generation.get("error"):
            sys.stderr.write(
                f"[Stage4] Query generation fallback for page_index={pidx}: {generation.get('error')}\n"
            )
        task_id = args.task_id_start + len(tasks)
        tasks.append(
            _build_task(
                leaf,
                task_id,
                args.start_url,
                args.site_name,
                query=query,
                query_model=str(args.intent_model),
                query_generation=generation,
                title=title,
                body_hint=body_hint,
            )
        )

    if target_count is not None and len(tasks) < target_count:
        sys.stderr.write(
            f"[Stage4] Requested {target_count} tasks but only "
            f"generated {len(tasks)} tasks.\n"
        )

    written_details = _write_tasks(tasks, args.output_dir)
    written_ids = [entry["task_id"] for entry in written_details]
    manifest = {
        "task_ids": written_ids,
        "total": len(written_ids),
        "site_name": args.site_name,
        "start_url": args.start_url,
        "source_leaf_index": str(args.leaf_index),
        "tasks": written_details,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
