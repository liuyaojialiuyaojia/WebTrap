#!/usr/bin/env python3
"""Generate page definitions from JSONL requests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections.abc import Iterable, Sequence

try:
    if __package__ in (None, ""):
        THIS_DIR = Path(__file__).resolve().parent
        if str(THIS_DIR) not in sys.path:
            sys.path.insert(0, str(THIS_DIR))
        from page_builder import PageRequest, SiblingInfo  # type: ignore
        from llm_page_builder import build_page, generate_page_drafts  # type: ignore
    else:
        from .page_builder import PageRequest, SiblingInfo
        from .llm_page_builder import build_page, generate_page_drafts
except Exception as e:  # noqa: BLE001
    raise ImportError(
        f"Failed to import node constructor modules: {e}. "
        "If you are running this as a script, prefer `python web/exp/01_node_constructor/llm_make_page.py`."
    ) from e


def _parse_breadcrumb(raw: Sequence[int] | str | None) -> Sequence[int]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    if isinstance(raw, str):
        stripped = raw.strip().strip("/")
        if not stripped:
            return []
        return [int(part) for part in stripped.split("/")]
    raise TypeError(f"Unsupported breadcrumb format: {raw!r}")


def _parse_title_path(raw: Sequence[str] | str | None) -> Sequence[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw if item is not None]
    if isinstance(raw, str):
        return [segment.strip() for segment in raw.split(",") if segment.strip()]
    raise TypeError(f"Unsupported ancestor_themes format: {raw!r}")


def _parse_siblings(raw: object) -> Sequence[SiblingInfo]:
    if raw is None:
        return []
    if isinstance(raw, (str, bytes)):
        raise TypeError(f"Unsupported siblings format: {raw!r}")
    if not isinstance(raw, Sequence):
        raise TypeError(f"Unsupported siblings format: {raw!r}")
    siblings: list[SiblingInfo] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        title = entry.get("title") or entry.get("theme")
        if title is None:
            continue
        summary_value = entry.get("summary")
        if summary_value is None:
            summary_value = entry.get("brief")
        siblings.append(
            SiblingInfo(
                title=str(title),
                summary=str(summary_value) if summary_value is not None else None,
            )
        )
    return siblings


def _iter_requests(
    jsonl_path: Path, default_seed: int | None
) -> Iterable[tuple[PageRequest, Path]]:
    for line_no, raw_line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:  # noqa: PERF203 - precise error context is valuable
            raise ValueError(f"Line {line_no} is not valid JSON") from exc

        try:
            page_id = payload["page_id"]
            page_index = int(payload["page_index"])
        except KeyError as exc:
            raise ValueError(f"Missing required key {exc} in line {line_no}") from exc

        breadcrumb = _parse_breadcrumb(payload.get("breadcrumb"))
        gen_mode = str(payload.get("gen_mode", "direct"))
        title_path = _parse_title_path(
            payload.get("title_path")
            or payload.get("ancestor_titles")
            or payload.get("ancestor_themes")
        )
        siblings = _parse_siblings(payload.get("siblings"))
        title_value = payload.get("title") or payload.get("theme")
        parent_summary = payload.get("parent_summary") or payload.get("father")
        seed_value = payload.get("seed", default_seed)
        batch_size_value = payload.get("batch_size") or payload.get("k")
        try:
            batch_size = int(batch_size_value) if batch_size_value is not None else 1
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Line {line_no} has invalid batch_size: {batch_size_value!r}") from exc
        if batch_size < 1:
            raise ValueError(f"Line {line_no} batch_size must be >= 1, got {batch_size}")
        request = PageRequest(
            page_id=page_id,
            page_index=page_index,
            breadcrumb=breadcrumb,
            gen_mode=gen_mode,
            title_hint=str(title_value) if title_value is not None else None,
            title_path=tuple(title_path),
            parent_summary=str(parent_summary) if parent_summary is not None else None,
            siblings=tuple(siblings),
            batch_size=batch_size,
            seed=int(seed_value) if seed_value is not None else None,
        )
        yield request, Path(f"{page_id}.json")


def generate_pages(
    input_path: Path,
    output_dir: Path,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float | None,
    seed: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for request, relative_path in _iter_requests(input_path, seed):
        target_path = output_dir / relative_path
        if request.gen_mode == "extend_breadth" and request.batch_size > 1:
            drafts = generate_page_drafts(
                request,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=request.seed,
            )
            payload: object = [draft.to_dict() for draft in drafts]
        else:
            metadata = build_page(
                request,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=request.seed,
            )
            payload = metadata.to_dict()

        target_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSONL file containing page prompts.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to store generated page metadata JSON files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name for the LLM generator.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=768,
        help="Maximum tokens to request from the LLM.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature for the LLM.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling parameter; leave unset to disable.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional global seed to make generation deterministic across runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_pages(
        args.input,
        args.output_dir,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
