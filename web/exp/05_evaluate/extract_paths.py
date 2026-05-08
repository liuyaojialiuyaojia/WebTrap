#!/usr/bin/env python3
"""Extract breadcrumb-style navigation paths from agent logs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE2_DIR = REPO_ROOT / "exp" / "02_nested_builder"
if str(STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_DIR))

from experiment_paths import resolve_experiment_root, resolve_path_within

_BREADCRUMB_RE = re.compile(r"Breadcrumb:\s*([^\n<]+)")
_TRACE_RE = re.compile(r"trace_(?P<task>\d+)(?:_r(?P<trial>\d+))?\.jsonl$")
_NAV_STATUS_RE = re.compile(r"Information navigation → page\s+(\d+)")
_HASH_URL_RE = re.compile(r"url:\s*[^\s]+#p(\d+)", re.IGNORECASE)
_INDEX_ROOT_RE = re.compile(r"url:\s*[^\s]*index\.html(?!#p)", re.IGNORECASE)


def _load_metadata(
    metadata_path: Path,
) -> Tuple[
    Dict[str, Dict[str, object]],
    Dict[str, Dict[str, object]],
    Dict[int, Dict[str, object]],
]:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    pages = payload.get("pages") if isinstance(payload, dict) else None
    if not isinstance(pages, list):
        raise ValueError(f"Unexpected page metadata structure in {metadata_path}")

    element_to_transition: Dict[str, Dict[str, object]] = {}
    breadcrumb_to_page: Dict[str, Dict[str, object]] = {}
    page_index_map: Dict[int, Dict[str, object]] = {}

    for page in pages:
        if not isinstance(page, dict):
            continue
        breadcrumb = str(page.get("breadcrumb", "/"))
        breadcrumb_to_page[breadcrumb] = page
        try:
            page_index = int(page.get("page_index", 0))
        except (TypeError, ValueError):
            page_index = 0
        page_index_map[page_index] = page
        for target in page.get("click_targets") or []:
            if not isinstance(target, dict):
                continue
            element_id = target.get("element_id")
            if not element_id:
                continue
            if element_id in element_to_transition:
                raise ValueError(f"Duplicate click target element_id '{element_id}' in metadata")
            element_to_transition[element_id] = {
                "source_breadcrumb": breadcrumb,
                "source_page": page.get("page_index"),
                "target_page": target.get("target_page"),
                "target_breadcrumb": target.get("target_breadcrumb"),
            }

    return element_to_transition, breadcrumb_to_page, page_index_map


def _parent_breadcrumb(breadcrumb: str) -> str | None:
    if breadcrumb in {"", "/"}:
        return None
    parts = [p for p in breadcrumb.strip("/").split("/") if p]
    if not parts:
        return "/"
    parent = "/" + "/".join(parts[:-1]) if parts[:-1] else "/"
    return parent


def _append_breadcrumb(history: List[str], breadcrumb: str) -> None:
    if not history or history[-1] != breadcrumb:
        history.append(breadcrumb)


def _breadcrumb_for(page_catalog: Dict[int, Dict[str, object]], page_index: Optional[int]) -> str:
    if page_index is None:
        return "/"
    page = page_catalog.get(page_index)
    if isinstance(page, dict):
        breadcrumb = page.get("breadcrumb")
        if isinstance(breadcrumb, str) and breadcrumb:
            return breadcrumb
    return "/"


def _detect_page_index(content: str) -> Optional[int]:
    if not content:
        return None
    status_matches = _NAV_STATUS_RE.findall(content)
    if status_matches:
        try:
            return int(status_matches[-1])
        except ValueError:
            pass
    hash_matches = _HASH_URL_RE.findall(content)
    if hash_matches:
        try:
            return int(hash_matches[-1])
        except ValueError:
            pass
    if _INDEX_ROOT_RE.search(content):
        return 0
    return None


def _normalize_tool_arguments(
    args_payload: object,
) -> Tuple[Dict[str, object], Optional[str], Optional[object]]:
    if isinstance(args_payload, str):
        try:
            parsed_args: object = json.loads(args_payload)
        except json.JSONDecodeError:
            return {}, "json_decode_error", args_payload
    elif isinstance(args_payload, dict):
        parsed_args = dict(args_payload)
    else:
        parsed_args = args_payload

    if isinstance(parsed_args, dict):
        return dict(parsed_args), None, None
    if isinstance(parsed_args, list):
        if len(parsed_args) == 1 and isinstance(parsed_args[0], dict):
            return dict(parsed_args[0]), "wrapped_single_item_list", parsed_args
        return {}, "unsupported_list_arguments", parsed_args
    return {}, f"unsupported_{type(parsed_args).__name__}_arguments", parsed_args


def _parse_tool_use_log(
    path: Path,
    element_index: Dict[str, Dict[str, object]],
    page_catalog: Dict[int, Dict[str, object]],
    *,
    interpolate_ancestors: bool = False,
) -> Dict[str, object]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return {
            "format": "gpt_web_tools",
            "breadcrumbs": ["/"],
            "click_sequence": [],
            "actions": [],
            "stopped": False,
            "current_page": None,
        }

    messages = json.loads(lines[-1])
    if not isinstance(messages, list):
        raise ValueError(f"Unexpected message log structure in {path}")

    breadcrumbs: List[str] = []
    click_sequence: List[Dict[str, object]] = []
    actions: List[Dict[str, object]] = []
    stopped = False

    current_page_index: Optional[int] = 0
    current_breadcrumb = _breadcrumb_for(page_catalog, current_page_index)
    _append_breadcrumb(breadcrumbs, current_breadcrumb)

    pending_click: Optional[Dict[str, object]] = None
    pending_nav: Optional[Dict[str, object]] = None

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            tool_calls = message.get("tool_calls") or []
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                func = tool_call.get("function") or {}
                tool_name = func.get("name")
                args_payload = func.get("arguments", "{}")
                parsed_args, argument_warning, raw_arguments = _normalize_tool_arguments(
                    args_payload
                )

                action_entry = {
                    "tool": tool_name,
                    "arguments": parsed_args,
                    "page_before": current_page_index,
                }
                if argument_warning is not None:
                    action_entry["argument_warning"] = argument_warning
                    if raw_arguments is not None:
                        action_entry["raw_arguments"] = raw_arguments

                if tool_name == "click":
                    element_id = str(parsed_args.get("element_id", ""))
                    action_entry["element_id"] = element_id
                    if element_id:
                        target = element_index.get(element_id)
                        if target:
                            action_entry["expected_target"] = target.get("target_breadcrumb")
                        pending_click = {
                            "element_id": element_id,
                            "source_page": current_page_index,
                        }
                        if target:
                            pending_click["expected_target"] = target.get("target_breadcrumb")
                    else:
                        pending_click = None
                elif tool_name == "goto":
                    url = parsed_args.get("url")
                    action_entry["url"] = url
                    pending_nav = {"type": "goto", "url": url} if url else None
                elif tool_name == "go_back":
                    pending_nav = {"type": "go_back"}
                elif tool_name == "stop":
                    stopped = True
                    pending_click = None
                    pending_nav = None
                else:
                    pending_nav = None

                actions.append(action_entry)
        elif role == "tool":
            content = message.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            if content.lstrip().startswith("ERROR:"):
                pending_click = None
                pending_nav = None
                continue
            new_page_index = _detect_page_index(content)
            if new_page_index is None:
                pending_click = None
                pending_nav = None
                continue
            if current_page_index == new_page_index:
                pending_click = None
                pending_nav = None
                continue

            source_index = current_page_index
            source_breadcrumb = _breadcrumb_for(page_catalog, source_index)
            current_page_index = new_page_index
            current_breadcrumb = _breadcrumb_for(page_catalog, current_page_index)

            # Optionally interpolate intermediate ancestors if path jumps multiple levels
            if interpolate_ancestors and isinstance(source_breadcrumb, str) and isinstance(current_breadcrumb, str):
                src_parts = [p for p in source_breadcrumb.strip("/").split("/") if p]
                tgt_parts = [p for p in current_breadcrumb.strip("/").split("/") if p]
                if len(tgt_parts) > len(src_parts) + 1:
                    # insert /a/b/c -> /a/b/c/d -> ... -> /a/b/c/d/e/f
                    prefix = src_parts
                    for k in range(len(src_parts) + 1, len(tgt_parts)):
                        b = "/" + "/".join(tgt_parts[:k])
                        _append_breadcrumb(breadcrumbs, b)
                        click_sequence.append({
                            "source": "/" + "/".join(tgt_parts[:k-1]) if k-1 > 0 else "/",
                            "target": b,
                            "type": "interpolated",
                        })

            _append_breadcrumb(breadcrumbs, current_breadcrumb)

            transition: Dict[str, object] = {
                "source": source_breadcrumb,
                "target": current_breadcrumb,
            }
            if pending_click is not None:
                transition["type"] = "click"
                transition["via"] = pending_click.get("element_id")
                transition["source_page"] = pending_click.get("source_page")
                if "expected_target" in pending_click:
                    transition["expected_target"] = pending_click.get("expected_target")
                pending_click = None
            elif pending_nav is not None:
                transition.update(pending_nav)
                transition.setdefault("type", pending_nav.get("type"))
                pending_nav = None
            else:
                transition["type"] = "observation"

            click_sequence.append(transition)

    return {
        "format": "gpt_web_tools",
        "breadcrumbs": breadcrumbs,
        "click_sequence": click_sequence,
        "actions": actions,
        "stopped": stopped,
        "current_page": current_page_index,
        "total_actions": len(actions),
    }


def _extract_from_file(path: Path) -> Dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = _BREADCRUMB_RE.findall(text)
    counts = Counter(matches)
    return {
        "paths": matches,
        "path_counts": dict(counts),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=None,
        help="Base directory for experiment artefacts (default: web/runs/trap_site_mvp).",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        help="Optional experiment identifier appended under --experiment-root.",
    )
    parser.add_argument(
        "--logs",
        type=Path,
        default=None,
        help="Directory containing conversation_render_*.html files. Defaults to experiment_root/agent_logs_pre_injection.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path summarising per-task and global path usage. Defaults to experiment_root/path_stats.json.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        help="Optional glob pattern for log discovery. Defaults depend on --format.",
    )
    parser.add_argument(
        "--format",
        choices=("gpt_web_tools", "webarena"),
        default="gpt_web_tools",
        help="Log format to parse (tool-call JSONL vs. HTML render logs).",
    )
    parser.add_argument(
        "--page-metadata",
        type=Path,
        default=None,
        help="Path to page_metadata.json for resolving click targets (gpt_web_tools format). Defaults to experiment_root/page_metadata.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_root = resolve_experiment_root(args.experiment_root, args.experiment_id)
    args.logs = resolve_path_within(args.logs, root=base_root, relative="agent_logs_pre_injection")
    args.out = resolve_path_within(args.out, root=base_root, relative="path_stats.json")
    args.page_metadata = resolve_path_within(args.page_metadata, root=base_root, relative="page_metadata.json")

    log_dir = args.logs
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    glob_pattern = args.glob
    if not glob_pattern:
        glob_pattern = "trace_*.jsonl" if args.format == "gpt_web_tools" else "conversation_render_*.html"

    per_file: Dict[str, Dict[str, object]] = {}
    global_counter: Counter[str] = Counter()
    transition_counter: Counter[Tuple[str, str]] = Counter()

    element_index: Dict[str, Dict[str, object]] = {}
    page_catalog: Dict[int, Dict[str, object]] = {}
    if args.format == "gpt_web_tools":
        if not args.page_metadata.exists():
            raise FileNotFoundError(f"Page metadata file not found: {args.page_metadata}")
        element_index, _, page_catalog = _load_metadata(args.page_metadata)

    for file_path in sorted(log_dir.glob(glob_pattern)):
        if not file_path.is_file():
            continue
        if args.format == "gpt_web_tools":
            data = _parse_tool_use_log(
                file_path,
                element_index,
                page_catalog,
                interpolate_ancestors=True,
            )
            match = _TRACE_RE.match(file_path.name)
            if match:
                data["task_id"] = int(match.group("task"))
                trial_str = match.group("trial")
                if trial_str is not None:
                    data["trial"] = int(trial_str)
            breadcrumbs = data.get("breadcrumbs", [])
            for breadcrumb in breadcrumbs:
                if isinstance(breadcrumb, str):
                    global_counter[breadcrumb] += 1
            for transition in data.get("click_sequence", []):
                if not isinstance(transition, dict):
                    continue
                source = transition.get("source") or "?"
                target = transition.get("target") or "?"
                transition_counter[(str(source), str(target))] += 1
            breadcrumbs = [b for b in data.get("breadcrumbs", []) if isinstance(b, str)]
            data["paths"] = breadcrumbs
            data["path_counts"] = dict(Counter(breadcrumbs))
        else:
            data = _extract_from_file(file_path)
            per_file[file_path.name] = data
            global_counter.update(data["paths"])  # type: ignore[arg-type]
            continue

        per_file[file_path.name] = data

    payload = {
        "files": per_file,
        "global_path_counts": dict(global_counter),
    }
    if transition_counter:
        payload["global_transition_counts"] = {
            f"{source}->{target}": count for (source, target), count in transition_counter.items()
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
