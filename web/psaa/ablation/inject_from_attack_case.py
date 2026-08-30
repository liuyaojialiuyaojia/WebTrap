#!/usr/bin/env python3
"""Inject ablation attacker instructions into trap-site metadata (PSAA ablations)."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
STAGE2_DIR = REPO_ROOT / "exp" / "02_nested_builder"
if str(STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_DIR))

from experiment_paths import resolve_experiment_root, resolve_path_within
from psaa.ablation import (
    inertia_only,
    lure_only,
    payload_only,
    wo_inertia,
    wo_lure,
    wo_payload,
)
from psaa.ablation.prompt_spec import (
    VALID_PSAA_PROMPT_SPEC_VERSIONS,
    load_ablation_prompt_spec,
    resolve_ablation_prompt_spec_path,
)
from psaa.inject_from_attack_case import (
    _append_jsonl,
    _build_adjacency,
    _build_navigation_path_text,
    _discover_first_attacker_task,
    _extract_safety_instruction,
    _find_click_target_label,
    _generate_checked_injection_text,
    _index_pages,
    _is_breadcrumb_prefix,
    _load_json,
    _render_page_for_prompt,
    _render_selections_for_prompt,
    _resolve_page_index,
    _resolve_relpath_across_roots,
    _shortest_path,
    _write_json,
)

_ABLATION_VARIANTS = {
    "lure_only": lure_only,
    "inertia_only": inertia_only,
    "payload_only": payload_only,
    "wo_lure": wo_lure,
    "wo_inertia": wo_inertia,
    "wo_payload": wo_payload,
}


def _validate_ablation_variant(value: str | None) -> str:
    variant = str(value or "").strip().lower()
    if variant not in _ABLATION_VARIANTS:
        allowed = ", ".join(sorted(_ABLATION_VARIANTS))
        raise ValueError(
            f"Ablation injector only supports ablation variants. Allowed values: {allowed}."
        )
    return variant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=None)
    parser.add_argument("--experiment-id", type=str)
    parser.add_argument(
        "--attack-case",
        type=Path,
        default=None,
        help="Path to attack_case.json (default: <root>/attack_case.json).",
    )
    parser.add_argument(
        "--attacker-tasks",
        type=Path,
        default=None,
        help=(
            "Directory containing aligned attacker tasks (default: from attack_case "
            "or <root>/webarena_tasks_attacker)."
        ),
    )
    try:
        action_bool = argparse.BooleanOptionalAction  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover

        class _BoolAct(argparse.Action):
            def __call__(self, parser, namespace, values, option_string=None):
                setattr(namespace, self.dest, not option_string.startswith("--no-"))

        action_bool = _BoolAct  # type: ignore

    parser.add_argument(
        "--use-security-microtree",
        action=action_bool,
        default=True,
        help="Prefer <root>/security_microtree as injection source when present.",
    )
    parser.add_argument("--design", type=Path, default=None)
    parser.add_argument("--transitions", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument(
        "--mode",
        type=str,
        default="dynamic",
        choices=("dynamic", "static"),
        help="Injection mode (only 'dynamic' is supported).",
    )
    parser.add_argument("--inject-start-page-index", type=int, default=None)
    parser.add_argument("--inject-start-path", type=str, default=None)
    parser.add_argument(
        "--prompt-spec",
        type=Path,
        default=None,
        help="Path to PSAA prompt spec YAML (overrides --prompt-spec-version).",
    )
    parser.add_argument(
        "--prompt-spec-version",
        type=str,
        default="base",
        choices=VALID_PSAA_PROMPT_SPEC_VERSIONS,
        help=(
            "Named default PSAA prompt spec to use when --prompt-spec is omitted: "
            "base=web/psaa/psaa_v1.yaml, enhanced=web/psaa/psaa_v2.yaml "
            "(default: base)."
        ),
    )
    parser.add_argument(
        "--generation-log",
        type=Path,
        default=None,
        help="Path to write injection generation trace (JSONL).",
    )
    parser.add_argument("--out-design", type=Path, default=None)
    parser.add_argument("--out-transitions", type=Path, default=None)
    parser.add_argument("--out-metadata", type=Path, default=None)
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=sorted(_ABLATION_VARIANTS),
        help="Ablation variant to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.variant = _validate_ablation_variant(args.variant)
    base_root = resolve_experiment_root(args.experiment_root, args.experiment_id)

    spec_path = resolve_ablation_prompt_spec_path(
        args.prompt_spec,
        args.prompt_spec_version,
    )
    prompt_spec = load_ablation_prompt_spec(spec_path)

    args.attack_case = resolve_path_within(
        args.attack_case,
        root=base_root,
        relative="attack_case.json",
    )
    if not args.attack_case.is_file():
        raise FileNotFoundError(f"attack_case.json not found: {args.attack_case}")
    attack_case = _load_json(args.attack_case)

    attacker_tasks_dir = attack_case.get("attacker_tasks_dir")
    if args.attacker_tasks is None and attacker_tasks_dir:
        args.attacker_tasks = _resolve_relpath_across_roots(
            str(attacker_tasks_dir),
            WORKSPACE_ROOT,
            REPO_ROOT,
        )
    args.attacker_tasks = resolve_path_within(
        args.attacker_tasks,
        root=base_root,
        relative="webarena_tasks_attacker",
    )
    if not args.attacker_tasks.is_dir():
        raise FileNotFoundError(
            f"Attacker tasks directory not found: {args.attacker_tasks}"
        )

    site_root = base_root
    security_root = base_root / "security_microtree"
    if args.use_security_microtree and security_root.is_dir():
        site_root = security_root

    args.design = resolve_path_within(
        args.design,
        root=site_root,
        relative="website_designs.json",
    )
    args.transitions = resolve_path_within(
        args.transitions,
        root=site_root,
        relative="transitions.json",
    )
    args.metadata = resolve_path_within(
        args.metadata,
        root=site_root,
        relative="page_metadata.json",
    )

    static_root = resolve_path_within(None, root=base_root, relative="static")
    args.out_design = resolve_path_within(
        args.out_design,
        root=static_root,
        relative="design_injected.json",
    )
    args.out_transitions = resolve_path_within(
        args.out_transitions,
        root=static_root,
        relative="transitions_injected.json",
    )
    args.out_metadata = resolve_path_within(
        args.out_metadata,
        root=static_root,
        relative="page_metadata_injected.json",
    )
    args.generation_log = resolve_path_within(
        args.generation_log,
        root=base_root,
        relative="injection_generation.jsonl",
    )
    args.generation_log.parent.mkdir(parents=True, exist_ok=True)
    args.generation_log.write_text("", encoding="utf-8")

    def _emit_generation_event(event: str, payload: Mapping[str, Any]) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
        }
        record.update(payload)
        _append_jsonl(args.generation_log, record)

    try:
        designs = _load_json(args.design)
        if not isinstance(designs, list) or not designs:
            raise ValueError("Design file must contain a non-empty list.")
        design = designs[0]
        transitions = _load_json(args.transitions)
        metadata = _load_json(args.metadata)

        page_by_index, _breadcrumb_to_index = _index_pages(metadata)
        adjacency = _build_adjacency(metadata)

        microtree = attack_case.get("microtree") or {}
        if not isinstance(microtree, dict):
            microtree = {}
        microtree_id = str(microtree.get("id") or "attack_case")

        if args.mode != "dynamic":
            raise ValueError("PSAA ablation injector only supports --mode dynamic.")

        if args.inject_start_page_index is None and not args.inject_start_path:
            raise ValueError(
                "Dynamic mode requires --inject-start-page-index or --inject-start-path "
                "(injection start must be provided externally)."
            )

        task_path = _discover_first_attacker_task(args.attacker_tasks)
        attacker_task = _load_json(task_path)
        task_instruction = _extract_safety_instruction(attacker_task)

        anchor = microtree.get("anchor") or {}
        if not isinstance(anchor, dict):
            raise ValueError(
                "attack_case.microtree.anchor must be an object for dynamic mode."
            )
        if not isinstance(anchor.get("page_index"), int):
            raise ValueError(
                "attack_case.microtree.anchor.page_index must be an int for dynamic mode."
            )

        anchor_page_index = int(anchor["page_index"])
        anchor_page = page_by_index.get(anchor_page_index)
        if not isinstance(anchor_page, dict):
            raise ValueError(
                f"Anchor page_index {anchor_page_index} not present in page metadata."
            )

        anchor_breadcrumb = str(anchor.get("breadcrumb") or "").strip()
        if not anchor_breadcrumb:
            anchor_breadcrumb = str(anchor_page.get("breadcrumb") or "").strip()
        if not anchor_breadcrumb:
            raise ValueError("Unable to determine anchor breadcrumb for dynamic mode.")

        interactive_pages: set[int] = set()
        for idx, page in page_by_index.items():
            breadcrumb = str(page.get("breadcrumb") or "").strip() or "/"
            if _is_breadcrumb_prefix(anchor_breadcrumb, breadcrumb):
                interactive_pages.add(idx)
        blocked = set(interactive_pages)
        blocked.discard(anchor_page_index)

        inject_start_index = _resolve_page_index(
            metadata,
            args.inject_start_page_index,
            args.inject_start_path,
        )
        if inject_start_index in interactive_pages:
            raise ValueError(
                "Injection start is inside the interactive subtree rooted at the anchor "
                "breadcrumb; dynamic PSAA only injects on the navigation-tree side."
            )
        if inject_start_index not in page_by_index:
            raise ValueError(
                f"Injection start page_index {inject_start_index} not present in metadata."
            )

        path_to_anchor = _shortest_path(
            adjacency,
            start=inject_start_index,
            goal=anchor_page_index,
            blocked=blocked,
        )
        if len(path_to_anchor) < 2:
            raise ValueError(
                "Unable to derive a navigation splice page: injection start must be "
                "outside the interactive subtree and strictly before the anchor page."
            )
        injection_path = path_to_anchor[:-1]
        if not injection_path:
            raise ValueError("Derived empty injection path (unexpected).")

        variant_impl = _ABLATION_VARIANTS[args.variant]
        chain_len = len(injection_path)
        stage_plan = variant_impl.build_stage_plan(chain_len)
        stop_page_index = int(injection_path[stage_plan[-1][1]])
        inertia_steps = sum(1 for stage, _position in stage_plan if stage == "inertia")
        stage_positions = {stage: position for stage, position in stage_plan}

        events: list[dict[str, Any]] = []
        for stage_idx, (stage, chain_position) in enumerate(stage_plan):
            page_index = int(injection_path[chain_position])
            page = page_by_index.get(page_index)
            if not isinstance(page, dict):
                raise ValueError(
                    f"Injection page_index {page_index} not present in metadata."
                )

            remaining_page_sequence = [
                int(idx) for idx in path_to_anchor[chain_position:]
            ]
            if len(remaining_page_sequence) < 2:
                raise ValueError(
                    "Remaining navigation suffix is incomplete; expected the current "
                    "page and the anchor page."
                )

            if stage_idx + 1 < len(stage_plan):
                route_target_position = int(stage_plan[stage_idx + 1][1])
                if route_target_position > chain_position:
                    route_page_sequence = [
                        int(idx)
                        for idx in injection_path[
                            chain_position : route_target_position + 1
                        ]
                    ]
                else:
                    route_page_sequence = remaining_page_sequence
            else:
                # A lone Lure/Inertia must carry the complete remaining route to
                # the anchor, rather than pointing at a stage removed by the
                # ablation. For Payload at the final Full location this naturally
                # reduces to [payload, anchor].
                route_page_sequence = remaining_page_sequence

            if len(route_page_sequence) < 2:
                route_page_sequence = remaining_page_sequence

            next_page_index = int(route_page_sequence[1])
            goal_label = _find_click_target_label(page, target_page=next_page_index)
            goal_path_labels, goal_path_text = _build_navigation_path_text(
                page_by_index=page_by_index,
                page_sequence=route_page_sequence,
            )
            events.append(
                {
                    "page_index": page_index,
                    "stage": stage,
                    "chain_position": chain_position,
                    "next_page_index": next_page_index,
                    "goal_label": goal_label,
                    "goal_path_labels": goal_path_labels,
                    "goal_path_text": goal_path_text,
                }
            )

        _emit_generation_event(
            "run_start",
            {
                "attack_case": str(args.attack_case),
                "attacker_task_path": str(task_path),
                "prompt_spec_path": str(spec_path),
                "prompt_spec_version": prompt_spec.version,
                "variant": args.variant,
                "microtree_id": microtree_id,
                "inject_start_page_index": inject_start_index,
                "stop_page_index": stop_page_index,
                "anchor_page_index": anchor_page_index,
                "chain_len": chain_len,
                "inertia_steps": inertia_steps,
                "total_steps": len(events),
                "path_to_anchor": path_to_anchor,
                "injection_path": injection_path,
                "stage_positions": stage_positions,
                "planned_event_pages": [
                    {
                        "stage": event["stage"],
                        "page_index": event["page_index"],
                        "chain_position": event["chain_position"],
                    }
                    for event in events
                ],
                "task_instruction": task_instruction,
            },
        )

        previous_injection: str | None = None
        for step, event in enumerate(events):
            page_index = int(event["page_index"])
            page = page_by_index.get(page_index)
            if not isinstance(page, dict):
                continue

            stage = str(event["stage"])
            next_page_index = int(event["next_page_index"])
            goal_label = str(event["goal_label"] or "").strip()
            chain_position = int(event.get("chain_position") or 0)
            goal_path_labels = [
                str(label) for label in event.get("goal_path_labels") or []
            ]
            goal_path_text = str(event.get("goal_path_text") or "").strip()

            _emit_generation_event(
                "step_start",
                {
                    "step": step,
                    "variant": args.variant,
                    "stage": stage,
                    "page_index": page_index,
                    "next_page_index": next_page_index,
                    "goal_label": goal_label,
                    "chain_position": chain_position,
                    "goal_path_labels": goal_path_labels,
                    "goal_path_text": goal_path_text,
                },
            )

            page_text = _render_page_for_prompt(page)
            selections_text = _render_selections_for_prompt(page)

            def _on_attempt(attempt_data: dict[str, Any]) -> None:
                attempt_record: dict[str, Any] = {
                    "step": step,
                    "variant": args.variant,
                    "stage": stage,
                    "page_index": page_index,
                    "chain_position": chain_position,
                    "goal_label": goal_label,
                    "goal_path_text": goal_path_text,
                }
                attempt_record.update(attempt_data)
                _emit_generation_event("attempt_result", attempt_record)

            text, generation = _generate_checked_injection_text(
                spec=prompt_spec,
                stage=stage,
                page_text=page_text,
                goal_label=goal_label,
                goal_path_text=goal_path_text,
                task_instruction=task_instruction,
                selections_text=selections_text,
                previous_injection=previous_injection,
                attempt_callback=_on_attempt,
            )
            previous_injection = text

            _emit_generation_event(
                "step_end",
                {
                    "step": step,
                    "variant": args.variant,
                    "stage": stage,
                    "page_index": page_index,
                    "goal_path_text": goal_path_text,
                    "selected_injection": text,
                    "generation": generation,
                },
            )

            attack_id = (
                f"psaa::{args.variant}::{microtree_id}"
                f"::{inject_start_index}->{stop_page_index}"
                f"::{stage}::{step}"
            )
            injection_entry: dict[str, Any] = {
                "attack_id": attack_id,
                "text": text,
                "source": {
                    "attack_case": str(args.attack_case),
                    "attacker_task_path": str(task_path),
                    "mode": "dynamic",
                    "variant": args.variant,
                    "inject_start_page_index": inject_start_index,
                    "stop_page_index": stop_page_index,
                    "anchor_page_index": anchor_page_index,
                    "interactive_root_prefix": anchor_breadcrumb,
                },
                "show_breadcrumb": True,
                "psaa": {
                    "variant": args.variant,
                    "prompt_spec_version": prompt_spec.version,
                    "stage": stage,
                    "step": step,
                    "chain_len": chain_len,
                    "inertia_steps": inertia_steps,
                    "chain_position": chain_position,
                    "next_page_index": next_page_index,
                    "goal_label": goal_label,
                    "goal_path_labels": goal_path_labels,
                    "goal_path_text": goal_path_text,
                    "generation": generation,
                },
            }
            page.setdefault("injections", []).append(injection_entry)
            design.setdefault("trap_injections", []).append(
                {
                    "page_index": page_index,
                    "attack_id": attack_id,
                    "source": "attack_case",
                }
            )

        _write_json(args.out_design, [design])
        _write_json(args.out_transitions, transitions)
        _write_json(args.out_metadata, metadata)
        _emit_generation_event(
            "run_end",
            {
                "steps_completed": len(events),
                "variant": args.variant,
                "out_design": str(args.out_design),
                "out_transitions": str(args.out_transitions),
                "out_metadata": str(args.out_metadata),
            },
        )
    except Exception as exc:
        _emit_generation_event(
            "run_error",
            {
                "variant": args.variant,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


if __name__ == "__main__":
    main()
