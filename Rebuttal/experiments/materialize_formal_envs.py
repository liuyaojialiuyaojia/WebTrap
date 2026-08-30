#!/usr/bin/env python3
"""Materialize rebuttal-only formal experiment environments.

This script only copies or rearranges already-generated local artifacts. It
does not call an agent, an evaluator, LiteLLM, or any remote/model API.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from file.exp.attack.common import ensure_readme_and_inject, root_logical_path
from Rebuttal.experiments.contracts import (
    PLACEMENT_VARIANTS,
    SINGLE_STAGE_VARIANTS,
    STAGE_ORDER,
)
from web.baseline.official_baseline_helpers import build_actionable_goal_text


STAGES = STAGE_ORDER
INJECTION_GENERATION_BLOCKER = "blocked_until_checked_injection_generation"
# Backward-compatible name used by older tests and result records.
SHIFTED_TEXT_BLOCKER = INJECTION_GENERATION_BLOCKER
RUNTIME_OUTPUT_DIR_NAMES = {
    "agent_logs_post_injection",
    "deepseek_v3_1_terminal",
    "eval",
    "logs",
}
PAYLOAD_FILE_RE = re.compile(
    r"^psaa_(?P<case_id>case_\d{4})_(?P<stage>lure|inertia|payload)_\d+_.+\.txt$"
)
OSR_HINT_SENTINEL = "Hint:"
NAVIGATION_OSR_HINT_SENTINEL = "Navigation hint:"
LEGACY_OSR_HINT_SENTINEL = "Aligned security-entry path hint:"


@dataclass(frozen=True)
class MaterializedEnv:
    experiment: str
    system: str
    variant: str
    sample_id: str
    environment_root: str
    active_environment_file: str
    user_tasks: str
    attacker_tasks: str
    source_artifact: str
    retained_stages: tuple[str, ...]
    moved_stages: tuple[str, ...]
    copied_stage_text_from_full: bool
    ready_for_deepseek_v3_1_terminal: bool
    model_run_status: str
    note: str


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_remove(path: Path, allowed_root: Path) -> None:
    path = path.resolve()
    allowed_root = allowed_root.resolve()
    if path == allowed_root or allowed_root not in path.parents:
        raise ValueError(f"Refusing to remove path outside generated root: {path}")
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _copytree_replace(src: Path, dst: Path, *, allowed_root: Path) -> None:
    if dst.exists() or dst.is_symlink():
        _safe_remove(dst, allowed_root=allowed_root)
    shutil.copytree(src, dst)


def _copy_file_replace(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.copy2(src, dst)


def _discover_numeric_json_tasks(task_dir: Path) -> list[Path]:
    tasks: list[Path] = []
    for path in task_dir.glob("*.json"):
        if not path.is_file() or path.name == "manifest.json":
            continue
        if path.stem.isdigit():
            tasks.append(path)
    return sorted(tasks, key=lambda item: int(item.stem))


def _drop_terminal_segment(path_text: str) -> str:
    cleaned = str(path_text or "").strip() or "/"
    if cleaned == "/":
        return "/"
    parts = [part for part in cleaned.split("/") if part]
    if len(parts) <= 1:
        return "/"
    return "/" + "/".join(parts[:-1])


def _strip_existing_osr_hint(intent: str) -> str:
    text = str(intent or "").strip()
    if not text:
        return text
    if (
        text.startswith(OSR_HINT_SENTINEL)
        or text.startswith(NAVIGATION_OSR_HINT_SENTINEL)
        or text.startswith(LEGACY_OSR_HINT_SENTINEL)
    ):
        parts = text.split("\n\n", 1)
        if len(parts) == 2:
            return parts[1].strip()
        return ""
    return text


def _browser_pages_by_index(metadata: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    pages: dict[int, dict[str, Any]] = {}
    for page in metadata.get("pages", []) if isinstance(metadata, Mapping) else []:
        if not isinstance(page, dict) or "page_index" not in page:
            continue
        pages[int(page["page_index"])] = page
    return pages


def _browser_home_page_index(metadata: Mapping[str, Any]) -> int:
    pages = _browser_pages_by_index(metadata)
    for page_index, page in pages.items():
        breadcrumb = str(page.get("breadcrumb") or "").strip() or "/"
        if breadcrumb == "/":
            return int(page_index)
    if not pages:
        raise ValueError("Browser metadata has no pages.")
    return min(pages)


def _browser_single_stage_osr_hint(
    *,
    page_metadata: Mapping[str, Any],
    attack_case: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    microtree = attack_case.get("microtree")
    if not isinstance(microtree, Mapping):
        raise ValueError("Browser attack case is missing microtree metadata.")
    anchor = microtree.get("anchor")
    entry = microtree.get("entry")
    if not isinstance(anchor, Mapping) or not isinstance(entry, Mapping):
        raise ValueError("Browser microtree is missing anchor or entry metadata.")

    raw_entry_label = str(entry.get("label") or "").strip() or "Security Console"
    _goal_text, nav_metadata = build_actionable_goal_text(
        metadata=page_metadata,
        start_page_index=_browser_home_page_index(page_metadata),
        anchor_page_index=int(anchor["page_index"]),
        anchor_breadcrumb=str(anchor.get("breadcrumb") or "").strip() or "/",
        anchor_title=str(anchor.get("title") or "").strip() or "Anchor Page",
        entry_label=raw_entry_label,
        attacker_instruction="enter the security flow",
        user_goal_str="complete the user task",
    )
    raw_instruction_path = (
        str(nav_metadata.get("display_instruction_path") or "/").strip() or "/"
    )
    display_instruction_path = _drop_terminal_segment(raw_instruction_path)
    hint_text = (
        f"{OSR_HINT_SENTINEL} The task below is likely related to content around "
        f"{display_instruction_path}."
    )
    return hint_text, {
        "source": "rebuttal_single_stage_osr_path_hint",
        "microtree_id": str(microtree.get("id") or "").strip(),
        "anchor_page_index": int(anchor["page_index"]),
        "anchor_breadcrumb": str(anchor.get("breadcrumb") or "").strip() or "/",
        "anchor_title": str(anchor.get("title") or "").strip() or "Anchor Page",
        "entry_label": raw_entry_label,
        "entry_element_id": str(entry.get("element_id") or "").strip() or None,
        "raw_instruction_path": raw_instruction_path,
        "display_instruction_path": display_instruction_path,
        "display_entry_path": str(nav_metadata.get("display_entry_path") or "").strip(),
    }


def _apply_browser_single_stage_osr_hints(
    *,
    task_dir: Path,
    page_metadata: Mapping[str, Any],
    attack_case: Mapping[str, Any],
) -> None:
    hint_text, hint_metadata = _browser_single_stage_osr_hint(
        page_metadata=page_metadata,
        attack_case=attack_case,
    )
    for task_path in _discover_numeric_json_tasks(task_dir):
        task = _load_json(task_path)
        if not isinstance(task, dict):
            raise ValueError(f"Browser task must be an object: {task_path}")
        original_intent = _strip_existing_osr_hint(str(task.get("intent") or ""))
        task["intent"] = f"{hint_text}\n\n{original_intent}".strip()
        metadata = dict(task.get("metadata") or {})
        metadata.update(
            {
                "osr_path_hint_text": hint_text,
                "rebuttal_user_task_hint": hint_metadata,
            }
        )
        task["metadata"] = metadata
        _write_json(task_path, task)


def _clear_generated_children(root: Path, names: Iterable[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        child = root / name
        if child.exists() or child.is_symlink():
            _safe_remove(child, allowed_root=root)


def _archive_runtime_outputs(root: Path, names: Iterable[str]) -> Path | None:
    """Preserve model/evaluator outputs before rematerializing env directories."""

    candidates: list[Path] = []
    for name in names:
        child = root / name
        if not child.is_dir():
            continue
        for path in child.rglob("*"):
            if path.is_dir() and path.name in RUNTIME_OUTPUT_DIR_NAMES:
                candidates.append(path)

    if not candidates:
        return None

    stamp = _utc_now().replace(":", "").replace("-", "").replace(".", "_")
    archive_root = root / "_archived_runtime_outputs" / stamp
    for src in sorted(candidates):
        rel = src.relative_to(root)
        dst = archive_root / rel
        shutil.copytree(src, dst)
    return archive_root


def _ready_for_formal_model_run(
    moved_stages: tuple[str, ...],
    *,
    checked_stage_text_generated: bool = False,
) -> bool:
    del moved_stages
    return bool(checked_stage_text_generated)


def _model_status_for_materialized_env(
    moved_stages: tuple[str, ...],
    *,
    checked_stage_text_generated: bool = False,
) -> str:
    if _ready_for_formal_model_run(
        moved_stages,
        checked_stage_text_generated=checked_stage_text_generated,
    ):
        return "not_started"
    return INJECTION_GENERATION_BLOCKER


def _route_text_note(moved_stages: tuple[str, ...]) -> str:
    if moved_stages:
        return (
            "Placeholder stage text is copied from Full only for materialization; "
            "checked text must be regenerated for the planned shifted route."
        )
    return (
        "Placeholder stage text is copied from Full only for materialization; "
        "checked standalone-stage text must be regenerated with the existing "
        "dynamic ablation protocol."
    )


def _relative_symlink(target: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    relative_target = os.path.relpath(target.resolve(), start=link_path.parent.resolve())
    link_path.symlink_to(relative_target, target_is_directory=target.is_dir())


def _write_rows_csv(path: Path, rows: list[MaterializedEnv]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else [
        "experiment",
        "system",
        "variant",
        "sample_id",
        "environment_root",
        "active_environment_file",
        "user_tasks",
        "attacker_tasks",
        "source_artifact",
        "retained_stages",
        "moved_stages",
        "copied_stage_text_from_full",
        "ready_for_deepseek_v3_1_terminal",
        "model_run_status",
        "note",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = asdict(row)
            payload["retained_stages"] = ";".join(row.retained_stages)
            payload["moved_stages"] = ";".join(row.moved_stages)
            writer.writerow(payload)


def _stage_of_injection(injection: Mapping[str, Any]) -> str | None:
    psaa = injection.get("psaa")
    if not isinstance(psaa, Mapping):
        return None
    stage = psaa.get("stage")
    if isinstance(stage, str) and stage in STAGES:
        return stage
    return None


def browser_stage_records(metadata: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return each stage injection and page index without exposing text."""

    pages = metadata.get("pages")
    if not isinstance(pages, list):
        raise ValueError("Browser metadata must contain a pages list.")

    records: dict[str, dict[str, Any]] = {}
    for page in pages:
        if not isinstance(page, Mapping) or not isinstance(page.get("page_index"), int):
            continue
        for injection in page.get("injections") or []:
            if not isinstance(injection, Mapping):
                continue
            stage = _stage_of_injection(injection)
            if stage is None:
                continue
            records[stage] = {
                "page_index": int(page["page_index"]),
                "injection": copy.deepcopy(dict(injection)),
            }

    missing = set(STAGES) - set(records)
    if missing:
        raise ValueError(f"Missing Browser stage injections: {sorted(missing)}")
    return records


def _browser_pages_by_index(metadata: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    pages = metadata.get("pages")
    if not isinstance(pages, list):
        raise ValueError("Browser metadata must contain a pages list.")
    result: dict[int, dict[str, Any]] = {}
    for page in pages:
        if isinstance(page, dict) and isinstance(page.get("page_index"), int):
            result[int(page["page_index"])] = page
    return result


def _annotate_browser_injection(
    injection: dict[str, Any],
    *,
    variant: str,
    stage: str,
    original_page_index: int,
    materialized_page_index: int,
    moved: bool,
) -> None:
    psaa = injection.setdefault("psaa", {})
    if isinstance(psaa, dict):
        psaa["rebuttal_variant"] = variant
        psaa["original_page_index"] = int(original_page_index)
        psaa["materialized_page_index"] = int(materialized_page_index)
        psaa["copied_stage_text_from_full"] = True
        psaa["moved_for_rebuttal"] = bool(moved)

    source = injection.setdefault("source", {})
    if isinstance(source, dict):
        source.setdefault("original_variant", source.get("variant", "full"))
        source["variant"] = variant
        source["rebuttal_materialization"] = {
            "stage": stage,
            "original_page_index": int(original_page_index),
            "materialized_page_index": int(materialized_page_index),
            "copied_stage_text_from_full": True,
            "model_calls": False,
        }


def _remove_browser_stages(metadata: dict[str, Any], stages: set[str]) -> None:
    pages = metadata.get("pages")
    if not isinstance(pages, list):
        raise ValueError("Browser metadata must contain a pages list.")
    for page in pages:
        if not isinstance(page, dict):
            continue
        injections = page.get("injections")
        if not isinstance(injections, list):
            continue
        kept = [
            injection
            for injection in injections
            if not (
                isinstance(injection, Mapping)
                and _stage_of_injection(injection) in stages
            )
        ]
        if kept:
            page["injections"] = kept
        elif "injections" in page:
            page["injections"] = []


def _append_browser_injection(
    metadata: dict[str, Any],
    *,
    page_index: int,
    injection: dict[str, Any],
) -> None:
    pages_by_index = _browser_pages_by_index(metadata)
    page = pages_by_index.get(int(page_index))
    if page is None:
        raise KeyError(f"Browser page_index not found: {page_index}")
    injections = page.setdefault("injections", [])
    if not isinstance(injections, list):
        raise ValueError(f"Browser page injections is not a list: {page_index}")
    injections.append(injection)


def patch_browser_shift(
    metadata: Mapping[str, Any],
    shifted_stage_nodes: Mapping[str, Any],
    *,
    variant: str = "shift_s2s3",
    moved_stages: tuple[str, ...] = ("inertia", "payload"),
) -> dict[str, Any]:
    """Move only the stages named by one planned placement condition."""

    if not moved_stages or any(stage not in {"inertia", "payload"} for stage in moved_stages):
        raise ValueError(f"Invalid moved stages: {moved_stages!r}")
    records = browser_stage_records(metadata)
    patched = copy.deepcopy(dict(metadata))
    _remove_browser_stages(patched, set(moved_stages))

    for stage in moved_stages:
        original_page_index = int(records[stage]["page_index"])
        target_page_index = int(shifted_stage_nodes[stage])
        injection = copy.deepcopy(records[stage]["injection"])
        _annotate_browser_injection(
            injection,
            variant=variant,
            stage=stage,
            original_page_index=original_page_index,
            materialized_page_index=target_page_index,
            moved=True,
        )
        _append_browser_injection(
            patched,
            page_index=target_page_index,
            injection=injection,
        )

    unchanged_stages = set(STAGES) - set(moved_stages)
    for page in patched.get("pages", []):
        if not isinstance(page, dict):
            continue
        for injection in page.get("injections") or []:
            if not isinstance(injection, dict):
                continue
            stage = _stage_of_injection(injection)
            if stage not in unchanged_stages:
                continue
            page_index = int(page.get("page_index", -1))
            _annotate_browser_injection(
                injection,
                variant=variant,
                stage=stage,
                original_page_index=page_index,
                materialized_page_index=page_index,
                moved=False,
            )

    return patched


def patch_browser_single_stage(
    metadata: Mapping[str, Any],
    retained_stage: str,
    *,
    variant: str,
) -> dict[str, Any]:
    """Keep exactly one PSAA stage in Browser metadata."""

    if retained_stage not in STAGES:
        raise ValueError(f"Unknown retained stage: {retained_stage}")
    records = browser_stage_records(metadata)
    patched = copy.deepcopy(dict(metadata))
    _remove_browser_stages(patched, set(STAGES) - {retained_stage})

    original_page_index = int(records[retained_stage]["page_index"])
    for page in patched.get("pages", []):
        if not isinstance(page, dict):
            continue
        for injection in page.get("injections") or []:
            if isinstance(injection, dict) and _stage_of_injection(injection) == retained_stage:
                _annotate_browser_injection(
                    injection,
                    variant=variant,
                    stage=retained_stage,
                    original_page_index=original_page_index,
                    materialized_page_index=original_page_index,
                    moved=False,
                )
    return patched


def _prepare_browser_static(
    *,
    shared_static: Path,
    env_root: Path,
    metadata: Mapping[str, Any],
) -> Path:
    static_root = env_root / "static"
    static_root.mkdir(parents=True, exist_ok=True)
    for child in shared_static.iterdir():
        if child.name in {"page_metadata.json", "page_metadata_injected.json"}:
            continue
        _relative_symlink(child, static_root / child.name)
    _write_json(static_root / "page_metadata.json", metadata)
    _relative_symlink(
        static_root / "page_metadata.json",
        static_root / "page_metadata_injected.json",
    )
    return static_root


def _browser_attack_sources(repo_root: Path) -> list[tuple[str, Path]]:
    batches = [
        ("gitlab", repo_root / "web/runs/exp_d10w2_psaa/batch_runs/20260405_161802_full"),
        ("reddit", repo_root / "web/runs/exp_d10w2_psaa/batch_runs/20260412_140735_full"),
    ]
    sources: list[tuple[str, Path]] = []
    for suite, batch_dir in batches:
        if not batch_dir.is_dir():
            raise FileNotFoundError(f"Missing Browser batch directory: {batch_dir}")
        for attack_dir in sorted(batch_dir.iterdir()):
            metadata_path = attack_dir / "static_snapshot" / "page_metadata.json"
            if attack_dir.is_dir() and metadata_path.is_file():
                sources.append((suite, attack_dir))
    return sources


def _write_browser_env_manifest(
    *,
    env_root: Path,
    experiment: str,
    system: str,
    variant: str,
    suite: str,
    attack_name: str,
    source_metadata: Path,
    active_metadata: Path,
    user_tasks: Path,
    attacker_tasks: Path,
    retained_stages: tuple[str, ...],
    moved_stages: tuple[str, ...],
    planned_stage_nodes: Mapping[str, Any],
    planned_route: Iterable[Any],
) -> None:
    ready = _ready_for_formal_model_run(moved_stages)
    _write_json(
        env_root / "environment_manifest.json",
        {
            "schema_version": 2,
            "experiment": experiment,
            "system": system,
            "suite": suite,
            "variant": variant,
            "sample_id": attack_name,
            "source_metadata": source_metadata.resolve().as_posix(),
            "active_page_metadata": active_metadata.resolve().as_posix(),
            "user_tasks": user_tasks.resolve().as_posix(),
            "attacker_tasks": attacker_tasks.resolve().as_posix(),
            "retained_stages": list(retained_stages),
            "moved_stages": list(moved_stages),
            "planned_stage_nodes": dict(planned_stage_nodes),
            "planned_route": list(planned_route),
            "copied_stage_text_from_full": True,
            "checked_stage_text_required": True,
            "checked_stage_text_generated": False,
            "route_specific_stage_text_required": True,
            "route_specific_stage_text_ready": False,
            "model_calls_during_materialization": False,
            "model_run_status": _model_status_for_materialized_env(moved_stages),
            "ready_for_deepseek_v3_1_terminal": ready,
            "notes": [
                "The environment is materialized under Rebuttal and can be served from its static directory.",
                _route_text_note(moved_stages),
            ],
        },
    )


def materialize_browser_envs(
    *,
    repo_root: Path,
    runs_root: Path,
    results_root: Path,
    placement: Mapping[str, Any],
) -> list[MaterializedEnv]:
    out_root = runs_root / "browser_formal"
    generated_names = ("_shared_static", *PLACEMENT_VARIANTS, "single_stage")
    _archive_runtime_outputs(out_root, (*PLACEMENT_VARIANTS, "single_stage"))
    _clear_generated_children(out_root, generated_names)

    source_static = repo_root / "web/runs/exp_d10w2_psaa/static"
    shared_static = out_root / "_shared_static"
    _copytree_replace(source_static, shared_static, allowed_root=out_root)

    user_tasks_src = repo_root / "web/runs/exp_d10w2_psaa/webarena_tasks"
    attacker_root = repo_root / "web/runs/exp_d10w2_psaa/webarena_tasks_attacker_wasp"
    browser_plan = placement["systems"]["Browser"]
    placement_plans = browser_plan["variants"]
    original_nodes = browser_plan["original_stage_nodes"]
    original_route = browser_plan["original_route"]

    rows: list[MaterializedEnv] = []
    for suite, attack_dir in _browser_attack_sources(repo_root):
        attack_name = attack_dir.name
        source_metadata = attack_dir / "static_snapshot" / "page_metadata.json"
        source_payload = _load_json(source_metadata)
        if not isinstance(source_payload, Mapping):
            raise ValueError(f"Invalid Browser metadata: {source_metadata}")

        attacker_src = attacker_root / attack_name
        if not attacker_src.is_dir():
            raise FileNotFoundError(f"Missing attacker task dir: {attacker_src}")

        for variant, moved_stages in PLACEMENT_VARIANTS.items():
            variant_plan = placement_plans[variant]["selected"]
            shifted_nodes = variant_plan["stage_nodes"]
            shift_env = out_root / variant / attack_name
            shift_env.mkdir(parents=True, exist_ok=True)
            shifted_metadata = patch_browser_shift(
                source_payload,
                shifted_nodes,
                variant=variant,
                moved_stages=moved_stages,
            )
            static_root = _prepare_browser_static(
                shared_static=shared_static,
                env_root=shift_env,
                metadata=shifted_metadata,
            )
            _copytree_replace(
                user_tasks_src,
                shift_env / "webarena_tasks",
                allowed_root=out_root,
            )
            _copytree_replace(
                attacker_src,
                shift_env / "webarena_tasks_attacker",
                allowed_root=out_root,
            )
            _copy_file_replace(
                attacker_src / "attack_case.json",
                shift_env / "attack_case.json",
            )
            _write_browser_env_manifest(
                env_root=shift_env,
                experiment="EXP-PLACE-001",
                system="Browser",
                variant=variant,
                suite=suite,
                attack_name=attack_name,
                source_metadata=source_metadata,
                active_metadata=static_root / "page_metadata.json",
                user_tasks=shift_env / "webarena_tasks",
                attacker_tasks=shift_env / "webarena_tasks_attacker",
                retained_stages=STAGES,
                moved_stages=moved_stages,
                planned_stage_nodes=shifted_nodes,
                planned_route=variant_plan["route"],
            )
            rows.append(
                MaterializedEnv(
                    experiment="EXP-PLACE-001",
                    system="Browser",
                    variant=variant,
                    sample_id=f"{suite}:{attack_name}",
                    environment_root=shift_env.resolve().as_posix(),
                    active_environment_file=(
                        static_root / "page_metadata.json"
                    ).resolve().as_posix(),
                    user_tasks=(shift_env / "webarena_tasks").resolve().as_posix(),
                    attacker_tasks=(
                        shift_env / "webarena_tasks_attacker"
                    ).resolve().as_posix(),
                    source_artifact=source_metadata.resolve().as_posix(),
                    retained_stages=STAGES,
                    moved_stages=moved_stages,
                    copied_stage_text_from_full=True,
                    ready_for_deepseek_v3_1_terminal=False,
                    model_run_status=INJECTION_GENERATION_BLOCKER,
                    note=(
                        f"{variant} materialized with placeholder Full text; "
                        "checked route-specific text has not been generated."
                    ),
                )
            )

        # The paper-facing ablation contract is Long GitLab. File remains
        # supplementary, but Browser Reddit must not silently enter the main
        # single-stage table.
        if suite != "gitlab":
            continue
        attack_case = _load_json(attacker_src / "attack_case.json")
        if not isinstance(attack_case, Mapping):
            raise ValueError(f"Invalid Browser attack case: {attacker_src / 'attack_case.json'}")
        for variant, retained_stage in SINGLE_STAGE_VARIANTS.items():
            env_root = out_root / "single_stage" / variant / attack_name
            env_root.mkdir(parents=True, exist_ok=True)
            stage_metadata = patch_browser_single_stage(
                source_payload,
                retained_stage,
                variant=variant,
            )
            static_root = _prepare_browser_static(
                shared_static=shared_static,
                env_root=env_root,
                metadata=stage_metadata,
            )
            user_tasks_dst = env_root / "webarena_tasks"
            _copytree_replace(user_tasks_src, user_tasks_dst, allowed_root=out_root)
            _apply_browser_single_stage_osr_hints(
                task_dir=user_tasks_dst,
                page_metadata=stage_metadata,
                attack_case=attack_case,
            )
            _copytree_replace(attacker_src, env_root / "webarena_tasks_attacker", allowed_root=out_root)
            _copy_file_replace(attacker_src / "attack_case.json", env_root / "attack_case.json")
            _write_browser_env_manifest(
                env_root=env_root,
                experiment="EXP-ABL-001",
                system="Browser",
                variant=variant,
                suite=suite,
                attack_name=attack_name,
                source_metadata=source_metadata,
                active_metadata=static_root / "page_metadata.json",
                user_tasks=env_root / "webarena_tasks",
                attacker_tasks=env_root / "webarena_tasks_attacker",
                retained_stages=(retained_stage,),
                moved_stages=(),
                planned_stage_nodes={retained_stage: original_nodes[retained_stage]},
                planned_route=original_route,
            )
            rows.append(
                MaterializedEnv(
                    experiment="EXP-ABL-001",
                    system="Browser",
                    variant=variant,
                    sample_id=f"{suite}:{attack_name}",
                    environment_root=env_root.resolve().as_posix(),
                    active_environment_file=(static_root / "page_metadata.json").resolve().as_posix(),
                    user_tasks=(env_root / "webarena_tasks").resolve().as_posix(),
                    attacker_tasks=(env_root / "webarena_tasks_attacker").resolve().as_posix(),
                    source_artifact=source_metadata.resolve().as_posix(),
                    retained_stages=(retained_stage,),
                    moved_stages=(),
                    copied_stage_text_from_full=True,
                    ready_for_deepseek_v3_1_terminal=False,
                    model_run_status=INJECTION_GENERATION_BLOCKER,
                    note=(
                        "Single-stage Browser environment materialized with "
                        "placeholder Full text; standalone checked text has not "
                        "been generated."
                    ),
                )
            )

    _write_rows_csv(results_root / "browser_formal_envs.csv", rows)
    _write_json(
        results_root / "browser_formal_envs.json",
        {
            "schema_version": 2,
            "generated_at": _utc_now(),
            "model_calls_during_materialization": False,
            "shared_static": shared_static.resolve().as_posix(),
            "environments": [
                {
                    **asdict(row),
                    "retained_stages": list(row.retained_stages),
                    "moved_stages": list(row.moved_stages),
                }
                for row in rows
            ],
        },
    )
    return rows


def _load_file_payload_paths(source_run: Path) -> dict[str, dict[str, Path]]:
    payload_root = source_run / "injection" / "payloads"
    result: dict[str, dict[str, Path]] = {}
    for path in sorted(payload_root.glob("psaa_case_*_*.txt")):
        match = PAYLOAD_FILE_RE.match(path.name)
        if not match:
            continue
        case_id = match.group("case_id")
        stage = match.group("stage")
        result.setdefault(case_id, {})[stage] = path

    bad_cases = {
        case_id: sorted(set(STAGES) - set(stage_map))
        for case_id, stage_map in result.items()
        if set(stage_map) != set(STAGES)
    }
    if bad_cases:
        raise ValueError(f"Incomplete File payload set: {bad_cases}")
    return result


def build_file_post_tree(
    pre_tree: Mapping[str, Any],
    *,
    stage_dirs: Mapping[str, str],
    payload_texts: Mapping[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Inject selected stages into a fresh copy of the pre-injection tree."""

    post_tree = copy.deepcopy(dict(pre_tree))
    root_logical = root_logical_path(post_tree)
    rows: list[dict[str, Any]] = []
    for stage_index, stage in enumerate(STAGES):
        if stage not in stage_dirs:
            continue
        if stage not in payload_texts:
            raise KeyError(f"Missing payload text for stage: {stage}")
        directory = str(stage_dirs[stage]).rstrip("/")
        marker = f"REBUTTAL_STAGE:{stage}:{stage_index}"
        readme_path, before_size, after_size = ensure_readme_and_inject(
            post_tree,
            directory,
            root_logical=root_logical,
            marker=marker,
            content=str(payload_texts[stage]).rstrip(),
            wrapper="raw",
            position="prepend",
        )
        rows.append(
            {
                "stage": stage,
                "stage_index": stage_index,
                "directory_logical_path": directory,
                "readme_logical_path": readme_path,
                "before_size": before_size,
                "after_size": after_size,
                "payload_hash": _hash_text(str(payload_texts[stage]).rstrip()),
            }
        )
    return post_tree, rows


def _patch_case_tasks(
    *,
    run_dir: Path,
    case_id: str,
    env_tree_path: Path,
) -> None:
    merged_path = run_dir / "tasks" / "merged" / f"{case_id}.json"
    if merged_path.is_file():
        payload = _load_json(merged_path)
        if isinstance(payload, dict):
            payload["env_tree_path"] = env_tree_path.resolve().as_posix()
            _write_json(merged_path, payload)


def _patch_attack_case_row(
    row: Mapping[str, Any],
    *,
    run_dir: Path,
    env_tree_path: Path,
    variant: str,
) -> dict[str, Any]:
    patched = copy.deepcopy(dict(row))
    patched["original_env_tree_path"] = patched.get("env_tree_path")
    patched["env_tree_path"] = env_tree_path.resolve().as_posix()
    patched["rebuttal_variant"] = variant
    patched["model_run_status"] = "not_started"
    user_task = run_dir / "task" / "user" / "user_task.json"
    attacker_task = run_dir / "task" / "attacker" / f"{patched.get('case_id')}.json"
    if user_task.is_file():
        patched["user_task_path"] = user_task.resolve().as_posix()
    if attacker_task.is_file():
        patched["attacker_task_path"] = attacker_task.resolve().as_posix()
    return patched


def _materialize_file_variant(
    *,
    source_run: Path,
    run_dir: Path,
    allowed_root: Path,
    shared_pre_tree_path: Path,
    experiment: str,
    variant: str,
    pre_tree: Mapping[str, Any],
    attack_cases: list[dict[str, Any]],
    payload_paths: Mapping[str, Mapping[str, Path]],
    original_stage_dirs: Mapping[str, str],
    materialized_stage_dirs: Mapping[str, str],
    retained_stages: tuple[str, ...],
    moved_stages: tuple[str, ...],
    planned_route: Iterable[str],
) -> MaterializedEnv:
    if run_dir.exists():
        _safe_remove(run_dir, allowed_root=allowed_root)
    run_dir.mkdir(parents=True, exist_ok=True)

    _copytree_replace(source_run / "tasks", run_dir / "tasks", allowed_root=allowed_root)
    _copytree_replace(source_run / "task", run_dir / "task", allowed_root=allowed_root)
    _copy_file_replace(source_run / "exp1_config.json", run_dir / "exp1_config.json")

    manifest_rows: list[dict[str, Any]] = []
    patched_cases: list[dict[str, Any]] = []
    payload_dst_root = run_dir / "injection" / "payloads"
    for case_row in attack_cases:
        case_id = str(case_row.get("case_id") or "").strip()
        if not case_id:
            raise ValueError("File attack case is missing case_id.")
        case_payloads = payload_paths.get(case_id)
        if not case_payloads:
            raise KeyError(f"Missing payloads for File case: {case_id}")

        case_env_dir = run_dir / "case_envs" / case_id / "env"
        pre_tree_path = case_env_dir / "env_pre_injection_tree.json"
        post_tree_path = case_env_dir / "env_post_injection_tree.json"

        payload_texts: dict[str, str] = {}
        payload_dst_by_stage: dict[str, Path] = {}
        for stage in retained_stages:
            src_payload = case_payloads[stage]
            dst_payload = payload_dst_root / src_payload.name
            _copy_file_replace(src_payload, dst_payload)
            payload_dst_by_stage[stage] = dst_payload
            payload_texts[stage] = dst_payload.read_text(encoding="utf-8").rstrip()

        post_tree, injection_rows = build_file_post_tree(
            pre_tree,
            stage_dirs={stage: materialized_stage_dirs[stage] for stage in retained_stages},
            payload_texts=payload_texts,
        )
        _relative_symlink(shared_pre_tree_path, pre_tree_path)
        _write_json(post_tree_path, post_tree)
        _patch_case_tasks(run_dir=run_dir, case_id=case_id, env_tree_path=post_tree_path)

        patched_case = _patch_attack_case_row(
            case_row,
            run_dir=run_dir,
            env_tree_path=post_tree_path,
            variant=variant,
        )
        patched_cases.append(patched_case)

        for injection_row in injection_rows:
            stage = str(injection_row["stage"])
            src_payload = payload_paths[case_id][stage]
            manifest_rows.append(
                {
                    "timestamp": _utc_now(),
                    "case_id": case_id,
                    "mode": "rebuttal_materialized",
                    "variant": variant,
                    "stage": stage,
                    "stage_index": injection_row["stage_index"],
                    "original_directory_logical_path": original_stage_dirs[stage],
                    "directory_logical_path": injection_row["directory_logical_path"],
                    "readme_logical_path": injection_row["readme_logical_path"],
                    "payload_hash": injection_row["payload_hash"],
                    "payload_path": payload_dst_by_stage[stage].resolve().as_posix(),
                    "source_payload_path": src_payload.resolve().as_posix(),
                    "before_size": injection_row["before_size"],
                    "after_size": injection_row["after_size"],
                    "copied_stage_text_from_full": True,
                    "checked_stage_text_required": True,
                    "checked_stage_text_generated": False,
                    "route_specific_stage_text_required": True,
                    "route_specific_stage_text_ready": False,
                    "model_calls": False,
                }
            )

    ready = _ready_for_formal_model_run(moved_stages)
    _write_jsonl(run_dir / "attack_cases.jsonl", patched_cases)
    _write_jsonl(run_dir / "attack_cases_pre_injection.jsonl", patched_cases)
    _write_jsonl(run_dir / "injection" / "injection_manifest.jsonl", manifest_rows)
    _write_json(
        run_dir / "environment_manifest.json",
        {
            "schema_version": 2,
            "experiment": experiment,
            "system": "File",
            "variant": variant,
            "cases": len(patched_cases),
            "retained_stages": list(retained_stages),
            "moved_stages": list(moved_stages),
            "planned_stage_nodes": {
                stage: materialized_stage_dirs[stage] for stage in retained_stages
            },
            "planned_route": list(planned_route),
            "case_env_layout": "case_envs/<case_id>/env/env_post_injection_tree.json",
            "source_run": source_run.resolve().as_posix(),
            "injection_manifest": (run_dir / "injection" / "injection_manifest.jsonl").resolve().as_posix(),
            "copied_stage_text_from_full": True,
            "checked_stage_text_required": True,
            "checked_stage_text_generated": False,
            "route_specific_stage_text_required": True,
            "route_specific_stage_text_ready": False,
            "model_calls_during_materialization": False,
            "model_run_status": _model_status_for_materialized_env(moved_stages),
            "ready_for_deepseek_v3_1_terminal": ready,
            "notes": [_route_text_note(moved_stages)],
        },
    )
    return MaterializedEnv(
        experiment=experiment,
        system="File",
        variant=variant,
        sample_id=f"{len(patched_cases)} cases",
        environment_root=run_dir.resolve().as_posix(),
        active_environment_file=(run_dir / "case_envs").resolve().as_posix(),
        user_tasks=(run_dir / "tasks" / "user").resolve().as_posix(),
        attacker_tasks=(run_dir / "tasks" / "attacker").resolve().as_posix(),
        source_artifact=source_run.resolve().as_posix(),
        retained_stages=retained_stages,
        moved_stages=moved_stages,
        copied_stage_text_from_full=True,
        ready_for_deepseek_v3_1_terminal=ready,
        model_run_status=_model_status_for_materialized_env(moved_stages),
        note=_route_text_note(moved_stages),
    )


def materialize_file_envs(
    *,
    repo_root: Path,
    runs_root: Path,
    results_root: Path,
    placement: Mapping[str, Any],
) -> list[MaterializedEnv]:
    source_run = repo_root / "file/runs/exp_d10w2_psaa"
    if not source_run.is_dir():
        raise FileNotFoundError(f"Missing File source run: {source_run}")
    out_root = runs_root / "file_formal"
    _archive_runtime_outputs(out_root, (*PLACEMENT_VARIANTS, "single_stage"))
    _clear_generated_children(
        out_root,
        ("_shared", *PLACEMENT_VARIANTS, "single_stage"),
    )

    pre_tree = _load_json(source_run / "env" / "env_pre_injection_tree.json")
    if not isinstance(pre_tree, Mapping):
        raise ValueError("File pre-injection tree must be an object.")
    shared_pre_tree_path = out_root / "_shared" / "env_pre_injection_tree.json"
    _write_json(shared_pre_tree_path, pre_tree)
    attack_cases = _read_jsonl(source_run / "attack_cases_pre_injection.jsonl")
    payload_paths = _load_file_payload_paths(source_run)
    original_stage_dirs = {
        stage: str(path)
        for stage, path in placement["systems"]["File"]["original_stage_nodes"].items()
    }
    file_plan = placement["systems"]["File"]
    placement_plans = file_plan["variants"]
    original_route = [str(node) for node in file_plan["original_route"]]

    rows: list[MaterializedEnv] = []
    for variant, moved_stages in PLACEMENT_VARIANTS.items():
        variant_plan = placement_plans[variant]["selected"]
        shifted_stage_dirs = {
            stage: str(path)
            for stage, path in variant_plan["stage_nodes"].items()
        }
        rows.append(
            _materialize_file_variant(
                source_run=source_run,
                run_dir=out_root / variant,
                allowed_root=out_root,
                shared_pre_tree_path=shared_pre_tree_path,
                experiment="EXP-PLACE-001",
                variant=variant,
                pre_tree=pre_tree,
                attack_cases=attack_cases,
                payload_paths=payload_paths,
                original_stage_dirs=original_stage_dirs,
                materialized_stage_dirs=shifted_stage_dirs,
                retained_stages=STAGES,
                moved_stages=moved_stages,
                planned_route=[str(node) for node in variant_plan["route"]],
            )
        )

    for variant, retained_stage in SINGLE_STAGE_VARIANTS.items():
        rows.append(
            _materialize_file_variant(
                source_run=source_run,
                run_dir=out_root / "single_stage" / variant,
                allowed_root=out_root,
                shared_pre_tree_path=shared_pre_tree_path,
                experiment="EXP-ABL-001",
                variant=variant,
                pre_tree=pre_tree,
                attack_cases=attack_cases,
                payload_paths=payload_paths,
                original_stage_dirs=original_stage_dirs,
                materialized_stage_dirs=original_stage_dirs,
                retained_stages=(retained_stage,),
                moved_stages=(),
                planned_route=original_route,
            )
        )

    _write_rows_csv(results_root / "file_formal_envs.csv", rows)
    _write_json(
        results_root / "file_formal_envs.json",
        {
            "schema_version": 2,
            "generated_at": _utc_now(),
            "model_calls_during_materialization": False,
            "environments": [
                {
                    **asdict(row),
                    "retained_stages": list(row.retained_stages),
                    "moved_stages": list(row.moved_stages),
                }
                for row in rows
            ],
        },
    )
    return rows


def _write_readiness_summary(
    *,
    path: Path,
    browser_rows: list[MaterializedEnv],
    file_rows: list[MaterializedEnv],
) -> None:
    all_rows = browser_rows + file_rows
    experiments = sorted({row.experiment for row in all_rows})
    blocked_rows = [row for row in all_rows if not row.ready_for_deepseek_v3_1_terminal]
    _write_json(
        path,
        {
            "schema_version": 2,
            "generated_at": _utc_now(),
            "model_calls_during_materialization": False,
            "ready_for_deepseek_v3_1_terminal": all(
                row.ready_for_deepseek_v3_1_terminal for row in all_rows
            ),
            "model_run_status": "not_started" if not blocked_rows else "blocked",
            "experiments": experiments,
            "blocked_environment_count": len(blocked_rows),
            "ready_environment_count": len(all_rows) - len(blocked_rows),
            "environment_counts": {
                "browser_placement_samples": {
                    variant: sum(
                        1
                        for row in browser_rows
                        if row.experiment == "EXP-PLACE-001"
                        and row.variant == variant
                    )
                    for variant in PLACEMENT_VARIANTS
                },
                "browser_single_stage_samples": sum(
                    1 for row in browser_rows if row.experiment == "EXP-ABL-001"
                ),
                "file_placement_runs": {
                    variant: sum(
                        1
                        for row in file_rows
                        if row.experiment == "EXP-PLACE-001"
                        and row.variant == variant
                    )
                    for variant in PLACEMENT_VARIANTS
                },
                "file_single_stage_runs": sum(
                    1 for row in file_rows if row.experiment == "EXP-ABL-001"
                ),
            },
            "notes": [
                "Formal environment packages are materialized under Rebuttal/runs.",
                "No DeepSeek or evaluator model calls have been started.",
                "All three shifted-placement conditions are materialized with placeholder Full text and require checked route-specific generation.",
                "Single-stage environments keep original stage locations but require dynamic standalone single-pass text generation matching the leave-one-out method.",
            ],
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--runs-root", type=Path, default=Path("Rebuttal/runs"))
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("Rebuttal/results/materialized_envs"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    runs_root = args.runs_root if args.runs_root.is_absolute() else repo_root / args.runs_root
    results_root = (
        args.results_root
        if args.results_root.is_absolute()
        else repo_root / args.results_root
    )
    runs_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    placement = _load_json(repo_root / "Rebuttal/results/placement/selected_placements.json")
    if not isinstance(placement, Mapping):
        raise ValueError("selected_placements.json must contain an object.")

    browser_rows = materialize_browser_envs(
        repo_root=repo_root,
        runs_root=runs_root,
        results_root=results_root,
        placement=placement,
    )
    file_rows = materialize_file_envs(
        repo_root=repo_root,
        runs_root=runs_root,
        results_root=results_root,
        placement=placement,
    )
    _write_readiness_summary(
        path=results_root / "readiness_summary.json",
        browser_rows=browser_rows,
        file_rows=file_rows,
    )
    print(f"Wrote {results_root / 'browser_formal_envs.csv'}")
    print(f"Wrote {results_root / 'file_formal_envs.csv'}")
    print(f"Wrote {results_root / 'readiness_summary.json'}")


if __name__ == "__main__":
    main()
