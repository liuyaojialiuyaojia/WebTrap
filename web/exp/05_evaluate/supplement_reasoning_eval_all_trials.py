#!/usr/bin/env python3
"""Supplement per-trial reasoning evaluator outputs without mutating original run artefacts.

This script produces fresh trial-level reasoning-evaluator artefacts under
`agent_logs_post_injection/reasoning_eval_all_trials/`.

Rules:
- Preserve the reasoning evaluator's historical semantics.
- Stop evaluating a trial immediately after the first compromised step.
- Reuse existing canonical `r01` reasoning results when they already match the
  canonical trace selected by Stage 05.
- For runs that were originally evaluated by the historical action-only
  evaluator, recompute all trials with the reasoning evaluator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluator_step_by_step_reasoning as reason_eval  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_SUBDIR = "reasoning_eval_all_trials"
TMP_SUFFIX = ".tmp"


@dataclass(frozen=True)
class TraceRef:
    log_dir: Path
    trace_path: Path
    task_index: int
    trial_number: int

    @property
    def trial_name(self) -> str:
        return f"r{self.trial_number:02d}"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_rel(path: Path | str) -> str:
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(candidate)


def _ensure_stage05_env_defaults() -> None:
    local_hosts = os.environ.get("LOCAL_NO_PROXY_HOSTS", "127.0.0.1,localhost")
    for key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(key, "")
        os.environ[key] = f"{local_hosts},{current}" if current else local_hosts
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    if base_url:
        os.environ.setdefault("OPENAI_BASE_URL", base_url)
        os.environ.setdefault("OPENAI_API_BASE", base_url)


def _iter_log_dirs_from_args(
    log_dirs: Iterable[str], log_dir_list: Optional[Path]
) -> list[Path]:
    resolved: list[Path] = [Path(item).resolve() for item in log_dirs]
    if log_dir_list is not None:
        for line in log_dir_list.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            resolved.append(Path(line).resolve())
    unique = sorted({path for path in resolved})
    if not unique:
        raise ValueError("No log directories provided.")
    return unique


def _schema_from_canonical_csv(csv_path: Path) -> str:
    cols = tuple(pd.read_csv(csv_path, nrows=0).columns)
    if "injection_in_context_window" in cols:
        return "historical_gpt_web_tools"
    if "tool_name" in cols and "tool_call_id" in cols:
        return "reasoning_eval"
    return "unknown"


def _discover_traces(log_dir: Path) -> list[TraceRef]:
    traces: list[TraceRef] = []
    for trace_path in sorted(log_dir.glob("trace_*_r*.jsonl")):
        task_index = reason_eval._task_index_from_trace_path(trace_path)
        trial_number = int(trace_path.stem.rsplit("_r", 1)[1])
        traces.append(
            TraceRef(
                log_dir=log_dir,
                trace_path=trace_path.resolve(),
                task_index=task_index,
                trial_number=trial_number,
            )
        )
    return traces


def _load_manifest(log_dir: Path) -> dict[str, Any]:
    manifest_path = log_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _resolve_tasks_folder(manifest: dict[str, Any]) -> Path:
    args = manifest.get("args") or {}
    tasks_value = args.get("tasks")
    if not tasks_value:
        raise ValueError("Manifest does not include args.tasks")
    return (REPO_ROOT / str(tasks_value)).resolve()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_canonical_trial_map(log_dir: Path) -> dict[int, int]:
    canonical_dir = log_dir / "canonical_logs"
    mapping: dict[int, int] = {}
    for canonical_trace in sorted(canonical_dir.glob("trace_*.jsonl")):
        task_index = reason_eval._task_index_from_trace_path(canonical_trace)
        canon_hash = _file_sha256(canonical_trace)
        matches: list[int] = []
        for trial_trace in sorted(log_dir.glob(f"trace_{task_index}_r*.jsonl")):
            if _file_sha256(trial_trace) == canon_hash:
                matches.append(int(trial_trace.stem.rsplit("_r", 1)[1]))
        if len(matches) != 1:
            raise RuntimeError(
                "Expected exactly one trial trace matching canonical trace: "
                f"log_dir={log_dir} task_index={task_index} matches={matches}"
            )
        mapping[task_index] = matches[0]
    return mapping


def _load_canonical_reasoning_rows(log_dir: Path) -> dict[int, list[dict[str, Any]]]:
    csv_path = log_dir / "canonical_logs" / "action_classifications.csv"
    df = pd.read_csv(csv_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    grouped: dict[int, list[dict[str, Any]]] = {}
    for task_index, group in df.groupby("task_index", sort=True):
        ordered = group.sort_values("step_index", kind="stable")
        grouped[int(task_index)] = ordered.to_dict(orient="records")
    return grouped


def _rows_to_final_label(rows: list[dict[str, Any]]) -> str:
    labels = {str(row.get("classification_result") or "") for row in rows}
    if "compromised" in labels:
        return "compromised"
    if "outright_refusal" in labels:
        return "outright_refusal"
    return "not_compromised"


def _binary_label(raw_final_label: str) -> str:
    return "compromised" if raw_final_label == "compromised" else "not_compromised"


def _first_compromised_step(rows: list[dict[str, Any]]) -> tuple[Optional[int], Optional[int]]:
    for row in rows:
        if str(row.get("classification_result") or "") == "compromised":
            step_index = int(row["step_index"])
            return step_index, step_index + 1
    return None, None


def _augment_rows(
    rows: list[dict[str, Any]],
    *,
    trace: TraceRef,
    source_kind: str,
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    trace_rel = _repo_rel(trace.trace_path)
    for row in rows:
        record = dict(row)
        original_source_file = str(record.get("source_file") or trace_rel)
        record["trial_name"] = trace.trial_name
        record["trial_number"] = trace.trial_number
        record["source_kind"] = source_kind
        record["original_source_file"] = original_source_file
        record["source_file"] = trace_rel
        output_rows.append(record)
    return output_rows


def _load_all_actions(trace: TraceRef, legitimate_objective: str) -> list[reason_eval.ReasoningAction]:
    return reason_eval.load_reasoning_actions_for_trace(
        trace.trace_path,
        task_index=trace.task_index,
        legitimate_objective=legitimate_objective,
    )


def _build_trial_result(
    *,
    trace: TraceRef,
    raw_final_label: str,
    source_kind: str,
    legitimate_objective: str,
    all_actions: list[reason_eval.ReasoningAction],
    classified_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    first_idx, first_num = _first_compromised_step(classified_rows)
    final_tool_name = all_actions[-1].tool_name if all_actions else None
    return {
        "task_index": trace.task_index,
        "trial_name": trace.trial_name,
        "trial_number": trace.trial_number,
        "trace_path": _repo_rel(trace.trace_path),
        "legitimate_objective": legitimate_objective,
        "source_kind": source_kind,
        "raw_final_label": raw_final_label,
        "binary_final_label": _binary_label(raw_final_label),
        "first_compromised_step_index": first_idx,
        "first_compromised_step_number": first_num,
        "evaluated_action_count": len(classified_rows),
        "total_tool_action_count": len(all_actions),
        "final_tool_name": final_tool_name,
        "final_action_is_stop": final_tool_name == "stop",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _output_ready(output_dir: Path) -> bool:
    return (
        output_dir.is_dir()
        and (output_dir / "meta.json").exists()
        and (output_dir / "trial_results.json").exists()
        and (output_dir / "action_classifications.csv").exists()
    )


def _process_log_dir(
    *,
    log_dir: Path,
    classifier: reason_eval.ReasoningClassifier,
    out_subdir: str,
    skip_existing: bool,
) -> str:
    output_dir = log_dir / out_subdir
    if skip_existing and _output_ready(output_dir):
        return "skipped_existing"

    manifest = _load_manifest(log_dir)
    tasks_folder = _resolve_tasks_folder(manifest)
    canonical_csv = log_dir / "canonical_logs" / "action_classifications.csv"
    schema = _schema_from_canonical_csv(canonical_csv)
    canonical_trial_map = (
        _build_canonical_trial_map(log_dir) if schema == "reasoning_eval" else {}
    )
    canonical_rows = (
        _load_canonical_reasoning_rows(log_dir) if schema == "reasoning_eval" else {}
    )

    tmp_dir = log_dir / f"{out_subdir}{TMP_SUFFIX}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    trial_results: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    reused_trials = 0
    newly_evaluated_trials = 0

    try:
        for trace in _discover_traces(log_dir):
            legitimate_objective = reason_eval._load_legitimate_objective(
                tasks_folder,
                trace.task_index,
            )
            all_actions = _load_all_actions(trace, legitimate_objective)
            if (
                schema == "reasoning_eval"
                and canonical_trial_map.get(trace.task_index) == trace.trial_number
            ):
                raw_rows = canonical_rows.get(trace.task_index, [])
                raw_final_label = _rows_to_final_label(raw_rows)
                augmented_rows = _augment_rows(
                    raw_rows,
                    trace=trace,
                    source_kind="reused_r01",
                )
                reused_trials += 1
            else:
                _, classified_actions = reason_eval.classify_task_actions(all_actions, classifier)
                raw_rows = [asdict(action) for action in classified_actions]
                raw_final_label = _rows_to_final_label(raw_rows)
                augmented_rows = _augment_rows(
                    raw_rows,
                    trace=trace,
                    source_kind="new_eval",
                )
                newly_evaluated_trials += 1

            trial_results.append(
                _build_trial_result(
                    trace=trace,
                    raw_final_label=raw_final_label,
                    source_kind=augmented_rows[0]["source_kind"] if augmented_rows else ("reused_r01" if schema == "reasoning_eval" and canonical_trial_map.get(trace.task_index) == trace.trial_number else "new_eval"),
                    legitimate_objective=legitimate_objective,
                    all_actions=all_actions,
                    classified_rows=raw_rows,
                )
            )
            action_rows.extend(augmented_rows)

        trial_results.sort(key=lambda row: (row["task_index"], row["trial_number"]))
        action_rows.sort(
            key=lambda row: (
                int(row["task_index"]),
                int(row["trial_number"]),
                int(row["step_index"]),
            )
        )

        counts = Counter(row["binary_final_label"] for row in trial_results)
        meta = {
            "generated_at_utc": _now_utc(),
            "status": "completed",
            "log_dir": _repo_rel(log_dir),
            "output_dir": _repo_rel(output_dir),
            "canonical_schema": schema,
            "tasks_folder": _repo_rel(tasks_folder),
            "script": _repo_rel(Path(__file__)),
            "reasoning_evaluator": _repo_rel(SCRIPT_DIR / "evaluator_step_by_step_reasoning.py"),
            "model": classifier.model,
            "temperature": classifier.temperature,
            "max_tokens": classifier.max_tokens,
            "num_retries": classifier.num_retries,
            "counts": dict(counts),
            "total_trials": len(trial_results),
            "reused_trials": reused_trials,
            "newly_evaluated_trials": newly_evaluated_trials,
            "total_written_action_rows": len(action_rows),
            "canonical_trial_map": {
                str(task_index): f"r{trial_number:02d}"
                for task_index, trial_number in sorted(canonical_trial_map.items())
            },
        }

        _write_csv(tmp_dir / "action_classifications.csv", action_rows)
        _write_json(
            tmp_dir / "trial_results.json",
            {
                "generated_at_utc": meta["generated_at_utc"],
                "counts": dict(counts),
                "trials": trial_results,
            },
        )
        _write_json(tmp_dir / "meta.json", meta)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        tmp_dir.rename(output_dir)
        return "completed"
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", action="append", default=[], help="Target agent_logs_post_injection directory.")
    parser.add_argument("--log-dir-list", type=Path, help="File containing one log directory per line.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--out-subdir", type=str, default=OUTPUT_SUBDIR)
    parser.add_argument("--force", action="store_true", help="Recompute outputs even if the output subdir already exists.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    _ensure_stage05_env_defaults()
    log_dirs = _iter_log_dirs_from_args(args.log_dir, args.log_dir_list)
    classifier = reason_eval.ReasoningClassifier(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    results = Counter()
    for log_dir in log_dirs:
        print(f"[reasoning-supplement] processing {log_dir}", flush=True)
        status = _process_log_dir(
            log_dir=log_dir,
            classifier=classifier,
            out_subdir=args.out_subdir,
            skip_existing=not args.force,
        )
        results[status] += 1
        print(f"[reasoning-supplement] {status} {log_dir}", flush=True)

    print(dict(results), flush=True)


if __name__ == "__main__":
    main()
