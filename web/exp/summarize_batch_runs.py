#!/usr/bin/env python3
"""
Summarize a batch_runs/<batch_id> directory into:

| ASR-end-to-end | ASR-intermediate | SR |

The script searches recursively for `metrics_post_injection.json` files (as written
by Stage 05 when `METRICS_OUT` is set to `<out_dir>/metrics_post_injection.json`).

Default outputs (written into the given batch_runs directory):
  - batch_summary.md
  - batch_summary.json

Usage:
  python web/exp/summarize_batch_runs.py web/runs/<exp_id>/batch_runs/<batch_id>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class MetricRow:
    attack: str
    variant: Optional[str]
    group: str
    rel_id: str
    metrics_path: str
    user_pass: int
    user_total: int
    attack_pass: Optional[int]
    attack_total: Optional[int]
    intermediate_compromised: Optional[int]
    intermediate_total: Optional[int]
    user_pass_by_trial: list[int]
    attack_pass_by_trial: list[int]


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Unable to parse JSON {path}: {exc}") from exc


def to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            try:
                return int(round(float(value)))
            except ValueError:
                return None
    return None


def ratio(numer: Optional[int], denom: Optional[int]) -> Optional[float]:
    if numer is None or denom is None or denom == 0:
        return None
    return numer / denom


def fmt_ratio(numer: Optional[int], denom: Optional[int]) -> str:
    rate = ratio(numer, denom)
    if rate is None:
        return "N/A"
    return f"{numer}/{denom} = {rate:.4f}"


def infer_rel_id(batch_dir: Path, metrics_path: Path) -> str:
    rel = metrics_path.relative_to(batch_dir)
    parts = rel.parts
    # Common layouts:
    #   PSAA:     <attack>/metrics_post_injection.json
    #   baseline: <attack>/<format>/metrics_post_injection.json
    #   baseline-per-task: <attack>/<format>/per_task/<task_id>/metrics_post_injection.json
    if len(parts) > 3 and parts[-1] == "metrics_post_injection.json":
        return "/".join(parts[:-1])
    if len(parts) >= 3 and parts[-1] == "metrics_post_injection.json":
        return "/".join(parts[:2])
    if len(parts) >= 2 and parts[-1] == "metrics_post_injection.json":
        return parts[0]
    return str(rel.parent)


def extract_intermediate_compromised(payload: dict[str, Any]) -> Optional[int]:
    summary = payload.get("summary")
    if isinstance(summary, dict):
        intermediate_value = to_int(summary.get("intermediate_compromised"))
        if intermediate_value is not None:
            return intermediate_value

    step_by_step = payload.get("step_by_step")
    if isinstance(step_by_step, dict):
        if "total_compromised" in step_by_step:
            return to_int(step_by_step.get("total_compromised"))
        if "compromised" in step_by_step:
            value = to_int(step_by_step.get("compromised"))
            return value if value is not None else 0
        # wasp/webarena_prompt_injections/evaluator_step_by_step.py writes a
        # `value_counts().to_dict()` payload. When there are 0 compromised tasks,
        # the `compromised` key is absent (e.g. {"not_compromised": N}).
        if "not_compromised" in step_by_step or "outright_refusal" in step_by_step:
            return 0
    return None


def extract_intermediate_total(
    payload: dict[str, Any], user_total: Optional[int]
) -> Optional[int]:
    if user_total is not None and user_total > 0:
        return user_total
    step_by_step = payload.get("step_by_step")
    if not isinstance(step_by_step, dict):
        return None
    total = 0
    saw_any = False
    for count_value in step_by_step.values():
        value_int = to_int(count_value)
        if value_int is None:
            continue
        total += value_int
        saw_any = True
    return total if saw_any and total > 0 else None


def parse_run_layout(
    batch_dir: Path, metrics_path: Path
) -> tuple[str, Optional[str], str, str]:
    rel = metrics_path.relative_to(batch_dir)
    parts = rel.parts
    # Common layouts:
    #   PSAA:     <attack>/metrics_post_injection.json
    #   baseline: <attack>/<variant>/metrics_post_injection.json
    attack = parts[0] if parts else "unknown"
    variant = parts[1] if len(parts) >= 3 else None
    group = variant or "psaa"
    rel_id = infer_rel_id(batch_dir, metrics_path)
    return attack, variant, group, rel_id


def extract_passes_by_trial(utility: Any) -> list[int]:
    if not isinstance(utility, dict):
        return []
    details = utility.get("details")
    if not isinstance(details, list) or not details:
        return []

    trial_counts: list[int] = []
    for detail in details:
        if not isinstance(detail, dict):
            trial_counts.append(0)
            continue
        trials = detail.get("trials")
        trial_counts.append(len(trials) if isinstance(trials, list) else 0)

    common_trials = min(trial_counts) if trial_counts else 0
    if common_trials <= 0:
        return []

    passes_by_trial: list[int] = []
    for trial_index in range(common_trials):
        passed_count = 0
        for detail in details:
            if not isinstance(detail, dict):
                continue
            trials = detail.get("trials")
            if not isinstance(trials, list) or trial_index >= len(trials):
                continue
            trial = trials[trial_index]
            if isinstance(trial, dict) and bool(trial.get("passed")):
                passed_count += 1
        passes_by_trial.append(passed_count)

    return passes_by_trial


def parse_metrics_file(batch_dir: Path, metrics_path: Path) -> MetricRow:
    payload = read_json(metrics_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at top-level: {metrics_path}")

    attack, variant, group, rel_id = parse_run_layout(batch_dir, metrics_path)

    user = payload.get("user_utility")
    attacker = payload.get("attacker_utility")

    user_pass = to_int(user.get("total_scores")) if isinstance(user, dict) else None
    user_total = to_int(user.get("cnt_tasks")) if isinstance(user, dict) else None
    user_pass_by_trial = extract_passes_by_trial(user)

    attack_pass = (
        to_int(attacker.get("total_scores")) if isinstance(attacker, dict) else None
    )
    attack_total = to_int(attacker.get("cnt_tasks")) if isinstance(attacker, dict) else None
    attack_pass_by_trial = extract_passes_by_trial(attacker)

    intermediate_compromised = extract_intermediate_compromised(payload)
    intermediate_total = extract_intermediate_total(payload, user_total)

    return MetricRow(
        attack=attack,
        variant=variant,
        group=group,
        rel_id=rel_id,
        metrics_path=str(metrics_path),
        user_pass=user_pass or 0,
        user_total=user_total or 0,
        attack_pass=attack_pass,
        attack_total=attack_total,
        intermediate_compromised=intermediate_compromised,
        intermediate_total=intermediate_total,
        user_pass_by_trial=user_pass_by_trial,
        attack_pass_by_trial=attack_pass_by_trial,
    )


def sum_optional_int(values: list[Optional[int]]) -> Optional[int]:
    total = 0
    saw_any = False
    for value in values:
        if value is None:
            continue
        total += value
        saw_any = True
    return total if saw_any else None


def sum_rows(rows: list[MetricRow]) -> dict[str, Optional[int]]:
    return {
        "user_pass": sum(row.user_pass for row in rows),
        "user_total": sum(row.user_total for row in rows),
        "attack_pass": sum_optional_int([row.attack_pass for row in rows]),
        "attack_total": sum_optional_int([row.attack_total for row in rows]),
        "intermediate_compromised": sum_optional_int(
            [row.intermediate_compromised for row in rows]
        ),
        "intermediate_total": sum_optional_int([row.intermediate_total for row in rows]),
    }


def totals_to_rates(totals: dict[str, Optional[int]]) -> dict[str, Optional[float]]:
    return {
        "asr_end_to_end": ratio(totals.get("attack_pass"), totals.get("attack_total")),
        "asr_intermediate": ratio(
            totals.get("intermediate_compromised"),
            totals.get("intermediate_total"),
        ),
        "sr": ratio(totals.get("user_pass"), totals.get("user_total")),
    }


def format_trial_name(trial_index: int) -> str:
    return f"r{trial_index + 1:02d}"


def group_common_trials(rows: list[MetricRow], passes_attr: str) -> int:
    counts: list[int] = []
    for row in rows:
        passes = getattr(row, passes_attr)
        if not isinstance(passes, list) or not passes:
            continue
        counts.append(len(passes))
    return min(counts) if counts else 0


def sum_one_time_totals(
    rows: list[MetricRow], trial_index: int
) -> dict[str, Optional[int]]:
    user_pass: Optional[int] = None
    user_total: Optional[int] = None
    attack_pass: Optional[int] = None
    attack_total: Optional[int] = None

    user_pass_values: list[int] = []
    user_total_values: list[int] = []
    for row in rows:
        if 0 <= trial_index < len(row.user_pass_by_trial):
            user_pass_values.append(row.user_pass_by_trial[trial_index])
            user_total_values.append(row.user_total)
    if user_pass_values:
        user_pass = sum(user_pass_values)
        user_total = sum(user_total_values)

    attack_pass_values: list[int] = []
    attack_total_values: list[int] = []
    for row in rows:
        if 0 <= trial_index < len(row.attack_pass_by_trial):
            attack_pass_values.append(row.attack_pass_by_trial[trial_index])
            if row.attack_total is not None:
                attack_total_values.append(row.attack_total)
    if attack_pass_values:
        attack_pass = sum(attack_pass_values)
        attack_total = sum_optional_int([*attack_total_values])

    intermediate_compromised = None
    intermediate_total = None
    if trial_index == 0:
        intermediate_compromised = sum_optional_int(
            [row.intermediate_compromised for row in rows]
        )
        intermediate_total = sum_optional_int([row.intermediate_total for row in rows])

    return {
        "user_pass": user_pass,
        "user_total": user_total,
        "attack_pass": attack_pass,
        "attack_total": attack_total,
        "intermediate_compromised": intermediate_compromised,
        "intermediate_total": intermediate_total,
    }


def pick_best_trial(
    trials: list[dict[str, Any]],
    metric: str,
    objective: str,
) -> Optional[dict[str, Any]]:
    candidates = []
    for entry in trials:
        rates = entry.get("rates", {})
        value = rates.get(metric)
        if value is None:
            continue
        candidates.append((value, entry))
    if not candidates:
        return None

    if objective == "max":
        best_value = max(value for value, _ in candidates)
    elif objective == "min":
        best_value = min(value for value, _ in candidates)
    else:
        raise ValueError(f"Unknown objective: {objective}")

    for value, entry in candidates:
        if value == best_value:
            return entry
    return None


def compute_group_summaries(rows: list[MetricRow]) -> dict[str, Any]:
    groups: dict[str, list[MetricRow]] = {}
    for row in rows:
        groups.setdefault(row.group, []).append(row)

    out: dict[str, Any] = {}
    for group, group_rows in sorted(groups.items(), key=lambda kv: kv[0]):
        totals = sum_rows(group_rows)
        rates = totals_to_rates(totals)

        user_trials = group_common_trials(group_rows, "user_pass_by_trial")
        attack_trials = group_common_trials(group_rows, "attack_pass_by_trial")
        max_trials = max(user_trials, attack_trials)

        one_time_trials: list[dict[str, Any]] = []
        for trial_index in range(max_trials):
            trial_totals = sum_one_time_totals(group_rows, trial_index)
            one_time_trials.append(
                {
                    "trial_index": trial_index,
                    "trial_name": format_trial_name(trial_index),
                    "totals": trial_totals,
                    "rates": totals_to_rates(trial_totals),
                }
            )

        selected_by_asr_end_to_end = pick_best_trial(
            one_time_trials, metric="asr_end_to_end", objective="max"
        )

        out[group] = {
            "metrics_files": len(group_rows),
            "totals": totals,
            "rates": rates,
            "one_time": {
                "user_trials_common": user_trials,
                "attack_trials_common": attack_trials,
                "trials": one_time_trials,
                "selected": {
                    "by": "asr_end_to_end_max",
                    "trial": selected_by_asr_end_to_end,
                },
            },
        }
    return out


def render_markdown(
    batch_dir: Path, totals: dict[str, Optional[int]], rows: list[MetricRow]
) -> str:
    now = datetime.now(timezone.utc).isoformat()
    lines: list[str] = []
    lines.append("# Batch Summary")
    lines.append("")
    lines.append(f"- batch_runs: `{batch_dir}`")
    lines.append(f"- generated_at_utc: `{now}`")
    lines.append(f"- metrics_files: `{len(rows)}`")
    lines.append(
        "- note: `ASR-intermediate` is computed from step-by-step classification on"
        " canonical logs (one trial per task, usually `r01`), so it may show `N/A` for"
        " one-time trials beyond `r01`."
    )
    lines.append("")

    group_summaries = compute_group_summaries(rows)
    lines.append("## Group Summary (best-of-n)")
    lines.append("")
    lines.append("| group | ASR-end-to-end | ASR-intermediate | SR |")
    lines.append("|---|---|---|---|")
    for group, summary in group_summaries.items():
        group_totals = summary.get("totals", {})
        lines.append(
            "| "
            + group
            + " | "
            + fmt_ratio(group_totals.get("attack_pass"), group_totals.get("attack_total"))
            + " | "
            + fmt_ratio(
                group_totals.get("intermediate_compromised"),
                group_totals.get("intermediate_total"),
            )
            + " | "
            + fmt_ratio(group_totals.get("user_pass"), group_totals.get("user_total"))
            + " |"
        )
    lines.append("")

    lines.append("## Group Summary (one-time by trial index)")
    lines.append("")
    lines.append("| group | trial | ASR-end-to-end | ASR-intermediate | SR |")
    lines.append("|---|---:|---|---|---|")
    for group, summary in group_summaries.items():
        one_time = summary.get("one_time", {})
        for trial in one_time.get("trials", []):
            trial_totals = trial.get("totals", {})
            lines.append(
                "| "
                + group
                + " | "
                + trial.get("trial_name", "N/A")
                + " | "
                + fmt_ratio(trial_totals.get("attack_pass"), trial_totals.get("attack_total"))
                + " | "
                + fmt_ratio(
                    trial_totals.get("intermediate_compromised"),
                    trial_totals.get("intermediate_total"),
                )
                + " | "
                + fmt_ratio(trial_totals.get("user_pass"), trial_totals.get("user_total"))
                + " |"
            )
    lines.append("")

    lines.append("## Group Selected One-time Trial")
    lines.append("")
    for group, summary in group_summaries.items():
        selected = summary.get("one_time", {}).get("selected", {})
        selected_trial = selected.get("trial")
        if selected_trial is None:
            continue
        selected_totals = selected_trial.get("totals", {})
        lines.append(f"- {group}")
        lines.append(
            f"  - selected (by ASR-end-to-end max): {selected_trial.get('trial_name')}"
        )
        lines.append(
            "  - ASR-end-to-end: "
            + fmt_ratio(
                selected_totals.get("attack_pass"), selected_totals.get("attack_total")
            )
        )
        lines.append(
            "  - ASR-intermediate: "
            + fmt_ratio(
                selected_totals.get("intermediate_compromised"),
                selected_totals.get("intermediate_total"),
            )
        )
        lines.append(
            "  - SR: "
            + fmt_ratio(selected_totals.get("user_pass"), selected_totals.get("user_total"))
        )
    lines.append("")

    lines.append("## Overall (all metrics files)")
    lines.append("")
    lines.append("| ASR-end-to-end | ASR-intermediate | SR |")
    lines.append("|---|---|---|")
    lines.append(
        "| "
        + fmt_ratio(totals.get("attack_pass"), totals.get("attack_total"))
        + " | "
        + fmt_ratio(
            totals.get("intermediate_compromised"), totals.get("intermediate_total")
        )
        + " | "
        + fmt_ratio(totals.get("user_pass"), totals.get("user_total"))
        + " |"
    )
    lines.append("")
    lines.append("## Per Run")
    lines.append("")
    lines.append("| run | ASR-end-to-end | ASR-intermediate | SR |")
    lines.append("|---|---|---|---|")
    for row in sorted(rows, key=lambda r: r.rel_id):
        lines.append(
            "| "
            + row.rel_id
            + " | "
            + fmt_ratio(row.attack_pass, row.attack_total)
            + " | "
            + fmt_ratio(row.intermediate_compromised, row.intermediate_total)
            + " | "
            + fmt_ratio(row.user_pass, row.user_total)
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_runs", type=Path, help="Path to batch_runs/<batch_id>.")
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Output markdown path (default: <batch_runs>/batch_summary.md).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Output json path (default: <batch_runs>/batch_summary.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_dir = args.batch_runs.expanduser().resolve()
    if not batch_dir.is_dir():
        raise SystemExit(f"ERROR: not a directory: {batch_dir}")

    metrics_paths = sorted(batch_dir.rglob("metrics_post_injection.json"))
    rows: list[MetricRow] = []
    errors: list[str] = []
    for metrics_path in metrics_paths:
        try:
            rows.append(parse_metrics_file(batch_dir, metrics_path))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{metrics_path}: {exc}")

    totals = sum_rows(rows)
    md = render_markdown(batch_dir, totals, rows)

    out_md = (args.out_md or (batch_dir / "batch_summary.md")).resolve()
    out_json = (args.out_json or (batch_dir / "batch_summary.json")).resolve()

    generated_at = datetime.now(timezone.utc).isoformat()
    out_md.write_text(md, encoding="utf-8")
    out_json.write_text(
        json.dumps(
            {
                "batch_runs": str(batch_dir),
                "generated_at_utc": generated_at,
                "metrics_files": len(rows),
                "totals": totals,
                "rates": totals_to_rates(totals),
                "groups": compute_group_summaries(rows),
                "rows": [asdict(row) for row in rows],
                "errors": errors,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote: {out_md}")
    print(f"Wrote: {out_json}")
    if errors:
        print(
            f"WARNING: {len(errors)} metrics file(s) failed to parse; see errors in {out_json}"
        )


if __name__ == "__main__":
    main()

# example
# python web/exp/summarize_batch_runs.py web/runs/<exp_id>/batch_runs/<batch_id>
