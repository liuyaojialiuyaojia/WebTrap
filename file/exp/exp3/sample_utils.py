#!/usr/bin/env python3
"""Helpers for working with exp2 per-sample artifacts.

exp2 writes per-sample traces and run summaries under:
  <run_dir>/logs/samples/<case_id>/

This module provides a small, robust API for locating those artifacts.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_TRACE_SAMPLE_RE = re.compile(r"^trace_sample_(\d{3})\.jsonl$")


@dataclass(frozen=True)
class SampleArtifact:
    case_id: str
    sample_idx: int
    trace_path: Path
    run_summary_path: Path


def _load_json_maybe(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_run_status(run_summary_path: Path) -> str:
    payload = _load_json_maybe(run_summary_path)
    return str(payload.get("status") or "missing")


def list_samples(run_dir: Path, case_id: str) -> list[SampleArtifact]:
    samples_dir = (run_dir / "logs" / "samples" / case_id).resolve()
    if not samples_dir.exists():
        return []

    sampling_path = samples_dir / "sampling.json"
    if sampling_path.exists():
        payload = _load_json_maybe(sampling_path)
        rows = payload.get("samples") or []
        out: list[SampleArtifact] = []
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                try:
                    sample_idx = int(row.get("sample_idx"))
                except Exception:
                    continue

                trace_path_raw = str(row.get("trace_path") or "").strip()
                run_summary_path_raw = str(row.get("run_summary_path") or "").strip()
                trace_path = (
                    Path(trace_path_raw)
                    if trace_path_raw
                    else samples_dir / f"trace_sample_{sample_idx:03d}.jsonl"
                )
                run_summary_path = (
                    Path(run_summary_path_raw)
                    if run_summary_path_raw
                    else samples_dir / f"run_sample_{sample_idx:03d}.json"
                )

                if not trace_path.is_absolute():
                    trace_path = (samples_dir / trace_path).resolve()
                if not run_summary_path.is_absolute():
                    run_summary_path = (samples_dir / run_summary_path).resolve()

                out.append(
                    SampleArtifact(
                        case_id=case_id,
                        sample_idx=sample_idx,
                        trace_path=trace_path,
                        run_summary_path=run_summary_path,
                    )
                )
        return sorted(out, key=lambda sample: sample.sample_idx)

    out: list[SampleArtifact] = []
    for trace_path in sorted(samples_dir.glob("trace_sample_*.jsonl"), key=lambda p: p.name):
        match = _TRACE_SAMPLE_RE.match(trace_path.name)
        if not match:
            continue
        sample_idx = int(match.group(1))
        run_summary_path = samples_dir / f"run_sample_{sample_idx:03d}.json"
        out.append(
            SampleArtifact(
                case_id=case_id,
                sample_idx=sample_idx,
                trace_path=trace_path.resolve(),
                run_summary_path=run_summary_path.resolve(),
            )
        )
    return sorted(out, key=lambda sample: sample.sample_idx)

